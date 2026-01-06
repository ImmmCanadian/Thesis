import os
from collections import defaultdict
from typing import Optional
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)

from model_detector import ResLight10Detector
from model_detector_tsm import ResLight10DetectorTSM
from model_classifiers import MediaPipeLSTM, MediaPipeGRU, MediaPipeTCN
from dataset_and_pipelines import IPNTwoStageDataset, DEFAULT_SAMPLE_SIZE


# --- Detector LightningModule wrapper -------------------------------------
class DetectorLightningModule(pl.LightningModule):

    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 4e-5,
        class_weights=(2.0, 1.0),
        input_channels: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResLight10Detector(in_channels=input_channels, num_classes=2)

        # Weight order follows class indices: 0 = background (negative), 1 = gesture (positive)
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )

        self.train_metrics = nn.ModuleDict({
            "acc": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
        })

        self.val_metrics = nn.ModuleDict({
            "acc": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
        })

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clips, labels = batch
        logits = self.model(clips)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics 
        for name, metric in self.train_metrics.items():
            metric.update(preds, labels)
            self.log(
                f"train/{name}",
                metric,
                prog_bar=(name == "acc"),
                on_step=False,
                on_epoch=True,
                batch_size=labels.size(0),
            )

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        clips, labels = batch
        logits = self.model(clips)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        preds = torch.argmax(logits, dim=1)
        
        # Update metrics 
        for name, metric in self.val_metrics.items():
            metric.update(preds, labels)
            self.log(
                f"val/{name}",
                metric,
                prog_bar=(name == "acc"),
                on_step=False,
                on_epoch=True,
                batch_size=labels.size(0),
            )

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=labels.size(0),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# --- Classifier LightningModule wrapper -----------------------------------
class ClassifierLightningModule(pl.LightningModule):

    def __init__(
        self,
        num_classes: int = 14,
        model_type: str = "gru",
        input_size: int = 63,
        model_input_size: int = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 4e-5,
        class_weights: Optional[torch.Tensor] = None,
        optimizer_type: str = "sgd",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.model_input_size = model_input_size or input_size

        # Enable Input Projection Layer
        if input_size != self.model_input_size:
            self.input_proj = nn.Linear(input_size, self.model_input_size)
        else:
            self.input_proj = nn.Identity()

        # Build our temporal model
        self.model = self._build_model(
            model_type, self.model_input_size, hidden_size, num_layers, dropout, num_classes
        )
        
        
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Metrics
        def make_metrics():
            return nn.ModuleDict({
                "acc": MulticlassAccuracy(num_classes=num_classes, average="micro"),
                "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "iou": MulticlassJaccardIndex(num_classes=num_classes, average="macro"),
            })

        self.train_metrics = make_metrics()
        self.val_metrics = make_metrics()

    def forward(self, x):
        x = self.input_proj(x)
        return self.model(x)

    def _build_model(self, model_type, input_size, hidden_size, num_layers, dropout, num_classes):
        projection_dim: Optional[int] = None
        
        if input_size > 100: # i.e. not mediapipe
            projection_dim = hidden_size * 2
        else:
            projection_dim = None

        model_type = model_type.lower()
        if model_type == "gru":
            return MediaPipeGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=num_classes,
                projection_dim=projection_dim,
            )
        elif model_type == "lstm":
            return MediaPipeLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=num_classes,
                projection_dim=projection_dim,
            )
        elif model_type == "tcn":
            return MediaPipeTCN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_levels=num_layers,
                dropout=dropout,
                num_classes=num_classes,
                projection_dim=projection_dim,
            )
        else:
            raise ValueError(f"Unsupported temporal model type: {model_type}")

    def training_step(self, batch, batch_idx):
        feats, labels = batch
        logits = self(feats)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        preds = torch.argmax(logits, dim=1)

        # Log metrics
        for name, metric in self.train_metrics.items():
            metric.update(preds, labels)
            self.log(
                f"train/{name}",
                metric,
                prog_bar=(name == "acc"),
                on_step=False,
                on_epoch=True,
            )

        # Log loss
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        feats, labels = batch
        logits = self(feats)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        preds = torch.argmax(logits, dim=1)

        # Log metrics
        for name, metric in self.val_metrics.items():
            metric.update(preds, labels)
            self.log(
                f"val/{name}",
                metric,
                prog_bar=(name == "acc"),
                on_step=False,
                on_epoch=True,
            )

        # Log loss
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=labels.size(0),
        )

    def configure_optimizers(self):
        optimizer_type = self.hparams.get('optimizer_type', 'sgd').lower()
        
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": True,
                },
            }
        
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-4
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Choose 'sgd' or 'adam'.")


    # --- Metric logging helpers -------------------------------------------
    # Print logging summary for classifier epochs for better readability
    def _get_metrics_log_path(self):
        """Generate log filename and folder based on model configuration."""
        model_type = self.hparams.get('model_type', 'unknown')
        num_layers = self.hparams.get('num_layers', 0)
        hidden_size = self.hparams.get('hidden_size', 0)
        optimizer_name = self.hparams.get('optimizer_type', 0)
        # Folder and filename base
        folder_name = f"{model_type}_{num_layers}L_{hidden_size}HS_{optimizer_name}"
        log_dir = os.path.join("training_logs", folder_name)
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{folder_name}_metrics.txt"
        return os.path.join(log_dir, filename)
    
    def on_train_start(self):
        """Write training configuration header to metrics file."""
        log_path = self._get_metrics_log_path()
        
        # Check if file already exists (resuming training)
        file_exists = os.path.exists(log_path)
        
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                if not file_exists:
                    # Write header only for new files
                    from datetime import datetime
                    f.write(f"\n{'='*70}\n")
                    f.write(f"🚀 TRAINING LOG\n")
                    f.write(f"{'='*70}\n")
                    f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model Configuration:\n")
                    f.write(f"  Model Type: {self.hparams.get('model_type', 'unknown')}\n")
                    f.write(f"  Num Layers: {self.hparams.get('num_layers', 'N/A')}\n")
                    f.write(f"  Hidden Size: {self.hparams.get('hidden_size', 'N/A')}\n")
                    f.write(f"  Num Classes: {self.hparams.get('num_classes', 'N/A')}\n")
                    f.write(f"  Learning Rate: {self.hparams.get('lr', 'N/A')}\n")
                    f.write(f"  Dropout: {self.hparams.get('dropout', 'N/A')}\n")
                    f.write(f"{'='*70}\n\n")
                else:
                    # Mark resumption
                    from datetime import datetime
                    f.write(f"\n{'='*70}\n")
                    f.write(f"🔄 TRAINING RESUMED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*70}\n\n")
        except Exception as e:
            print(f"Warning: Could not write header to metrics file: {e}")

    def on_validation_epoch_end(self):
        """Pretty summary after each validation epoch."""
        if self.trainer.sanity_checking:
            return

        epoch = self.current_epoch
        
        # Prepare output lines
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"📊 EPOCH {epoch} SUMMARY")
        lines.append(f"{'='*70}")

        metrics_to_show = {
            'val/loss': 'Validation Loss',
            'val/recall': 'Average Recall', 
            'val/acc': 'Accuracy',
            'val/precision': 'Precision (avg)',
            'val/f1': 'F1 Score (avg)',
            'val/iou': 'IoU (avg)',
            'train/loss_epoch': 'Training Loss',
            'train/acc': 'Training Accuracy',
            'train/recall': 'Training Recall (avg)',
        }

        for metric_key, display_name in metrics_to_show.items():
            if metric_key in self.trainer.callback_metrics:
                value = self.trainer.callback_metrics[metric_key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if 'loss' in metric_key:
                    line = f"  {display_name:.<35} {value:.4f}"
                else:
                    line = f"  {display_name:.<35} {value:.4f} ({value*100:.2f}%)"
                lines.append(line)

        lines.append(f"{'='*70}\n")
        
        # Print to console
        for line in lines:
            print(line)
        
        # Write to file
        try:
            log_path = self._get_metrics_log_path()
            with open(log_path, 'a', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
        except Exception as e:
            print(f"Warning: Could not write metrics to file: {e}")
    
    def on_train_end(self):
        """Write final training summary to metrics file."""
        try:
            from datetime import datetime
            log_path = self._get_metrics_log_path()
            
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"TRAINING COMPLETED\n")
                f.write(f"{'='*70}\n")
                f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Epochs: {self.current_epoch + 1}\n")
                
                # Write best metrics if available
                if hasattr(self.trainer.checkpoint_callback, 'best_model_score'):
                    best_score = self.trainer.checkpoint_callback.best_model_score
                    if best_score is not None:
                        f.write(f"Best Validation Metric: {best_score:.4f}\n")
                
                f.write(f"{'='*70}\n\n")
                
            print(f"\nTraining metrics saved to: {log_path}")
        except Exception as e:
            print(f"Warning: Could not write training end summary: {e}")


# --- Two-stage data module -------------------------------------------------
class TwoStageDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_annotation_file: str,
        video_root: str,
        stage: str,
        val_annotation_file: Optional[str] = None,
        detector_window: int = 8,
        classifier_window: int = 30,
        stride: int = 5,
        feature_type: str = "mediapipe",
        cache_dir: Optional[str] = None,
        features_cache_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        include_background: bool = False,
        use_elastic_distortion: bool = False,
        val_split: float = 0.1,
        val_seed: int = 42,
        multiprocessing_context=None,
        use_extracted_frames: bool = False,
        frames_root: str = "data/frames",
    ):
        super().__init__()
        assert stage in {"detector", "classifier"}, "stage must be 'detector' or 'classifier'"
        self.stage = stage
        self.train_annotation = train_annotation_file
        self.val_annotation = val_annotation_file if val_annotation_file else None
        self.video_root = video_root
        self.detector_window = detector_window
        self.classifier_window = classifier_window
        self.stride = stride
        self.feature_type = feature_type
        self.features_cache_dir = features_cache_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_size = sample_size
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.include_background = include_background
        self.use_elastic_distortion = use_elastic_distortion
        self.val_split = max(0.0, val_split)
        self.val_seed = val_seed
        self.multiprocessing_context = multiprocessing_context
        self.use_extracted_frames = use_extracted_frames
        self.frames_root = frames_root
        
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.feature_dim = None
        self.num_classes = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        base_kwargs = dict(
            annotation_file=self.train_annotation,
            video_root=self.video_root,
            stage=self.stage,
            detector_window=self.detector_window,
            classifier_window=self.classifier_window,
            stride=self.stride,
            feature_type=self.feature_type,
            cache_dir=self.cache_dir,
            sample_size=self.sample_size,
            features_cache_dir=self.features_cache_dir,
            include_background=self.include_background,
            use_elastic_distortion=self.use_elastic_distortion,
            use_extracted_frames=self.use_extracted_frames,
            frames_root=self.frames_root,
        )

        if self.val_annotation is None:
            self.train_dataset = IPNTwoStageDataset(
                mode="train",
                max_samples=self.max_train_samples,
                **base_kwargs,
            )

            self.val_dataset = None
            samples = getattr(self.train_dataset, "samples", [])
            if samples and self.val_split > 0.0:
                video_to_indices = defaultdict(list)
                for idx, sample in enumerate(samples):
                    video_to_indices[sample["video"]].append(idx)

                videos = list(video_to_indices.keys())
                num_videos = len(videos)
                if num_videos >= 2:
                    generator = torch.Generator().manual_seed(self.val_seed)
                    perm = torch.randperm(num_videos, generator=generator).tolist()
                    requested_val = int(round(num_videos * self.val_split))
                    val_video_count = max(1, min(requested_val, num_videos - 1)) if self.val_split > 0 else 0

                    if val_video_count > 0:
                        val_video_indices = {videos[i] for i in perm[:val_video_count]}
                        val_indices = []
                        train_indices = []
                        for video, idxs in video_to_indices.items():
                            if video in val_video_indices:
                                val_indices.extend(idxs)
                            else:
                                train_indices.extend(idxs)

                        if val_indices and train_indices:
                            self.train_dataset.restrict_to_indices(train_indices)
                            self.val_dataset = IPNTwoStageDataset(
                                mode="val",
                                max_samples=self.max_val_samples,
                                subset_indices=val_indices,
                                **base_kwargs,
                            )
        else:
            self.train_dataset = IPNTwoStageDataset(
                mode="train",
                max_samples=self.max_train_samples,
                **base_kwargs,
            )
            self.val_dataset = IPNTwoStageDataset(
                annotation_file=self.val_annotation,
                video_root=self.video_root,
                stage=self.stage,
                detector_window=self.detector_window,
                classifier_window=self.classifier_window,
                stride=self.stride,
                mode="val",
                feature_type=self.feature_type,
                cache_dir=self.cache_dir,
                features_cache_dir=self.features_cache_dir,
                sample_size=self.sample_size,
                max_samples=self.max_val_samples,
                include_background=self.include_background,
                use_elastic_distortion=self.use_elastic_distortion,
                use_extracted_frames=self.use_extracted_frames,
                frames_root=self.frames_root,
            )

        self.feature_dim = getattr(self.train_dataset, "feature_dim", None)
        self.num_classes = getattr(self.train_dataset, "num_classes", None)
        
        self.class_weights = None
        if self.stage == "classifier" and hasattr(self.train_dataset, "get_class_distribution"):
            class_dist = self.train_dataset.get_class_distribution()
            if class_dist:
                beta = 0.999
                class_ids = sorted(class_dist.keys())
                counts = torch.tensor([class_dist[i] for i in class_ids], dtype=torch.float32)
                effective_num = 1.0 - torch.pow(beta, counts)
                effective_num = torch.clamp(effective_num, min=1e-7)
                weights = (1.0 - beta) / effective_num
                weights = weights / weights.sum() * len(class_ids)

                self.class_weights = weights

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            multiprocessing_context=self.multiprocessing_context,
        )

    def val_dataloader(self):
        if getattr(self, "val_dataset", None) is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            multiprocessing_context=self.multiprocessing_context,
        )