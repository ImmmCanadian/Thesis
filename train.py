import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import os
import warnings

# --- Environment and warning suppression ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', message='.*persistent_workers.*')
warnings.filterwarnings('ignore', message='.*SymbolDatabase.GetPrototype.*')


import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)

from lightning_modules import (
    DetectorLightningModule,
    ClassifierLightningModule,
    TwoStageDataModule,
)


# --- Trainer helpers -------------------------------------------------------
def _trainer_kwargs():
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


def _build_logger(stage: str, save_root: Path) -> TensorBoardLogger:
    return TensorBoardLogger(save_dir=str(save_root), name=stage)


def _build_progress_callback(refresh_rate: int, style: str) -> Optional[pl.callbacks.Callback]:
    if refresh_rate <= 0 or style == "none":
        return None
    return TQDMProgressBar(refresh_rate=refresh_rate, leave=True)


# --- CLI parsing -----------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train detector and classifier for two-stage gesture system."
    )
    
    # Data paths
    parser.add_argument("--train-annotations", default="annotations/Annot_TrainList.txt")
    parser.add_argument(
        "--val-annotations",
        default=None,
        help="Optional validation annotations file. When omitted, validation is derived from a split of the training annotations.",
    )
    parser.add_argument("--video-root", default="data/videos")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--output-dir", default="models", help="Directory to save trained models")
    
    # Pre-computed features cache
    parser.add_argument(
        "--features-cache-dir",
        type=str,
        default=None,
        help="Directory containing pre-extracted features (.npy files). "
             "Enables 50-100x faster training by loading features from disk "
             "instead of extracting on-the-fly. Use precompute_features.py first."
    )

    # Model architecture
    parser.add_argument("--feature-type", choices=["mediapipe", "yolo_mobilenet"], 
                       default="mediapipe", help="Feature extraction method for classifier")
    parser.add_argument("--temporal-model", choices=["lstm", "gru", "tcn"], 
                       default="lstm", help="Temporal model architecture")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-classes", type=int, default=14)

    # Dataset parameters
    parser.add_argument("--detector-window", type=int, default=8)
    parser.add_argument("--classifier-window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--detector-use-extracted-frames",
        action="store_true",
        help="Use pre-extracted frames from data/frames/ instead of loading from videos. "
             "Frames should be in data/frames/VIDEO_NAME/VIDEO_NAME_XXXXXX.jpg format. "
             "This significantly speeds up training by avoiding video file I/O."
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        default="data/frames",
        help="Root directory containing pre-extracted frames (default: data/frames)"
    )
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit number of training samples for quick smoke tests")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Limit number of validation samples for quick smoke tests")
    parser.add_argument("--progress-refresh-rate", type=int, default=20,
                        help="Number of batches between progress bar updates")
    parser.add_argument(
        "--classifier-include-background",
        action="store_true",
        help="Include background (no-gesture) windows when training the classifier",
    )
    parser.add_argument(
        "--progress-style",
        choices=["tqdm", "none", "auto", "rich"],
        default="tqdm",
        help="Progress bar implementation (rich styles fall back to tqdm).",
    )

    # Training parameters
    parser.add_argument("--detector-lr", type=float, default=0.1)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--max-detector-epochs", type=int, default=30)
    parser.add_argument("--max-classifier-epochs", type=int, default=50)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of training samples to allocate to validation when --val-annotations is not provided.",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=42,
        help="Random seed controlling the train/validation split when derived from training data.",
    )
    
    # Optimizer configuration
    parser.add_argument(
        "--optimizer-type",
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer type for classifier. 'sgd' uses SGD with momentum and CosineAnnealing scheduler (default). "
             "'adam' uses Adam with constant learning rate (paper's approach)."
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (only used when --optimizer-type=sgd)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=4e-5,
        help="Weight decay for optimizer"
    )
    
    # Early stopping
    parser.add_argument("--early-stopping", action="store_true",
                       help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")

    # Training control
    parser.add_argument("--skip-detector", action="store_true", 
                       help="Skip detector training stage")
    parser.add_argument("--skip-classifier", action="store_true",
                       help="Skip classifier training stage")
    parser.add_argument("--resume-detector", type=str, default=None,
                       help="Path to detector checkpoint to resume from")
    parser.add_argument("--resume-classifier", type=str, default=None,
                       help="Path to classifier checkpoint to resume from")
    
    return parser.parse_args()


# --- Orchestrator ----------------------------------------------------------
def train_two_stage_system(args: argparse.Namespace):
    """Train detector and classifier using PyTorch Lightning wrappers."""

    # Prepare cache folders and model output directories
    cache_root = Path(args.cache_root)
    detector_cache = cache_root / "detector"
    classifier_cache = cache_root / "classifier"
    detector_cache.mkdir(parents=True, exist_ok=True)
    classifier_cache.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_root = Path("lightning_logs")

    # --- Detector training loop -------------------------------------------
    if not args.skip_detector:
        print("\n" + "=" * 60)
        print("TRAINING DETECTOR")
        print("=" * 60)

        # Build detector datamodule and report dataset sizes
        detector_datamodule = TwoStageDataModule(
            train_annotation_file=args.train_annotations,
            val_annotation_file=args.val_annotations,
            video_root=args.video_root,
            stage="detector",
            detector_window=args.detector_window,
            classifier_window=args.classifier_window,
            stride=args.stride,
            cache_dir=str(detector_cache),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            val_split=args.val_split,
            val_seed=args.val_seed,
            use_extracted_frames=args.detector_use_extracted_frames,
            frames_root=args.frames_root,
        )
        
        detector_datamodule.setup("fit")
        detector_train_count = len(detector_datamodule.train_dataset)
        detector_val_count = (
            len(detector_datamodule.val_dataset)
            if detector_datamodule.val_dataset is not None
            else 0
        )
        print(f"Detector samples -> train: {detector_train_count}, val: {detector_val_count}")
        
        # Initialize detector Lightning module
        detector_module = DetectorLightningModule(
            lr=args.detector_lr,
        )
        detector_progress = _build_progress_callback(
            args.progress_refresh_rate, args.progress_style
        )

        # Configure callbacks and logging
        callbacks = []
        if detector_progress is not None:
            callbacks.append(detector_progress)

        detector_ckpt_dir = output_dir / "detector"
        detector_ckpt_dir.mkdir(parents=True, exist_ok=True)

        detector_best_ckpt = ModelCheckpoint(
            dirpath=detector_ckpt_dir,
            filename="detector-best_{epoch:02d}_{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        )
        detector_epoch_ckpt = ModelCheckpoint(
            dirpath=detector_ckpt_dir,
            filename="detector-epoch{epoch:02d}",
            save_top_k=-1,
            every_n_epochs=1,
        )

        callbacks.extend([detector_best_ckpt, detector_epoch_ckpt, LearningRateMonitor(logging_interval='epoch')])
        
        if args.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    patience=args.patience,
                    mode="min",
                )
            )
        
        detector_logger = _build_logger("detector", log_root)
        detector_trainer = pl.Trainer(
            max_epochs=args.max_detector_epochs,
            default_root_dir="lightning_logs/detector",
            log_every_n_steps=args.log_frequency,
            logger=detector_logger,
            callbacks=callbacks,
            enable_progress_bar=detector_progress is not None,
            **_trainer_kwargs(),
        )
        
        if args.resume_detector:
            detector_trainer.fit(
                detector_module, 
                datamodule=detector_datamodule,
                ckpt_path=args.resume_detector
            )
        else:
            detector_trainer.fit(detector_module, datamodule=detector_datamodule)
        
        detector_trainer.save_checkpoint(output_dir / "detector_final.ckpt")
        torch.save(detector_module.model.state_dict(), output_dir / "detector_final.pth")
        print(f"\nDetector saved to {output_dir / 'detector_final.pth'}")

    # --- Classifier training loop ----------------------------------------
    if not args.skip_classifier:
        print("\n" + "=" * 60)
        print("TRAINING CLASSIFIER")
        print("=" * 60)
        print(f"Feature type: {args.feature_type}")
        print(f"Temporal model: {args.temporal_model}")
        print(f"Optimizer: {args.optimizer_type.upper()}")
        if args.optimizer_type == "sgd":
            print(f"  - Learning rate: {args.classifier_lr} (with CosineAnnealing scheduler)")
            print(f"  - Momentum: {args.momentum}")
        else:  # adam
            print(f"  - Learning rate: {args.classifier_lr} (constant)")
        print(f"  - Weight decay: {args.weight_decay}")
        
        if args.features_cache_dir:
            print(f"Features cache: {args.features_cache_dir}")
        else:
            print("No features cache provided; extracting on-the-fly.")

        # Use spawn context when workers rely on CUDA/MediaPipe
        classifier_mp_context = None
        if args.num_workers > 0 and args.feature_type in ["mediapipe", "yolo_mobilenet"]:
            import multiprocessing
            classifier_mp_context = multiprocessing.get_context('spawn')
        
        # Build classifier datamodule and gather metadata
        classifier_datamodule = TwoStageDataModule(
            train_annotation_file=args.train_annotations,
            val_annotation_file=args.val_annotations,
            video_root=args.video_root,
            stage="classifier",
            detector_window=args.detector_window,
            classifier_window=args.classifier_window,
            stride=args.stride,
            feature_type=args.feature_type,
            cache_dir=str(classifier_cache),
            features_cache_dir=args.features_cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            include_background=args.classifier_include_background,
            val_split=args.val_split,
            val_seed=args.val_seed,
            multiprocessing_context=classifier_mp_context
        )
        
        classifier_datamodule.setup("fit")
        classifier_train_count = len(classifier_datamodule.train_dataset)
        classifier_val_count = (
            len(classifier_datamodule.val_dataset)
            if classifier_datamodule.val_dataset is not None
            else 0
        )
        print(f"Classifier samples -> train: {classifier_train_count}, val: {classifier_val_count}")
        feature_dim = getattr(classifier_datamodule, "feature_dim", 63) or 63
        detected_num_classes = getattr(classifier_datamodule, "num_classes", None)
        classifier_num_classes = args.num_classes
        if detected_num_classes:
            classifier_num_classes = detected_num_classes
            if detected_num_classes != args.num_classes:
                print(
                    f"Detected {detected_num_classes} classes from dataset; overriding CLI num_classes={args.num_classes}."
                )
        print(f"Feature dimension: {feature_dim}")
        
        class_weights = getattr(classifier_datamodule, "class_weights", None)
        if class_weights is not None:
            print("Using class weights to handle imbalanced data")
        
        classifier_dropout = args.dropout
        if args.temporal_model == "gru" and args.feature_type == "yolo_mobilenet" and args.optimizer_type == "adam":
            classifier_dropout = 0.3

        # Initialize classifier Lightning module
        classifier_module = ClassifierLightningModule(
            num_classes=classifier_num_classes,
            lr=args.classifier_lr,
            model_type=args.temporal_model,
            input_size=feature_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=classifier_dropout,
            class_weights=class_weights,  
            optimizer_type=args.optimizer_type,  
            momentum=args.momentum,  
            weight_decay=args.weight_decay,  
        )
        classifier_progress = _build_progress_callback(
            args.progress_refresh_rate, args.progress_style
        )

        # Configure callbacks and logging
        callbacks = []
        if classifier_progress is not None:
            callbacks.append(classifier_progress)

        metrics_base = f"{args.temporal_model}_{args.num_layers}L_{args.hidden_size}HS_{args.optimizer_type}"
        classifier_subdir = output_dir / "classifier"
        os.makedirs(classifier_subdir, exist_ok=True)
        best_classifier_ckpt = ModelCheckpoint(
            dirpath=classifier_subdir,
            filename=f"{metrics_base}_best_epoch{{epoch:02d}}_loss{{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        )
        epoch_classifier_ckpt = ModelCheckpoint(
            dirpath=classifier_subdir,
            filename=f"{metrics_base}_epoch{{epoch:02d}}",
            save_top_k=-1,
            every_n_epochs=1,
        )

        callbacks.extend([best_classifier_ckpt, epoch_classifier_ckpt, LearningRateMonitor(logging_interval='epoch')])
        
        if args.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    patience=args.patience,
                    mode="min",
                )
            )
        
        classifier_logger = _build_logger("classifier", log_root)
        
        classifier_trainer = pl.Trainer(
            max_epochs=args.max_classifier_epochs,
            default_root_dir="lightning_logs/classifier",
            log_every_n_steps=args.log_frequency,
            logger=classifier_logger,
            callbacks=callbacks,
            enable_progress_bar=classifier_progress is not None,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            **_trainer_kwargs(),
        )
        
        if args.resume_classifier:
            classifier_trainer.fit(
                classifier_module,
                datamodule=classifier_datamodule,
                ckpt_path=args.resume_classifier
            )
        else:
            classifier_trainer.fit(classifier_module, datamodule=classifier_datamodule)
        
        ckpt_path = classifier_subdir / f"{metrics_base}_final.ckpt"
        pth_path = classifier_subdir / f"{metrics_base}_final.pth"
        classifier_trainer.save_checkpoint(ckpt_path)
        torch.save(classifier_module.model.state_dict(), pth_path)
        print(f"\nClassifier saved to {pth_path}")

    # --- Training summary -------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    cli_args = _parse_args()
    train_two_stage_system(cli_args)