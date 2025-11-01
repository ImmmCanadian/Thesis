import cv2
import torch
import numpy as np
import time
import math
import random
from torch.utils.data import Dataset
import mediapipe as mp
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Tuple, Sequence
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from ultralytics import YOLO

IPN_RGB_MEAN = [0.450097, 0.422493, 0.390098]
IPN_RGB_STD = [0.152967, 0.148480, 0.157761]
DEFAULT_SAMPLE_SIZE = 112


# --- Detector image transform pipeline ------------------------------------
class DetectorTrainTransform:

    def __init__(self, sample_size: int, use_elastic_distortion: bool = False):
        self.sample_size = int(sample_size)
        self.use_elastic_distortion = use_elastic_distortion
        self.scale = (0.7, 1.0)
        self.ratio = (0.85, 1.15)
        # Elastic transform removed as per research recommendations
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IPN_RGB_MEAN, std=IPN_RGB_STD)
        self.clear_parameters()

    def clear_parameters(self):
        self._params_ready = False
        self._crop_params = None
        self._flip = False

    def randomize_parameters(self, frame: np.ndarray) -> None:
        if frame is None:
            frame = np.zeros((self.sample_size, self.sample_size, 3), dtype=np.uint8)
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            frame = np.transpose(frame, (1, 2, 0))
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(frame)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_img, self.scale, self.ratio
        )
        self._crop_params = (i, j, h, w)
        self._flip = random.random() < 0.5
        self._params_ready = True

    def _ensure_params(self, frame: np.ndarray) -> None:
        if not self._params_ready:
            self.randomize_parameters(frame)

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        self._ensure_params(frame)
        pil_img = Image.fromarray(frame)
        i, j, h, w = self._crop_params
        pil_img = F.resized_crop(
            pil_img,
            top=i,
            left=j,
            height=h,
            width=w,
            size=(self.sample_size, self.sample_size),
        )
        if self._flip:
            pil_img = F.hflip(pil_img)
        tensor = self.to_tensor(pil_img)
        tensor = self.normalize(tensor)
        return tensor


# --- MediaPipe landmark extractor -----------------------------------------
class MediaPipeExtractor:

    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3):
        self._hands_kwargs = dict(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )
        self.hands = None
        self.feature_dim = 63
        self._ensure_hands()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["hands"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.hands = None
        self._ensure_hands()

    def _ensure_hands(self):
        if self.hands is None:
            self.hands = mp.solutions.hands.Hands(**self._hands_kwargs)

    def extract(self, frame):
        self._ensure_hands()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(21, 3)
            center = landmarks.mean(axis=0)
            landmarks -= center
            return landmarks.flatten().astype(np.float32)
        else:
            return np.zeros(63, dtype=np.float32)


# --- YOLO + MobileNet feature extractor -----------------------------------
class YOLOMobileNetExtractor:

    def __init__(
        self,
        detector_weights: str = "weights/hand_detect_best.pt",
        fallback_weights: str = "yolo11n.pt",
        mobilenet_weights: MobileNet_V2_Weights = MobileNet_V2_Weights.IMAGENET1K_V1,
        device: Optional[torch.device] = None,
        use_fp16: bool = False,
        
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        if use_fp16 and self.device.type != "cuda":
            warnings.warn("FP16 requested but CUDA device unavailable; running in FP32 instead.")
        

        self._yolo_device = self.device

        detector_weights_path = Path(detector_weights)
        if not detector_weights_path.exists():
            warnings.warn(
                f"Detector weights not found at '{detector_weights_path}'. Falling back to '{fallback_weights}'."
            )
            detector_weights_path = Path(fallback_weights)

        self.yolo_weights = detector_weights_path
        # Load YOLO detector (PyTorch) for metadata and optional inference
        self.yolo = YOLO(str(self.yolo_weights))
        
        if not self.use_fp16:
            self.yolo.fuse()
        self.yolo.to(self.device)
        yolo_model = getattr(self.yolo, "model", None)
        if hasattr(yolo_model, "device"):
            self._yolo_device = torch.device(yolo_model.device)
        else:
            self._yolo_device = self.device

        # Prepare MobileNetV2 for feature extraction
        mobilenet = mobilenet_v2(weights=mobilenet_weights)
        mobilenet.classifier = torch.nn.Identity()
        mobilenet.eval()
        self.mobilenet = mobilenet.to(self.device)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Store normalization parameters as tensors for faster processing
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.feature_dim = 1280

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device))


    def _select_hand_box(self, result, frame_shape):
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.data.numel() == 0:
            return None

        names = self.yolo.names if hasattr(self.yolo, "names") else {}
        chosen_idx = None
        chosen_conf = 0.0
        chosen_area = 0

        for i, data_row in enumerate(boxes.data):
            cls = int(data_row[5].item())
            conf = float(data_row[4].item())
            class_name = names.get(cls, None)

            if class_name not in {"hand", "Hand", "hand_box"}:
                continue
            

            x1, y1, x2, y2 = data_row[:4].tolist()
            area = (x2 - x1) * (y2 - y1)

            if conf > chosen_conf or (conf == chosen_conf and area > chosen_area):
                chosen_idx = i
                chosen_conf = conf
                chosen_area = area

        if chosen_idx is None:
            return None

        box_tensor = boxes.xyxy[chosen_idx]
        x1, y1, x2, y2 = box_tensor.tolist()
        x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
        x1_int = max(0, x1_int)
        y1_int = max(0, y1_int)
        x2_int = min(frame_shape[1], x2_int)
        y2_int = min(frame_shape[0], y2_int)
        return (x1_int, y1_int, x2_int, y2_int)

    def fast_mobilenet_preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_float = crop_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(crop_float).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = F.resize(tensor, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
        tensor = (tensor - self.imagenet_mean) / self.imagenet_std
        if self.use_fp16:
            tensor = tensor.half()
        return tensor

    def extract_with_box(self, frame: np.ndarray, return_timings: bool = False):
        total_start = time.perf_counter()
        detect_start = total_start
        
        results = self.yolo.predict(frame, verbose=False, imgsz=640)
        result = results[0] if len(results) > 0 else None
        detect_time = time.perf_counter() - detect_start

        box = None
        if result is not None:
            box = self._select_hand_box(result, frame.shape)

        if box is not None:
            x1, y1, x2, y2 = box
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
                crop_bgr = frame

            mobilenet_start = time.perf_counter()
            tensor = self.fast_mobilenet_preprocess(crop_bgr)

            with torch.no_grad():
                features = self.mobilenet(tensor)
                if isinstance(features, tuple):
                    features = features[0]
                if features.dim() == 4:
                    features = self.pool(features)
                features = features.view(features.size(0), -1)

            embedding = features.squeeze(0).cpu().numpy().astype(np.float32)
            mobilenet_time = time.perf_counter() - mobilenet_start
        else:
            embedding = np.zeros(self.feature_dim, dtype=np.float32)
            mobilenet_time = 0.0
        total_time = time.perf_counter() - total_start
        timings = None
        if return_timings:
            timings = {
                "yolo": detect_time,
                "mobilenet": mobilenet_time,
                "total": total_time,
            }
        return embedding, box, timings
    
    def runtime_devices(self) -> Dict[str, str]:
        try:
            mobilenet_device = next(self.mobilenet.parameters()).device
        except StopIteration:
            mobilenet_device = self.device
        devices = {
            "yolo": str(self._yolo_device or self.device),
            "mobilenet": str(mobilenet_device),
        }
        
        return devices


# --- Two-stage dataset -----------------------------------------------------
class IPNTwoStageDataset(Dataset):

    def __init__(
        self,
        annotation_file,
        video_root,
        stage="detector",
        detector_window=8,
        classifier_window=30,
        stride=5,
        mode='train',
        feature_type='mediapipe',
        cache_dir=None,
        features_cache_dir=None,  
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        max_samples: Optional[int] = None,
        include_background: bool = False,
        use_elastic_distortion: bool = False,
        subset_indices: Optional[Sequence[int]] = None,
    ):
        self.detector_window = detector_window
        self.classifier_window = classifier_window
        self.stride = stride
        self.mode = mode
        self.video_root = Path(video_root)
        self.stage = stage
        self.feature_type = feature_type
        self.cache_dir = None
        if cache_dir and self.stage != 'classifier':
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.cache_dir = cache_path
        
        
        self.features_cache_dir = Path(features_cache_dir) if features_cache_dir else None
        if self.features_cache_dir and self.stage == 'classifier' and not self.features_cache_dir.exists():
            warnings.warn(f"Features cache directory not found: {self.features_cache_dir}")

        self.sample_size = int(sample_size)
        self.include_background = include_background
        self.use_elastic_distortion = use_elastic_distortion
        self.detector_min_positive_ratio = 0.7 if self.stage == 'detector' else 0.0
        self._subset_indices = list(subset_indices) if subset_indices is not None else None
        
        # Initialize extractors
        self.feature_extractor = None
        self.feature_dim = None

        self.frame_transform = self._build_frame_transform()
        self.max_samples = max_samples if (max_samples is None or max_samples > 0) else None
        
        if self.stage == 'classifier':
            cache_feature_dim = None
            if self.features_cache_dir and self.features_cache_dir.exists():
                for npy_file in self.features_cache_dir.glob("*.npy"):
                    try:
                        sample_features = np.load(npy_file)
                        cache_feature_dim = sample_features.shape[1]
                        print(f"🚀 Using pre-computed features from: {self.features_cache_dir}")
                        print(f"   Feature dimension: {cache_feature_dim} (detected from cache)")
                        break
                    except Exception as e:
                        warnings.warn(f"Could not read cache file {npy_file}: {e}")
                        continue
            if feature_type == 'mediapipe':
                self.feature_extractor = MediaPipeExtractor(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.3
                )
            elif feature_type == 'yolo_mobilenet':
                self.feature_extractor = YOLOMobileNetExtractor(
                    detector_weights="weights/hand_detect_best.pt",
                    fallback_weights="yolo11n.pt",
                    device=None,
                    use_fp16=False
                )
            else:
                raise ValueError(f"Unsupported feature_type '{feature_type}'")
            if cache_feature_dim is not None:
                self.feature_dim = cache_feature_dim
            else:
                self.feature_dim = getattr(self.feature_extractor, "feature_dim", None)
                if self.feature_dim is None:
                    dummy = np.zeros((112, 112, 3), dtype=np.uint8)
                    sample = self.feature_extractor.extract(dummy)
                    self.feature_dim = int(np.prod(sample.shape))
        
        else:
            # Detector stage
            self.feature_extractor = None
            self.feature_dim = None
        
        # Parse annotations and index samples
        self.segments = self._parse_annotations(annotation_file)
        self.samples = self._index_samples()
        
        # Restrict to subset indices if needed
        if self._subset_indices is not None:
            self.samples = [self.samples[i] for i in self._subset_indices]
        
        # Apply max_samples limit
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
        
        #Display class distribution
        if self.stage == 'classifier':
            label_counts = defaultdict(int)
            for s in self.samples:
                label_counts[s['label']] += 1
            self.num_classes = len(label_counts)
            print(f"[{self.mode}] Loaded {len(self.samples)} samples across {self.num_classes} classes")
            print(f"   Class distribution: {dict(label_counts)}")
        else:
            label_counts = defaultdict(int)
            for s in self.samples:
                label_counts[s['label']] += 1
            print(f"[{self.mode}] Loaded {len(self.samples)} samples")
            print(f"   Label distribution: {dict(label_counts)}")

    def restrict_to_indices(self, indices):
        self.samples = [self.samples[i] for i in indices]
        
    # Build appropriate spatial transforms per stage
    def _build_frame_transform(self):
        if self.stage != 'detector':
            return None

        if self.mode == 'train':
            return DetectorTrainTransform(
                sample_size=self.sample_size,
                use_elastic_distortion=self.use_elastic_distortion,
            )
        else:
            resize_size = int(self.sample_size * 1.14)
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.sample_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IPN_RGB_MEAN, std=IPN_RGB_STD),
            ])

    # Parse raw annotation file into structured segments
    def _parse_annotations(self, annotation_file):
        segments = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) < 5:
                    continue
                video = parts[0]
                class_id = int(parts[2]) - 1  # 0-indexed
                start = int(parts[3])
                end = int(parts[4])
                segments.append({'video': video, 'class_id': class_id, 'start': start, 'end': end})
        return segments

    # Enumerate clip samples for detector and classifier stages
    def _index_samples(self):
        samples = []
        video_segments = defaultdict(list)
        for seg in self.segments:
            video_segments[seg['video']].append(seg)

        if self.stage == 'classifier':
            all_positive_segments = []
            for segments in video_segments.values():
                all_positive_segments.extend([s for s in segments if s['class_id'] != 0])
            
            unique_classes = sorted(set(s['class_id'] for s in all_positive_segments))
            
           
            if self.include_background:
                class_map = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_classes)}
                background_label = 0
                self.num_classes = len(unique_classes) + 1  
            else:
                class_map = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}
                self.num_classes = len(unique_classes)

        for video, segments in video_segments.items():
            video_path = self.video_root / (video + '.avi')
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if self.stage == 'detector':
                min_overlap_frames = max(
                    1,
                    int(math.ceil(self.detector_window * self.detector_min_positive_ratio))
                    if self.detector_min_positive_ratio > 0.0
                    else 1,
                )
                for i in range(0, max(0, total_frames - self.detector_window + 1), self.stride):
                    window_start = i
                    window_end = i + self.detector_window - 1
                    gesture_frames = 0
                    if self.detector_min_positive_ratio > 0.0:
                        for ann in segments:
                            if ann['class_id'] == 0:
                                continue
                            overlap_start = max(window_start, ann['start'])
                            overlap_end = min(window_end, ann['end'])
                            if overlap_end >= overlap_start:
                                gesture_frames += overlap_end - overlap_start + 1
                                if gesture_frames >= min_overlap_frames:
                                    break
                        has_gesture = gesture_frames >= min_overlap_frames
                    else:
                        has_gesture = False
                        for ann in segments:
                            if ann['class_id'] != 0 and not (
                                window_start > ann['end'] or window_end < ann['start']
                            ):
                                has_gesture = True
                                break
                    samples.append({'video': video, 'start': window_start, 'end': window_end + 1, 'label': 1 if has_gesture else 0})
            elif self.stage == 'classifier':
                positive_segments = [s for s in segments if s['class_id'] != 0]
                
                stride = max(1, self.stride * 2)
                for ann in positive_segments:
                    gesture_length = ann['end'] - ann['start']
                    
                    if gesture_length < self.classifier_window:
                        # For gestures shorter than window, create one sample
                        # _extract_features will handle padding
                        samples.append({
                            'video': video, 
                            'start': ann['start'], 
                            'end': ann['end'] + 1,  # ✅ FIX #4: Make end exclusive (annotations are inclusive)
                            'label': class_map[ann['class_id']],
                            'is_short': True  # Mark for padding
                        })
                    else:
                        # Normal sliding window for gestures >= window size
                        start_idx = ann['start']
                        end_idx = max(ann['start'], ann['end'] - self.classifier_window + 1)
                        for i in range(start_idx, end_idx + 1, stride):
                            
                            samples.append({
                                'video': video, 
                                'start': i, 
                                'end': i + self.classifier_window, 
                                'label': class_map[ann['class_id']],
                                'is_short': False
                            })

                if self.include_background:
                    gesture_intervals = [(ann['start'], ann['end']) for ann in positive_segments]
                    background_stride = max(1, self.stride * 2)
                    max_start = max(0, total_frames - self.classifier_window + 1)
                    for i in range(0, max_start + 1, background_stride):
                        window_start = i
                        window_end = i + self.classifier_window
                        overlaps_gesture = False
                        for start, end in gesture_intervals:
                            if not (window_end <= start or window_start >= end):
                                overlaps_gesture = True
                                break
                        if overlaps_gesture:
                            continue
                        samples.append({'video': video, 'start': window_start, 'end': window_end, 'label': background_label})

        return samples

    def __len__(self):
        return len(self.samples)
    
    # Provide label distribution for class-weight computation
    def get_class_distribution(self):
        if self.stage != 'classifier':
            return None
        
        from collections import Counter
        labels = [sample['label'] for sample in self.samples]
        return Counter(labels)

    # Load preprocessed clip tensors for detector training
    def _load_clip(self, video, start, end):
        video_path = self.video_root / (video + '.avi')
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        transform = self.frame_transform
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(start, end):
            ret, frame = cap.read()
            if not ret:
                fallback = np.zeros((self.sample_size, self.sample_size, 3), dtype=np.uint8)
                if transform is not None:
                    if hasattr(transform, "randomize_parameters") and not frames:
                        transform.randomize_parameters(fallback)
                    frame_tensor = transform(fallback)
                else:
                    frame_tensor = torch.zeros(3, self.sample_size, self.sample_size)
                frames.append(frame_tensor)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if transform is not None:
                if hasattr(transform, "randomize_parameters") and not frames:
                    transform.randomize_parameters(frame_rgb)
                frame_tensor = transform(frame_rgb)
            else:
                resized = cv2.resize(frame_rgb, (self.sample_size, self.sample_size))
                frame_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
            frames.append(frame_tensor)
        cap.release()
        if transform is not None and hasattr(transform, "clear_parameters"):
            transform.clear_parameters()
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        return clip

    # Fetch raw RGB frames without preprocessing (classifier feature extraction)
    def _load_raw_clip(self, video, start, end):
        video_path = self.video_root / (video + '.avi')
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(start, end):
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.sample_size, self.sample_size, 3), dtype=np.uint8))
                continue
            frames.append(frame)
        cap.release()
        return frames

    # Extract or retrieve feature embeddings for classifier stage
    def _extract_features(self, video, start, end):
        """Extract features for a video segment, using cache if available."""
        
       
        if self.features_cache_dir is not None:
            cache_file = self.features_cache_dir / f"{video}.npy"
            if cache_file.exists():
                try:
                    # Load full video features
                    all_features = np.load(cache_file)  # Shape: (total_frames, feature_dim)
                    
                    # Extract the requested window
                    features = all_features[start:end]
                    
                    target_length = self.classifier_window if self.stage == 'classifier' else (end - start)
                    if len(features) < target_length:
                        num_missing = target_length - len(features)
                        # Always pad with zeros (represents "no hand" state)
                        padding = np.zeros((num_missing, all_features.shape[1]), dtype=np.float32)
                        features = np.concatenate([features, padding], axis=0)
                    
                    return features.astype(np.float32)
                
                except Exception as e:
                    warnings.warn(f"Failed to load cached features for {video}: {e}. Extracting on-the-fly.")
        
        
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not initialized and no cached features available")
        
        raw_frames = self._load_raw_clip(video, start, end)
        features = []
        for frame in raw_frames:
            if hasattr(self.feature_extractor, "extract_with_box"):
                embedding, _, _ = self.feature_extractor.extract_with_box(frame)
                features.append(embedding)
            else:
                feature = self.feature_extractor.extract(frame)
                if isinstance(feature, tuple):
                    feature = feature[0]
                features.append(feature)
        features = np.stack(features, axis=0)
        
        
        target_length = self.classifier_window if self.stage == 'classifier' else (end - start)
        if len(features) < target_length:
            num_missing = target_length - len(features)
            # Always pad with zeros (represents "no hand" state)
            padding = np.zeros((num_missing, features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        
        return features

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.stage == 'detector':
            clip = self._load_clip(sample['video'], sample['start'], sample['end'])
            clip = clip.permute(1, 0, 2, 3)  # (T,C,H,W) -> (C,T,H,W)
            return clip.float(), torch.tensor(sample['label']).long()
        elif self.stage == 'classifier':
            features = self._extract_features(sample['video'], sample['start'], sample['end'])
            return torch.from_numpy(features).float(), torch.tensor(sample['label']).long()