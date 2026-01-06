"""
Pre-extract and cache features for fast classifier training.

This script extracts MediaPipe or YOLO+MobileNet features from all videos
and saves them to disk. Training then loads pre-computed features instead
of extracting on-the-fly, resulting in 50-100x speedup.

Usage:
    python precompute_features.py \
        --annotation-file annotations/Annot_TrainList.txt \
        --video-root data/videos \
        --output-dir cache/mediapipe \
        --feature-type mediapipe \
        --num-workers 8
"""

import argparse
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import your feature extractors
import sys
sys.path.append('.')
from dataset_and_pipelines import MediaPipeExtractor, YOLOMobileNetExtractor


def parse_annotations(annotation_file: str):
    """Parse annotation file to get list of videos."""
    videos = set()
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Handle both space and comma separators
            # Extract just the video name (first field before space or comma)
            parts = line.replace(',', ' ').split()
            if parts:
                video = parts[0]
                videos.add(video)
    return sorted(videos)


def extract_video_features(args_tuple):
    """Extract features for a single video (worker function)."""
    if len(args_tuple) == 4:
        video_name, video_root, feature_type, output_dir = args_tuple
        mobilenet_variant = "v3_large_imagenet"
    else:
        video_name, video_root, feature_type, output_dir, mobilenet_variant = args_tuple
    
    video_path = Path(video_root) / (video_name + '.avi')
    if not video_path.exists():
        return video_name, False, f"Video not found: {video_path}"
    
    # Initialize feature extractor (per-process)
    if feature_type == 'mediapipe':
        extractor = MediaPipeExtractor(static_image_mode=False, max_num_hands=1)
    elif feature_type == 'yolo_mobilenet':
        extractor = YOLOMobileNetExtractor(
            detector_weights="weights/hand_detect_best.pt",
            fallback_weights="yolo11n.pt",
            mobilenet_variant=mobilenet_variant,
        )
    else:
        return video_name, False, f"Unknown feature type: {feature_type}"
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return video_name, False, "Video has 0 frames"
    
    # Extract features frame-by-frame
    features = []
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            # Use last valid feature or zeros
            if features:
                features.append(features[-1])
            else:
                feature_dim = extractor.feature_dim
                features.append(np.zeros(feature_dim, dtype=np.float32))
            continue
        
        # Extract feature
        if hasattr(extractor, 'extract_with_box'):
            feature, _, _ = extractor.extract_with_box(frame)
        else:
            feature = extractor.extract(frame)
            if isinstance(feature, tuple):
                feature = feature[0]
        
        features.append(feature)
    
    cap.release()
    
    # Stack into array: (T, feature_dim)
    features = np.stack(features, axis=0).astype(np.float32)
    
    # Save to cache
    output_path = Path(output_dir) / f"{video_name}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    
    return video_name, True, f"{features.shape[0]} frames, dim={features.shape[1]}"


def main():
    parser = argparse.ArgumentParser(description="Pre-extract features for fast training")
    parser.add_argument("--annotation-file", required=True, help="Annotation file")
    parser.add_argument("--video-root", required=True, help="Root directory of videos")
    parser.add_argument("--output-dir", required=True, help="Output directory for cached features")
    parser.add_argument("--feature-type", choices=["mediapipe", "yolo_mobilenet"], 
                       default="mediapipe", help="Feature extraction method")
    parser.add_argument(
        "--mobilenet-backbone",
        type=str,
        default="v3_large_imagenet",
        choices=["v2_imagenet", "v3_small_imagenet", "v3_large_imagenet"],
        help="MobileNet backbone to use when --feature-type=yolo_mobilenet.",
    )
    parser.add_argument("--num-workers", type=int, default=4, 
                       help="Number of parallel workers")
    parser.add_argument("--force", action="store_true",
                       help="Re-extract even if cache exists")
    
    args = parser.parse_args()
    
    # Parse videos from annotation file
    print(f"Parsing annotation file: {args.annotation_file}")
    videos = parse_annotations(args.annotation_file)
    print(f"Found {len(videos)} unique videos")
    
    # Check which videos already have cached features
    output_dir = Path(args.output_dir)
    if not args.force:
        videos_to_process = []
        for video in videos:
            cache_file = output_dir / f"{video}.npy"
            if not cache_file.exists():
                videos_to_process.append(video)
        print(f"Skipping {len(videos) - len(videos_to_process)} already cached videos")
        videos = videos_to_process
    
    if not videos:
        print("All videos already cached! Use --force to re-extract.")
        return
    
    print(f"\nExtracting features for {len(videos)} videos...")
    print(f"Feature type: {args.feature_type}")
    if args.feature_type == "yolo_mobilenet":
        print(f"MobileNet backbone: {args.mobilenet_backbone}")
    print(f"Output dir: {args.output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"{'='*60}\n")
    
    # Prepare worker arguments
    worker_args = [
        (video, args.video_root, args.feature_type, args.output_dir, args.mobilenet_backbone)
        for video in videos
    ]
    
    # Use spawn context for MediaPipe (safer)
    ctx = mp.get_context('spawn')
    
    # Process videos in parallel
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(extract_video_features, arg): arg[0] 
                  for arg in worker_args}
        
        with tqdm(total=len(videos), desc="Extracting features") as pbar:
            for future in as_completed(futures):
                video_name = futures[future]
                try:
                    video, success, message = future.result()
                    if success:
                        success_count += 1
                        pbar.set_postfix_str(f"✓ {message}")
                    else:
                        error_count += 1
                        tqdm.write(f"Error {video}: {message}")
                except Exception as e:
                    error_count += 1
                    tqdm.write(f"Error {video_name}: {e}")
                pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"Success: {success_count}/{len(videos)}")
    if error_count > 0:
        print(f"Errors: {error_count}/{len(videos)}")
    print(f"{'='*60}")
    print(f"\nCached features saved to: {args.output_dir}")
    print(f"\nTo use cached features during training, modify your dataset to load from:")
    print(f"  {output_dir / '<video_name>.npy'}")


if __name__ == "__main__":
    main()