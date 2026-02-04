"""
Day 3: MTCNN Face Extraction Pipeline
Extracts 224x224 face crops from video datasets using facenet-pytorch
"""

import os
import cv2
import json
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Dict
import hashlib

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
CONFIDENCE_THRESHOLD = 0.95
TARGET_SIZE = 224
FRAMES_PER_VIDEO = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceExtractor:
    def __init__(self, device: torch.device = DEVICE, target_size: int = TARGET_SIZE):
        self.device = device
        self.target_size = target_size
        self.mtcnn = MTCNN(
            image_size=target_size,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False
        )
        logger.info(f"MTCNN initialized on {device}")
        
    def extract_from_frame(self, frame: np.ndarray) -> Tuple[Image.Image, float]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        face_tensor, prob = self.mtcnn(pil_img, return_prob=True)
        
        if face_tensor is not None and prob is not None:
            face_img = Image.fromarray(
                (face_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            return face_img, float(prob)
        
        return None, 0.0
    
    def process_video(self, video_path: Path, output_dir: Path, label: str) -> Dict:
        video_id = video_path.stem
        output_label_dir = output_dir / label
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return {"video": video_id, "status": "failed", "faces": 0}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return {"video": video_id, "status": "empty", "faces": 0}
        
        frame_indices = np.linspace(0, total_frames-1, min(FRAMES_PER_VIDEO, total_frames), dtype=int)
        
        faces_extracted = 0
        metadata = {
            "video": video_id,
            "label": label,
            "total_frames": total_frames,
            "sampled_frames": len(frame_indices),
            "faces": [],
            "status": "success"
        }
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            face_img, conf = self.extract_from_frame(frame)
            if face_img and conf >= CONFIDENCE_THRESHOLD:
                frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
                filename = f"{video_id}_frame{idx}_{frame_hash}.jpg"
                filepath = output_label_dir / filename
                
                face_img.save(filepath, quality=95)
                faces_extracted += 1
                
                metadata["faces"].append({
                    "frame": int(idx),
                    "confidence": round(conf, 4),
                    "filename": filename,
                    "path": str(filepath.relative_to(output_dir.parent))
                })
        
        cap.release()
        metadata["faces_count"] = faces_extracted
        return metadata


def process_video_wrapper(args):
    video_path, output_dir, label, device_id = args
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id % torch.cuda.device_count())
    
    extractor = FaceExtractor(device=DEVICE)
    return extractor.process_video(video_path, output_dir, label)


def main():
    parser = argparse.ArgumentParser(description="Extract faces from video dataset")
    parser.add_argument("--input", type=str, default="data/raw/celeb-df/videos")
    parser.add_argument("--output", type=str, default="data/processed/celeb-df/faces")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()))
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_tasks = []
    for label in ["real", "fake"]:
        video_dir = input_dir / label
        if not video_dir.exists():
            logger.warning(f"Directory not found: {video_dir}")
            continue
            
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        if args.limit:
            videos = videos[:args.limit]
            
        logger.info(f"Found {len(videos)} {label} videos")
        
        for i, video_path in enumerate(videos):
            device_id = i % torch.cuda.device_count() if torch.cuda.is_available() else 0
            video_tasks.append((video_path, output_dir, label, device_id))
    
    logger.info(f"Processing {len(video_tasks)} videos with {args.workers} workers...")
    all_metadata = []
    
    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_video_wrapper, video_tasks),
                total=len(video_tasks),
                desc="Extracting faces"
            ))
            all_metadata.extend(results)
    else:
        extractor = FaceExtractor()
        for task in tqdm(video_tasks, desc="Extracting faces"):
            result = extractor.process_video(task[0], task[1], task[2])
            all_metadata.append(result)
    
    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "total_videos": len(all_metadata),
            "total_faces": sum(m["faces_count"] for m in all_metadata),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "target_size": TARGET_SIZE,
            "videos": all_metadata
        }, f, indent=2)
    
    total_faces = sum(m["faces_count"] for m in all_metadata)
    successful_videos = sum(1 for m in all_metadata if m["status"] == "success" and m["faces_count"] > 0)
    
    logger.info(f"Extraction complete:")
    logger.info(f"  Videos processed: {len(all_metadata)}")
    logger.info(f"  Videos with faces: {successful_videos}")
    logger.info(f"  Total faces extracted: {total_faces}")
    
    real_count = len(list((output_dir / "real").glob("*.jpg"))) if (output_dir / "real").exists() else 0
    fake_count = len(list((output_dir / "fake").glob("*.jpg"))) if (output_dir / "fake").exists() else 0
    
    print(f"\n{'='*50}")
    print(f"DAY 3 CHECKPOINT")
    print(f"{'='*50}")
    print(f"Real faces extracted: {real_count}")
    print(f"Fake faces extracted: {fake_count}")
    print(f"Total faces: {real_count + fake_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
