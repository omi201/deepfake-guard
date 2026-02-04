"""
Day 3: MTCNN Face Extraction Pipeline
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
import hashlib

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.5
TARGET_SIZE = 224
FRAMES_PER_VIDEO = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceExtractor:
    def __init__(self, device=DEVICE):
        self.device = device
        self.mtcnn = MTCNN(
            image_size=TARGET_SIZE,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False
        )
        
    def extract_from_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        face_tensor, prob = self.mtcnn(pil_img, return_prob=True)
        
        if face_tensor is not None and prob is not None:
            face_img = Image.fromarray((face_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            return face_img, float(prob)
        return None, 0.0
    
    def process_video(self, video_path: Path, output_dir: Path, label: str):
        video_id = video_path.stem
        output_label_dir = output_dir / label
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"video": video_id, "status": "failed", "faces_count": 0}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return {"video": video_id, "status": "empty", "faces_count": 0}
        
        frame_indices = np.linspace(0, total_frames-1, min(FRAMES_PER_VIDEO, total_frames), dtype=int)
        faces_extracted = 0
        metadata = {"video": video_id, "label": label, "faces": []}
        
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
                metadata["faces"].append({"frame": int(idx), "confidence": round(conf, 4), "filename": filename})
        
        cap.release()
        metadata["faces_count"] = faces_extracted
        metadata["status"] = "success"
        return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/celeb-df/videos")
    parser.add_argument("--output", type=str, default="data/processed/celeb-df/faces")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect videos
    video_list = []
    for label in ["real", "fake"]:
        video_dir = input_dir / label
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            for video_path in videos:
                video_list.append((video_path, output_dir, label))
    
    logger.info(f"Processing {len(video_list)} videos on {DEVICE}...")
    
    # Single-process (avoid CUDA fork issues)
    extractor = FaceExtractor()
    all_metadata = []
    
    for video_path, out_dir, label in tqdm(video_list, desc="Extracting faces"):
        result = extractor.process_video(video_path, out_dir, label)
        all_metadata.append(result)
    
    # Save metadata
    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "total_videos": len(all_metadata), 
            "total_faces": sum(m["faces_count"] for m in all_metadata), 
            "videos": all_metadata
        }, f, indent=2)
    
    real_count = len(list((output_dir / "real").glob("*.jpg"))) if (output_dir / "real").exists() else 0
    fake_count = len(list((output_dir / "fake").glob("*.jpg"))) if (output_dir / "fake").exists() else 0
    
    print(f"\n{'='*50}")
    print(f"DAY 3 CHECKPOINT")
    print(f"{'='*50}")
    print(f"Real faces: {real_count}")
    print(f"Fake faces: {fake_count}")
    print(f"Total: {real_count + fake_count}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
