"""Generate synthetic test videos for Day 3 pipeline verification"""
import cv2
import numpy as np
from pathlib import Path

def create_synthetic_video(output_path, num_frames=30, is_fake=False):
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 30, (width, height))
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        center = (width//2 + int(20*np.sin(i*0.1)), height//2 + int(10*np.cos(i*0.1)))
        
        # Face ellipse
        color = (128, 128, 128) if is_fake else (200, 180, 160)
        cv2.ellipse(frame, center, (100, 120), 0, 0, 360, color, -1)
        
        # Eyes
        cv2.circle(frame, (center[0]-40, center[1]-20), 10, (50, 50, 50), -1)
        cv2.circle(frame, (center[0]+40, center[1]-20), 10, (50, 50, 50), -1)
        
        # Mouth
        cv2.ellipse(frame, (center[0], center[1]+40), (30, 15), 0, 0, 180, (100, 50, 50), 2)
        
        if is_fake:
            noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)
        
        out.write(frame)
    out.release()

def main():
    base_dir = Path("data/raw/celeb-df/videos")
    for label in ["real", "fake"]:
        output_dir = base_dir / label
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            video_path = output_dir / f"test_{label}_{i:03d}.mp4"
            create_synthetic_video(video_path, is_fake=(label=="fake"))
            print(f"Created: {video_path}")

if __name__ == "__main__":
    main()
