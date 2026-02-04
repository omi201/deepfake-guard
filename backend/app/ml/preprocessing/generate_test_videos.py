"""Generate synthetic test videos for Day 3"""
import cv2
import numpy as np
from pathlib import Path

def create_video(path, is_fake=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        center = (320 + int(20*np.sin(i*0.1)), 240 + int(10*np.cos(i*0.1)))
        color = (128, 128, 128) if is_fake else (200, 180, 160)
        cv2.ellipse(frame, center, (100, 120), 0, 0, 360, color, -1)
        cv2.circle(frame, (center[0]-40, center[1]-20), 10, (50, 50, 50), -1)
        cv2.circle(frame, (center[0]+40, center[1]-20), 10, (50, 50, 50), -1)
        if is_fake:
            frame = cv2.add(frame, np.random.normal(0, 5, frame.shape).astype(np.uint8))
        out.write(frame)
    out.release()

base = Path("data/raw/celeb-df/videos")
for label in ["real", "fake"]:
    for i in range(5):
        create_video(base / label / f"test_{label}_{i:03d}.mp4", is_fake=(label=="fake"))
