#!/usr/bin/env python3
"""
FaceForensics++ c23 Dataset Downloader
Requires academic email registration at https://faceforensics.com
"""

from pathlib import Path

def main():
    print("""
    ============================================================
    FaceForensics++ c23 Download Instructions
    ============================================================
    
    STEP 1: Register at https://faceforensics.com
    STEP 2: Wait for approval email (usually same day)
    STEP 3: Download using official script provided in email
    
    Expected structure:
    data/raw/faceforensics++/
    ├── original_sequences/youtube/     (1000 real .mp4 files)
    └── manipulated_sequences/Deepfakes/ (1000 fake .mp4 files)
    ============================================================
    """)
    
    root = Path("data/raw/faceforensics++")
    real_path = root / "original_sequences" / "youtube"
    fake_path = root / "manipulated_sequences" / "Deepfakes"
    
    if real_path.exists() and fake_path.exists():
        real_count = len(list(real_path.glob("*.mp4")))
        fake_count = len(list(fake_path.glob("*.mp4")))
        print(f"Status: {real_count}/1000 real, {fake_count}/1000 fake")

if __name__ == "__main__":
    main()
