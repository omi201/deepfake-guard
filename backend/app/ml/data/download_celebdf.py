#!/usr/bin/env python3
"""
Celeb-DF Dataset Downloader
No academic email required - just fill Google Form
"""

from pathlib import Path
import os

def main():
    print("""
    ============================================================
    Celeb-DF (v2) Download Instructions
    ============================================================
    
    STEP 1: Request Access
    - Go to: https://github.com/yuezunli/celeb-deepfakeforensics
    - Click "Download" link or fill Google Form
    - Wait for approval email (usually 24-48 hours)
    
    STEP 2: Download Links
    - You'll receive Google Drive links via email
    - Celeb-DF-v2 has ~6000 videos (~12GB total)
    
    STEP 3: Download Method A (Manual)
    - Download 'Celeb-real' and 'Celeb-synthesis' folders
    - Extract to: data/raw/celeb-df/videos/
    
    STEP 4: Download Method B (gdown - if you have links)
    pip install gdown
    gdown [link-to-Celeb-real] -O data/raw/celeb-df/videos/Celeb-real.zip
    gdown [link-to-Celeb-synthesis] -O data/raw/celeb-df/videos/Celeb-synthesis.zip
    
    Expected Structure:
    data/raw/celeb-df/
    └── videos/
        ├── Celeb-real/          (~900 real videos)
        └── Celeb-synthesis/     (~5600 fake videos)
    ============================================================
    """)
    
    # Check current status
    root = Path("data/raw/celeb-df/videos")
    if root.exists():
        real_path = root / "Celeb-real"
        fake_path = root / "Celeb-synthesis"
        
        real_count = len(list(real_path.glob("*.mp4"))) if real_path.exists() else 0
        fake_count = len(list(fake_path.glob("*.mp4"))) if fake_path.exists() else 0
        
        print(f"Current status: {real_count} real, {fake_count} fake videos")
        
        if real_count > 0 or fake_count > 0:
            print(f"Total size: {sum(f.stat().st_size for f in root.rglob('*.mp4')) / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
