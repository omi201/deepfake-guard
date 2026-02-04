#!/usr/bin/env python3
"""
Celeb-DF Dataset Verification
Day 2 Alternative Checkpoint (No .edu required)
"""

from pathlib import Path

def verify_celebdf():
    """Verify Celeb-DF v2 download"""
    root = Path("data/raw/celeb-df/videos")
    
    if not root.exists():
        print("❌ FAIL: Directory not found:", root.absolute())
        return False
    
    real_path = root / "Celeb-real"
    fake_path = root / "Celeb-synthesis"
    
    # Check Real videos
    if real_path.exists():
        real_videos = list(real_path.glob("*.mp4"))
        real_count = len(real_videos)
        print(f"📹 Celeb-real videos: {real_count} (~900 expected)")
    else:
        print(f"❌ Celeb-real not found")
        real_count = 0
    
    # Check Fake videos  
    if fake_path.exists():
        fake_videos = list(fake_path.glob("*.mp4"))
        fake_count = len(fake_videos)
        print(f"🎭 Celeb-synthesis videos: {fake_count} (~5600 expected)")
    else:
        print(f"❌ Celeb-synthesis not found")
        fake_count = 0
    
    # Total size
    all_videos = []
    if real_path.exists():
        all_videos.extend(list(real_path.glob("*.mp4")))
    if fake_path.exists():
        all_videos.extend(list(fake_path.glob("*.mp4")))
        
    if all_videos:
        total_size_gb = sum(f.stat().st_size for f in all_videos) / (1024**3)
        print(f"💾 Total size: {total_size_gb:.2f} GB (expected ~12GB)")
    
    # Checkpoint (Celeb-DF has variable counts, just check >0 for now)
    if real_count > 100 and fake_count > 500:
        print("\n✅ CHECKPOINT PASSED: Celeb-DF ready for Day 3")
        return True
    else:
        print(f"\n❌ CHECKPOINT FAILED: Download incomplete")
        print("   Visit https://github.com/yuezunli/celeb-deepfakeforensics")
        return False

if __name__ == "__main__":
    print("=== Celeb-DF v2 Verification (Day 2 Alternative) ===\n")
    
    if verify_celebdf():
        exit(0)
    else:
        exit(1)
