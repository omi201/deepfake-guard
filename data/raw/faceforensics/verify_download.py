#!/usr/bin/env python3
"""
Day 2 Verification Checkpoint
Validates FaceForensics++ c23 download integrity
"""

from pathlib import Path

def verify_faceforensics_structure():
    """Verify 1000 real + 1000 fake videos exist"""
    root = Path("data/raw/faceforensics++")
    
    if not root.exists():
        print("❌ FAIL: Directory not found:", root.absolute())
        return False
    
    real_path = root / "original_sequences" / "youtube"
    fake_path = root / "manipulated_sequences" / "Deepfakes"
    
    results = {}
    
    # Check Real videos
    if real_path.exists():
        real_videos = list(real_path.glob("*.mp4"))
        real_count = len(real_videos)
        results['real'] = real_count
        print(f"📹 Real videos: {real_count}/1000")
    else:
        print(f"❌ Real path not found: {real_path}")
        results['real'] = 0
    
    # Check Fake videos  
    if fake_path.exists():
        fake_videos = list(fake_path.glob("*.mp4"))
        fake_count = len(fake_videos)
        results['fake'] = fake_count
        print(f"🎭 Fake videos: {fake_count}/1000")
    else:
        print(f"❌ Fake path not found: {fake_path}")
        results['fake'] = 0
    
    # Total size
    all_videos = []
    if real_path.exists():
        all_videos.extend(list(real_path.glob("*.mp4")))
    if fake_path.exists():
        all_videos.extend(list(fake_path.glob("*.mp4")))
        
    if all_videos:
        total_size_gb = sum(f.stat().st_size for f in all_videos) / (1024**3)
        print(f"💾 Total size: {total_size_gb:.2f} GB (expected ~40GB)")
    
    # Checkpoint
    if results.get('real') == 1000 and results.get('fake') == 1000:
        print("\n✅ CHECKPOINT PASSED: Day 2 Complete")
        return True
    else:
        print(f"\n❌ CHECKPOINT FAILED: {results.get('real', 0)}/1000 real, {results.get('fake', 0)}/1000 fake")
        return False

if __name__ == "__main__":
    print("=== FaceForensics++ c23 Verification (Day 2 Checkpoint) ===\n")
    
    if verify_faceforensics_structure():
        exit(0)
    else:
        exit(1)
