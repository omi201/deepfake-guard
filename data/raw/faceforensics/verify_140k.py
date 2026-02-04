#!/usr/bin/env python3
"""
140k Real vs Fake Faces Dataset Verification
Day 2 Checkpoint (Primary Dataset)
"""

from pathlib import Path

def verify_140k():
    """Verify 140k dataset structure"""
    root = Path("data/processed/140k/real_vs_fake/real-vs-fake")
    
    if not root.exists():
        print("❌ FAIL: 140k dataset not found")
        return False
    
    splits = ['train', 'test', 'valid']
    classes = ['real', 'fake']
    total_images = 0
    
    for split in splits:
        print(f"\n📁 {split.upper()}")
        for cls in classes:
            path = root / split / cls
            if path.exists():
                count = len(list(path.glob("*.jpg")))
                total_images += count
                print(f"   {cls}: {count}")
            else:
                print(f"   ❌ {cls}: missing")
    
    print(f"\n💾 TOTAL: {total_images} images")
    
    if total_images == 140000:
        print("\n✅ CHECKPOINT PASSED: Day 2 Complete")
        print("   140k dataset verified and ready for training")
        print("   Can proceed to Day 3 (Data Loading Pipeline)")
        return True
    else:
        print(f"\n❌ Expected 140000, found {total_images}")
        return False

if __name__ == "__main__":
    print("=== 140k Real vs Fake Dataset Verification ===\n")
    
    if verify_140k():
        exit(0)
    else:
        exit(1)
