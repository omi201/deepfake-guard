# backend/app/ml/data/organize_celebdf.py
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_DIR = "data/processed/celeb-df/faces"
TARGET_DIR = "data/processed/celeb-df/faces"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def create_structure():
    """Create train/test directories"""
    for split in ['train', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(TARGET_DIR, split, label)
            os.makedirs(path, exist_ok=True)
            print(f"📁 Created: {path}")

def split_and_copy():
    """Split data and copy to train/test"""
    random.seed(RANDOM_SEED)
    
    for label in ['real', 'fake']:
        source_path = os.path.join(SOURCE_DIR, label)
        
        if not os.path.exists(source_path):
            print(f"❌ Source not found: {source_path}")
            continue
            
        # Get all images
        images = [f for f in os.listdir(source_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        images.sort()
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        print(f"\n📊 {label.upper()}:")
        print(f"   Total: {len(images)} | Train: {len(train_images)} | Test: {len(test_images)}")
        
        # Copy train
        print(f"   Copying train...")
        for img in tqdm(train_images, desc=f"{label} train"):
            src = os.path.join(source_path, img)
            dst = os.path.join(TARGET_DIR, 'train', label, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        # Copy test
        print(f"   Copying test...")
        for img in tqdm(test_images, desc=f"{label} test"):
            src = os.path.join(source_path, img)
            dst = os.path.join(TARGET_DIR, 'test', label, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

def verify_counts():
    """Verify the split"""
    print("\n✅ Verification:")
    total_train = 0
    total_test = 0
    for split in ['train', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(TARGET_DIR, split, label)
            count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"   {split}/{label}: {count} images")
            if split == 'train':
                total_train += count
            else:
                total_test += count
    print(f"\n📈 Total: {total_train} train + {total_test} test = {total_train + total_test} images")

if __name__ == "__main__":
    print("🗂️  Organizing Celeb-DF Dataset")
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}/train and {TARGET_DIR}/test")
    print(f"Split Ratio: {TRAIN_RATIO:.0%} train / {1-TRAIN_RATIO:.0%} test")
    
    # Check existing data
    real_path = os.path.join(SOURCE_DIR, 'real')
    fake_path = os.path.join(SOURCE_DIR, 'fake')
    real_count = len([f for f in os.listdir(real_path) if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(real_path) else 0
    fake_count = len([f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(fake_path) else 0
    
    print(f"\n📈 Source Counts: {real_count} real, {fake_count} fake")
    
    if real_count == 0 or fake_count == 0:
        print("❌ No source images found! Check extraction paths.")
        exit(1)
    
    input("\n⚠️  This will COPY files (not move). Press Enter to continue or Ctrl+C to abort...")
    
    create_structure()
    split_and_copy()
    verify_counts()
    
    print("\n🎉 Celeb-DF organization complete!")
    print("Ready for combined training with 140k dataset!")
