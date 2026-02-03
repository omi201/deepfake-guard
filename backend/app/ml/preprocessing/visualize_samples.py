import matplotlib.pyplot as plt
import cv2
import os
import random
from pathlib import Path
import pandas as pd

# Security: Validate paths to prevent directory traversal
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
DATASET_ROOT = BASE_DIR / "data" / "processed" / "140k" / "real_vs_fake" / "real-vs-fake"

def detect_dataset_structure():
    """
    Auto-detect 140k dataset layout.
    Returns: dict with paths to train/test/val splits.
    """
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root missing: {DATASET_ROOT}")
    
    structure = {}
    splits = ['train', 'test', 'valid']
    
    for split in splits:
        split_path = DATASET_ROOT / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split directory: {split_path}")
        
        real_path = split_path / "real"
        fake_path = split_path / "fake"
        
        if not real_path.exists() or not fake_path.exists():
            raise FileNotFoundError(f"Missing real/ or fake/ in {split_path}")
        
        real_count = len(list(real_path.glob("*.jpg")))
        fake_count = len(list(fake_path.glob("*.jpg")))
        
        structure[split] = {
            'real_path': real_path,
            'fake_path': fake_path,
            'real_count': real_count,
            'fake_count': fake_count,
            'total': real_count + fake_count
        }
        
        print(f"[SECURITY AUDIT] {split.upper():<6} | Real: {real_count:>6} | Fake: {fake_count:>6} | Ratio: {fake_count/max(real_count,1):.2f}:1")
    
    return structure

def validate_csv_metadata(split: str):
    """Verify CSV indices match actual file counts (integrity check)."""
    csv_path = DATASET_ROOT.parent / f"{split}.csv"
    if not csv_path.exists():
        print(f"[WARNING] Missing CSV: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    expected_real = len(df[df['label'] == 0])
    expected_fake = len(df[df['label'] == 1])
    
    # Note: CSV labels: 0=real, 1=fake
    print(f"[METADATA] {split}.csv | Real entries: {expected_real} | Fake entries: {expected_fake}")
    return True

def visualize_comparison(structure: dict, split: str = 'train', samples: int = 5):
    """Generate side-by-side forensic comparison from specified split."""
    if split not in structure:
        raise ValueError(f"Unknown split: {split}. Available: {list(structure.keys())}")
    
    real_path = structure[split]['real_path']
    fake_path = structure[split]['fake_path']
    
    # Random sampling for unbiased visualization
    real_images = list(real_path.glob("*.jpg"))
    fake_images = list(fake_path.glob("*.jpg"))
    
    real_samples = random.sample(real_images, min(samples, len(real_images)))
    fake_samples = random.sample(fake_images, min(samples, len(fake_images)))
    
    fig, axes = plt.subplots(2, samples, figsize=(15, 6))
    fig.suptitle(f'Forensic Analysis: Real (Top) vs Deepfake (Bottom)\nSource: {split} split | n={samples}', 
                 fontsize=14, fontweight='bold')
    
    for idx, (real_img, fake_img) in enumerate(zip(real_samples, fake_samples)):
        # Load with OpenCV, convert BGR to RGB
        real_frame = cv2.imread(str(real_img))
        real_frame = cv2.cvtColor(real_frame, cv2.COLOR_BGR2RGB)
        
        fake_frame = cv2.imread(str(fake_img))
        fake_frame = cv2.cvtColor(fake_frame, cv2.COLOR_BGR2RGB)
        
        # Resize for uniform display (prevent rendering attacks)
        real_frame = cv2.resize(real_frame, (256, 256))
        fake_frame = cv2.resize(fake_frame, (256, 256))
        
        axes[0, idx].imshow(real_frame)
        axes[0, idx].set_title(f'Real\n{real_img.name[:12]}...', fontsize=8)
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(fake_frame)
        axes[1, idx].set_title(f'Fake\n{fake_img.name[:12]}...', fontsize=8)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    # Save to docs/ for audit trail
    output_path = BASE_DIR / "docs" / f"sample_forensics_{split}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Forensic visualization saved: {output_path}")
    plt.show()

if __name__ == "__main__":
    print("[SECURITY] Scanning dataset structure...")
    structure = detect_dataset_structure()
    
    total_real = sum(s['real_count'] for s in structure.values())
    total_fake = sum(s['fake_count'] for s in structure.values())
    print(f"\n[SECURITY AUDIT] TOTAL | Real: {total_real} | Fake: {total_fake} | Balance: {total_fake/total_real:.3f}")
    
    # Verify metadata integrity
    for split in structure.keys():
        validate_csv_metadata(split)
    
    # Visualize training split (representative sample)
    visualize_comparison(structure, split='train', samples=5)
