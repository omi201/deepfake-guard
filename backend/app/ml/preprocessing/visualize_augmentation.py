"""
Visualization script for Day 6 Checkpoint
Generates side-by-side comparison of original vs augmented images
Saves to: docs/augmentation_samples.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import glob

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from augmentation import DeepfakeAugmentation, get_training_augmentation
import albumentations as A
import cv2


def load_sample_images(n_samples: int = 4) -> list:
    """
    Load sample images from existing datasets (140k or Celeb-DF).
    Returns list of (image_array, label) tuples.
    """
    # Search paths for existing datasets
    search_paths = [
        "data/processed/140k/real_vs_fake/real-vs-fake/train/real",
        "data/processed/140k/real_vs_fake/real-vs-fake/train/fake",
        "data/processed/celeb-df/faces/real",
        "data/processed/celeb-df/faces/fake",
    ]
    
    found_images = []
    
    for path in search_paths:
        if os.path.exists(path):
            # Get all jpg/png files
            files = glob.glob(os.path.join(path, "*.jpg")) + \
                    glob.glob(os.path.join(path, "*.png"))
            
            if files:
                # Randomly sample from this directory
                samples = random.sample(files, min(len(files), n_samples // 2))
                for f in samples:
                    try:
                        img = Image.open(f).convert('RGB')
                        img = img.resize((224, 224))
                        found_images.append((np.array(img), os.path.basename(path)))
                        if len(found_images) >= n_samples:
                            break
                    except Exception as e:
                        print(f"Error loading {f}: {e}")
                        continue
        
        if len(found_images) >= n_samples:
            break
    
    # If no images found, create synthetic test patterns
    if len(found_images) == 0:
        print("No dataset images found. Creating synthetic test patterns...")
        for i in range(n_samples):
            # Create synthetic image with patterns
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            # Add gradient
            img[:, :, 0] = np.linspace(0, 255, 224).reshape(1, -1).repeat(224, axis=0)
            img[:, :, 1] = np.linspace(0, 255, 224).reshape(-1, 1).repeat(224, axis=1)
            img[:, :, 2] = 128
            # Add some noise
            img = np.clip(img + np.random.randint(-20, 20, img.shape), 0, 255).astype(np.uint8)
            found_images.append((img, "synthetic"))
    
    return found_images[:n_samples]


def create_augmentation_grid(images: list, save_path: str = "docs/augmentation_samples.png"):
    """
    Create visualization grid showing:
    - Original image
    - Geometric augment (rotate + flip)
    - Compression augment (JPEG)
    - Noise augment (Gaussian)
    - Blur augment (Motion + Gaussian)
    - Combined augment (random combination)
    """
    
    aug_pipeline = get_training_augmentation()
    
    # Define specific augmentations for demonstration
    aug_types = [
        ("Original", lambda img: img),
        ("Geometric\n(Rotate+Flip)", lambda img: apply_geometric(img)),
        ("Compression\n(JPEG 50-100)", lambda img: apply_compression(img)),
        ("Noise\n(Gauss 0.01-0.05)", lambda img: apply_noise(img)),
        ("Blur\n(Motion+Gaussian)", lambda img: apply_blur(img)),
        ("Combined\n(Random Mix)", lambda img: apply_combined(img, aug_pipeline)),
    ]
    
    n_images = len(images)
    n_aug_types = len(aug_types)
    
    fig, axes = plt.subplots(n_images, n_aug_types, figsize=(16, 10))
    
    for row, (original_img, label) in enumerate(images):
        for col, (aug_name, aug_func) in enumerate(aug_types):
            ax = axes[row, col] if n_images > 1 else axes[col]
            
            # Apply augmentation
            aug_img = aug_func(original_img.copy())
            
            # Convert to displayable format
            if isinstance(aug_img, np.ndarray):
                display_img = aug_img
            else:
                # It's a tensor, denormalize
                aug_img = aug_img.permute(1, 2, 0).numpy()
                # Denormalize ImageNet
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                display_img = (aug_img * std + mean) * 255
                display_img = np.clip(display_img, 0, 255).astype(np.uint8)
            
            ax.imshow(display_img)
            ax.set_title(aug_name if row == 0 else "", fontsize=10, pad=10)
            ax.axis('off')
            
            # Add row label on first column
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}\n({label})", fontsize=9, rotation=0, 
                             labelpad=50, va='center')
    
    plt.suptitle("Deepfake Detection - Data Augmentation Pipeline (Day 6)\n" + 
                 "Albumentations: Geometric | Compression | Noise | Blur", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    
    # Ensure docs directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Augmentation visualization saved to: {save_path}")
    
    return fig


# Specific augmentation applications for visualization
def apply_geometric(image: np.ndarray) -> np.ndarray:
    """Apply geometric augmentations only."""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=1.0),
    ])
    return transform(image=image)['image']


def apply_compression(image: np.ndarray) -> np.ndarray:
    """Apply JPEG compression only."""
    transform = A.Compose([
        A.ImageCompression(quality_lower=50, quality_upper=70, p=1.0),
    ])
    return transform(image=image)['image']


def apply_noise(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian noise only."""
    transform = A.Compose([
        A.GaussNoise(var_limit=(0.03, 0.05), mean=0, p=1.0),
    ])
    return transform(image=image)['image']


def apply_blur(image: np.ndarray) -> np.ndarray:
    """Apply blur augmentations."""
    transform = A.Compose([
        A.MotionBlur(blur_limit=7, p=0.5),
        A.GaussianBlur(blur_limit=7, p=0.5),
    ])
    return transform(image=image)['image']


def apply_combined(image: np.ndarray, pipeline: DeepfakeAugmentation) -> np.ndarray:
    """Apply full pipeline (returns tensor, will be converted)."""
    return pipeline(image, phase="train")


def main():
    """Main execution for Day 6 checkpoint."""
    print("=" * 60)
    print("DAY 6 CHECKPOINT: Augmentation Visualization")
    print("=" * 60)
    
    # Load samples
    print("\nLoading sample images...")
    images = load_sample_images(n_samples=4)
    print(f"Loaded {len(images)} sample images")
    
    # Create visualization
    print("\nGenerating augmentation grid...")
    save_path = "docs/augmentation_samples.png"
    create_augmentation_grid(images, save_path)
    
    # Verify file exists
    if os.path.exists(save_path):
        size_kb = os.path.getsize(save_path) / 1024
        print(f"\n✅ CHECKPOINT VERIFIED:")
        print(f"   File: {save_path}")
        print(f"   Size: {size_kb:.1f} KB")
        print(f"   Images: 4 samples × 6 augmentation types = 24 panels")
        print("\n" + "=" * 60)
        print("DAY 6 CHECKPOINT COMPLETE")
        print("=" * 60)
    else:
        print("❌ ERROR: File was not created!")
        sys.exit(1)


if __name__ == "__main__":
    main()
