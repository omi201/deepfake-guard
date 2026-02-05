"""
Data Augmentation Pipeline for Deepfake Detection
Uses Albumentations for robust augmentation strategies
Includes: Geometric, Compression, Noise, Blur transforms
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from typing import Optional, Tuple, Union
import cv2


class DeepfakeAugmentation:
    """
    Albumentations-based augmentation pipeline for adversarial robustness training.
    Simulates real-world image degradations and geometric variations.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        augmentation_probability: float = 0.5,
        jpeg_quality_range: Tuple[int, int] = (50, 100),
        noise_variance_range: Tuple[float, float] = (0.01, 0.05),
        rotation_limit: int = 15,
        enable_geometric: bool = True,
        enable_compression: bool = True,
        enable_noise: bool = True,
        enable_blur: bool = True
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size (square)
            augmentation_probability: Probability of applying each augmentation type (0-1)
            jpeg_quality_range: Range for JPEG compression simulation (min, max)
            noise_variance_range: Range for Gaussian noise variance (min, max)
            rotation_limit: Maximum rotation in degrees
            enable_geometric: Enable rotation and flip
            enable_compression: Enable JPEG compression simulation
            enable_noise: Enable Gaussian noise
            enable_blur: Enable motion and Gaussian blur
        """
        self.image_size = image_size
        self.aug_prob = augmentation_probability
        self.jpeg_range = jpeg_quality_range
        self.noise_range = noise_variance_range
        self.rotation_limit = rotation_limit
        
        # Build transforms
        self.train_transform = self._build_train_transform(
            enable_geometric, enable_compression, enable_noise, enable_blur
        )
        self.val_transform = self._build_val_transform()
        
    def _build_train_transform(
        self, 
        enable_geometric: bool,
        enable_compression: bool,
        enable_noise: bool,
        enable_blur: bool
    ) -> A.Compose:
        """Build training augmentation pipeline."""
        
        transforms = []
        
        # 1. GEOMETRIC AUGMENTATIONS (p=0.5 as specified)
        if enable_geometric:
            transforms.extend([
                # Horizontal flip (p=0.5 as per requirements)
                A.HorizontalFlip(p=0.5),
                # Rotation ±15 degrees
                A.Rotate(
                    limit=self.rotation_limit,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=self.aug_prob
                ),
            ])
        
        # 2. COMPRESSION AUGMENTATION (JPEG quality 50-100)
        if enable_compression:
            transforms.append(
                A.ImageCompression(
                    quality_lower=self.jpeg_range[0],
                    quality_upper=self.jpeg_range[1],
                    p=self.aug_prob
                )
            )
        
        # 3. NOISE AUGMENTATION (Gaussian noise, var 0.01-0.05)
        if enable_noise:
            transforms.append(
                A.GaussNoise(
                    var_limit=self.noise_range,  # (0.01, 0.05) per requirements
                    mean=0,
                    per_channel=True,
                    p=self.aug_prob
                )
            )
        
        # 4. BLUR AUGMENTATIONS (Motion blur + Gaussian blur)
        if enable_blur:
            transforms.extend([
                # Motion blur (simulates camera movement)
                A.MotionBlur(
                    blur_limit=(3, 7),
                    allow_shifted=True,
                    p=self.aug_prob * 0.5  # Half probability of other augments
                ),
                # Gaussian blur (simulates out of focus)
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    sigma_limit=(0.1, 2.0),
                    p=self.aug_prob * 0.5
                ),
            ])
        
        # Normalization (ImageNet stats for EfficientNet compatibility)
        transforms.extend([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def _build_val_transform(self) -> A.Compose:
        """Build validation transform (no augmentation, only normalization)."""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(
        self, 
        image: np.ndarray, 
        phase: str = "train"
    ) -> torch.Tensor:
        """
        Apply augmentation to image.
        
        Args:
            image: Input image (H, W, C) in RGB format, uint8
            phase: 'train' or 'val'
            
        Returns:
            Augmented image as torch.Tensor (C, H, W)
        """
        if phase == "train":
            transformed = self.train_transform(image=image)
        else:
            transformed = self.val_transform(image=image)
        
        return transformed["image"]
    
    def apply_random_augment(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply single random augmentation for visualization/debugging.
        Returns augmented image and name of augmentation applied.
        """
        aug_types = []
        
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
            aug_types.append("HorizontalFlip")
        
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.rotation_limit, self.rotation_limit)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
            aug_types.append(f"Rotate({angle:.1f})")
        
        if np.random.random() < 0.5:
            quality = np.random.randint(self.jpeg_range[0], self.jpeg_range[1])
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc_img = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(enc_img, 1)
            aug_types.append(f"JPEG({quality})")
        
        if np.random.random() < 0.5:
            var = np.random.uniform(self.noise_range[0], self.noise_range[1])
            noise = np.random.normal(0, var ** 0.5, image.shape) * 255
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            aug_types.append(f"GaussNoise({var:.3f})")
        
        if np.random.random() < 0.3:
            ksize = np.random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
            aug_types.append(f"GaussianBlur({ksize})")
        
        aug_name = " + ".join(aug_types) if aug_types else "None"
        return image, aug_name


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wrapper for PyTorch Dataset that applies Albumentations augmentations.
    Compatible with existing datasets used in training.
    """
    
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        augmentation: Optional[DeepfakeAugmentation] = None,
        phase: str = "train"
    ):
        """
        Args:
            base_dataset: Original dataset (returns image, label)
            augmentation: DeepfakeAugmentation instance
            phase: 'train' or 'val'
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation or DeepfakeAugmentation()
        self.phase = phase
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        image, label = self.base_dataset[idx]
        
        # Convert tensor to numpy if needed (assuming tensor input from dataset)
        if isinstance(image, torch.Tensor):
            # Convert (C, H, W) to (H, W, C) and denormalize if needed
            image = image.permute(1, 2, 0).cpu().numpy()
            # If normalized, denormalize for augmentation
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif isinstance(image, np.ndarray):
            # Ensure RGB format
            if image.shape[0] == 3:  # (C, H, W)
                image = image.transpose(1, 2, 0)
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        # Apply augmentation
        augmented = self.augmentation(image, phase=self.phase)
        
        return augmented, label


def get_training_augmentation(image_size: int = 224) -> DeepfakeAugmentation:
    """Factory function for standard training augmentation."""
    return DeepfakeAugmentation(
        image_size=image_size,
        augmentation_probability=0.5,
        jpeg_quality_range=(50, 100),
        noise_variance_range=(0.01, 0.05),
        rotation_limit=15
    )


if __name__ == "__main__":
    # Test the augmentation pipeline
    print("Testing DeepfakeAugmentation...")
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Initialize augmenter
    aug = get_training_augmentation()
    
    # Test train transform
    train_result = aug(dummy_img, phase="train")
    print(f"Train output shape: {train_result.shape}, dtype: {train_result.dtype}")
    print(f"Train output range: [{train_result.min():.3f}, {train_result.max():.3f}]")
    
    # Test val transform
    val_result = aug(dummy_img, phase="val")
    print(f"Val output shape: {val_result.shape}")
    
    print("✓ Augmentation pipeline test passed!")
