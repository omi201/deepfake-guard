"""
FGSM Vulnerability Assessment
Security Objective: Verify baseline model susceptibility to single-step adversarial attacks
"""
import torch
import torch.nn as nn
import foolbox as fb
import numpy as np
import json
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.meso4 import Meso4

# CONFIGURATION - Corrected filename
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_PATH = "data/models/meso4_baseline_best.pth"  # CORRECT: Using best checkpoint
RESULTS_PATH = "data/models/fgsm_baseline_results.json"
DATASET_PATH = "data/processed/140k/real_vs_fake"
BATCH_SIZE = 32
EPSILONS = [0.0, 0.01, 0.03, 0.05, 0.1]  # L-infinity bounds

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=1000):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"
        
        real_files = list(real_dir.glob("*.jpg"))[:max_samples//2]
        fake_files = list(fake_dir.glob("*.jpg"))[:max_samples//2]
        
        for f in real_files:
            self.samples.append(f)
            self.labels.append(0)
        for f in fake_files:
            self.samples.append(f)
            self.labels.append(1)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    print("[SECURITY] Starting FGSM vulnerability assessment...")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Target Model: {BASELINE_PATH}")
    
    # Load baseline model
    model = Meso4()
    checkpoint = torch.load(BASELINE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"[CHECKPOINT] Model loaded successfully")
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = DeepfakeDataset(DATASET_PATH, transform=transform, max_samples=1000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[CHECKPOINT] Dataset loaded: {len(dataset)} samples")
    
    # Foolbox model wrapper
    fmodel = fb.PyTorchModel(model, bounds=(-1, 1), preprocessing=dict(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-1
    ))
    
    # FGSM Attack
    attack = fb.attacks.LinfFastGradientAttack()
    results = {}
    
    print("[SECURITY] Executing FGSM attacks...")
    for epsilon in EPSILONS:
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if epsilon == 0.0:
                # Clean accuracy
                with torch.no_grad():
                    outputs = model(images)
                    preds = (outputs.squeeze() > 0.5).long()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            else:
                # Adversarial accuracy
                raw_advs, clipped_advs, success = attack(
                    fmodel, images, labels, epsilons=epsilon
                )
                
                with torch.no_grad():
                    outputs = model(clipped_advs)
                    preds = (outputs.squeeze() > 0.5).long()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
        
        accuracy = correct / total
        results[f"epsilon_{epsilon}"] = {
            "accuracy": accuracy,
            "samples": total
        }
        print(f"[RESULT] ε={epsilon}: Accuracy={accuracy:.4f}")
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            "attack_type": "FGSM",
            "model": "meso4_baseline_best",
            "samples": len(dataset),
            "results": results
        }, f, indent=2)
    
    print(f"[CHECKPOINT] Results saved: {RESULTS_PATH}")
    print("[SECURITY] Phase 1 complete.")

if __name__ == "__main__":
    main()
