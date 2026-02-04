"""
Phase 1: FGSM Attack Verification (CUDA-FIXED)
Tests baseline model vulnerability to Fast Gradient Sign Method
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
import foolbox as fb
import numpy as np
from PIL import Image
import json
import gc

print(f"{'='*60}")
print("PHASE 1: FGSM ATTACK VERIFICATION")
print(f"{'='*60}")

DEVICE = torch.device("cuda")

# ============================================================
# MESO4 MODEL (224x224 version)
# ============================================================
class Meso4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(64*14*14, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)

# ============================================================
# DATASET
# ============================================================
class TestDataset(Dataset):
    def __init__(self, max_samples=500):  # REDUCED for memory
        self.samples = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform = transform
        
        base_path = Path("data/processed/140k/real_vs_fake/real-vs-fake/valid")
        real_files = list((base_path / "real").iterdir())[:250]
        fake_files = list((base_path / "fake").iterdir())[:250]
        
        for f in real_files:
            self.samples.append((f, 0))
        for f in fake_files:
            self.samples.append((f, 1))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

print("[1/4] Loading model...")
model = Meso4().to(DEVICE)
checkpoint_path = "data/models/meso4_baseline_best.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Loaded: {checkpoint_path}")
print(f"  Original accuracy: {checkpoint.get('val_accuracy', 'N/A')}")

print("[2/4] Loading test data...")
test_dataset = TestDataset(max_samples=500)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # SMALLER BATCH

images = []
labels = []
for imgs, lbls in test_loader:
    images.append(imgs)
    labels.append(lbls)
    
images = torch.cat(images).to(DEVICE)
labels = torch.cat(labels).to(DEVICE)

print(f"✓ Test samples: {len(images)} (Real: {(labels==0).sum().item()}, Fake: {(labels==1).sum().item()})")

print("[3/4] Testing clean accuracy...")
with torch.no_grad():
    clean_preds = model(images).argmax(dim=1)
    clean_acc = (clean_preds == labels).float().mean().item()
print(f"✓ Clean Accuracy: {clean_acc:.2%}")

# FGSM Attack setup - SEQUENTIAL PROCESSING
attack = fb.attacks.FGSM()
epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]

print("[4/4] Running FGSM attacks (sequential, memory-cleared)...")
results = {"clean_accuracy": clean_acc, "attacks": {}}

for eps in epsilons:
    if eps == 0.0:
        continue
    
    print(f"\n  Testing ε = {eps}...")
    torch.cuda.empty_cache()  # CLEAR MEMORY BEFORE EACH EPSILON
    gc.collect()
    
    try:
        # Recreate model wrapper fresh each time (memory leak fix)
        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1), preprocessing=preprocessing)
        
        _, advs, success = attack(fmodel, images, labels, epsilons=eps)
        robust_acc = 1 - success.float().mean().item()
        
        results["attacks"][f"eps_{eps}"] = {
            "robust_accuracy": robust_acc,
            "attack_success_rate": 1 - robust_acc
        }
        print(f"    Robust Accuracy: {robust_acc:.2%} (Drop: {clean_acc - robust_acc:.2%})")
        
        # Clear variables
        del fmodel, advs, success
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"    ⚠️  Error at ε={eps}: {str(e)[:50]}...")
        results["attacks"][f"eps_{eps}"] = {"error": str(e)}

# Save results
results_path = "data/models/fgsm_baseline_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("FGSM VERIFICATION COMPLETE")
print(f"{'='*60}")
print(f"Results saved: {results_path}")
print(f"\nSummary:")
print(f"  Clean:     {clean_acc:.2%}")

for eps in [0.01, 0.03, 0.05, 0.1]:
    if f"eps_{eps}" in results["attacks"]:
        if "robust_accuracy" in results["attacks"][f"eps_{eps}"]:
            acc = results["attacks"][f"eps_{eps}"]["robust_accuracy"]
            print(f"  ε={eps:<5} -> {acc:.2%} ({clean_acc - acc:+.2%} drop)")
        else:
            print(f"  ε={eps:<5} -> [ERROR]")
    else:
        print(f"  ε={eps:<5} -> [MISSING]")

if "eps_0.01" in results["attacks"] and results["attacks"]["eps_0.01"]["robust_accuracy"] < clean_acc * 0.5:
    print(f"\n🔴 CRITICAL VULNERABILITY: >50% drop at ε=0.01")
    print("   Adversarial training URGENTLY required.")
