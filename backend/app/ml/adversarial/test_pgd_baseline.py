"""
Phase 2: PGD Attack Verification
Tests baseline model vulnerability to Projected Gradient Descent (stronger than FGSM)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
import foolbox as fb
from PIL import Image
import json
import gc

print(f"{'='*60}")
print("PHASE 2: PGD ATTACK VERIFICATION")
print("Parameters: 40 steps, α=0.01, ε=0.03 (L-infinity)")
print(f"{'='*60}")

DEVICE = torch.device("cuda")

# ============================================================
# MESO4 MODEL
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
    def __init__(self, max_samples=200):
        self.samples = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform = transform
        
        base_path = Path("data/processed/140k/real_vs_fake/real-vs-fake/valid")
        real_files = list((base_path / "real").iterdir())[:100]
        fake_files = list((base_path / "fake").iterdir())[:100]
        
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

print("[1/3] Loading model...")
model = Meso4().to(DEVICE)
checkpoint = torch.load("data/models/meso4_baseline_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Baseline accuracy: {checkpoint.get('val_accuracy', 'N/A')}")

print("[2/3] Loading test data (200 samples - PGD is slow)...")
test_dataset = TestDataset()
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

images = []
labels = []
for imgs, lbls in test_loader:
    images.append(imgs)
    labels.append(lbls)
    
images = torch.cat(images).to(DEVICE)
labels = torch.cat(labels).to(DEVICE)
print(f"✓ Test samples: {len(images)}")

# Clean accuracy
print("[3/3] Testing...")
with torch.no_grad():
    clean_preds = model(images).argmax(dim=1)
    clean_acc = (clean_preds == labels).float().mean().item()
print(f"✓ Clean Accuracy: {clean_acc:.2%}")

# PGD Attack - Iterative (stronger than FGSM)
# PGD is essentially FGSM repeated 40 times with step size 0.01
print("\n[4/3] Running PGD attack (this takes 1-2 minutes)...")
torch.cuda.empty_cache()
gc.collect()

preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
fmodel = fb.PyTorchModel(model, bounds=(-1, 1), preprocessing=preprocessing)

# PGD parameters: 40 steps, step size 0.01, epsilon 0.03
attack = fb.attacks.PGD(steps=40, rel_stepsize=0.25)  # rel_stepsize=0.25 means step=0.25*eps=0.0075
epsilon = 0.03

print(f"  Parameters: 40 iterations, ε={epsilon}, step size ~0.0075")
_, advs, success = attack(fmodel, images, labels, epsilons=epsilon)

pgd_acc = 1 - success.float().mean().item()
print(f"\n{'='*60}")
print("PGD ATTACK RESULTS")
print(f"{'='*60}")
print(f"Clean Accuracy:     {clean_acc:.2%}")
print(f"PGD Robust Accuracy: {pgd_acc:.2%}")
print(f"Attack Success Rate: {(1-pgd_acc):.2%}")
print(f"Performance Drop:    {clean_acc - pgd_acc:.2%}")

# Save results
results = {
    "attack_type": "PGD",
    "parameters": {"steps": 40, "epsilon": epsilon, "rel_stepsize": 0.25},
    "clean_accuracy": clean_acc,
    "pgd_robust_accuracy": pgd_acc,
    "attack_success_rate": 1 - pgd_acc,
    "vulnerability_severity": "CRITICAL" if pgd_acc < 0.1 else "HIGH" if pgd_acc < 0.3 else "MODERATE"
}

with open("data/models/pgd_baseline_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: data/models/pgd_baseline_results.json")
print(f"\nSeverity: {results['vulnerability_severity']}")

if pgd_acc < 0.05:
    print("🔴 CRITICAL: Model collapses under PGD (near 0% robust accuracy)")
    print("   Adversarial training (Madry Protocol) required immediately.")
elif pgd_acc < 0.2:
    print("🟠 HIGH: Model highly vulnerable to iterative attacks.")
else:
    print("🟡 MODERATE: Some resistance, but still vulnerable.")
