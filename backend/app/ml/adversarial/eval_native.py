"""
Ultra-Fast Evaluation - Native PyTorch (No Foolbox)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import json
import time

print(f"{'='*70}")
print("FAST EVALUATION - NATIVE PYTORCH")
print(f"{'='*70}")

DEVICE = torch.device("cuda")

class Meso4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(64*14*14, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)

class TestDataset(Dataset):
    def __init__(self):
        self.samples = []
        t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
        self.transform = t
        base = Path("data/processed/140k/real_vs_fake/real-vs-fake/valid")
        for f in list((base / "real").iterdir())[:50]: self.samples.append((f, 0))
        for f in list((base / "fake").iterdir())[:50]: self.samples.append((f, 1))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert('RGB')), label

def fgsm_attack(model, x, y, eps):
    """Fast FGSM"""
    x_adv = x.clone().detach().requires_grad_(True)
    out = model(x_adv)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    grad = x_adv.grad.sign()
    return (x + eps * grad).detach()

def pgd_attack(model, x, y, eps, steps=7):
    """Fast PGD"""
    alpha = eps / 4
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        out = model(x + delta)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return (x + delta).detach()

def evaluate(model_path, name):
    print(f"\nEvaluating {name}...")
    model = Meso4().to(DEVICE)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt.get('state_dict') or ckpt.get('model_state_dict'))
    model.eval()
    
    # Load data
    ds = TestDataset()
    loader = DataLoader(ds, batch_size=100, shuffle=False)
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    print(f"  Samples: {len(x)}")
    
    # Clean
    with torch.no_grad():
        clean = (model(x).argmax(1) == y).float().mean().item()
    print(f"  Clean: {clean:.2%}")
    
    # FGSM
    results = {"clean": clean, "fgsm": {}, "pgd7": 0}
    for eps in [0.01, 0.03, 0.05, 0.1]:
        x_adv = fgsm_attack(model, x, y, eps)
        with torch.no_grad():
            acc = (model(x_adv).argmax(1) == y).float().mean().item()
        results["fgsm"][f"eps_{eps}"] = acc
        print(f"  FGSM ε={eps}: {acc:.2%}")
    
    # PGD-7
    x_adv = pgd_attack(model, x, y, 0.03, 7)
    with torch.no_grad():
        pgd = (model(x_adv).argmax(1) == y).float().mean().item()
    results["pgd7"] = pgd
    print(f"  PGD-7: {pgd:.2%}")
    
    return results

# Run
start = time.time()
baseline = evaluate("data/models/meso4_baseline_best.pth", "BASELINE")
hardened = evaluate("data/models/meso4_pgd_hardened.pth", "HARDENED")

# Report
print(f"\n{'='*70}")
print("DAY 1 REPORT")
print(f"{'='*70}")
print(f"{'Attack':<12} {'Baseline':<12} {'Hardened':<12} {'Δ':<12}")
print("-" * 50)
print(f"{'Clean':<12} {baseline['clean']:<12.2%} {hardened['clean']:<12.2%} {hardened['clean']-baseline['clean']:+12.2%}")
for eps in [0.01, 0.03, 0.05, 0.1]:
    b, h = baseline['fgsm'][f'eps_{eps}'], hardened['fgsm'][f'eps_{eps}']
    print(f"{'FGSM'+str(eps):<12} {b:<12.2%} {h:<12.2%} {h-b:+12.2%}")
print(f"{'PGD-7':<12} {baseline['pgd7']:<12.2%} {hardened['pgd7']:<12.2%} {hardened['pgd7']-baseline['pgd7']:+12.2%}")

# Save
with open("data/models/day1_final.json", 'w') as f:
    json.dump({"baseline": baseline, "hardened": hardened}, f, indent=2)

print(f"\nTime: {time.time()-start:.1f}s")
print("✅ Day 1 Complete - Ready for Day 2")
