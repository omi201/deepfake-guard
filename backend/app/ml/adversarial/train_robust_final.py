"""
Phase 3 FINAL: Robust Adversarial Training (Production Ready)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json

print(f"{'='*70}")
print("ADVERSARIAL TRAINING - FINAL IMPLEMENTATION")
print(f"{'='*70}")

DEVICE = torch.device("cuda")

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

class AdvDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        base_path = Path("data/processed/140k/real_vs_fake/real-vs-fake/train")
        real_files = list((base_path / "real").iterdir())[:4000]
        fake_files = list((base_path / "fake").iterdir())[:4000]
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

def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, steps=7):
    """
    Stable PGD implementation - creates fresh graph every time
    """
    model.eval()
    
    # Clone inputs to avoid modifying originals
    x = images.clone().detach()
    y = labels.clone().detach()
    
    # Initialize perturbation on SAME device as x
    delta = torch.zeros_like(x, requires_grad=True)
    delta.data.uniform_(-eps, eps)
    
    for _ in range(steps):
        # Forward pass
        outputs = model(x + delta)
        loss = nn.CrossEntropyLoss()(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Update delta
        with torch.no_grad():
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.data = torch.clamp(x + delta.data, -1, 1) - x
        
        # Zero grad for next iteration
        if delta.grad is not None:
            delta.grad.zero_()
    
    return (x + delta).detach()

# Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

print("[1/3] Loading data...")
dataset = AdvDataset(transform)
train_ds, val_ds = random_split(dataset, [7200, 800])
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

print("[2/3] Model setup...")
model = Meso4().to(DEVICE)
ckpt = torch.load("data/models/meso4_baseline_best.pth")
model.load_state_dict(ckpt['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)
criterion = nn.CrossEntropyLoss()

best_robust = 0.0
save_path = "data/models/meso4_pgd_hardened.pth"

print("[3/3] Training...")
for epoch in range(10):
    steps, mix = (3, 0.5) if epoch < 3 else (5, 0.6) if epoch < 6 else (7, 0.7)
    print(f"\nEpoch {epoch+1}/10 | PGD-{steps} | Mix {int(mix*100)}%")
    
    # Training
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for imgs, lbls in tqdm(train_loader, desc="Train"):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        
        # Generate adversarial examples
        with torch.enable_grad():
            advs = pgd_attack(model, imgs, lbls, steps=steps)
        
        # Mix adversarial and clean
        n = int(imgs.size(0) * mix)
        mixed = torch.cat([advs[:n], imgs[n:]])
        targets = torch.cat([lbls[:n], lbls[n:]])
        
        # Train
        optimizer.zero_grad()
        outputs = model(mixed)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        correct += (model(imgs).argmax(1) == lbls).sum().item()
        total += lbls.size(0)
    
    train_acc = correct / total
    
    # Validation
    model.eval()
    val_clean = 0
    val_robust = 0
    val_total = 0
    
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        
        # Clean
        with torch.no_grad():
            val_clean += (model(imgs).argmax(1) == lbls).sum().item()
        
        # Robust (PGD-7)
        with torch.enable_grad():
            advs = pgd_attack(model, imgs, lbls, steps=7)
        
        with torch.no_grad():
            val_robust += (model(advs).argmax(1) == lbls).sum().item()
        
        val_total += lbls.size(0)
    
    clean_acc = val_clean / val_total
    robust_acc = val_robust / val_total
    
    print(f"  Clean: {clean_acc:.2%} | Robust: {robust_acc:.2%}")
    
    if robust_acc > best_robust:
        best_robust = robust_acc
        torch.save({
            'state_dict': model.state_dict(),
            'robust_acc': robust_acc,
            'clean_acc': clean_acc,
            'epoch': epoch+1
        }, save_path)
        print(f"  💾 SAVED: {best_robust:.2%}")
    
    if robust_acc >= 0.85:
        print("🎉 TARGET REACHED!")
        break
    
    scheduler.step(robust_acc)

print(f"\n{'='*70}")
print(f"DONE! Best Robust Accuracy: {best_robust:.2%}")
print(f"{'='*70}")
