"""
Phase 3c: Manual PGD Adversarial Training (NO FOOLBOX for training loop)
Uses native PyTorch for speed and stability
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
print("PHASE 3c: MANUAL PGD ADVERSARIAL TRAINING")
print("Native PyTorch implementation - faster, no wrapper issues")
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

class AdversarialDataset(Dataset):
    def __init__(self, transform=None, max_samples=8000):
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
        return self.transform(img) if self.transform else img, label

def pgd_attack(model, images, labels, eps=0.03, alpha=0.0075, steps=7):
    """
    Manual PGD implementation - no Foolbox needed
    eps: max perturbation (0.03)
    alpha: step size (eps/4 = 0.0075)
    steps: number of iterations
    """
    model.eval()
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    # Random initialization within epsilon ball
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta.requires_grad = True
    
    for _ in range(steps):
        outputs = model(images + delta)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        loss.backward()
        
        # Gradient ascent on delta (maximize loss)
        delta.data = delta.data + alpha * delta.grad.sign()
        
        # Project back to epsilon ball
        delta.data = torch.clamp(delta.data, -eps, eps)
        
        # Ensure images + delta stay in valid range (-1 to 1 due to normalization)
        delta.data = torch.clamp(images + delta.data, -1, 1) - images
        
        delta.grad.zero_()
    
    return images + delta.detach()

# Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print("[1/3] Loading dataset...")
full_dataset = AdversarialDataset(transform, max_samples=8000)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)  # Bigger batch
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

print("[2/3] Initializing model...")
model = Meso4().to(DEVICE)
checkpoint = torch.load("data/models/meso4_baseline_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
print("✓ Warm start from baseline")

optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam for stability
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
criterion = nn.CrossEntropyLoss()

best_robust_acc = 0.0
SAVE_PATH = "data/models/meso4_pgd_hardened.pth"

print(f"[3/3] Starting manual PGD training...")
print(f"{'='*70}")

history = {"epochs": [], "clean_acc": [], "robust_acc": [], "train_loss": []}

for epoch in range(10):
    # Curriculum
    if epoch < 3:
        steps, mix_ratio = 3, 0.5
    elif epoch < 6:
        steps, mix_ratio = 5, 0.6
    else:
        steps, mix_ratio = 7, 0.7
    
    print(f"\nEpoch {epoch+1}/10 | Manual PGD-{steps} | Mix: {mix_ratio*100:.0f}%")
    
    model.train()
    total_loss = 0
    clean_correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Training")
    
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Generate adversarial examples MANUALLY
        advs = pgd_attack(model, images, labels, eps=0.03, alpha=0.0075, steps=steps)
        
        # Mix
        batch_size = images.size(0)
        n_adv = int(batch_size * mix_ratio)
        
        mixed_images = torch.cat([advs[:n_adv], images[n_adv:]])
        mixed_labels = torch.cat([labels[:n_adv], labels[n_adv:]])
        
        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = criterion(outputs, mixed_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        clean_correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})
    
    train_clean_acc = clean_correct / total
    avg_loss = total_loss / len(train_loader)
    
    # Validation
    print(f"  Running validation...")
    model.eval()
    val_clean_correct = 0
    val_robust_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
            
            # Clean accuracy
            val_outputs = model(val_images)
            val_clean_correct += (val_outputs.argmax(dim=1) == val_labels).sum().item()
            
            # Robust accuracy - generate PGD-7 attacks
            val_advs = pgd_attack(model, val_images, val_labels, eps=0.03, alpha=0.0075, steps=7)
            robust_outputs = model(val_advs)
            val_robust_correct += (robust_outputs.argmax(dim=1) == val_labels).sum().item()
            val_total += val_labels.size(0)
    
    val_clean_acc = val_clean_correct / val_total
    val_robust_acc = val_robust_correct / val_total
    
    history["epochs"].append(epoch+1)
    history["clean_acc"].append(val_clean_acc)
    history["robust_acc"].append(val_robust_acc)
    history["train_loss"].append(avg_loss)
    
    print(f"  Summary: Loss={avg_loss:.4f} | Clean={val_clean_acc:.2%} | Robust={val_robust_acc:.2%}")
    
    if val_robust_acc > best_robust_acc:
        best_robust_acc = val_robust_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'robust_accuracy': val_robust_acc,
            'clean_accuracy': val_clean_acc,
            'epoch': epoch+1,
        }, SAVE_PATH)
        print(f"  💾 NEW BEST: {best_robust_acc:.2%}")
    
    if val_robust_acc >= 0.85:
        print(f"\n🎉 TARGET REACHED!")
        break
    
    scheduler.step(val_robust_acc)  # LR scheduler based on robust accuracy

with open("data/models/adversarial_training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*70}")
print("MANUAL PGD TRAINING COMPLETE")
print(f"Best Robust Accuracy: {best_robust_acc:.2%}")
print(f"{'='*70}")
