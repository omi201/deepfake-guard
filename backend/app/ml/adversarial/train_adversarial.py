"""
Phase 3: Adversarial Training - Madry Protocol (FIXED)
Minimax: min_θ E[max_δ L(θ, x+δ)]
Inner loop: PGD-7, Outer loop: 10 epochs
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
import foolbox as fb
import numpy as np

print(f"{'='*70}")
print("PHASE 3: ADVERSARIAL TRAINING (Madry Protocol)")
print("PGD-7 Inner Loop | 10 Epochs | Target: >85% Robust Accuracy")
print(f"{'='*70}")

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
class AdversarialDataset(Dataset):
    def __init__(self, transform=None, max_samples=8000):
        self.samples = []
        self.transform = transform
        
        print("Loading training data for adversarial training...")
        base_path = Path("data/processed/140k/real_vs_fake/real-vs-fake/train")
        real_files = list((base_path / "real").iterdir())[:4000]
        fake_files = list((base_path / "fake").iterdir())[:4000]
        
        for f in real_files:
            self.samples.append((f, 0))
        for f in fake_files:
            self.samples.append((f, 1))
            
        print(f"Loaded {len(self.samples)} samples (4000 real, 4000 fake)")
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ============================================================
# SETUP
# ============================================================
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

print("[1/4] Loading dataset...")
full_dataset = AdversarialDataset(transform, max_samples=8000)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

print("[2/4] Initializing model...")
model = Meso4().to(DEVICE)

# Load baseline weights
try:
    baseline_ckpt = torch.load("data/models/meso4_baseline_best.pth")
    model.load_state_dict(baseline_ckpt['model_state_dict'])
    print("✓ Warm start from baseline")
except:
    print("! Training from scratch")

preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
fmodel = fb.PyTorchModel(model, bounds=(-1, 1), preprocessing=preprocessing)

attack = fb.attacks.PGD(steps=7, rel_stepsize=0.25)
epsilon_train = 0.03

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
criterion = nn.CrossEntropyLoss()

best_robust_acc = 0.0
SAVE_PATH = "data/models/meso4_pgd_hardened.pth"

print(f"[3/4] Starting adversarial training...")
print(f"     Epochs: 10 | Batch: 32 | PGD-7 | ε={epsilon_train}")
print(f"{'='*70}")

history = {
    "epochs": [],
    "clean_acc": [],
    "robust_acc": [],
    "train_loss": []
}

for epoch in range(10):
    model.train()
    total_loss = 0
    clean_correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
    
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # INNER LOOP: Generate adversarial examples
        model.eval()
        torch.set_grad_enabled(True)  # Ensure gradients for attack
        _, advs, _ = attack(fmodel, images, labels, epsilons=epsilon_train)
        torch.set_grad_enabled(False)  # Disable temporarily
        model.train()
        
        # OUTER LOOP: Train on mix
        mix_ratio = 0.7
        batch_size = images.size(0)
        n_adv = int(batch_size * mix_ratio)
        
        mixed_images = torch.cat([advs[:n_adv], images[n_adv:]])
        mixed_labels = torch.cat([labels[:n_adv], labels[n_adv:]])
        
        optimizer.zero_grad()
        torch.set_grad_enabled(True)  # Re-enable for training
        outputs = model(mixed_images)
        loss = criterion(outputs, mixed_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.set_grad_enabled(False)
        
        total_loss += loss.item()
        clean_correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})
    
    train_clean_acc = clean_correct / total
    avg_loss = total_loss / len(train_loader)
    
    # VALIDATION - FIXED: Enable gradients for attack
    print(f"\n  Running validation...")
    model.eval()
    val_clean_correct = 0
    val_robust_correct = 0
    val_total = 0
    
    for val_images, val_labels in val_loader:
        val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
        
        # Clean accuracy (no grad needed)
        with torch.no_grad():
            val_outputs = model(val_images)
            val_clean_correct += (val_outputs.argmax(dim=1) == val_labels).sum().item()
        
        # Robust accuracy - NEEDS GRADIENTS for attack
        torch.set_grad_enabled(True)
        _, val_advs, val_success = attack(fmodel, val_images, val_labels, epsilons=epsilon_train)
        torch.set_grad_enabled(False)
        
        val_robust_correct += (val_success == False).sum().item()
        val_total += val_labels.size(0)
    
    val_clean_acc = val_clean_correct / val_total
    val_robust_acc = val_robust_correct / val_total
    
    history["epochs"].append(epoch+1)
    history["clean_acc"].append(val_clean_acc)
    history["robust_acc"].append(val_robust_acc)
    history["train_loss"].append(avg_loss)
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_loss:.4f} | Train Clean: {train_clean_acc:.2%}")
    print(f"  Val Clean:  {val_clean_acc:.2%} | Val Robust: {val_robust_acc:.2%}")
    
    if val_robust_acc > best_robust_acc:
        best_robust_acc = val_robust_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'robust_accuracy': val_robust_acc,
            'clean_accuracy': val_clean_acc,
            'epoch': epoch+1,
            'epsilon': epsilon_train,
            'pgd_steps': 7,
            'protocol': 'madry_pgd7'
        }, SAVE_PATH)
        print(f"  💾 SAVED (Best Robust: {best_robust_acc:.2%})")
    
    if val_robust_acc >= 0.85:
        print(f"\n🎉 TARGET REACHED: {val_robust_acc:.2%} robust accuracy!")
        break
    
    scheduler.step()
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

with open("data/models/adversarial_training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*70}")
print("ADVERSARIAL TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Best Robust Accuracy: {best_robust_acc:.2%}")
print(f"Target: >85% | Status: {'✅ ACHIEVED' if best_robust_acc >= 0.85 else '⚠️ BELOW TARGET'}")
print(f"Model saved: {SAVE_PATH}")
print(f"{'='*70}")
