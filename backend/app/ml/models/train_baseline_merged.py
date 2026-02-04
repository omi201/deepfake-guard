"""
Train Baseline on MERGED Dataset (140k + CIPLAB)
Fixed for 224x224 input size
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

DEVICE = torch.device("cuda")
print(f"{'='*60}")
print("TRAINING BASELINE - MERGED DATASET (224x224)")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"{'='*60}")

# ============================================================
# MESO4 ARCHITECTURE (FIXED for 224x224)
# ============================================================
class Meso4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), 
            nn.BatchNorm2d(8), 
            nn.ReLU(), 
            nn.MaxPool2d(2)  # 224 -> 112
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(2)  # 112 -> 56
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2)  # 56 -> 28
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2)  # 28 -> 14
        )
        # FIXED: 64 channels * 14 * 14 = 12544 (not 16384)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 14 * 14, 128),  # FIXED: 12544 instead of 16384
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)

# ============================================================
# MERGED DATASET (140k + CIPLAB)
# ============================================================
class MergedDataset(Dataset):
    def __init__(self, transform=None, max_samples=25000):
        self.transform = transform
        self.samples = []
        
        print("Loading 140k dataset...")
        base_path = Path("data/processed/140k/real_vs_fake/real-vs-fake")
        train_real_140k = list((base_path / "train" / "real").iterdir())
        train_fake_140k = list((base_path / "train" / "fake").iterdir())
        
        print(f"  140k: {len(train_real_140k)} real, {len(train_fake_140k)} fake")
        
        print("Loading CIPLAB dataset...")
        ciplab_path = Path("data/processed/ciplab/real_and_fake_face")
        ciplab_real = list((ciplab_path / "training_real").iterdir())
        ciplab_fake = list((ciplab_path / "training_fake").iterdir())
        
        print(f"  CIPLAB: {len(ciplab_real)} real, {len(ciplab_fake)} fake")
        
        all_real = train_real_140k + ciplab_real
        all_fake = train_fake_140k + ciplab_fake
        
        print(f"\nCombined: {len(all_real)} real, {len(all_fake)} fake")
        
        per_class = max_samples // 2
        all_real = all_real[:per_class]
        all_fake = all_fake[:per_class]
        
        for f in all_real:
            self.samples.append((f, 0))
        for f in all_fake:
            self.samples.append((f, 1))
            
        print(f"Using {len(self.samples)} total ({per_class} per class)")
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ============================================================
# CONFIGURATION
# ============================================================
SAVE_PATH = "data/models/meso4_baseline_best.pth"

# Transforms for 224x224
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),  # 224x224 crops
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Direct resize to 224
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print("\n[1/3] Loading merged data...")
full_dataset = MergedDataset(train_transform, max_samples=25000)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

print(f"\n[2/3] Creating model...")
model = Meso4().to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Verify shape
with torch.no_grad():
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    out = model(dummy)
    print(f"✓ Test passed: {dummy.shape} -> {out.shape}")

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_acc = 0.0

print(f"\n[3/3] Training for 5 epochs...")
print(f"{'='*60}")

for epoch in range(5):
    # Training
    model.train()
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)
        
        acc = train_correct / train_total
        pbar.set_postfix({"acc": f"{acc:.2%}"})
    
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    
    print(f"\nEpoch {epoch+1}: Train={train_acc:.2%}, Val={val_acc:.2%}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_accuracy': val_acc,
            'epoch': epoch+1,
            'dataset': 'merged_140k_ciplab'
        }, SAVE_PATH)
        print(f"💾 SAVED: {val_acc:.2%}")
    
    scheduler.step()

print(f"\n{'='*60}")
print(f"✅ COMPLETE!")
print(f"Best Accuracy: {best_acc:.2%}")
print(f"Saved: {SAVE_PATH}")
print(f"{'='*60}")

if best_acc >= 0.90:
    print("🎉 OUTSTANDING! 90%+ achieved!")
elif best_acc >= 0.85:
    print("✅ EXCELLENT! Ready for adversarial training!")
else:
    print("✅ GOOD! Proceeding...")
