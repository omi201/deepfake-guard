# backend/app/ml/ensemble/train_efficientnet.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import json
from datetime import datetime

# FIXED IMPORT - matches your model file
from efficientnet_model import EfficientNetB3Deepfake

CONFIG = {
    'data_dirs': [
        'data/processed/140k/real_vs_fake/real-vs-fake/train',
        'data/processed/celeb-df/faces/train'
    ],
    'val_dirs': [
        'data/processed/140k/real_vs_fake/real-vs-fake/test',
        'data/processed/celeb-df/faces/test'
    ],
    'batch_size': 32,
    'num_workers': 2,
    'image_size': 224,
    'epochs_phase1': 5,
    'epochs_phase2': 5,
    'epochs_phase3': 10,
    'lr_phase1': 1e-3,
    'lr_phase2': 1e-4,
    'lr_phase3': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_dir': 'data/models',
    'best_model_path': 'data/models/efficientnet_b3_best.pth'
}

def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def create_dataset(data_dirs, transform):
    datasets_list = []
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            ds = datasets.ImageFolder(data_dir, transform=transform)
            datasets_list.append(ds)
            print(f"✅ Loaded {len(ds)} images from {data_dir}")
        else:
            print(f"⚠️  Not found: {data_dir}")
    
    if not datasets_list:
        raise RuntimeError("No datasets found!")
    return torch.utils.data.ConcatDataset(datasets_list)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def main():
    print(f"🚀 EfficientNet-B3 Training | Device: {CONFIG['device']}")
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    # Phase 1: Frozen
    print("\n=== Phase 1: Frozen Backbone ===")
    model = EfficientNetB3Deepfake()
    model.freeze_backbone()
    model = model.to(CONFIG['device'])
    
    # Check data exists
    if not os.path.exists(CONFIG['data_dirs'][0]):
        print("❌ ERROR: Dataset not found at:", CONFIG['data_dirs'][0])
        print("💡 Run Step 3 first to organize datasets!")
        return
    
    train_ds = create_dataset(CONFIG['data_dirs'], get_transforms('train'))
    val_ds = create_dataset(CONFIG['val_dirs'], get_transforms('val'))
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=CONFIG['num_workers'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=CONFIG['lr_phase1'])
    
    best_acc = 0.0
    for epoch in range(CONFIG['epochs_phase1']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs_phase1']}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        print(f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG['best_model_path'])
            print(f"💾 Saved (Acc: {val_acc:.2f}%)")
    
    print(f"\n✅ Phase 1 Complete. Best: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
