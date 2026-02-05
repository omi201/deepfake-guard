"""
Train Meso4 Baseline Model
Dataset: 140k Real vs Fake
Target: 80% accuracy (Day 4)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from meso4 import Meso4

# Configuration
CONFIG = {
    'data_dir': 'data/processed/140k/real_vs_fake/real-vs-fake/train',
    'val_dir': 'data/processed/140k/real_vs_fake/real-vs-fake/test',
    'batch_size': 32,
    'num_workers': 2,
    'image_size': 224,
    'epochs': 10,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'data/models/meso4_baseline_best.pth'
}

def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc='Training'):
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
    print(f"🚀 Training Meso4 Baseline")
    print(f"Device: {CONFIG['device']}")
    
    # Model
    model = Meso4().to(CONFIG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Data
    train_dataset = datasets.ImageFolder(CONFIG['data_dir'], transform=get_transforms('train'))
    val_dataset = datasets.ImageFolder(CONFIG['val_dir'], transform=get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=CONFIG['num_workers'])
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    best_acc = 0.0
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        
        print(f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG['model_path'])
            print(f"💾 Saved (Acc: {val_acc:.2f}%)")
    
    print(f"\n✅ Training Complete. Best: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
