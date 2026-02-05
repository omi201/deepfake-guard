# Optimized training script with proper DataLoader settings
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import json
from datetime import datetime
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
    'num_workers': 4,  # INCREASED from 2 to 4
    'prefetch_factor': 2,  # ADDED: Prefetch batches per worker
    'image_size': 224,
    'epochs_phase2': 5,
    'epochs_phase3': 10,
    'lr_phase2': 1e-4,
    'lr_phase3': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_dir': 'data/models',
    'phase1_model': 'data/models/efficientnet_b3_best.pth',
    'final_model_path': 'data/models/efficientnet_b3_final.pth'
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

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision for speed
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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

def train_phase(model, train_loader, val_loader, criterion, optimizer, epochs, device, phase_name, scaler=None):
    print(f"\n{'='*60}")
    print(f"🚀 {phase_name}")
    print(f"{'='*60}")
    
    best_acc = 0.0
    history = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'phase': phase_name
            }, CONFIG['final_model_path'])
            print(f"💾 Saved: {val_acc:.2f}%")
    
    return history, best_acc

def main():
    print(f"🎯 EfficientNet-B3 Fast Training")
    print(f"Device: {CONFIG['device']}")
    
    # Check if Phase 1 completed successfully
    if not os.path.exists(CONFIG['phase1_model']):
        print(f"❌ Phase 1 model not found: {CONFIG['phase1_model']}")
        print("Run Phase 1 training first!")
        return
    
    # Load data with optimized settings
    print("\n📂 Loading datasets...")
    train_ds = create_dataset(CONFIG['data_dirs'], get_transforms('train'))
    val_ds = create_dataset(CONFIG['val_dirs'], get_transforms('val'))
    
    print(f"\n🔧 Creating DataLoaders (workers={CONFIG['num_workers']}, prefetch={CONFIG['prefetch_factor']})...")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        prefetch_factor=CONFIG['prefetch_factor'],
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        prefetch_factor=CONFIG['prefetch_factor'],
        persistent_workers=True
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    
    # Phase 2
    print(f"\n📥 Loading Phase 1: {CONFIG['phase1_model']}")
    model = EfficientNetB3Deepfake()
    checkpoint = torch.load(CONFIG['phase1_model'])
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(CONFIG['device'])
    model.unfreeze_top_layers(30)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr_phase2'],
        weight_decay=1e-4
    )
    
    history2, best_acc2 = train_phase(
        model, train_loader, val_loader, criterion, optimizer,
        CONFIG['epochs_phase2'], CONFIG['device'], "Phase 2: Partial Unfreeze", scaler
    )
    
    # Phase 3
    print(f"\n🔄 Phase 3...")
    model.unfreeze_all()
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_phase3'], weight_decay=1e-4)
    
    history3, best_acc3 = train_phase(
        model, train_loader, val_loader, criterion, optimizer,
        CONFIG['epochs_phase3'], CONFIG['device'], "Phase 3: Full Fine-tune", scaler
    )
    
    print(f"\n{'='*60}")
    print("🎉 COMPLETE")
    print(f"Phase 2 Best: {best_acc2:.2f}%")
    print(f"Phase 3 Best: {best_acc3:.2f}%")
    print(f"🏆 FINAL: {max(best_acc2, best_acc3):.2f}%")
    
    results = {
        'date': datetime.now().isoformat(),
        'phase2_best': best_acc2,
        'phase3_best': best_acc3,
        'final_best': max(best_acc2, best_acc3),
        'history_phase2': history2,
        'history_phase3': history3
    }
    
    with open('data/models/efficientnet_history.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
