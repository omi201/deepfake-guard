import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from backend.app.ml.models.meso4 import Meso4


def get_combined_dataset():
    """Load combined dataset with verified labels"""
    path_140k = "data/processed/140k/real_vs_fake/real-vs-fake/"
    path_celeb = "data/processed/celeb-df/faces/"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load 140k
    train_140k = datasets.ImageFolder(os.path.join(path_140k, "train"), transform=transform)
    test_140k = datasets.ImageFolder(os.path.join(path_140k, "test"), transform=transform)
    print(f"140k: {len(train_140k)} train, {len(test_140k)} test")
    
    # Load Celeb-DF
    train_celeb = datasets.ImageFolder(os.path.join(path_celeb, "train"), transform=transform)
    test_celeb = datasets.ImageFolder(os.path.join(path_celeb, "test"), transform=transform)
    print(f"Celeb-DF: {len(train_celeb)} train, {len(test_celeb)} test")
    
    # Combine
    full_train = ConcatDataset([train_140k, train_celeb])
    full_test = ConcatDataset([test_140k, test_celeb])
    
    print(f"Combined: {len(full_train)} train, {len(full_test)} test")
    print(f"Labels: Fake=0, Real=1")
    
    return full_train, full_test


def train_meso4():
    print("=" * 60)
    print("Retraining Meso4 - Correct Labels (Fake=0, Real=1)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    train_dataset, test_dataset = get_combined_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    model = Meso4(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nTraining {epochs} epochs (~30-40 mins)...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Validating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(test_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}: Train={train_acc:.2f}% | Val={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_acc,
                'history': history,
                'labels': {'fake': 0, 'real': 1}
            }
            os.makedirs('data/models', exist_ok=True)
            torch.save(checkpoint, 'data/models/meso4_baseline_best.pth')
            print(f"  ✓ Saved: {best_acc:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Done! Best: {best_acc:.2f}%")
    print(f"{'='*60}")
    
    with open('data/models/meso4_history_corrected.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train_meso4()
