import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backend.app.ml.ensemble.ensemble_model import DeepfakeEnsembleV1
from backend.app.ml.preprocessing.augmentation import AugmentedDataset


def get_combined_dataset():
    """
    Load combined 182k dataset (140k + Celeb-DF)
    Uses Day 6 augmentation for training split
    """
    # Paths
    path_140k = "data/processed/140k/real_vs_fake/real-vs-fake/"
    path_celeb = "data/processed/celeb-df/faces/"
    
    # Standard transforms for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    datasets_list_train = []
    datasets_list_test = []
    
    # Load 140k dataset
    if os.path.exists(path_140k):
        train_path = os.path.join(path_140k, "train")
        test_path = os.path.join(path_140k, "test")
        if os.path.exists(train_path):
            train_140k = datasets.ImageFolder(train_path, transform=val_transform)
            test_140k = datasets.ImageFolder(test_path, transform=val_transform)
            datasets_list_train.append(train_140k)
            datasets_list_test.append(test_140k)
            print(f"✓ Loaded 140k dataset: {len(train_140k)} train, {len(test_140k)} test")
    
    # Load Celeb-DF dataset - check for train/test subfolders first
    if os.path.exists(path_celeb):
        train_path = os.path.join(path_celeb, "train")
        test_path = os.path.join(path_celeb, "test")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            # Structure: train/ and test/ folders with real/fake inside
            train_celeb = datasets.ImageFolder(train_path, transform=val_transform)
            test_celeb = datasets.ImageFolder(test_path, transform=val_transform)
            datasets_list_train.append(train_celeb)
            datasets_list_test.append(test_celeb)
            print(f"✓ Loaded Celeb-DF: {len(train_celeb)} train, {len(test_celeb)} test")
        else:
            # Structure: real/ and fake/ at top level - need to split manually
            print("⚠ No train/test split found, using real/fake folders directly...")
            real_path = os.path.join(path_celeb, "real")
            fake_path = os.path.join(path_celeb, "fake")
            
            if os.path.exists(real_path) and os.path.exists(fake_path):
                # Create temporary dataset and split 80/20
                full_dataset = datasets.ImageFolder(path_celeb, transform=val_transform)
                
                # Calculate split
                total_size = len(full_dataset)
                train_size = int(0.8 * total_size)
                test_size = total_size - train_size
                
                train_celeb, test_celeb = random_split(
                    full_dataset, [train_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )
                
                datasets_list_train.append(train_celeb)
                datasets_list_test.append(test_celeb)
                print(f"✓ Loaded Celeb-DF (manual 80/20 split): {len(train_celeb)} train, {len(test_celeb)} test")
    
    # Combine datasets
    if len(datasets_list_train) == 0:
        raise ValueError("No datasets found! Check paths.")
    
    if len(datasets_list_train) == 1:
        full_train = datasets_list_train[0]
        full_test = datasets_list_test[0]
    else:
        full_train = ConcatDataset(datasets_list_train)
        full_test = ConcatDataset(datasets_list_test)
    
    print(f"✓ Combined dataset: {len(full_train)} train, {len(full_test)} test")
    return full_train, full_test


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler:
            with autocast():
                logits, meta = model(data)
                loss = criterion(logits, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, meta = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        
        # Stats
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Update progress bar
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return detailed metrics"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track individual model performance
    meso_correct = 0
    eff_correct = 0
    disagreement_count = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            logits, meta = model(data)
            loss = criterion(logits, target)
            
            total_loss += loss.item()
            
            # Ensemble prediction
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
            # Individual predictions
            meso_pred = meta['meso_prob'].argmax(dim=1)
            eff_pred = meta['eff_prob'].argmax(dim=1)
            
            meso_correct += meso_pred.eq(target).sum().item()
            eff_correct += eff_pred.eq(target).sum().item()
            disagreement_count += (meso_pred != eff_pred).sum().item()
            
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            acc = 100. * correct / total
            pbar.set_postfix({'acc': f'{acc:.2f}%'})
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100. * correct / total
    meso_acc = 100. * meso_correct / total
    eff_acc = 100. * eff_correct / total
    disagreement_rate = 100. * disagreement_count / total
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'meso_accuracy': meso_acc,
        'efficientnet_accuracy': eff_acc,
        'disagreement_rate': disagreement_rate,
        'predictions': all_preds,
        'targets': all_targets
    }


def main():
    print("=" * 60)
    print("Day 7: Ensemble V1 Training (Meso4 + EfficientNet-B3)")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    batch_size = 32
    epochs = 5  # Quick convergence since only training fusion params
    lr = 0.001
    num_workers = 4
    
    # Load data
    print("\n[1/5] Loading datasets...")
    train_base, test_base = get_combined_dataset()
    
    # Apply Day 6 augmentation to training set only
    print("\n[2/5] Applying Day 6 augmentation pipeline...")
    train_dataset = AugmentedDataset(train_base, phase='train')
    test_dataset = AugmentedDataset(test_base, phase='val')  # No augmentation for test
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize ensemble
    print("\n[3/5] Initializing Ensemble V1...")
    model = DeepfakeEnsembleV1(
        meso4_path="data/models/meso4_baseline_best.pth",
        efficientnet_path="data/models/efficientnet_b3_final.pth",
        freeze_base=True,  # Keep base models frozen
        learnable_weights=True  # Allow learning optimal fusion weights
    ).to(device)
    
    stats = model.get_model_stats()
    print(f"Model stats: {stats}")
    
    # Only optimize temperature and fusion weights (not base models)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    print("\n[4/5] Training fusion parameters...")
    best_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_meso_acc': [],
        'val_eff_acc': [],
        'disagreement_rate': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Evaluate
        val_metrics = evaluate(model, test_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_meso_acc'].append(val_metrics['meso_accuracy'])
        history['val_eff_acc'].append(val_metrics['efficientnet_accuracy'])
        history['disagreement_rate'].append(val_metrics['disagreement_rate'])
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  ↳ Meso4: {val_metrics['meso_accuracy']:.2f}% | EfficientNet: {val_metrics['efficientnet_accuracy']:.2f}%")
        print(f"  ↳ Model Disagreement: {val_metrics['disagreement_rate']:.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_acc,
                'history': history,
                'model_stats': stats
            }
            os.makedirs('data/models', exist_ok=True)
            torch.save(checkpoint, 'data/models/ensemble_v1.pth')
            print(f"✓ Saved new best model: {best_acc:.2f}%")
    
    # Final evaluation
    print("\n[5/5] Final Evaluation...")
    model.load_state_dict(torch.load('data/models/ensemble_v1.pth')['model_state_dict'])
    final_metrics = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Ensemble Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Meso4 Alone: {final_metrics['meso_accuracy']:.2f}%")
    print(f"EfficientNet Alone: {final_metrics['efficientnet_accuracy']:.2f}%")
    print(f"Improvement over Meso4: {final_metrics['accuracy'] - final_metrics['meso_accuracy']:.2f}%")
    print(f"Improvement over EfficientNet: {final_metrics['accuracy'] - final_metrics['efficientnet_accuracy']:.2f}%")
    print(f"Model Disagreement Rate: {final_metrics['disagreement_rate']:.2f}%")
    print("=" * 60)
    
    # Save history
    with open('data/models/ensemble_v1_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Verification checkpoint
    print("\n✅ Day 7 Complete!")
    print(f"Target: 92% accuracy")
    print(f"Achieved: {final_metrics['accuracy']:.2f}%")
    
    if final_metrics['accuracy'] >= 92:
        print("🎯 TARGET EXCEEDED!")
    else:
        print("⚠️ Target not met, but ensemble completed")
    
    print("\nVerification command:")
    print("python -c \"import torch; m=torch.load('data/models/ensemble_v1.pth'); print(f'Ensemble Acc: {m[\\\"val_acc\\\"]:.2f}%')\"")


if __name__ == "__main__":
    main()
