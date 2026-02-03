import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json
from datetime import datetime
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent))
from meso4 import Meso4

class SecureTrainer:
    def __init__(self, data_dir: str, batch_size: int = 32, epochs: int = 5, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SECURITY] Compute device: {self.device}")
        
        if self.device.type == "cpu":
            print("[WARNING] CUDA unavailable - training will be extremely slow")
        
        # Reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.model = Meso4(num_classes=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Security: Weight decay prevents overfitting to adversarial artifacts
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.best_val_acc = 0.0
        self.training_log = []
        
        self.setup_data(data_dir, batch_size)
    
    def setup_data(self, data_dir: str, batch_size: int):
        """Setup train/val data pipeline."""
        data_path = Path(data_dir)
        
        # Verify structure matches our discovery
        train_path = data_path / "train"
        valid_path = data_path / "valid"
        
        if not train_path.exists() or not valid_path.exists():
            raise FileNotFoundError(
                f"Expected train/ and valid/ in {data_path}\n"
                f"Found: {list(data_path.glob('*'))}"
            )
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Augmentation for training only
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        train_dataset = datasets.ImageFolder(root=str(train_path), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(valid_path), transform=transform)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        print(f"[DATA] Training samples: {len(train_dataset)}")
        print(f"[DATA] Validation samples: {len(val_dataset)}")
        print(f"[DATA] Classes: {train_dataset.classes}")
        print(f"[DATA] Class-to-index: {train_dataset.class_to_idx}")
    
    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            # Security: Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"[TRAIN] Epoch {epoch} | Batch {batch_idx:>3}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        # Per-class accuracy tracking
        class_correct = [0, 0]
        class_total = [0, 0]
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Per-class metrics
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_correct[label] += (predicted[i] == targets[i]).item()
                    class_total[label] += 1
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Class names: assuming alphabetical (fake=0, real=1) or check class_to_idx
        print(f"[VALIDATE] Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
        for i, name in enumerate(['Fake', 'Real']):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                print(f"[VALIDATE] Class '{name}': {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Secure checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
            'training_log': self.training_log,
            'timestamp': datetime.now().isoformat(),
            'architecture': 'Meso4',
            'security_note': 'BASELINE - Not adversarially hardened (Day 0)',
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        # Save latest
        latest_path = Path("data/models/meso4_latest.pth")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, latest_path)
        
        # Save best only if improved
        if is_best:
            best_path = Path("data/models/meso4_baseline_best.pth")
            torch.save(checkpoint, best_path)
            print(f"[SECURITY] Best model saved: {best_path} (Acc: {val_acc:.2f}%)")
    
    def run(self):
        print("\n" + "="*60)
        print("[SECURITY] BASELINE TRAINING INITIATED (Day 0)")
        print("[SECURITY] No adversarial defense - establishing clean accuracy")
        print("="*60 + "\n")
        
        for epoch in range(1, self.epochs + 1):
            print(f"\n[EPOCH {epoch}/{self.epochs}]")
            print("-" * 40)
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            log_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'timestamp': datetime.now().isoformat()
            }
            self.training_log.append(log_entry)
            
            print(f"[METRICS] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"[METRICS] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save if best
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"[SECURITY] New best validation accuracy: {val_acc:.2f}%")
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Early stopping check (simple)
            if epoch > 3 and val_acc < 55.0:
                print("[WARNING] Validation accuracy critically low - check data loading")
        
        # Final report
        print("\n" + "="*60)
        print("[SECURITY] TRAINING COMPLETE")
        print(f"[SECURITY] Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*60)
        
        # Save training log
        log_path = Path("data/models/training_log_baseline.json")
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(f"[SECURITY] Audit log saved: {log_path}")

if __name__ == "__main__":
    # Dataset path using discovered structure
    dataset_path = Path.home() / "deepfake-guard" / "data" / "processed" / "140k" / "real_vs_fake" / "real-vs-fake"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    trainer = SecureTrainer(
        data_dir=str(dataset_path),
        batch_size=32,      # Reduce to 16 if GPU memory < 8GB
        epochs=2,           # Start with 2 for verification (increase to 10 for production)
        lr=0.001
    )
    trainer.run()
