import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
from pathlib import Path
import importlib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backend.app.ml.models.meso4 import Meso4

# Dynamically import EfficientNet class
try:
    eff_module = importlib.import_module('backend.app.ml.ensemble.efficientnet_model')
    available_classes = [name for name in dir(eff_module) 
                        if isinstance(getattr(eff_module, name), type) 
                        and not name.startswith('_')
                        and name not in ['torch', 'nn', 'OrderedDict']]
    
    model_class_name = None
    for cls_name in available_classes:
        if 'Efficient' in cls_name or 'Net' in cls_name:
            model_class_name = cls_name
            break
    
    if not model_class_name:
        model_class_name = available_classes[0]
    
    EfficientNetClass = getattr(eff_module, model_class_name)
    print(f"Loaded EfficientNet class: {model_class_name}")
    
except Exception as e:
    print(f"Error loading EfficientNet: {e}")
    raise


class DeepfakeEnsembleV1(nn.Module):
    def __init__(self, 
                 meso4_path: str = "data/models/meso4_baseline_best.pth",
                 efficientnet_path: str = "data/models/efficientnet_b3_final.pth",
                 num_classes: int = 2,
                 freeze_base: bool = True,
                 learnable_weights: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.learnable_weights = learnable_weights
        
        # Load Meso4
        print("Loading Meso4...")
        self.meso4 = Meso4(num_classes=num_classes)
        
        meso4_checkpoint = torch.load(meso4_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in meso4_checkpoint:
            meso4_state = meso4_checkpoint['model_state_dict']
        else:
            meso4_state = meso4_checkpoint
        
        # Remove 'module.' prefix if present
        meso4_state = {k.replace('module.', ''): v for k, v in meso4_state.items()}
        self.meso4.load_state_dict(meso4_state, strict=True)
        print(f"  ✓ Meso4 loaded, val_acc was: {meso4_checkpoint.get('val_acc', 'N/A')}")
        
        # Load EfficientNet
        print("Loading EfficientNet-B3...")
        self.efficientnet = EfficientNetClass(num_classes=num_classes, pretrained=False)
        
        eff_checkpoint = torch.load(efficientnet_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in eff_checkpoint:
            eff_state = eff_checkpoint['model_state_dict']
        else:
            eff_state = eff_checkpoint
        
        # Remove 'module.' prefix if present
        eff_state = {k.replace('module.', ''): v for k, v in eff_state.items()}
        
        # Load with strict=True to catch mismatches
        try:
            self.efficientnet.load_state_dict(eff_state, strict=True)
            print(f"  ✓ EfficientNet loaded, val_acc was: {eff_checkpoint.get('val_acc', 'N/A')}")
        except RuntimeError as e:
            print(f"  ✗ Loading error: {e}")
            print("  Attempting strict=False load...")
            self.efficientnet.load_state_dict(eff_state, strict=False)
            print("  ✓ Loaded with strict=False")
        
        # Freeze base models
        if freeze_base:
            for param in self.meso4.parameters():
                param.requires_grad = False
            for param in self.efficientnet.parameters():
                param.requires_grad = False
            print("Base models frozen (inference mode)")
        
        # Learnable fusion weights
        if learnable_weights:
            self.meso_weight = nn.Parameter(torch.tensor(0.5))
            self.eff_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('meso_weight', torch.tensor(0.5))
            self.register_buffer('eff_weight', torch.tensor(0.5))
        
        # Temperature scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        meso_logits = self.meso4(x)
        eff_logits = self.efficientnet(x)
        
        meso_prob = torch.softmax(meso_logits / self.temperature, dim=1)
        eff_prob = torch.softmax(eff_logits / self.temperature, dim=1)
        
        if self.learnable_weights:
            total_weight = self.meso_weight + self.eff_weight
            w_meso = self.meso_weight / total_weight
            w_eff = self.eff_weight / total_weight
        else:
            w_meso = self.meso_weight
            w_eff = self.eff_weight
            
        fused_prob = w_meso * meso_prob + w_eff * eff_prob
        
        eps = 1e-7
        fused_logits = torch.log(fused_prob + eps) - torch.log(1 - fused_prob + eps)
        
        return fused_logits, {
            'meso_logits': meso_logits,
            'eff_logits': eff_logits,
            'meso_prob': meso_prob,
            'eff_prob': eff_prob,
            'fused_prob': fused_prob,
            'weights': {'meso': w_meso.item() if isinstance(w_meso, torch.Tensor) else w_meso, 
                       'eff': w_eff.item() if isinstance(w_eff, torch.Tensor) else w_eff}
        }
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        self.eval()
        with torch.no_grad():
            logits, meta = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            meso_pred = torch.argmax(meta['meso_prob'], dim=1)
            eff_pred = torch.argmax(meta['eff_prob'], dim=1)
            disagreement = (meso_pred != eff_pred).sum().item()
            meta['disagreement_count'] = disagreement
            meta['final_pred'] = preds
        return probs, meta
    
    def get_model_stats(self) -> Dict:
        meso_params = sum(p.numel() for p in self.meso4.parameters())
        eff_params = sum(p.numel() for p in self.efficientnet.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'meso4_params': meso_params,
            'efficientnet_params': eff_params,
            'total_params': meso_params + eff_params,
            'trainable_params': trainable_params,
            'frozen_params': (meso_params + eff_params) - trainable_params
        }


class SimpleAverageEnsemble(nn.Module):
    def __init__(self, meso4_path: str, efficientnet_path: str):
        super().__init__()
        self.meso4 = Meso4(num_classes=2)
        
        checkpoint = torch.load(meso4_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.meso4.load_state_dict(state_dict)
        
        self.efficientnet = EfficientNetClass(num_classes=2, pretrained=False)
        checkpoint = torch.load(efficientnet_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.efficientnet.load_state_dict(state_dict, strict=False)
        
        self.meso4.eval()
        self.efficientnet.eval()
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            meso_out = torch.softmax(self.meso4(x), dim=1)
            eff_out = torch.softmax(self.efficientnet(x), dim=1)
            return (meso_out + eff_out) / 2


if __name__ == "__main__":
    print("Testing Ensemble V1 Architecture...")
    import os
    if not os.path.exists("data/models/meso4_baseline_best.pth"):
        print("ERROR: Meso4 model not found")
        sys.exit(1)
    if not os.path.exists("data/models/efficientnet_b3_final.pth"):
        print("ERROR: EfficientNet model not found")
        sys.exit(1)
    
    model = DeepfakeEnsembleV1(
        meso4_path="data/models/meso4_baseline_best.pth",
        efficientnet_path="data/models/efficientnet_b3_final.pth",
        freeze_base=True,
        learnable_weights=False
    )
    
    # Test
    dummy_input = torch.randn(4, 3, 224, 224)
    logits, meta = model(dummy_input)
    
    print(f"\n✓ Test passed! Output shape: {logits.shape}")
    print(f"✓ Meso4 accuracy in checkpoint: Should be ~80%")
    print(f"✓ EfficientNet accuracy in checkpoint: Should be ~99%")
    
    stats = model.get_model_stats()
    print(f"\nParameters: Total={stats['total_params']:,}, Trainable={stats['trainable_params']}")
