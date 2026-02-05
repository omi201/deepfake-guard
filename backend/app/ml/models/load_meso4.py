"""
Utility to load Meso4 models from checkpoints
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from meso4 import Meso4

def load_meso4(checkpoint_path, device='cpu'):
    """
    Load Meso4 model from checkpoint
    Handles both full checkpoint and state_dict formats
    """
    model = Meso4()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state_dict from checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
        else:
            state_dict = checkpoint
            metadata = {}
    else:
        state_dict = checkpoint
        metadata = {}
    
    # Load weights (non-strict for BatchNorm buffers)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, metadata

if __name__ == "__main__":
    # Test Day 4
    print("Loading Day 4 baseline...")
    model, meta = load_meso4('data/models/meso4_baseline_best.pth')
    print(f"✅ Loaded (Acc: {meta.get('val_accuracy', 'N/A')})")
    
    # Test inference
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        print(f"   Prediction: {'Real' if pred == 0 else 'Fake'} ({probs[0][pred]:.2%})")
    
    # Test Day 11
    print("\nLoading Day 11 PGD hardened...")
    model2, meta2 = load_meso4('data/models/meso4_pgd_hardened.pth')
    print(f"✅ Loaded (Robust Acc: {meta2.get('robust_acc', 'N/A')})")
