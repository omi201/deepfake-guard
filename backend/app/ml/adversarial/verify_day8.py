"""
Day 8 Verification: Check FGSM Implementation Readiness
"""

import os
import torch
import sys

def check_day8_readiness():
    """Verify all components are ready for FGSM testing"""
    
    print("🔍 Day 8 Verification Checklist")
    print("=" * 50)
    
    # Check files exist
    checks = {
        "Ensemble Model": "data/models/ensemble_v1.pth",
        "Meso4 Model": "data/models/meso4_baseline_best.pth", 
        "EfficientNet Model": "data/models/efficientnet_b3_final.pth",
        "FGSM Script": "backend/app/ml/adversarial/test_fgsm_ensemble.py",
        "Meso4 Architecture": "backend/app/ml/models/meso4.py",
        "EfficientNet Architecture": "backend/app/ml/ensemble/efficientnet_model.py",
        "Test Data": "data/processed/140k/real_vs_fake/real-vs-fake/test"
    }
    
    all_good = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name:<25} {path}")
        if not exists:
            all_good = False
    
    print("=" * 50)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available (will use CPU - slower)")
    
    # Check model can load
    try:
        ensemble_data = torch.load("data/models/ensemble_v1.pth", map_location="cpu")
        acc = ensemble_data.get("val_acc", "N/A")
        print(f"✅ Ensemble model loads successfully (Acc: {acc})")
    except Exception as e:
        print(f"❌ Failed to load ensemble: {e}")
        all_good = False
    
    print("=" * 50)
    
    if all_good:
        print("🚀 Ready to run FGSM attacks!")
        print("\nExecute:")
        print("  python backend/app/ml/adversarial/test_fgsm_ensemble.py")
        print("\nExpected duration: ~15 minutes on GPU")
        print("Expected result: Accuracy drop from 99% to <50%")
    else:
        print("⚠️  Some components missing. Check paths above.")
    
    return all_good

if __name__ == "__main__":
    check_day8_readiness()
