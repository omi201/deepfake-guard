"""
Day 8: FGSM Attack Implementation against Ensemble V1
Target: Demonstrate vulnerability of 99.93% accuracy ensemble
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from PIL import Image
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.app.ml.models.meso4 import Meso4
from backend.app.ml.ensemble.efficientnet_model import EfficientNetB3Deepfake

class SimpleAverageEnsemble(nn.Module):
    """Ensemble V1: Simple average of Meso4 and EfficientNet"""
    def __init__(self, meso4_model, efficientnet_model):
        super(SimpleAverageEnsemble, self).__init__()
        self.meso4 = meso4_model
        self.efficientnet = efficientnet_model
        
    def forward(self, x):
        # Get logits from both models
        logits_meso4 = self.meso4(x)
        logits_eff = self.efficientnet(x)
        
        # Convert to probabilities and average
        prob_meso4 = F.softmax(logits_meso4, dim=1)
        prob_eff = F.softmax(logits_eff, dim=1)
        
        # Simple average fusion
        ensemble_prob = (prob_meso4 + prob_eff) / 2.0
        return ensemble_prob

def fgsm_attack(image, epsilon, data_grad):
    """
    Fast Gradient Sign Method
    x_adv = x + ε * sign(∇_x J(θ, x, y))
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    
    # Adding clipping to maintain [0,1] range (assuming normalized input)
    # Note: If using ImageNet normalization, we need to handle this carefully
    perturbed_image = torch.clamp(perturbed_image, -2.5, 2.5)  # Rough ImageNet bounds
    
    return perturbed_image

def test_fgsm_ensemble(epsilon, model, device, test_loader, max_batches=50):
    """
    Test ensemble against FGSM attack with specific epsilon
    Returns accuracy on adversarial examples
    """
    correct = 0
    total = 0
    original_correct = 0
    adversarial_examples = []
    
    model.eval()
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
            
        data, target = data.to(device), target.to(device)
        
        # Set requires_grad attribute of tensor
        data.requires_grad = True
        
        # Forward pass
        output = model(data)
        
        # Get initial prediction
        init_pred = output.argmax(dim=1)
        
        # Check if prediction is correct
        correct_mask = (init_pred == target)
        original_correct += correct_mask.sum().item()
        
        # Calculate loss
        loss = F.cross_entropy(output, target)
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Calculate gradients of model in backward pass
        loss.backward()
        
        # Collect datagrad
        data_grad = data.grad.data
        
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        
        # Re-classify the perturbed image
        output_adv = model(perturbed_data)
        
        # Get new prediction
        final_pred = output_adv.argmax(dim=1)
        
        # Check for adversarial success (model fooled)
        correct += (final_pred == target).sum().item()
        total += target.size(0)
        
        # Save some adversarial examples for visualization (first batch only)
        if batch_idx == 0 and len(adversarial_examples) < 5:
            for i in range(min(5, len(data))):
                adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                orig_ex = data[i].squeeze().detach().cpu().numpy()
                adversarial_examples.append({
                    'original': orig_ex,
                    'adversarial': adv_ex,
                    'epsilon': epsilon,
                    'original_pred': init_pred[i].item(),
                    'adversarial_pred': final_pred[i].item(),
                    'true_label': target[i].item(),
                    'index': i
                })
    
    # Calculate final accuracy
    adversarial_acc = correct / total
    clean_acc = original_correct / total
    
    return adversarial_acc, clean_acc, adversarial_examples

def visualize_adversarial_examples(examples, save_path):
    """Create visualization grid of adversarial examples"""
    fig = plt.figure(figsize=(15, 10))
    
    for i, example in enumerate(examples):
        # Original image
        plt.subplot(3, 5, i+1)
        orig = example['original']
        # Denormalize if using ImageNet stats
        orig = denormalize(orig)
        plt.imshow(np.transpose(orig, (1, 2, 0)))
        plt.title(f"Original\nPred: {example['original_pred']}\nTrue: {example['true_label']}")
        plt.axis('off')
        
        # Adversarial image
        plt.subplot(3, 5, i+6)
        adv = example['adversarial']
        adv = denormalize(adv)
        # Clip for display
        adv = np.clip(adv, 0, 1)
        plt.imshow(np.transpose(adv, (1, 2, 0)))
        plt.title(f"Adversarial (ε={example['epsilon']})\nPred: {example['adversarial_pred']}")
        plt.axis('off')
        
        # Perturbation (magnified)
        plt.subplot(3, 5, i+11)
        perturbation = np.abs(np.transpose(adv, (1, 2, 0)) - np.transpose(orig, (1, 2, 0)))
        plt.imshow(perturbation)
        plt.title(f"Perturbation\n(magnified)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved to {save_path}")

def denormalize(tensor):
    """Denormalize ImageNet normalized tensor for visualization"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor

def main():
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {DEVICE}")
    
    # Epsilon values to test (as per Day 8 requirements)
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]
    
    # Model paths
    ENSEMBLE_PATH = "data/models/ensemble_v1.pth"
    MESO4_PATH = "data/models/meso4_baseline_best.pth"
    EFFNET_PATH = "data/models/efficientnet_b3_final.pth"
    
    # Load individual models
    print("📦 Loading Meso4...")
    meso4 = Meso4()
    meso4.load_state_dict(torch.load(MESO4_PATH, map_location=DEVICE)['model_state_dict'])
    meso4 = meso4.to(DEVICE)
    meso4.eval()
    
    print("📦 Loading EfficientNet-B3...")
    effnet = EfficientNetB3Deepfake()
    effnet.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE)['model_state_dict'])
    effnet = effnet.to(DEVICE)
    effnet.eval()
    
    print("🔗 Creating Ensemble V1...")
    ensemble = SimpleAverageEnsemble(meso4, effnet).to(DEVICE)
    ensemble.eval()
    
    # Data loading
    print("📂 Loading test dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use combined dataset test split
    test_dataset = ImageFolder('data/processed/140k/real_vs_fake/real-vs-fake/test', transform=transform)
    
    # Subset for faster testing (Day 8: Use 1000 samples)
    indices = random.sample(range(len(test_dataset)), min(1000, len(test_dataset)))
    test_subset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"🧪 Testing on {len(test_subset)} samples")
    print(f"🎯 Target: Reduce accuracy from 99.93% to <50%")
    print("=" * 60)
    
    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Ensemble V1 (Meso4 + EfficientNet)',
        'dataset_size': len(test_subset),
        'epsilon_values': [],
        'clean_accuracies': [],
        'adversarial_accuracies': [],
        'attack_success_rate': []
    }
    
    all_examples = []
    
    # Run FGSM for each epsilon
    for eps in epsilons:
        print(f"\n🔥 Testing FGSM with ε = {eps}")
        
        if eps == 0.0:
            # Clean accuracy (no attack)
            acc, _, _ = test_fgsm_ensemble(eps, ensemble, DEVICE, test_loader)
            print(f"   ✅ Clean Accuracy: {acc*100:.2f}%")
            results['clean_accuracies'].append(acc)
            results['adversarial_accuracies'].append(acc)
            results['attack_success_rate'].append(0.0)
        else:
            # Adversarial attack
            adv_acc, clean_acc, examples = test_fgsm_ensemble(eps, ensemble, DEVICE, test_loader)
            attack_success = (clean_acc - adv_acc) / clean_acc * 100 if clean_acc > 0 else 0
            
            print(f"   ⚠️  Clean Accuracy: {clean_acc*100:.2f}%")
            print(f"   🎯 Adversarial Accuracy: {adv_acc*100:.2f}%")
            print(f"   💥 Attack Success Rate: {attack_success:.2f}%")
            
            results['epsilon_values'].append(eps)
            results['clean_accuracies'].append(clean_acc)
            results['adversarial_accuracies'].append(adv_acc)
            results['attack_success_rate'].append(attack_success)
            
            if examples:
                all_examples.extend(examples)
                
            # Save specific epsilon examples
            if examples:
                visualize_adversarial_examples(
                    examples, 
                    f'docs/day8_fgsm/fgsm_epsilon_{eps:.2f}_examples.png'
                )
    
    # Generate summary plot
    plt.figure(figsize=(10, 6))
    plt.plot([0] + results['epsilon_values'], [acc*100 for acc in results['clean_accuracies']], 
             'o-', label='Clean Accuracy', linewidth=2)
    plt.plot([0] + results['epsilon_values'], [acc*100 for acc in results['adversarial_accuracies']], 
             's--', label='Adversarial Accuracy', linewidth=2, color='red')
    plt.axhline(y=50, color='orange', linestyle=':', label='Target Threshold (50%)')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Accuracy (%)')
    plt.title('FGSM Attack on Ensemble V1\n(99.93% → Vulnerable)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.savefig('docs/day8_fgsm/accuracy_vs_epsilon.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    with open('docs/day8_fgsm/fgsm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DAY 8 SUMMARY - FGSM VULNERABILITY PROVEN")
    print("=" * 60)
    print(f"{'Epsilon':<10} {'Clean Acc':<12} {'Adv Acc':<12} {'Success Rate':<15}")
    print("-" * 60)
    for i, eps in enumerate([0.0] + results['epsilon_values']):
        clean = results['clean_accuracies'][i] * 100
        adv = results['adversarial_accuracies'][i] * 100
        succ = results['attack_success_rate'][i] if i > 0 else 0
        print(f"{eps:<10.2f} {clean:<12.2f} {adv:<12.2f} {succ:<15.2f}")
    
    final_drop = (results['clean_accuracies'][0] - results['adversarial_accuracies'][-1]) * 100
    print("-" * 60)
    print(f"🎯 Total Accuracy Drop: {final_drop:.2f}% (from {results['clean_accuracies'][0]*100:.2f}% to {results['adversarial_accuracies'][-1]*100:.2f}%)")
    
    if results['adversarial_accuracies'][-1] < 0.5:
        print("✅ TARGET ACHIEVED: Ensemble accuracy reduced below 50%")
    else:
        print("⚠️  TARGET NOT MET: Ensemble still robust at ε=0.1")
    
    print("\n📁 Artifacts saved:")
    print("   • docs/day8_fgsm/fgsm_results.json")
    print("   • docs/day8_fgsm/accuracy_vs_epsilon.png")
    print("   • docs/day8_fgsm/fgsm_epsilon_*.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
