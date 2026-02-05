"""
Meso4 Deepfake Detection Model
Exact architecture matching checkpoint
Parameters: 1,631,030
Input: 224x224 RGB
"""

import torch
import torch.nn as nn

class Meso4(nn.Module):
    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        
        # Blocks (layer1-4 match checkpoint naming)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier - Sequential with indices 0-5 to match checkpoint
        # Checkpoint has classifier.2 and classifier.5
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),           # 0
            nn.Identity(),             # 1 (placeholder to match index)
            nn.Linear(12544, 128),     # 2 (matches checkpoint)
            nn.ReLU(),                 # 3
            nn.Dropout(0.5),           # 4
            nn.Linear(128, num_classes) # 5 (matches checkpoint)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = Meso4()
    total_params = sum(p.numel() for p in model.parameters())
    
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"✅ Meso4 Model Test")
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Parameters: {total_params:,} (expected: 1,631,030)")
