import torch
import torch.nn as nn
import torch.nn.functional as F

class Meso4(nn.Module):
    """
    Meso4 Deepfake Detection CNN
    Architecture optimized for detecting face manipulation at the mesoscopic level.
    Reference: "MesoNet: a Compact Facial Video Forgery Detection Network"
    """
    def __init__(self, num_classes: int = 2):
        super(Meso4, self).__init__()
        
        # Security: Explicit seeding for reproducible initialization
        torch.manual_seed(42)
        
        # Block 1: Feature extraction (low-level edges/textures)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Block 2: Mid-level features
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Block 3: Deep features
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        # Calculate flattened size: 256 -> 128 -> 64 -> 32 -> 8 (after 4 pools)
        # But with pool4 being 4x4: 32 -> 8, so 16 channels * 8 * 8 = 1024
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 3, 256, 256]
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = Meso4()
    test_input = torch.randn(2, 3, 256, 256)
    output = model(test_input)
    
    print(f"[ARCHITECTURE TEST] Input: {test_input.shape}")
    print(f"[ARCHITECTURE TEST] Output: {output.shape}")
    print(f"[ARCHITECTURE TEST] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify gradients flow
    loss = output.sum()
    loss.backward()
    print("[SECURITY] Gradient flow verified.")
    print("[SECURITY] Meso4 architecture initialized successfully.")
