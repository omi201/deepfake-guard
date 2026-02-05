import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class EfficientNetB3Deepfake(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(EfficientNetB3Deepfake, self).__init__()
        
        # Load backbone with or without pretrained weights
        if pretrained:
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        # Replace classifier to match the saved model architecture
        # Based on error, saved model has: Dropout -> Linear(1536, 512) -> ... -> Linear(512, 2)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),  # Intermediate layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(512, num_classes)  # Final layer
        )
    
    def forward(self, x):
        return self.backbone(x)
