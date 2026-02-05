# backend/app/ml/ensemble/efficientnet_model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class EfficientNetB3Deepfake(nn.Module):  # NOTE: B3 in name
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(EfficientNetB3Deepfake, self).__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        print("🔒 Backbone frozen")
    
    def unfreeze_top_layers(self, num_layers=30):
        for param in self.backbone.features[-num_layers:].parameters():
            param.requires_grad = True
        print(f"🔓 Top {num_layers} layers unfrozen")
    
    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Full unfreeze")

if __name__ == "__main__":
    model = EfficientNetB3Deepfake()
    model.freeze_backbone()
    x = torch.randn(2, 3, 224, 224)
    print(f"Test output shape: {model(x).shape}")  # Should be [2, 2]
