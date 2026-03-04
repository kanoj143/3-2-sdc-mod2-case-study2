import torch
import torch.nn as nn
from torchvision import models

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(DiseaseClassifier, self).__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

def get_classifier(config):
    model = DiseaseClassifier(num_classes=config['model']['num_classes'])
    return model
