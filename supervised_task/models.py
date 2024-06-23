import numpy as np
from torch import nn, flatten
import torch.nn.functional as F
import torch
import torchvision

class Model1(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes))
        self.model.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        return x

class Model2(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_ftrs = self.model.features[8][0].out_channels
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 640),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(640, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Model3(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.efficient_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficient_model.classifier[1].out_features = 640
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(640, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        x = self.efficient_model(x)
        x = self.classifier(x)
        return x

class Model4(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
            nn.Softmax(dim=1))
        self.model.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        return x
