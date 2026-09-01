import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ImageModel(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT

        self.backbone = efficientnet_b0(
            weights=weights
        )

        # Number of features coming from EfficientNet
        in_features = self.backbone.classifier[1].in_features

        # Replace original ImageNet classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )


    def forward(self, x):

        return self.backbone(x)