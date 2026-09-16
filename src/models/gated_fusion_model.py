import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class GatedFusionModel(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        # Image branch
        weights = EfficientNet_B0_Weights.DEFAULT

        self.image_model = efficientnet_b0(weights=weights)

        # Remove original classifier
        image_features = self.image_model.classifier[1].in_features
        self.image_model.classifier = nn.Identity()

        # Project image features to 128 dimensions
        self.image_projection = nn.Sequential(
            nn.Linear(image_features, 128),
            nn.ReLU()
        )

        # Metadata branch
        self.metadata_model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.Sigmoid()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, image, metadata):

        # Extract image features
        image_features = self.image_model(image)

        # Project image features
        image_features = self.image_projection(image_features)

        # Extract metadata features
        metadata_features = self.metadata_model(metadata)

        # Combine both modalities to calculate the gate
        combined = torch.cat(
            [image_features, metadata_features],
            dim=1
        )

        gate = self.gate(combined)

        # Adaptive fusion
        fused_features = (
            gate * image_features
            + (1 - gate) * metadata_features
        )

        # Classification
        output = self.classifier(fused_features)

        return output