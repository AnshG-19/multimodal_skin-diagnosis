import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class FusionModel(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        # ====================================================
        # IMAGE BRANCH
        # ====================================================

        self.image_model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # EfficientNet-B0 originally outputs 1000 classes.
        # Replace classifier with Identity to obtain features.

        image_features = self.image_model.classifier[1].in_features

        self.image_model.classifier = nn.Identity()


        # ====================================================
        # METADATA BRANCH
        # ====================================================

        self.metadata_model = nn.Sequential(

            nn.Linear(3, 32),
            nn.ReLU(),

            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Dropout(0.2)
        )

        metadata_features = 64


        # ====================================================
        # FUSION CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                image_features + metadata_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(self, image, metadata):

        # Image features
        image_features = self.image_model(image)

        # Metadata features
        metadata_features = self.metadata_model(metadata)

        # Concatenate both modalities
        fused_features = torch.cat(
            [image_features, metadata_features],
            dim=1
        )

        # Final prediction
        output = self.classifier(fused_features)

        return output