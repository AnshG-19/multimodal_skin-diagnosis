import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import (
    HAM10000Dataset,
    val_test_transform
)

from src.models.gated_fusion_model import GatedFusionModel


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = HAM10000Dataset(
    "data/processed/test.csv",
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# LOAD MODEL
# ============================================================

model = GatedFusionModel(
    num_classes=7
)

model.load_state_dict(
    torch.load(
        "checkpoints/gated_fusion_model_best.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Best gated fusion model loaded.")


# ============================================================
# GATE ANALYSIS
# ============================================================

all_gate_values = []

all_labels = []

all_predictions = []


with torch.no_grad():

    for images, metadata, labels in tqdm(
        test_loader,
        desc="Analyzing gates"
    ):

        images = images.to(device)
        metadata = metadata.to(device)


        # ----------------------------------------------------
        # Image branch
        # ----------------------------------------------------

        image_features = model.image_model(
            images
        )

        image_features = model.image_projection(
            image_features
        )


        # ----------------------------------------------------
        # Metadata branch
        # ----------------------------------------------------

        metadata_features = model.metadata_model(
            metadata
        )


        # ----------------------------------------------------
        # Calculate gate
        # ----------------------------------------------------

        combined = torch.cat(
            [
                image_features,
                metadata_features
            ],
            dim=1
        )

        gate = model.gate(
            combined
        )


        # ----------------------------------------------------
        # Final fused representation
        # ----------------------------------------------------

        fused_features = (
            gate * image_features
            +
            (1 - gate) * metadata_features
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        outputs = model.classifier(
            fused_features
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # Store values
        # ----------------------------------------------------

        all_gate_values.append(
            gate.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

all_gate_values = np.concatenate(
    all_gate_values,
    axis=0
)

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)


# ============================================================
# BASIC CHECK
# ============================================================

print()

print(
    "Gate shape:",
    all_gate_values.shape
)

print(
    "Expected:",
    (len(test_dataset), 128)
)


# ============================================================
# OVERALL GATE STATISTICS
# ============================================================

mean_gate = np.mean(
    all_gate_values
)

median_gate = np.median(
    all_gate_values
)

min_gate = np.min(
    all_gate_values
)

max_gate = np.max(
    all_gate_values

)

std_gate = np.std(
    all_gate_values
)


print()
print("=" * 70)
print("OVERALL GATE ANALYSIS")
print("=" * 70)

print()

print(
    f"Mean gate value:    {mean_gate:.4f}"
)

print(
    f"Median gate value:  {median_gate:.4f}"
)

print(
    f"Minimum gate value: {min_gate:.4f}"
)

print(
    f"Maximum gate value: {max_gate:.4f}"
)

print(
    f"Gate std deviation: {std_gate:.4f}"
)


# ============================================================
# MODALITY CONTRIBUTION
# ============================================================

image_weight = all_gate_values

metadata_weight = 1.0 - all_gate_values


mean_image_weight = np.mean(
    image_weight
)

mean_metadata_weight = np.mean(
    metadata_weight
)


print()
print("=" * 70)
print("MODALITY CONTRIBUTION")
print("=" * 70)

print()

print(
    f"Average image contribution:    "
    f"{mean_image_weight:.4f} "
    f"({mean_image_weight * 100:.2f}%)"
)

print(
    f"Average metadata contribution: "
    f"{mean_metadata_weight:.4f} "
    f"({mean_metadata_weight * 100:.2f}%)"
)


# ============================================================
# GATE DISTRIBUTION
# ============================================================

image_dominant = np.mean(
    all_gate_values > 0.5
)

metadata_dominant = np.mean(
    all_gate_values < 0.5
)

balanced = np.mean(
    (
        all_gate_values >= 0.4
    )
    &
    (
        all_gate_values <= 0.6
    )
)


print()
print("=" * 70)
print("GATE DISTRIBUTION")
print("=" * 70)

print()

print(
    f"Image-dominant gates (> 0.5): "
    f"{image_dominant * 100:.2f}%"
)

print(
    f"Metadata-dominant gates (< 0.5): "
    f"{metadata_dominant * 100:.2f}%"
)

print(
    f"Balanced gates (0.4 - 0.6): "
    f"{balanced * 100:.2f}%"
)


# ============================================================
# PER-DIMENSION STATISTICS
# ============================================================

mean_gate_per_dimension = np.mean(
    all_gate_values,
    axis=0
)

median_gate_per_dimension = np.median(
    all_gate_values,
    axis=0
)


print()
print("=" * 70)
print("PER-DIMENSION GATE STATISTICS")
print("=" * 70)

print()

print(
    f"Mean of dimension means: "
    f"{np.mean(mean_gate_per_dimension):.4f}"
)

print(
    f"Minimum dimension mean:   "
    f"{np.min(mean_gate_per_dimension):.4f}"
)

print(
    f"Maximum dimension mean:   "
    f"{np.max(mean_gate_per_dimension):.4f}"
)


# ============================================================
# SAVE GATE VALUES
# ============================================================

results_dir = "results"

import os

os.makedirs(
    results_dir,
    exist_ok=True
)


# Save every gate dimension for every test sample

gate_columns = [
    f"gate_{i + 1}"
    for i in range(128)
]

gate_dataframe = pd.DataFrame(
    all_gate_values,
    columns=gate_columns
)

gate_dataframe.insert(
    0,
    "prediction",
    all_predictions
)

gate_dataframe.insert(
    0,
    "label",
    all_labels
)

gate_dataframe.to_csv(
    "results/gated_fusion_gate_values.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "Metric": [
        "Mean Gate",
        "Median Gate",
        "Minimum Gate",
        "Maximum Gate",
        "Gate Std",
        "Average Image Contribution",
        "Average Metadata Contribution",
        "Image-Dominant Gates (%)",
        "Metadata-Dominant Gates (%)",
        "Balanced Gates (%)"
    ],

    "Value": [
        mean_gate,
        median_gate,
        min_gate,
        max_gate,
        std_gate,
        mean_image_weight,
        mean_metadata_weight,
        image_dominant * 100,
        metadata_dominant * 100,
        balanced * 100
    ]
}

summary_dataframe = pd.DataFrame(
    summary
)

summary_dataframe.to_csv(
    "results/gated_fusion_gate_summary.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("Gate analysis complete.")
print("=" * 70)

print()

print(
    "Saved:"
)

print(
    "results/gated_fusion_gate_values.csv"
)

print(
    "results/gated_fusion_gate_summary.csv"
)