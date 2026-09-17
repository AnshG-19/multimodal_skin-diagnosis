from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.dataset import (
    HAM10000Dataset,
    val_test_transform,
)
from src.models.gated_fusion_model import GatedFusionModel


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

ROOT = Path(r"C:\multimodal_skin_diagnosis")

VAL_CSV = ROOT / "data" / "processed" / "val.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "gated_fusion_modality_dropout_best.pth"
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
NUM_CLASSES = 7
NUM_BINS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Collect model logits
# --------------------------------------------------

def get_logits(model, loader):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():

        for images, metadata, labels in loader:

            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)

            logits = model(images, metadata)

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    return logits, labels


# --------------------------------------------------
# Expected Calibration Error
# --------------------------------------------------

def calculate_ece(logits, labels, num_bins=10):

    probabilities = torch.softmax(logits, dim=1)

    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )

    accuracies = predictions.eq(labels)

    ece = 0.0

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    for i in range(num_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        if mask.sum() == 0:
            continue

        bin_accuracy = accuracies[mask].float().mean()
        bin_confidence = confidences[mask].mean()

        bin_probability = mask.float().mean()

        ece += (
            torch.abs(bin_accuracy - bin_confidence)
            * bin_probability
        )

    return ece.item()


# --------------------------------------------------
# Multiclass Brier Score
# --------------------------------------------------

def calculate_brier_score(logits, labels):

    probabilities = torch.softmax(logits, dim=1)

    targets = torch.zeros_like(probabilities)

    targets.scatter_(
        1,
        labels.unsqueeze(1),
        1
    )

    brier_score = torch.mean(
        torch.sum(
            (probabilities - targets) ** 2,
            dim=1
        )
    )

    return brier_score.item()


# --------------------------------------------------
# Temperature Scaling
# --------------------------------------------------

class TemperatureScaler(nn.Module):

    def __init__(self):
        super().__init__()

        self.temperature = nn.Parameter(
            torch.ones(1)
        )

    def forward(self, logits):

        temperature = self.temperature

        return logits / temperature


def fit_temperature(logits, labels):

    scaler = TemperatureScaler()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [scaler.temperature],
        lr=0.01,
        max_iter=50
    )

    def closure():

        optimizer.zero_grad()

        scaled_logits = scaler(logits)

        loss = criterion(
            scaled_logits,
            labels
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    return scaler


# --------------------------------------------------
# Reliability diagram data
# --------------------------------------------------

def get_reliability_data(
    logits,
    labels,
    num_bins=10
):

    probabilities = torch.softmax(logits, dim=1)

    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )

    accuracies = predictions.eq(labels)

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    for i in range(num_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        count = mask.sum().item()

        if count == 0:
            bin_confidences.append(np.nan)
            bin_accuracies.append(np.nan)
        else:
            bin_confidences.append(
                confidences[mask].mean().item()
            )

            bin_accuracies.append(
                accuracies[mask]
                .float()
                .mean()
                .item()
            )

        bin_counts.append(count)

    return (
        bin_confidences,
        bin_accuracies,
        bin_counts
    )


# --------------------------------------------------
# Plot reliability diagram
# --------------------------------------------------

def plot_reliability_diagram(
    original_logits,
    calibrated_logits,
    labels
):

    original_conf, original_acc, _ = (
        get_reliability_data(
            original_logits,
            labels,
            NUM_BINS
        )
    )

    calibrated_conf, calibrated_acc, _ = (
        get_reliability_data(
            calibrated_logits,
            labels,
            NUM_BINS
        )
    )

    x = np.linspace(
        0.05,
        0.95,
        NUM_BINS
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.plot(
        x,
        original_acc,
        marker="o",
        label="Before Temperature Scaling"
    )

    plt.plot(
        x,
        calibrated_acc,
        marker="o",
        label="After Temperature Scaling"
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    plt.title("Reliability Diagram")

    plt.legend()

    plt.grid(True)

    output_path = (
        RESULTS_DIR
        / "gated_fusion_reliability_diagram.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Reliability diagram saved to:")
    print(output_path)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Device:", DEVICE)

    # -----------------------------
    # Load datasets
    # -----------------------------

    val_dataset = HAM10000Dataset(
        csv_file=VAL_CSV,
        transform=val_test_transform
    )

    test_dataset = HAM10000Dataset(
        csv_file=TEST_CSV,
        transform=val_test_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------
    # Load model
    # -----------------------------

    model = GatedFusionModel(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)

    print("\nModel loaded successfully.")

    # -----------------------------
    # Get validation logits
    # -----------------------------

    print("\nCollecting validation predictions...")

    val_logits, val_labels = get_logits(
        model,
        val_loader
    )

    # -----------------------------
    # Get test logits
    # -----------------------------

    print("Collecting test predictions...")

    test_logits, test_labels = get_logits(
        model,
        test_loader
    )

    # -----------------------------
    # Before calibration
    # -----------------------------

    original_val_ece = calculate_ece(
        val_logits,
        val_labels,
        NUM_BINS
    )

    original_val_brier = calculate_brier_score(
        val_logits,
        val_labels
    )

    original_test_ece = calculate_ece(
        test_logits,
        test_labels,
        NUM_BINS
    )

    original_test_brier = calculate_brier_score(
        test_logits,
        test_labels
    )

    # -----------------------------
    # Fit temperature
    # -----------------------------

    print("\nFitting temperature scaling...")

    scaler = fit_temperature(
        val_logits,
        val_labels
    )

    temperature = scaler.temperature.item()

    print(
        f"Learned temperature: {temperature:.4f}"
    )

    # -----------------------------
    # Apply temperature
    # -----------------------------

    calibrated_val_logits = scaler(
        val_logits
    ).detach()

    calibrated_test_logits = scaler(
        test_logits
    ).detach()

    # -----------------------------
    # After calibration
    # -----------------------------

    calibrated_val_ece = calculate_ece(
        calibrated_val_logits,
        val_labels,
        NUM_BINS
    )

    calibrated_val_brier = calculate_brier_score(
        calibrated_val_logits,
        val_labels
    )

    calibrated_test_ece = calculate_ece(
        calibrated_test_logits,
        test_labels,
        NUM_BINS
    )

    calibrated_test_brier = calculate_brier_score(
        calibrated_test_logits,
        test_labels
    )

    # -----------------------------
    # Print results
    # -----------------------------

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    print("\nValidation:")
    print(
        f"ECE before: {original_val_ece:.4f}"
    )
    print(
        f"ECE after : {calibrated_val_ece:.4f}"
    )

    print(
        f"Brier before: {original_val_brier:.4f}"
    )
    print(
        f"Brier after : {calibrated_val_brier:.4f}"
    )

    print("\nTest:")
    print(
        f"ECE before: {original_test_ece:.4f}"
    )
    print(
        f"ECE after : {calibrated_test_ece:.4f}"
    )

    print(
        f"Brier before: {original_test_brier:.4f}"
    )
    print(
        f"Brier after : {calibrated_test_brier:.4f}"
    )

    # -----------------------------
    # Save results
    # -----------------------------

    results = pd.DataFrame([
        {
            "Dataset": "Validation",
            "ECE Before": original_val_ece,
            "ECE After": calibrated_val_ece,
            "Brier Before": original_val_brier,
            "Brier After": calibrated_val_brier
        },
        {
            "Dataset": "Test",
            "ECE Before": original_test_ece,
            "ECE After": calibrated_test_ece,
            "Brier Before": original_test_brier,
            "Brier After": calibrated_test_brier
        }
    ])

    results_path = (
        RESULTS_DIR
        / "gated_fusion_calibration_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # -----------------------------
    # Save temperature
    # -----------------------------

    temperature_path = (
        RESULTS_DIR
        / "temperature_scaling.txt"
    )

    with open(
        temperature_path,
        "w"
    ) as f:

        f.write(
            f"Learned temperature: {temperature:.6f}\n"
        )

    # -----------------------------
    # Reliability diagram
    # -----------------------------

    plot_reliability_diagram(
        test_logits,
        calibrated_test_logits,
        test_labels
    )

    print("\nResults saved to:")
    print(results_path)

    print("\nTemperature saved to:")
    print(temperature_path)


if __name__ == "__main__":
    main()