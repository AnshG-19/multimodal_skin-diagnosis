from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import (
    HAM10000Dataset,
    val_test_transform
)

from src.models.image_model import ImageModel


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(
    r"C:\Users\ANSH\OneDrive\Desktop\multimodal_skin_diagnosis"
)

TEST_FILE = ROOT / "data" / "processed" / "test.csv"

MODEL_FILE = (
    ROOT /
    "checkpoints" /
    "image_model_best.pth"
)

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# =====================================================
# HEADER
# =====================================================

print("=" * 65)
print("              IMAGE-ONLY TEST EVALUATION")
print("=" * 65)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Test file: {TEST_FILE}"
)

print(
    f"Model: {MODEL_FILE}"
)


# =====================================================
# LOAD TEST DATASET
# =====================================================

print("\nLoading test dataset...")

test_dataset = HAM10000Dataset(
    TEST_FILE,
    transform=val_test_transform
)

print(
    f"Test samples: {len(test_dataset)}"
)


# =====================================================
# TEST DATALOADER
# =====================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(
    f"Test batches: {len(test_loader)}"
)


# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading model...")

model = ImageModel(
    num_classes=7
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

# Our training script saves a dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# =====================================================
# TESTING
# =====================================================

print("\nRunning test evaluation...\n")

all_labels = []
all_predictions = []
all_probabilities = []


for images, metadata, labels in test_loader:

    images = images.to(
        DEVICE,
        non_blocking=True
    )

    labels = labels.to(
        DEVICE,
        non_blocking=True
    )

    with torch.no_grad():

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )


    all_labels.extend(
        labels.cpu().numpy()
    )

    all_predictions.extend(
        predictions.cpu().numpy()
    )

    all_probabilities.extend(
        probabilities.cpu().numpy()
    )


y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

y_prob = np.array(
    all_probabilities
)


# =====================================================
# MAIN METRICS
# =====================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# =====================================================
# MULTICLASS ROC-AUC
# =====================================================

try:

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

except ValueError:

    roc_auc = float("nan")


# =====================================================
# PRINT RESULTS
# =====================================================

print("=" * 65)
print("                    TEST RESULTS")
print("=" * 65)

print(
    f"Accuracy             : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy    : {balanced_accuracy:.4f}"
)

print(
    f"Macro Precision      : {macro_precision:.4f}"
)

print(
    f"Macro Recall         : {macro_recall:.4f}"
)

print(
    f"Macro F1             : {macro_f1:.4f}"
)

print(
    f"Weighted F1          : {weighted_f1:.4f}"
)

print(
    f"Macro ROC-AUC        : {roc_auc:.4f}"
)


# =====================================================
# PER-CLASS REPORT
# =====================================================

print("\n")
print("=" * 65)
print("                 PER-CLASS RESULTS")
print("=" * 65)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("=" * 65)
print("                  CONFUSION MATRIX")
print("=" * 65)

print(cm)


# =====================================================
# SAVE RESULTS DIRECTORY
# =====================================================

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# SAVE CONFUSION MATRIX
# =====================================================

plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "EfficientNet-B0 Image-Only Confusion Matrix"
)

plt.tight_layout()

cm_path = (
    RESULTS_DIR /
    "image_only_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# =====================================================
# SAVE METRICS
# =====================================================

metrics_path = (
    RESULTS_DIR /
    "image_only_metrics.txt"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "IMAGE-ONLY EFFICIENTNET-B0\n"
    )

    f.write(
        "===========================\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: {balanced_accuracy:.4f}\n"
    )

    f.write(
        f"Macro Precision: {macro_precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: {macro_recall:.4f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.4f}\n"
    )

    f.write(
        f"Weighted F1: {weighted_f1:.4f}\n"
    )

    f.write(
        f"Macro ROC-AUC: {roc_auc:.4f}\n"
    )

    f.write(
        "\n\nPER-CLASS RESULTS\n"
    )

    f.write(
        report
    )


# =====================================================
# COMPLETE
# =====================================================

print("\n")
print("=" * 65)
print("                 EVALUATION COMPLETE")
print("=" * 65)

print(
    f"Confusion matrix saved to:\n{cm_path}"
)

print(
    f"\nMetrics saved to:\n{metrics_path}"
)

print("=" * 65)