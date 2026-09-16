import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import numpy as np

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
    print("GPU:", torch.cuda.get_device_name(0))


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

print("Test samples:", len(test_dataset))


# ============================================================
# MODEL
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
# CLASS NAMES
# ============================================================

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []
all_probabilities = []


with torch.no_grad():

    progress_bar = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    from tqdm import tqdm

    for images, metadata, labels in tqdm(
        progress_bar,
        desc="Testing"
    ):

        images = images.to(device)
        metadata = metadata.to(device)

        outputs = model(
            images,
            metadata
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# Convert to NumPy arrays

all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

balanced_accuracy = balanced_accuracy_score(
    all_labels,
    all_predictions
)

macro_precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# ROC-AUC
# ============================================================

try:

    macro_auc = roc_auc_score(
        all_labels,
        all_probabilities,
        multi_class="ovr",
        average="macro"
    )

except ValueError:

    macro_auc = float("nan")


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print()
print("=" * 60)
print("GATED FUSION TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy:          {accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision:   {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall:      {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1:          {macro_f1 * 100:.2f}%"
)

print(
    f"Macro ROC-AUC:     {macro_auc * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()

print(
    "             " +
    " ".join(
        f"{name:>7}"
        for name in class_names
    )
)

for i, row in enumerate(cm):

    print(
        f"{class_names[i]:>7}     " +
        " ".join(
            f"{value:7d}"
            for value in row
        )
    )


# ============================================================
# FINISHEDgit 
# ============================================================

print()
print("=" * 60)
print("Evaluation complete")
print("=" * 60)