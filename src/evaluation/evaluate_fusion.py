import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.data.dataset import HAM10000Dataset
from src.models.fusion_model import FusionModel


# ============================================================
# CONFIG
# ============================================================

TEST_CSV = "data/processed/test.csv"
MODEL_PATH = "checkpoints/fusion_model_best.pth"

BATCH_SIZE = 16
NUM_CLASSES = 7

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("        MULTIMODAL FUSION TEST EVALUATION")
print("=" * 60)

print(f"Device: {device}")


# ============================================================
# DATASET
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = HAM10000Dataset(
    TEST_CSV,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Test samples: {len(test_dataset)}")
print(f"Test batches: {len(test_loader)}")


# ============================================================
# LOAD MODEL
# ============================================================

model = FusionModel(
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:
        model.load_state_dict(checkpoint)

else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, metadata, labels in test_loader:

        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        outputs = model(
            images,
            metadata
        )

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


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)


# ============================================================
# METRICS
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
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("                 TEST RESULTS")
print("=" * 60)

print(f"Accuracy             : {accuracy:.4f}")
print(f"Balanced Accuracy    : {balanced_accuracy:.4f}")
print(f"Macro Precision      : {macro_precision:.4f}")
print(f"Macro Recall         : {macro_recall:.4f}")
print(f"Macro F1             : {macro_f1:.4f}")
print(f"Macro ROC-AUC        : {macro_auc:.4f}")


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("             PER-CLASS RESULTS")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("=" * 60)
print("              CONFUSION MATRIX")
print("=" * 60)

print("\nClass order:")
print(CLASS_NAMES)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("          FUSION EVALUATION COMPLETE")
print("=" * 60)