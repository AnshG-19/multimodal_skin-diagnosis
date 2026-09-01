from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import (
    HAM10000Dataset,
    train_transform,
    val_test_transform
)

from src.models.metadata_model import MetadataModel


# =====================================================
# CONFIG
# =====================================================

ROOT = Path(
    r"C:\Users\ANSH\OneDrive\Desktop\multimodal_skin_diagnosis"
)

TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
VAL_FILE = ROOT / "data" / "processed" / "val.csv"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL = (
    CHECKPOINT_DIR /
    "metadata_model_best.pth"
)


# =====================================================
# HEADER
# =====================================================

print("=" * 65)
print("             METADATA-ONLY MODEL")
print("=" * 65)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# =====================================================
# DATASETS
# =====================================================

train_dataset = HAM10000Dataset(
    TRAIN_FILE,
    transform=train_transform
)

val_dataset = HAM10000Dataset(
    VAL_FILE,
    transform=val_test_transform
)

print(
    f"Training samples: {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)


# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# =====================================================
# MODEL
# =====================================================

model = MetadataModel(
    num_classes=7
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================================
# TRAINING
# =====================================================

best_f1 = 0.0


for epoch in range(EPOCHS):

    print("\n" + "=" * 65)
    print(
        f"                 EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("=" * 65)

    # -------------------------------------------------
    # TRAIN
    # -------------------------------------------------

    model.train()

    train_loss = 0.0
    train_labels = []
    train_predictions = []

    train_bar = tqdm(
        train_loader,
        desc=f"Training {epoch + 1}/{EPOCHS}",
        unit="batch",
        dynamic_ncols=True
    )

    for images, metadata, labels in train_bar:

        metadata = metadata.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(metadata)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() *
            labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_labels.extend(
            labels.detach().cpu().numpy()
        )

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss /= len(train_dataset)

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )

    train_f1 = f1_score(
        train_labels,
        train_predictions,
        average="macro",
        zero_division=0
    )


    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_labels = []
    val_predictions = []

    val_bar = tqdm(
        val_loader,
        desc=f"Validation {epoch + 1}/{EPOCHS}",
        unit="batch",
        dynamic_ncols=True
    )

    with torch.no_grad():

        for images, metadata, labels in val_bar:

            metadata = metadata.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(metadata)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item() *
                labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


    val_loss /= len(val_dataset)

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro",
        zero_division=0
    )


    # -------------------------------------------------
    # RESULTS
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("EPOCH RESULTS")
    print("-" * 60)

    print(
        f"Train Loss        : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy    : {train_accuracy:.4f}"
    )

    print(
        f"Train Macro F1    : {train_f1:.4f}"
    )

    print(
        f"Validation Loss   : {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Validation Macro F1: {val_f1:.4f}"
    )

    print("-" * 60)


    # -------------------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_f1
            },
            BEST_MODEL
        )

        print("✓ New best model saved!")


# =====================================================
# COMPLETE
# =====================================================

print("\n")
print("=" * 65)
print("             METADATA TRAINING COMPLETE")
print("=" * 65)

print(
    f"Best Validation Macro F1: {best_f1:.4f}"
)

print(
    f"Model saved to: {BEST_MODEL}"
)