from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.dataset import (
    HAM10000Dataset,
    train_transform,
    val_test_transform
)

from src.models.image_model import ImageModel


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(
    r"C:\Users\ANSH\OneDrive\Desktop\multimodal_skin_diagnosis"
)

TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
VAL_FILE = ROOT / "data" / "processed" / "val.csv"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("        IMAGE-ONLY SKIN DIAGNOSIS")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")


# =====================================================
# DATASET
# =====================================================

print("\nLoading datasets...")

train_dataset = HAM10000Dataset(
    TRAIN_FILE,
    transform=train_transform
)

val_dataset = HAM10000Dataset(
    VAL_FILE,
    transform=val_test_transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


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

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# =====================================================
# MODEL
# =====================================================

print("\nLoading EfficientNet-B0...")

model = ImageModel(
    num_classes=7
)

model = model.to(DEVICE)

print("Model loaded successfully.")


# =====================================================
# CLASS WEIGHTS
# =====================================================

class_counts = torch.tensor(
    [
        222,    # akiec
        349,    # bcc
        770,    # bkl
        92,     # df
        771,    # mel
        4662,   # nv
        93      # vasc
    ],
    dtype=torch.float32
)

weights = 1.0 / class_counts

weights = (
    weights /
    weights.sum()
    * len(class_counts)
)

weights = weights.to(DEVICE)


criterion = nn.CrossEntropyLoss(
    weight=weights
)


# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =====================================================
# LEARNING RATE SCHEDULER
# =====================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# =====================================================
# TRAINING
# =====================================================

best_f1 = 0.0

for epoch in range(EPOCHS):

    print("\n")
    print("=" * 60)
    print(f"                 EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------

    model.train()

    train_predictions = []
    train_labels = []

    total_loss = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Training {epoch + 1}/{EPOCHS}",
        unit="batch",
        dynamic_ncols=True
    )

    for images, metadata, labels in train_bar:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_labels.extend(
            labels.detach().cpu().numpy()
        )

        # Update progress bar

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )


    # -------------------------------------------------
    # TRAIN METRICS
    # -------------------------------------------------

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )

    train_f1 = f1_score(
        train_labels,
        train_predictions,
        average="macro"
    )

    average_train_loss = (
        total_loss /
        len(train_loader)
    )


    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    model.eval()

    val_predictions = []
    val_labels = []

    val_loss = 0.0

    val_bar = tqdm(
        val_loader,
        desc=f"Validation {epoch + 1}/{EPOCHS}",
        unit="batch",
        dynamic_ncols=True
    )


    with torch.no_grad():

        for images, metadata, labels in val_bar:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

            val_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


    # -------------------------------------------------
    # VALIDATION METRICS
    # -------------------------------------------------

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro"
    )

    average_val_loss = (
        val_loss /
        len(val_loader)
    )


    # -------------------------------------------------
    # SCHEDULER
    # -------------------------------------------------

    scheduler.step(val_f1)


    # -------------------------------------------------
    # EPOCH SUMMARY
    # -------------------------------------------------

    print("\n")
    print("-" * 60)
    print(f"EPOCH {epoch + 1} RESULTS")
    print("-" * 60)

    print(
        f"Train Loss       : {average_train_loss:.4f}"
    )

    print(
        f"Train Accuracy   : {train_accuracy:.4f}"
    )

    print(
        f"Train Macro F1   : {train_f1:.4f}"
    )

    print(
        f"Validation Loss   : {average_val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Validation Macro F1: {val_f1:.4f}"
    )

    print(
        f"Learning Rate     : "
        f"{optimizer.param_groups[0]['lr']:.2e}"
    )

    print("-" * 60)


    # -------------------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        checkpoint_dir = (
            ROOT / "checkpoints"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_accuracy": val_accuracy
            },
            checkpoint_dir /
            "image_model_best.pth"
        )

        print("✓ New best model saved!")

    else:

        print(
            f"No improvement. "
            f"Best Macro F1: {best_f1:.4f}"
        )


# =====================================================
# COMPLETE
# =====================================================

print("\n")
print("=" * 60)
print("              TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Validation Macro F1: {best_f1:.4f}"
)

print(
    "Model saved to:"
)

print(
    ROOT /
    "checkpoints" /
    "image_model_best.pth"
)

print("=" * 60)