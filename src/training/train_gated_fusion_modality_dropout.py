import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import (
    HAM10000Dataset,
    train_transform,
    val_test_transform,
    CLASS_TO_INDEX
)

from src.models.gated_fusion_model import GatedFusionModel


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

# Probability of removing the complete metadata modality
METADATA_DROPOUT_PROB = 0.30

NUM_CLASSES = 7

CHECKPOINT_PATH = (
    "checkpoints/"
    "gated_fusion_modality_dropout_best.pth"
)


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
# DATASETS
# ============================================================

train_dataset = HAM10000Dataset(
    "data/processed/train.csv",
    transform=train_transform
)

val_dataset = HAM10000Dataset(
    "data/processed/val.csv",
    transform=val_test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# MODEL
# ============================================================

model = GatedFusionModel(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# The CSV contains the class name in the "dx" column.
# HAM10000Dataset converts these class names into numeric
# labels internally using CLASS_TO_INDEX.
#
# We reproduce that mapping here to calculate class weights.

train_df = pd.read_csv(
    "data/processed/train.csv"
)

labels = train_df["dx"].map(
    CLASS_TO_INDEX
).values

class_counts = np.bincount(
    labels,
    minlength=NUM_CLASSES
)

class_weights = (
    1.0 /
    torch.tensor(
        class_counts,
        dtype=torch.float32
    )
)

# Normalize weights so their mean is 1
class_weights = (
    class_weights /
    class_weights.mean()
)

class_weights = class_weights.to(device)

print()
print("Class counts:")

for class_name, class_index in CLASS_TO_INDEX.items():

    print(
        f"{class_name:6s}: "
        f"{class_counts[class_index]}"
    )

print()
print("Class weights:")

for class_name, class_index in CLASS_TO_INDEX.items():

    print(
        f"{class_name:6s}: "
        f"{class_weights[class_index].item():.4f}"
    )


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_val_f1 = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print()
    print("=" * 70)
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print("=" * 70)


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    train_loss = 0.0

    train_predictions = []
    train_labels = []

    dropped_samples = 0
    total_samples = 0


    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, metadata, labels_batch in progress_bar:

        images = images.to(device)
        metadata = metadata.to(device)
        labels_batch = labels_batch.to(device)


        # ----------------------------------------------------
        # MODALITY DROPOUT
        # ----------------------------------------------------
        #
        # Randomly remove the COMPLETE metadata modality
        # for approximately 30% of training samples.
        #
        # age          -> 0.0
        # sex          -> 2.0
        # localization -> 13.0
        #
        # Image is never removed.
        # ----------------------------------------------------

        dropout_mask = (
            torch.rand(
                metadata.size(0),
                device=device
            )
            < METADATA_DROPOUT_PROB
        )


        if dropout_mask.any():

            metadata = metadata.clone()

            metadata[dropout_mask, 0] = 0.0
            metadata[dropout_mask, 1] = 2.0
            metadata[dropout_mask, 2] = 13.0

            dropped_samples += (
                dropout_mask.sum().item()
            )


        total_samples += images.size(0)


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        optimizer.zero_grad()

        outputs = model(
            images,
            metadata
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels_batch
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        train_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_labels.extend(
            labels_batch.detach().cpu().numpy()
        )


        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    # ========================================================
    # TRAINING METRICS
    # ========================================================

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

    dropout_percentage = (
        dropped_samples /
        total_samples
    ) * 100


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_predictions = []
    val_labels = []


    with torch.no_grad():

        progress_bar = tqdm(
            val_loader,
            desc="Validation"
        )


        for images, metadata, labels_batch in progress_bar:

            images = images.to(device)
            metadata = metadata.to(device)
            labels_batch = labels_batch.to(device)


            # IMPORTANT:
            #
            # No modality dropout during validation.
            #
            # Validation uses complete metadata.

            outputs = model(
                images,
                metadata
            )


            loss = criterion(
                outputs,
                labels_batch
            )


            val_loss += (
                loss.item() *
                images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_labels.extend(
                labels_batch.cpu().numpy()
            )


            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

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


    # ========================================================
    # LEARNING RATE UPDATE
    # ========================================================

    scheduler.step(
        val_f1
    )

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        f"Metadata dropout this epoch: "
        f"{dropout_percentage:.2f}%"
    )

    print(
        f"Train Loss:      "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Train Macro F1:  "
        f"{train_f1:.4f}"
    )

    print(
        f"Val Loss:        "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy:    "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Val Macro F1:    "
        f"{val_f1:.4f}"
    )

    print(
        f"Learning Rate:   "
        f"{current_lr:.2e}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        os.makedirs(
            "checkpoints",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print()
        print(
            "Best model saved."
        )

        print(
            f"Best Val Macro F1: "
            f"{best_val_f1:.4f}"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 70)
print("Training complete.")
print("=" * 70)

print()

print(
    f"Best Validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print()

print(
    "Saved checkpoint:"
)

print(
    CHECKPOINT_PATH
)