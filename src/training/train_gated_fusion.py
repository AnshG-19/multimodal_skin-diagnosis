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

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# DATASET TEST
# ============================================================

test_image, test_metadata, test_label = train_dataset[0]

print()
print("Dataset test:")
print("Image type:", type(test_image))
print("Image shape:", test_image.shape)
print("Metadata type:", type(test_metadata))
print("Metadata shape:", test_metadata.shape)
print("Label:", test_label)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = GatedFusionModel(num_classes=7)
model = model.to(device)


# ============================================================
# CLASS WEIGHTS
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

class_counts = []

for class_name in class_names:

    count = (
        train_dataset.df["dx"] == class_name
    ).sum()

    class_counts.append(count)

class_counts = torch.tensor(
    class_counts,
    dtype=torch.float32
)

class_weights = 1.0 / class_counts

class_weights = (
    class_weights
    / class_weights.sum()
    * len(class_weights)
)

class_weights = class_weights.to(device)

print()
print("Class counts:")
print(class_counts)

print()
print("Class weights:")
print(class_weights)


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
    lr=5e-5,
    weight_decay=1e-4
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
# TRAINING SETTINGS
# ============================================================

num_epochs = 10
best_val_f1 = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(num_epochs):

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("=" * 60)

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    train_predictions = []
    train_true_labels = []

    train_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )

    for images, metadata, labels in progress_bar:

        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            images,
            metadata
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_true_labels.extend(
            labels.detach().cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss /= len(train_loader)

    train_accuracy = accuracy_score(
        train_true_labels,
        train_predictions
    )

    train_f1 = f1_score(
        train_true_labels,
        train_predictions,
        average="macro",
        zero_division=0
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_predictions = []
    val_true_labels = []

    val_loss = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            val_loader,
            desc="Validation"
        )

        for images, metadata, labels in progress_bar:

            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            outputs = model(
                images,
                metadata
            )

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

            val_true_labels.extend(
                labels.cpu().numpy()
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    val_loss /= len(val_loader)

    val_accuracy = accuracy_score(
        val_true_labels,
        val_predictions
    )

    val_f1 = f1_score(
        val_true_labels,
        val_predictions,
        average="macro",
        zero_division=0
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Train Macro F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val Macro F1: {val_f1:.4f}"
    )

    print(
        f"Learning Rate: {current_lr:.7f}"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            "checkpoints/gated_fusion_model_best.pth"
        )

        print(
            f"Best model saved! "
            f"Val Macro F1: {best_val_f1:.4f}"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 60)
print("Training complete")
print(
    f"Best Validation Macro F1: "
    f"{best_val_f1:.4f}"
)
print("=" * 60)