import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import HAM10000Dataset
from src.models.fusion_model import FusionModel


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "fusion_model_best.pth"
)

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.00005

NUM_CLASSES = 7

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("        MULTIMODAL FUSION TRAINING")
print("=" * 65)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = HAM10000Dataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = HAM10000Dataset(
    VAL_CSV,
    transform=val_transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ============================================================
# DATALOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

model = FusionModel(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# HAM10000 is highly imbalanced.
# These weights reduce the dominance of NV.

class_counts = torch.tensor([
    222,   # akiec
    349,   # bcc
    770,   # bkl
    92,    # df
    771,   # mel
    4662,  # nv
    93     # vasc
], dtype=torch.float32)

class_weights = 1.0 / class_counts

class_weights = (
    class_weights /
    class_weights.sum()
) * NUM_CLASSES

class_weights = class_weights.to(device)

print("\nClass weights:")
for name, weight in zip(CLASS_NAMES, class_weights):
    print(f"{name:6s}: {weight.item():.4f}")


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
    weight_decay=1e-4
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# BEST SCORE
# ============================================================

best_val_f1 = 0.0

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    print("\n")
    print("=" * 65)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 65)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    train_labels = []
    train_predictions = []


    progress = tqdm(
        train_loader,
        desc=f"Training {epoch + 1}/{EPOCHS}",
        unit="batch"
    )


    for images, metadata, labels in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        metadata = metadata.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(
            images,
            metadata
        )


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Statistics
        train_loss += (
            loss.item() *
            images.size(0)
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


        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
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


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_labels = []
    val_predictions = []


    progress = tqdm(
        val_loader,
        desc=f"Validation {epoch + 1}/{EPOCHS}",
        unit="batch"
    )


    with torch.no_grad():

        for images, metadata, labels in progress:

            images = images.to(
                device,
                non_blocking=True
            )

            metadata = metadata.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            outputs = model(
                images,
                metadata
            )


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item() *
                images.size(0)
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


            progress.set_postfix(
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


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(val_f1)


    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("-" * 60)
    print("EPOCH RESULTS")
    print("-" * 60)

    print(f"Train Loss         : {train_loss:.4f}")
    print(f"Train Accuracy     : {train_accuracy:.4f}")
    print(f"Train Macro F1     : {train_f1:.4f}")

    print(f"Validation Loss    : {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Macro F1: {val_f1:.4f}")

    print(
        f"Learning Rate      : "
        f"{optimizer.param_groups[0]['lr']:.2e}"
    )

    print("-" * 60)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("✓ New best fusion model saved!")


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 65)
print("          MULTIMODAL FUSION TRAINING COMPLETE")
print("=" * 65)

print(
    f"Best Validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Model saved to: "
    f"{os.path.abspath(BEST_MODEL_PATH)}"
)

print("=" * 65)