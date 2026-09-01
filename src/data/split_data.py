from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ==========================================
# Paths
# ==========================================

ROOT = Path(
    r"C:\Users\ANSH\OneDrive\Desktop\multimodal_skin_diagnosis"
)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

METADATA_FILE = RAW_DIR / "HAM10000_metadata.csv"


# Create processed directory
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Load metadata
# ==========================================

df = pd.read_csv(METADATA_FILE)

print("Total samples:", len(df))
print("Total lesions:", df["lesion_id"].nunique())


# ==========================================
# First split: 85% train+validation
#         15% test
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_val_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["lesion_id"]
    )
)

train_val = df.iloc[train_val_idx].copy()
test = df.iloc[test_idx].copy()


# ==========================================
# Second split:
# 85% train+validation
#     ↓
# 82.35% train
# 17.65% validation
#
# Final:
# ~70% train
# ~15% validation
# ~15% test
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1765,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        train_val,
        groups=train_val["lesion_id"]
    )
)

train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()


# ==========================================
# Save splits
# ==========================================

train.to_csv(
    PROCESSED_DIR / "train.csv",
    index=False
)

val.to_csv(
    PROCESSED_DIR / "val.csv",
    index=False
)

test.to_csv(
    PROCESSED_DIR / "test.csv",
    index=False
)


# ==========================================
# Print results
# ==========================================

print("\n========== SPLIT RESULTS ==========")

print("Train samples:", len(train))
print("Validation samples:", len(val))
print("Test samples:", len(test))

print("\nTrain lesions:", train["lesion_id"].nunique())
print("Validation lesions:", val["lesion_id"].nunique())
print("Test lesions:", test["lesion_id"].nunique())


# ==========================================
# Check for lesion leakage
# ==========================================

train_lesions = set(train["lesion_id"])
val_lesions = set(val["lesion_id"])
test_lesions = set(test["lesion_id"])


print("\n========== LEAKAGE CHECK ==========")

print(
    "Train ∩ Validation:",
    len(train_lesions & val_lesions)
)

print(
    "Train ∩ Test:",
    len(train_lesions & test_lesions)
)

print(
    "Validation ∩ Test:",
    len(val_lesions & test_lesions)
)


# ==========================================
# Class distribution
# ==========================================

print("\n========== TRAIN CLASS DISTRIBUTION ==========")
print(train["dx"].value_counts())

print("\n========== VALIDATION CLASS DISTRIBUTION ==========")
print(val["dx"].value_counts())

print("\n========== TEST CLASS DISTRIBUTION ==========")
print(test["dx"].value_counts())