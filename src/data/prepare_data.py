from pathlib import Path
import pandas as pd


# -----------------------------
# Paths
# -----------------------------

ROOT = Path(r"C:\Users\ANSH\OneDrive\Desktop\multimodal_skin_diagnosis")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

METADATA_FILE = RAW_DIR / "HAM10000_metadata.csv"

IMAGE_DIR_1 = RAW_DIR / r"C:\Users\ANSH\Downloads\archive\HAM10000_images_part_1"
IMAGE_DIR_2 = RAW_DIR / r"C:\Users\ANSH\Downloads\archive\HAM10000_images_part_2"


# -----------------------------
# Load metadata
# -----------------------------

df = pd.read_csv(METADATA_FILE)

print("\n========== DATASET ==========")

print("Number of rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# -----------------------------
# Classes
# -----------------------------

print("\n========== CLASSES ==========")

print(df["dx"].value_counts())


# -----------------------------
# Missing values
# -----------------------------

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


# -----------------------------
# Image verification
# -----------------------------

print("\n========== IMAGE CHECK ==========")

image_dirs = [
    IMAGE_DIR_1,
    IMAGE_DIR_2
]

image_paths = {}

for image_dir in image_dirs:

    if not image_dir.exists():
        print("WARNING: Directory not found:", image_dir)
        continue

    for image_path in image_dir.glob("*.jpg"):
        image_paths[image_path.stem] = image_path


print("Images found:", len(image_paths))


# -----------------------------
# Match images with CSV
# -----------------------------

df["image_path"] = df["image_id"].map(
    lambda x: str(image_paths.get(x, ""))
)

missing_images = (df["image_path"] == "").sum()

print("Images missing from CSV:", missing_images)


# -----------------------------
# Final information
# -----------------------------

print("\n========== SUMMARY ==========")

print("Samples:", len(df))
print("Images found:", len(df) - missing_images)
print("Missing images:", missing_images)

print("\nDiagnosis distribution:")
print(df["dx"].value_counts())

print("\nMetadata missing values:")
print(df[["age", "sex", "localization"]].isnull().sum())