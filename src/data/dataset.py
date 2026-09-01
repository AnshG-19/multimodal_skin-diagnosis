from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# =========================================================
# Paths
# =========================================================

ROOT = Path(
    r"C:\Users\ANSH\OneDrive\Desktop\multimodal_skin_diagnosis"
)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


IMAGE_DIRS = [
    RAW_DIR / "HAM10000_images_part_1",
    RAW_DIR / "HAM10000_images_part_2"
]


# =========================================================
# Class mapping
# =========================================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# =========================================================
# Find image paths
# =========================================================

def build_image_index():

    image_index = {}

    for image_dir in IMAGE_DIRS:

        if not image_dir.exists():
            continue

        for image_path in image_dir.glob("*.jpg"):

            image_index[image_path.stem] = str(image_path)

    return image_index


# =========================================================
# Metadata preprocessing
# =========================================================

def prepare_metadata(df):

    df = df.copy()

    # Fill missing age with median age
    median_age = df["age"].median()

    df["age"] = df["age"].fillna(median_age)

    # Normalize age to approximately 0-1
    df["age"] = df["age"] / 100.0

    # Encode sex
    sex_mapping = {
        "male": 0,
        "female": 1,
        "unknown": 2
    }

    df["sex"] = df["sex"].fillna("unknown")
    df["sex"] = df["sex"].map(sex_mapping)

    # Encode localization
    locations = sorted(
        df["localization"].dropna().unique()
    )

    location_mapping = {
        location: index
        for index, location in enumerate(locations)
    }

    df["localization"] = (
        df["localization"]
        .fillna("unknown")
        .map(location_mapping)
    )

    return df


# =========================================================
# Dataset
# =========================================================

class HAM10000Dataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.df = pd.read_csv(csv_file)

        self.df = prepare_metadata(self.df)

        self.image_index = build_image_index()

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        # -----------------------------------------
        # Image
        # -----------------------------------------

        image_id = row["image_id"]

        image_path = self.image_index[image_id]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)


        # -----------------------------------------
        # Metadata
        # -----------------------------------------

        age = float(row["age"])

        sex = int(row["sex"])

        localization = int(row["localization"])


        metadata = torch.tensor(
            [
                age,
                sex,
                localization
            ],
            dtype=torch.float32
        )


        # -----------------------------------------
        # Label
        # -----------------------------------------

        label = CLASS_TO_INDEX[row["dx"]]

        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return image, metadata, label


# =========================================================
# Image transforms
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])