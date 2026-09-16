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
    r"C:\multimodal_skin_diagnosis"
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

            print(
                f"WARNING: Image directory not found: "
                f"{image_dir}"
            )

            continue

        image_files = list(
            image_dir.glob("*.jpg")
        )

        print(
            f"Found {len(image_files)} images in "
            f"{image_dir.name}"
        )

        for image_path in image_files:

            image_id = image_path.stem

            image_index[image_id] = str(image_path)

    print(
        f"Total indexed images: {len(image_index)}"
    )

    return image_index


# =========================================================
# Metadata preprocessing
# =========================================================

def prepare_metadata(df):

    df = df.copy()


    # -----------------------------------------------------
    # Age
    # -----------------------------------------------------

    median_age = df["age"].median()

    df["age"] = df["age"].fillna(
        median_age
    )

    df["age"] = df["age"] / 100.0


    # -----------------------------------------------------
    # Sex
    # -----------------------------------------------------

    sex_mapping = {
        "male": 0,
        "female": 1,
        "unknown": 2
    }

    df["sex"] = (
        df["sex"]
        .fillna("unknown")
        .map(sex_mapping)
    )

    # Safety in case an unexpected value occurs
    df["sex"] = df["sex"].fillna(2)


    # -----------------------------------------------------
    # Localization
    # -----------------------------------------------------

    # Fixed mapping used for all datasets
    location_mapping = {
        "abdomen": 0,
        "acral": 1,
        "back": 2,
        "chest": 3,
        "ear": 4,
        "face": 5,
        "foot": 6,
        "genital": 7,
        "hand": 8,
        "lower extremity": 9,
        "neck": 10,
        "scalp": 11,
        "trunk": 12,
        "unknown": 13,
        "upper extremity": 14
    }

    df["localization"] = (
        df["localization"]
        .fillna("unknown")
        .map(location_mapping)
    )

    # Unknown/unmapped locations
    df["localization"] = (
        df["localization"]
        .fillna(13)
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

        # -----------------------------------------------
        # Read CSV
        # -----------------------------------------------

        self.df = pd.read_csv(csv_file)


        # -----------------------------------------------
        # Prepare metadata
        # -----------------------------------------------

        self.df = prepare_metadata(
            self.df
        )


        # -----------------------------------------------
        # Build image index
        # -----------------------------------------------

        self.image_index = build_image_index()


        # -----------------------------------------------
        # Transform
        # -----------------------------------------------

        self.transform = transform


        # -----------------------------------------------
        # Check missing images
        # -----------------------------------------------

        missing_images = []

        for image_id in self.df["image_id"]:

            if image_id not in self.image_index:

                missing_images.append(
                    image_id
                )


        if len(missing_images) > 0:

            print(
                f"\nWARNING: "
                f"{len(missing_images)} images "
                f"from {csv_file} were not found."
            )

            print(
                "First missing images:"
            )

            print(
                missing_images[:10]
            )

        else:

            print(
                f"All {len(self.df)} images found "
                f"for {csv_file}"
            )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]


        # =================================================
        # Image
        # =================================================

        image_id = row["image_id"]


        if image_id not in self.image_index:

            raise FileNotFoundError(
                f"Image '{image_id}' was not found "
                f"in the HAM10000 image directories."
            )


        image_path = self.image_index[
            image_id
        ]


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        # =================================================
        # Metadata
        # =================================================

        age = float(
            row["age"]
        )

        sex = int(
            row["sex"]
        )

        localization = int(
            row["localization"]
        )


        metadata = torch.tensor(
            [
                age,
                sex,
                localization
            ],
            dtype=torch.float32
        )


        # =================================================
        # Label
        # =================================================

        label = CLASS_TO_INDEX[
            row["dx"]
        ]


        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return image, metadata, label


# =========================================================
# Training transforms
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(
        20
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# Validation / Test transforms
# =========================================================

val_test_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])