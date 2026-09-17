import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.dataset import (
    HAM10000Dataset,
    val_test_transform
)

from src.models.gated_fusion_model import GatedFusionModel


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT_PATH = (
    "checkpoints/"
    "gated_fusion_modality_dropout_best.pth"
)

RESULTS_DIR = "results"

OUTPUT_FILE = (
    "results/"
    "gated_fusion_modality_dropout_missing_modality.csv"
)

BATCH_SIZE = 16
NUM_CLASSES = 7


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
# TEST DATASET
# ============================================================

test_dataset = HAM10000Dataset(
    "data/processed/test.csv",
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# LOAD MODEL
# ============================================================

model = GatedFusionModel(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print()
print(
    "Modality-dropout gated fusion model loaded."
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)


# ============================================================
# CLASS NAMES
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


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_condition(
    condition_name,
    metadata_mode
):

    print()
    print("=" * 70)
    print(
        f"EVALUATING: {condition_name}"
    )
    print("=" * 70)


    all_labels = []
    all_predictions = []
    all_probabilities = []


    with torch.no_grad():

        progress_bar = tqdm(
            test_loader,
            desc=condition_name
        )


        for images, metadata, labels in progress_bar:

            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)


            # =================================================
            # APPLY MISSING-MODALITY CONDITION
            # =================================================

            metadata = metadata.clone()


            if metadata_mode == "complete":

                # Keep metadata unchanged
                pass


            elif metadata_mode == "age_missing":

                # Age is represented as normalized age.
                # 0.0 is used to represent missing age.

                metadata[:, 0] = 0.0


            elif metadata_mode == "sex_missing":

                # 2.0 represents unknown sex.

                metadata[:, 1] = 2.0


            elif metadata_mode == "localization_missing":

                # 13.0 represents unknown localization.

                metadata[:, 2] = 13.0


            elif metadata_mode == "all_missing":

                # Remove the complete metadata modality.

                metadata[:, 0] = 0.0
                metadata[:, 1] = 2.0
                metadata[:, 2] = 13.0


            else:

                raise ValueError(
                    f"Unknown metadata mode: "
                    f"{metadata_mode}"
                )


            # =================================================
            # MODEL PREDICTION
            # =================================================

            outputs = model(
                images,
                metadata
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            # =================================================
            # STORE RESULTS
            # =================================================

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    all_probabilities = np.array(
        all_probabilities
    )


    # ========================================================
    # METRICS
    # ========================================================

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


    # ========================================================
    # MACRO ROC-AUC
    # ========================================================

    try:

        macro_auc = roc_auc_score(
            all_labels,
            all_probabilities,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        macro_auc = float("nan")


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        f"Accuracy:          "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision:   "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall:      "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1:          "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Macro ROC-AUC:     "
        f"{macro_auc * 100:.2f}%"
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "Condition": condition_name,
        "Accuracy": accuracy * 100,
        "Balanced Accuracy": balanced_accuracy * 100,
        "Macro Precision": macro_precision * 100,
        "Macro Recall": macro_recall * 100,
        "Macro F1": macro_f1 * 100,
        "Macro ROC-AUC": macro_auc * 100
    }


# ============================================================
# RUN FIVE CONDITIONS
# ============================================================

results = []


# ------------------------------------------------------------
# 1. COMPLETE METADATA
# ------------------------------------------------------------

results.append(
    evaluate_condition(
        "Complete",
        "complete"
    )
)


# ------------------------------------------------------------
# 2. AGE MISSING
# ------------------------------------------------------------

results.append(
    evaluate_condition(
        "Age Missing",
        "age_missing"
    )
)


# ------------------------------------------------------------
# 3. SEX MISSING
# ------------------------------------------------------------

results.append(
    evaluate_condition(
        "Sex Missing",
        "sex_missing"
    )
)


# ------------------------------------------------------------
# 4. LOCALIZATION MISSING
# ------------------------------------------------------------

results.append(
    evaluate_condition(
        "Localization Missing",
        "localization_missing"
    )
)


# ------------------------------------------------------------
# 5. ALL METADATA MISSING
# ------------------------------------------------------------

results.append(
    evaluate_condition(
        "All Metadata Missing",
        "all_missing"
    )
)


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_dataframe = pd.DataFrame(
    results
)


# ============================================================
# CALCULATE PERFORMANCE CHANGE
# ============================================================

complete_f1 = results_dataframe.loc[
    results_dataframe["Condition"] == "Complete",
    "Macro F1"
].iloc[0]


results_dataframe["Macro F1 Change"] = (
    results_dataframe["Macro F1"]
    - complete_f1
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

results_dataframe.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print()
print("=" * 100)
print("FIVE-CONDITION MISSING-MODALITY RESULTS")
print("=" * 100)

print()

print(
    results_dataframe.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# PRINT ROBUSTNESS SUMMARY
# ============================================================

print()
print("=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)

print()

for _, row in results_dataframe.iterrows():

    condition = row["Condition"]
    f1 = row["Macro F1"]
    change = row["Macro F1 Change"]

    print(
        f"{condition:25s} "
        f"Macro F1: {f1:.2f}% "
        f"Change: {change:+.2f}"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("Evaluation complete.")
print("=" * 70)

print()

print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)