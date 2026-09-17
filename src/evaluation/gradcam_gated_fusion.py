from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.data.dataset import (
    HAM10000Dataset,
    val_test_transform,
    CLASS_NAMES
)

from src.models.gated_fusion_model import GatedFusionModel


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(r"C:\multimodal_skin_diagnosis")

TEST_CSV = ROOT / "data" / "processed" / "test.csv"

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "gated_fusion_modality_dropout_best.pth"
)

OUTPUT_DIR = ROOT / "results" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 1

# Number of examples to save for each class
EXAMPLES_PER_CLASS = 2


# ImageNet normalization used during training
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None

        # Register forward hook
        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, input, output):

        self.activations = output

        # We need gradients of this activation
        self.activations.retain_grad()

    def generate(self, image, metadata, class_index):

        self.model.zero_grad()

        # Forward pass
        output = self.model(
            image,
            metadata
        )

        # Select the class we want to explain
        score = output[0, class_index]

        # Backpropagate
        score.backward()

        # Get activation gradients
        gradients = self.activations.grad

        # Remove batch dimension
        activations = self.activations[0]
        gradients = gradients[0]

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(1, 2)
        )

        # Weighted combination of feature maps
        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )

        for i in range(activations.shape[0]):

            cam += (
                weights[i]
                * activations[i]
            )

        # ReLU
        cam = torch.relu(cam)

        # Convert to numpy
        cam = cam.detach().cpu().numpy()

        # Normalize between 0 and 1
        if cam.max() > cam.min():

            cam = (
                cam - cam.min()
            ) / (
                cam.max() - cam.min()
            )

        else:

            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):

        self.forward_hook.remove()


# ============================================================
# IMAGE PROCESSING
# ============================================================

def denormalize(image_tensor):

    image = image_tensor.cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    image = image * STD + MEAN

    image = np.clip(
        image,
        0,
        1
    )

    return image


def resize_heatmap(
    heatmap,
    height,
    width
):

    heatmap_tensor = torch.tensor(
        heatmap,
        dtype=torch.float32
    )

    heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)

    heatmap_tensor = torch.nn.functional.interpolate(
        heatmap_tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    return heatmap_tensor[0, 0].numpy()


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_gradcam_visualization(
    image,
    heatmap,
    true_class,
    predicted_class,
    confidence,
    output_path
):

    height, width, _ = image.shape

    heatmap = resize_heatmap(
        heatmap,
        height,
        width
    )

    plt.figure(
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    plt.subplot(1, 3, 1)

    plt.imshow(image)

    plt.title(
        f"Original\nTrue: {true_class}"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    plt.subplot(1, 3, 2)

    plt.imshow(
        heatmap,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    plt.subplot(1, 3, 3)

    plt.imshow(image)

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GRAD-CAM EXPLAINABILITY")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = HAM10000Dataset(
        csv_file=TEST_CSV,
        transform=val_test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GatedFusionModel(
        num_classes=7
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(DEVICE)

    model.eval()

    print("\nModel loaded successfully.")

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = model.image_model.features[-1]

    print(
        "\nGrad-CAM target layer:",
        target_layer
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    # --------------------------------------------------------
    # Keep track of examples
    # --------------------------------------------------------

    class_counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    records = []

    # --------------------------------------------------------
    # Process test images
    # --------------------------------------------------------

    for index, (
        image,
        metadata,
        label
    ) in enumerate(test_loader):

        image = image.to(DEVICE)
        metadata = metadata.to(DEVICE)
        label = label.to(DEVICE)

        # Need gradients for Grad-CAM
        image.requires_grad = False

        # Forward pass
        model.zero_grad()

        output = model(
            image,
            metadata
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        true_index = label.item()
        predicted_index = prediction.item()

        true_class = CLASS_NAMES[true_index]
        predicted_class = CLASS_NAMES[predicted_index]

        confidence_value = confidence.item()

        # ----------------------------------------------------
        # Only save correctly classified examples
        # ----------------------------------------------------

        if predicted_index != true_index:
            continue

        if class_counts[true_class] >= EXAMPLES_PER_CLASS:
            continue

        print(
            f"\nGenerating Grad-CAM:"
            f"\n  Image: {index}"
            f"\n  True: {true_class}"
            f"\n  Prediction: {predicted_class}"
            f"\n  Confidence: {confidence_value:.2%}"
        )

        # ----------------------------------------------------
        # Generate Grad-CAM
        # ----------------------------------------------------

        heatmap = gradcam.generate(
            image,
            metadata,
            predicted_index
        )

        # ----------------------------------------------------
        # Convert original image
        # ----------------------------------------------------

        original_image = denormalize(
            image[0]
        )

        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        class_number = (
            class_counts[true_class] + 1
        )

        filename = (
            f"{true_class}_"
            f"{class_number}_"
            f"conf_{confidence_value:.2f}.png"
        )

        output_path = (
            OUTPUT_DIR / filename
        )

        save_gradcam_visualization(
            original_image,
            heatmap,
            true_class,
            predicted_class,
            confidence_value,
            output_path
        )

        class_counts[true_class] += 1

        records.append({
            "test_index": index,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence_value,
            "output_file": str(output_path)
        })

        # ----------------------------------------------------
        # Check whether we have enough examples
        # ----------------------------------------------------

        finished = all(
            count >= EXAMPLES_PER_CLASS
            for count in class_counts.values()
        )

        if finished:
            break

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    results = pd.DataFrame(records)

    csv_path = (
        OUTPUT_DIR
        / "gradcam_results.csv"
    )

    results.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRAD-CAM COMPLETE")
    print("=" * 60)

    print("\nExamples generated:")

    for class_name in CLASS_NAMES:

        print(
            f"{class_name}: "
            f"{class_counts[class_name]}"
        )

    print("\nImages saved to:")

    print(OUTPUT_DIR)

    print("\nResults CSV:")

    print(csv_path)


if __name__ == "__main__":
    main()