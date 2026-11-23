
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes):
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT
    )

    # Replace final classification layer
    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    return model


def main():

    parser = argparse.ArgumentParser(
        description="Train the Smart Waste Classifier"
    )

    parser.add_argument("--data_dir", default="data/garbage")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--val_split", type=float, default=0.2)

    args = parser.parse_args()

    set_seed()

    # Select GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========== DEVICE INFORMATION ==========")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
    else:
        print("GPU not detected. Using CPU.")

    print("========================================\n")

    # Dataset directory
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir.resolve()}"
        )

    # Training image transformations
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    # Validation transformations
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    # Load datasets
    train_base = datasets.ImageFolder(
        data_dir,
        transform=train_transform
    )

    val_base = datasets.ImageFolder(
        data_dir,
        transform=val_transform
    )

    if len(train_base.classes) < 2:
        raise ValueError(
            "At least two class folders are required."
        )

    if len(train_base) < 10:
        raise ValueError(
            "Very few images found in dataset."
        )

    # Split dataset into training and validation
    num_samples = len(train_base)

    val_size = max(
        1,
        int(num_samples * args.val_split)
    )

    train_size = num_samples - val_size

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        num_samples,
        generator=generator
    ).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = Subset(
        train_base,
        train_indices
    )

    val_data = Subset(
        val_base,
        val_indices
    )

    print(f"Classes ({len(train_base.classes)}): {train_base.classes}")
    print(f"Total Images: {num_samples}")
    print(f"Training Images: {train_size}")
    print(f"Validation Images: {val_size}")

    # Data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Build model and move it to GPU/CPU
    model = build_model(
        len(train_base.classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val_accuracy = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    # Training loop
    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            # Move images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Validation
        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                # Move validation data to GPU
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_running_loss += (
                    loss.item() * images.size(0)
                )

                val_correct += (
                    outputs.argmax(1) == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total

        # Store training history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"\nEpoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2%}"
        )

        # Save best model
        if val_accuracy >= best_val_accuracy:

            best_val_accuracy = val_accuracy

            Path("models").mkdir(exist_ok=True)

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_base.classes,
                "image_size": args.image_size,
                "val_accuracy": best_val_accuracy,
            }, "models/waste_classifier.pth")

    # Save training history
    Path("outputs").mkdir(exist_ok=True)

    with open(
        "outputs/training_history.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(history, f, indent=2)

    # Accuracy graph
    epochs_range = range(1, args.epochs + 1)

    plt.figure()

    plt.plot(
        epochs_range,
        history["train_accuracy"],
        label="Training accuracy"
    )

    plt.plot(
        epochs_range,
        history["val_accuracy"],
        label="Validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig("outputs/accuracy.png")
    plt.close()

    # Loss graph
    plt.figure()

    plt.plot(
        epochs_range,
        history["train_loss"],
        label="Training loss"
    )

    plt.plot(
        epochs_range,
        history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig("outputs/loss.png")
    plt.close()

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2%}")
    print("Saved Model: models/waste_classifier.pth")
    print("Saved Graphs:")
    print(" - outputs/accuracy.png")
    print(" - outputs/loss.png")
    print("========================================")


if __name__ == "__main__":
    main()
