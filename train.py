import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config
from model import RockPaperScissorsCNN
from dataset_loader import get_data_loader
def compute_class_weights(train_dir):
    """
    Counts samples per class in the training folder and returns inverse-frequency
    weights as a tensor.  This ensures the loss function penalises the model more
    for getting the minority classes wrong, preventing the "always predicts rock"
    bias that comes from an imbalanced dataset.
    """
    # Use a plain ImageFolder with no transforms just to read labels
    raw_dataset = datasets.ImageFolder(root=train_dir)
    # Verify that PyTorch's detected class order matches config.CLASS_NAMES
    detected = raw_dataset.classes
    if detected != config.CLASS_NAMES:
        print("─" * 55)
        print("  ⚠  CLASS NAME MISMATCH!")
        print(f"  PyTorch detected: {detected}")
        print(f"  config.CLASS_NAMES: {config.CLASS_NAMES}")
        print("  Please update CLASS_NAMES in config.py to match the list above,")
        print("  then re-run training.")
        print("─" * 55)
    # Count samples per class index
    counts = [0] * config.NUM_CLASSES
    for _, label in raw_dataset:
        counts[label] += 1
    print(f"  Samples per class: { {detected[i]: counts[i] for i in range(len(detected))} }")
    # Inverse-frequency weights: rare classes get a higher weight
    total = sum(counts)
    weights = torch.tensor(
        [total / (config.NUM_CLASSES * c) for c in counts],
        dtype=torch.float
    )
    return weights
def train_model():
    """
    Sets up dataset loaders, instantiates the custom CNN, applies class-weighted
    loss to handle imbalanced datasets, runs the training loop with per-epoch
    validation metrics, and saves the trained weights.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # ── 1. Data Loaders ──────────────────────────────────────
    try:
        train_loader = get_data_loader(train=True)
        print(f"Training starting...  Classes: {config.CLASS_NAMES}")
        print(f"Device: {config.DEVICE}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return
    val_loader = None
    try:
        val_loader = get_data_loader(train=False)
        print("Validation loader ready.")
    except Exception:
        print("Info: No validation folder found – training only.")
    # ── 2. Model ─────────────────────────────────────────────
    model = RockPaperScissorsCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    print("\nComputing class weights to fix any imbalance...")
    class_weights = compute_class_weights(config.TRAIN_DIR).to(config.DEVICE)
    print(f"  Class weights: { {config.CLASS_NAMES[i]: f'{class_weights[i].item():.3f}' for i in range(config.NUM_CLASSES)} }")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # ── 4. Optimizer ─────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Optional: reduce LR automatically when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    print(f"\nTraining for {config.EPOCHS} epochs...\n")
    best_val_accuracy = 0.0
    # ── 5. Training Loop ─────────────────────────────────────
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_train   += labels.size(0)
            correct_train += (preds == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc  = 100.0 * correct_train / total_train
        print(f"Epoch {epoch+1:>2}/{config.EPOCHS}  |  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.1f}%", end="")
        # ── 6. Validation ─────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val   = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds  = torch.max(outputs, 1)
                    total_val   += labels.size(0)
                    correct_val += (preds == labels).sum().item()
            val_epoch_loss = val_loss / len(val_loader)
            val_acc = 100.0 * correct_val / total_val
            print(f"  |  Val Loss: {val_epoch_loss:.4f}  Val Acc: {val_acc:.1f}%", end="")
            # Step the LR scheduler based on validation loss
            scheduler.step(val_epoch_loss)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), config.MODEL_PATH)
                torch.save(model.state_dict(), f"{config.MODEL_PATH}.pth")
                print("  ✓ Best model saved!", end="")
        print()  # Newline after each epoch summary
    # ── 7. Final Save ─────────────────────────────────────────
    # If no validation set existed we save here instead
    if val_loader is None:
        torch.save(model.state_dict(), config.MODEL_PATH)
        torch.save(model.state_dict(), f"{config.MODEL_PATH}.pth")
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.1f}%")
    print(f"Weights saved to: {os.path.abspath(config.MODEL_PATH)}")
if __name__ == "__main__":
    train_model()
