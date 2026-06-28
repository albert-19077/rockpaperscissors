import os
import torch
# ─────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────
# Automatically use GPU (CUDA) if available, otherwise fall back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)   # Input resolution expected by the CNN
BATCH_SIZE    = 32           # Number of images per mini-batch
LEARNING_RATE = 0.0003       # Lowered from 0.001 — prevents collapsing to one class
EPOCHS        = 25           # Increased for better convergence with the improved model
NUM_CLASSES  = 3
CLASS_NAMES  = ["paper", "rock", "scissors"]   # ← Update if diagnose.py shows a different order
# ─────────────────────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
# Support both "validation" (our naming) and "valid" (Roboflow default)
_val_candidates = [
    os.path.join(DATASET_DIR, "validation"),
    os.path.join(DATASET_DIR, "valid"),
]
VAL_DIR = next((p for p in _val_candidates if os.path.exists(p)), _val_candidates[0])
# ─────────────────────────────────────────────────────────────
# OUTPUT / SAVE PATHS
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR  = "weights"
MODEL_NAME  = "RockPaperScissors_CNN"
MODEL_PATH  = os.path.join(OUTPUT_DIR, MODEL_NAME)
ONNX_MODEL_NAME = "modelimiz.onnx"
CONFIDENCE_THRESHOLD = 0.60
