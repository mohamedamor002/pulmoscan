import os
import torch
from unet3d import UNet3D

# Get script directory and use absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'best_model.pt')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'metrics')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Looking for model at: {MODEL_PATH}")

# Get model file size without loading it
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Model size: {model_size_mb:.2f} MB")

# Create model with attention=False to match original parameters
# Older model likely had smaller dimensions
model = UNet3D(n_channels=1, n_classes=1, use_attention=False)

# Count parameters without loading weights
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), 'w') as f:
    f.write(f"Model size: {model_size_mb:.2f} MB\n")
    f.write(f"Number of parameters: {num_params}\n")

print(f"Metrics saved to: {os.path.join(OUTPUT_DIR, 'metrics.txt')}")