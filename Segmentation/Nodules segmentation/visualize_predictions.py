import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm

from unet3d import UNet3D
from train_nodule_segmentation import NoduleDataset

def load_model(model_path, device):
    """Load a trained model from checkpoint"""
    model = UNet3D(n_channels=1, n_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}, trained for {checkpoint['epoch']} epochs")
    print(f"Validation Dice: {checkpoint.get('val_dice', 'N/A')}")
    return model

def visualize_sample(model, dataset, idx, device, save_dir=None):
    """Visualize model predictions for a sample"""
    # Get input and ground truth
    volume, mask = dataset[idx]
    
    # Add batch dimension and move to device
    volume_batch = volume.unsqueeze(0).to(device)
    
    # Generate prediction
    with torch.no_grad():
        prediction = model(volume_batch)
        prediction = torch.sigmoid(prediction)
        prediction_binary = (prediction > 0.5).float()
    
    # Move to CPU and remove batch dimension
    prediction = prediction.cpu().squeeze(0)
    prediction_binary = prediction_binary.cpu().squeeze(0)
    
    # Convert to numpy for visualization
    volume_np = volume.squeeze(0).numpy()
    mask_np = mask.squeeze(0).numpy()
    prediction_np = prediction.squeeze(0).numpy()
    prediction_binary_np = prediction_binary.squeeze(0).numpy()
    
    # Create 3 views: axial, coronal, sagittal
    views = ['axial', 'coronal', 'sagittal']
    slices = []
    
    # Find the center slice for each dimension where the mask has the most positive pixels
    z_sum = np.sum(mask_np, axis=(1, 2))
    y_sum = np.sum(mask_np, axis=(0, 2))
    x_sum = np.sum(mask_np, axis=(0, 1))
    
    # If no positive pixels, use the center slice
    z_center = np.argmax(z_sum) if np.max(z_sum) > 0 else mask_np.shape[0] // 2
    y_center = np.argmax(y_sum) if np.max(y_sum) > 0 else mask_np.shape[1] // 2
    x_center = np.argmax(x_sum) if np.max(x_sum) > 0 else mask_np.shape[2] // 2
    
    slices = [z_center, y_center, x_center]
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Set title
    fig.suptitle(f"Sample {idx}", fontsize=16)
    
    for i, (view, center_slice) in enumerate(zip(views, slices)):
        # Get the slices
        if view == 'axial':
            volume_slice = volume_np[center_slice, :, :]
            mask_slice = mask_np[center_slice, :, :]
            prediction_slice = prediction_np[center_slice, :, :]
            prediction_binary_slice = prediction_binary_np[center_slice, :, :]
        elif view == 'coronal':
            volume_slice = volume_np[:, center_slice, :]
            mask_slice = mask_np[:, center_slice, :]
            prediction_slice = prediction_np[:, center_slice, :]
            prediction_binary_slice = prediction_binary_np[:, center_slice, :]
        else:  # sagittal
            volume_slice = volume_np[:, :, center_slice]
            mask_slice = mask_np[:, :, center_slice]
            prediction_slice = prediction_np[:, :, center_slice]
            prediction_binary_slice = prediction_binary_np[:, :, center_slice]
        
        # Display CT
        axes[i, 0].imshow(volume_slice, cmap='gray')
        axes[i, 0].set_title(f"{view.capitalize()} CT")
        axes[i, 0].axis('off')
        
        # Display ground truth
        axes[i, 1].imshow(volume_slice, cmap='gray')
        mask_overlay = np.ma.masked_where(mask_slice < 0.5, mask_slice)
        axes[i, 1].imshow(mask_overlay, cmap='hot', alpha=0.5)
        axes[i, 1].set_title(f"{view.capitalize()} Ground Truth")
        axes[i, 1].axis('off')
        
        # Display probability prediction
        axes[i, 2].imshow(volume_slice, cmap='gray')
        pred_overlay = prediction_slice
        axes[i, 2].imshow(pred_overlay, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        axes[i, 2].set_title(f"{view.capitalize()} Prediction (Prob)")
        axes[i, 2].axis('off')
        
        # Display binary prediction
        axes[i, 3].imshow(volume_slice, cmap='gray')
        pred_bin_overlay = np.ma.masked_where(prediction_binary_slice < 0.5, prediction_binary_slice)
        axes[i, 3].imshow(pred_bin_overlay, cmap='hot', alpha=0.5)
        axes[i, 3].set_title(f"{view.capitalize()} Prediction (Binary)")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Save if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = Path(save_dir) / f"prediction_sample_{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize Nodule Segmentation Predictions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/Luna16', help='Path to Luna16 dataset')
    parser.add_argument('--annotations', type=str, default='src/Nodules segmentation/processed_annotations.csv', 
                        help='Path to processed annotations CSV file')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='src/Nodules segmentation/Visualizations/Predictions', 
                        help='Directory to save visualizations')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64], 
                        help='Patch size (z, y, x)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create dataset
    dataset = NoduleDataset(
        data_dir=args.data_dir,
        annotations_file=args.annotations,
        patch_size=tuple(args.patch_size),
        transform=None  # No transforms for inference
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize samples
    for i in range(min(args.num_samples, len(dataset))):
        visualize_sample(model, dataset, i, device, args.save_dir)

if __name__ == "__main__":
    main() 