#!/usr/bin/env python
"""
Simple test script for UNet3D nodule detection.
"""

import argparse
import logging
from pathlib import Path
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch.utils.data import DataLoader

from train import UNet3D, NoduleDataset, NoduleTransforms, calculate_nodule_detection_accuracy

def visualize_nodule(volume, gt_coords, pred_coords, output_dir, uid, idx):
    # """
    # run the script with the following command:
    # python "src\Nodules detection\test_nodules.py" `       
    # >>   -m "src\Nodules detection\models\best_model.pth" `
    # >>   -d "data\Luna16" `
    # >>   -o "src\Nodules detection\visualizations" `
    # >>   -n 5
    # Create visualization of a nodule with axial, coronal, and sagittal views.
    # Args:
    #     volume: 3D numpy array (D, H, W) containing the CT patch
    #     gt_coords: Ground truth coordinates [z, y, x, r] normalized
    #     pred_coords: Predicted coordinates [z, y, x, r] normalized
    #     output_dir: Directory to save visualizations
    #     uid: Series UID
    #     idx: Patch index
    # """
    
    # Remove channel dimension if present
    if len(volume.shape) == 4:
        volume = volume[0]
    
    # Get dimensions
    D, H, W = volume.shape
    
    # Convert normalized coordinates to voxel coordinates
    gt_z, gt_y, gt_x, gt_r = gt_coords
    gt_z, gt_y, gt_x = int(gt_z * D), int(gt_y * H), int(gt_x * W)
    gt_r = int(gt_r * min(D, H, W))
    
    pred_z, pred_y, pred_x, _ = pred_coords
    pred_z, pred_y, pred_x = int(pred_z * D), int(pred_y * H), int(pred_x * W)
    
    # Ensure valid coordinates
    gt_z = min(max(gt_z, 0), D-1)
    gt_y = min(max(gt_y, 0), H-1)
    gt_x = min(max(gt_x, 0), W-1)
    pred_z = min(max(pred_z, 0), D-1)
    pred_y = min(max(pred_y, 0), H-1)
    pred_x = min(max(pred_x, 0), W-1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial view (Z plane)
    axes[0].imshow(volume[gt_z], cmap='gray')
    axes[0].add_patch(Circle((gt_x, gt_y), gt_r, fill=False, color='red', linestyle='-', linewidth=2, label='GT'))
    axes[0].plot(pred_x, pred_y, 'g+', markersize=10, markeredgewidth=2, label='Pred')
    axes[0].set_title(f'Axial (Z={gt_z})')
    axes[0].axis('off')
    
    # Coronal view (Y plane)
    axes[1].imshow(volume[:, gt_y, :].T, cmap='gray')
    axes[1].add_patch(Circle((gt_x, gt_z), gt_r, fill=False, color='red', linestyle='-', linewidth=2))
    axes[1].plot(pred_x, pred_z, 'g+', markersize=10, markeredgewidth=2)
    axes[1].set_title(f'Coronal (Y={gt_y})')
    axes[1].axis('off')
    
    # Sagittal view (X plane)
    axes[2].imshow(volume[:, :, gt_x].T, cmap='gray')
    axes[2].add_patch(Circle((gt_y, gt_z), gt_r, fill=False, color='red', linestyle='-', linewidth=2))
    axes[2].plot(pred_y, pred_z, 'g+', markersize=10, markeredgewidth=2)
    axes[2].set_title(f'Sagittal (X={gt_x})')
    axes[2].axis('off')
    
    # Add legend to the first subplot
    axes[0].legend(loc='upper left')
    
    # Add series info as a subtitle
    plt.suptitle(f'Series: {uid}, Patch: {idx}', fontsize=14)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{uid.split(".")[-1]}_{idx}_visualization.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # Also save the numpy array
    np_output_path = os.path.join(output_dir, f'{uid.split(".")[-1]}_{idx}_volume.npy')
    np.save(np_output_path, volume)
    
    return output_path

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test nodule detection model")
    parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", "-d", type=str, required=True, help="Path to data directory")
    parser.add_argument("--num-ct", "-n", type=int, default=5, help="Number of CT scans to test")
    # Add output directory parameter
    parser.add_argument("--output-dir", "-o", type=str, default="nodule_visualizations", 
                        help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("test")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving visualizations to {output_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = NoduleDataset(
        data_dir=args.data_dir,
        annotations_file=Path(__file__).parent / "processed_annotations.csv",
        patch_size=(64, 64, 64),
        transform=NoduleTransforms(p=0.0)  # No augmentation for testing
    )
    
    # Load model
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Find unique CT series
    unique_series = list(dict.fromkeys([s for s, _ in dataset.samples]))
    logger.info(f"Found {len(unique_series)} unique CT series")
    
    # Test on limited number of CTs
    test_series = unique_series[:args.num_ct]
    logger.info(f"Testing on {len(test_series)} series: {test_series}")
    
    # Track results
    correct = 0
    total = 0
    
    # Process each series
    for series_uid in test_series:
        # Find all patches for this series
        patch_indices = [i for i, (s, _) in enumerate(dataset.samples) if s == series_uid]
        logger.info(f"Series {series_uid} has {len(patch_indices)} nodule patches")
        
        # Process each patch
        for idx in patch_indices:
            # Get data
            volume, mask, det_target = dataset[idx]
            
            # Move to device and add batch dimension
            volume_gpu = volume.unsqueeze(0).to(device)
            mask_gpu = mask.unsqueeze(0).to(device)
            
            # Forward pass
            with torch.no_grad():
                _, det_pred = model(volume_gpu)
            
            # Check if detection is correct (>50% hit rate)
            accuracy = calculate_nodule_detection_accuracy(
                det_pred, mask_gpu, volume_shape=volume_gpu.shape[2:]
            )
            is_hit = accuracy > 50.0
            
            # Create and save visualization
            vis_path = visualize_nodule(
                volume.cpu().numpy(), 
                det_target.cpu().numpy(), 
                det_pred.squeeze(0).cpu().numpy(),
                output_dir,
                series_uid, 
                idx
            )
            
            # Log prediction details
            logger.info(
                f"Series {series_uid}, Patch {idx}: "
                f"GT={np.round(det_target.numpy(), 3).tolist()}, "
                f"Pred={np.round(det_pred.squeeze(0).cpu().numpy(), 3).tolist()}, "
                f"Hit={is_hit} ({accuracy:.1f}%), "
                f"Visualization: {vis_path}"
            )
            
            # Update counters
            correct += int(is_hit)
            total += 1
    
    # Final results
    if total > 0:
        logger.info(f"Overall Detection Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
    else:
        logger.info("No patches tested")

if __name__ == "__main__":
    main()