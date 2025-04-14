import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import argparse
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib

from unet3d import UNet3D
from dataset import LungDataset

def dice_score(y_true, y_pred):
    """Calculate Dice score for binary segmentation"""
    smooth = 1e-5
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def iou_score(y_true, y_pred):
    """Calculate IoU (Jaccard index) for binary segmentation"""
    smooth = 1e-5
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def sensitivity(y_true, y_pred):
    """Calculate sensitivity (recall) for binary segmentation"""
    smooth = 1e-5
    true_positives = np.sum(y_true * y_pred)
    return (true_positives + smooth) / (np.sum(y_true) + smooth)

def specificity(y_true, y_pred):
    """Calculate specificity for binary segmentation"""
    smooth = 1e-5
    true_negatives = np.sum((1 - y_true) * (1 - y_pred))
    return (true_negatives + smooth) / (np.sum(1 - y_true) + smooth)

def save_slice_comparison(ct_slice, true_mask_slice, pred_mask_slice, save_path, idx):
    """Save a visualization of the segmentation results for a single slice"""
    plt.figure(figsize=(15, 5))
    
    # CT scan
    plt.subplot(1, 3, 1)
    plt.imshow(ct_slice, cmap='gray')
    plt.title('CT Scan')
    plt.axis('off')
    
    # Ground truth
    plt.subplot(1, 3, 2)
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(true_mask_slice, alpha=0.5, cmap='Reds')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(pred_mask_slice, alpha=0.5, cmap='Blues')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path / f'comparison_{idx}.png')
    plt.close()

def save_3d_volume(volume, save_path, filename):
    """Save a 3D volume as a NIfTI file"""
    nifti_img = nib.Nifti1Image(volume, np.eye(4))
    nib.save(nifti_img, save_path / filename)

def validate_model(model, loader, device, save_dir=None, save_volumes=False, max_samples_to_visualize=5):
    """Validate model on the dataset"""
    model.eval()
    
    metrics = {
        'dice': [],
        'iou': [],
        'sensitivity': [],
        'specificity': []
    }
    
    # Create directory for visualizations if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        vis_dir = save_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        if save_volumes:
            vol_dir = save_dir / 'volumes'
            vol_dir.mkdir(exist_ok=True)
    
    samples_visualized = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Validation")):
            # Get data
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            series_id = batch['series_id']
            
            # Forward pass
            pred = model(ct)
            
            # Convert to binary predictions
            pred_binary = (pred > 0.5).float()
            
            # Calculate metrics for each sample in batch
            for j in range(ct.shape[0]):
                y_true = mask[j, 0].cpu().numpy()
                y_pred = pred_binary[j, 0].cpu().numpy()
                
                # Calculate and store metrics
                metrics['dice'].append(dice_score(y_true, y_pred))
                metrics['iou'].append(iou_score(y_true, y_pred))
                metrics['sensitivity'].append(sensitivity(y_true, y_pred))
                metrics['specificity'].append(specificity(y_true, y_pred))
                
                # Save visualizations for a limited number of samples
                if save_dir and samples_visualized < max_samples_to_visualize:
                    # Get a middle slice for visualization (along each axis)
                    z_mid = y_true.shape[0] // 2
                    y_mid = y_true.shape[1] // 2
                    x_mid = y_true.shape[2] // 2
                    
                    # Save axial slice (top view)
                    save_slice_comparison(
                        ct[j, 0, z_mid].cpu().numpy(),
                        y_true[z_mid],
                        y_pred[z_mid],
                        vis_dir, 
                        f"{series_id[j]}_axial"
                    )
                    
                    # Save coronal slice (front view)
                    save_slice_comparison(
                        ct[j, 0, :, y_mid, :].cpu().numpy(),
                        y_true[:, y_mid, :],
                        y_pred[:, y_mid, :],
                        vis_dir,
                        f"{series_id[j]}_coronal"
                    )
                    
                    # Save sagittal slice (side view)
                    save_slice_comparison(
                        ct[j, 0, :, :, x_mid].cpu().numpy(),
                        y_true[:, :, x_mid],
                        y_pred[:, :, x_mid],
                        vis_dir,
                        f"{series_id[j]}_sagittal"
                    )
                    
                    # Save full volumes if requested
                    if save_volumes:
                        save_3d_volume(ct[j, 0].cpu().numpy(), vol_dir, f"{series_id[j]}_ct.nii.gz")
                        save_3d_volume(y_true, vol_dir, f"{series_id[j]}_true_mask.nii.gz")
                        save_3d_volume(y_pred, vol_dir, f"{series_id[j]}_pred_mask.nii.gz")
                    
                    samples_visualized += 1
    
    # Calculate and return average metrics
    results = {}
    for metric, values in metrics.items():
        if values:  # Check if list is not empty
            results[metric] = np.mean(values)
            results[f"{metric}_std"] = np.std(values)
        else:
            results[metric] = float('nan')
            results[f"{metric}_std"] = float('nan')
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Validate 3D UNet for lung segmentation')
    parser.add_argument('--data_dir', type=str, default='data/Luna16',
                      help='Path to LUNA16 dataset')
    parser.add_argument('--model_path', type=str, default='unet3d_models/best_model.pt',
                      help='Path to saved model')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for validation')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='validation_results',
                      help='Directory to save validation results')
    parser.add_argument('--save_volumes', action='store_true',
                      help='Save full 3D volumes as NIfTI files')
    parser.add_argument('--num_vis', type=int, default=5,
                      help='Number of samples to visualize')
    parser.add_argument('--val_limit', type=int, default=10,
                      help='Maximum number of samples to validate (helpful for large datasets)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert data_dir to Path
    data_dir = Path(args.data_dir)
    print(f"Loading dataset from: {data_dir.absolute()}")
    
    # Debug: Check if directories exist
    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir.absolute()} does not exist!")
        return
        
    # Check for subsets
    found_subsets = []
    for i in range(10):
        subset_dir = data_dir / f"subset{i}"
        if subset_dir.exists():
            found_subsets.append(i)
    
    print(f"Found subsets: {found_subsets}")
    
    # Check for mask directory
    mask_dir = data_dir
    mask_subdir = mask_dir / "seg-lungs-LUNA16" / "seg-lungs-LUNA16"
    if mask_subdir.exists():
        print(f"Found mask directory: {mask_subdir}")
    else:
        print(f"ERROR: Mask directory {mask_subdir} not found!")
    
    # Create dataset with more explicit error handling
    try:
        full_dataset = LungDataset(str(data_dir), str(data_dir))
        dataset_size = len(full_dataset)
        print(f"Successfully loaded dataset with {dataset_size} samples")
    
        if dataset_size == 0:
            print("ERROR: Dataset is empty (len=0). Cannot proceed with validation.")
            return
            
        # Limit validation set size if requested
        val_size = min(dataset_size, args.val_limit) if args.val_limit > 0 else dataset_size
        if val_size < dataset_size:
            print(f"Using {val_size} of {dataset_size} samples for validation")
            indices = list(range(val_size))
            val_dataset = Subset(full_dataset, indices)
        else:
            val_dataset = full_dataset
            
        # Create data loader
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Load model
        model = UNet3D(n_channels=1, n_classes=1).to(device)
        
        print(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with Dice score: {checkpoint['dice_score']:.4f}")
        
        # Validate model
        results = validate_model(
            model, 
            val_loader, 
            device, 
            save_dir=args.save_dir, 
            save_volumes=args.save_volumes,
            max_samples_to_visualize=args.num_vis
        )
        
        # Print results
        print("\nValidation Results:")
        print(f"Dice Score: {results['dice']:.4f} ± {results['dice_std']:.4f}")
        print(f"IoU Score: {results['iou']:.4f} ± {results['iou_std']:.4f}")
        print(f"Sensitivity: {results['sensitivity']:.4f} ± {results['sensitivity_std']:.4f}")
        print(f"Specificity: {results['specificity']:.4f} ± {results['specificity_std']:.4f}")
        
        # Save metrics to CSV
        with open(save_dir / 'metrics.csv', 'w') as f:
            f.write('Metric,Value,StdDev\n')
            for metric in ['dice', 'iou', 'sensitivity', 'specificity']:
                if metric in results and not np.isnan(results[metric]):
                    f.write(f"{metric},{results[metric]:.6f},{results[f'{metric}_std']:.6f}\n")
                    
        print(f"\nResults saved to {save_dir}")

    except Exception as e:
        print(f"ERROR loading or processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPossible solutions:")
        print("1. Check if the dataset directory structure matches the expected structure")
        print("2. Check if you have read permissions for the dataset files")
        print("3. Inspect the LungDataset class for assumptions about file extensions or naming")

if __name__ == '__main__':
    main()