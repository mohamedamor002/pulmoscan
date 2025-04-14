import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import argparse
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import nibabel as nib
import io
from PIL import Image

from unet3d import UNet3D
from dataset import LungDataset

def dice_score(y_true, y_pred):
    """Calculate Dice score for binary segmentation"""
    smooth = 1e-5
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def create_3d_visualization(ct_volume, true_mask, pred_mask, save_path, series_id, downsample=2):
    """
    Create a 3D visualization of the predicted mask and ground truth
    with the ct volume as a context
    
    Args:
        ct_volume: 3D numpy array of CT scan
        true_mask: 3D numpy array of ground truth mask
        pred_mask: 3D numpy array of predicted mask
        save_path: directory to save visualization
        series_id: ID of the series for naming
        downsample: factor to downsample the volume for faster rendering
    """
    # Create directory if it doesn't exist
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Downsample volumes for more efficient processing
    ct_ds = ct_volume[::downsample, ::downsample, ::downsample]
    true_ds = true_mask[::downsample, ::downsample, ::downsample]
    pred_ds = pred_mask[::downsample, ::downsample, ::downsample]
    
    # Create 3D figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    # Extract surface mesh from mask using marching cubes algorithm
    verts, faces, _, _ = measure.marching_cubes(true_ds, level=0.5)
    
    # Plot surface
    ax1.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                   triangles=faces, color='red', alpha=0.5)
    ax1.set_title('Ground Truth Mask')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Scale axes equally and set reasonable limits
    max_range = np.array([verts[:, 0].max()-verts[:, 0].min(),
                        verts[:, 1].max()-verts[:, 1].min(),
                        verts[:, 2].max()-verts[:, 2].min()]).max() / 2.0
    mid_x = (verts[:, 0].max()+verts[:, 0].min()) * 0.5
    mid_y = (verts[:, 1].max()+verts[:, 1].min()) * 0.5
    mid_z = (verts[:, 2].max()+verts[:, 2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Plot prediction
    ax2 = fig.add_subplot(122, projection='3d')
    # Extract surface mesh from mask using marching cubes
    verts, faces, _, _ = measure.marching_cubes(pred_ds, level=0.5)
    
    # Plot surface
    ax2.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                   triangles=faces, color='blue', alpha=0.5)
    ax2.set_title('Predicted Mask')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Scale axes to match first plot
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewpoint
    for ax in [ax1, ax2]:
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path / f'3d_visualization_{series_id}.png', dpi=200)
    plt.close(fig)
    
    return save_path / f'3d_visualization_{series_id}.png'

def create_3d_animation_gif(ct_volume, true_mask, pred_mask, save_path, series_id, angles=None, duration=100, downsample=3):
    """
    Create a GIF animation directly showing the 3D model rotating
    
    Args:
        ct_volume: 3D numpy array of CT scan
        true_mask: 3D numpy array of ground truth mask
        pred_mask: 3D numpy array of predicted mask
        save_path: directory to save visualization
        series_id: ID of the series for naming
        angles: list of azimuthal angles to render
        duration: duration of each frame in milliseconds
        downsample: factor to downsample the volume for faster rendering
    """
    # Create directory if it doesn't exist
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Output GIF file path
    gif_path = save_path / f'animation_{series_id}.gif'
    
    # Downsample for faster rendering
    ct_ds = ct_volume[::downsample, ::downsample, ::downsample]
    true_ds = true_mask[::downsample, ::downsample, ::downsample]
    pred_ds = pred_mask[::downsample, ::downsample, ::downsample]
    
    # Default angles if none provided
    if angles is None:
        angles = range(0, 360, 10)  # 36 frames for a full rotation
    
    print(f"Creating 3D animation GIF for {series_id} with {len(angles)} frames...")
    
    # Get surface meshes once for efficiency
    true_verts, true_faces, _, _ = measure.marching_cubes(true_ds, level=0.5)
    pred_verts, pred_faces, _, _ = measure.marching_cubes(pred_ds, level=0.5)
    
    # Determine common axis scales
    all_verts = np.vstack([true_verts, pred_verts])
    max_range = np.array([all_verts[:, 0].max()-all_verts[:, 0].min(),
                        all_verts[:, 1].max()-all_verts[:, 1].min(),
                        all_verts[:, 2].max()-all_verts[:, 2].min()]).max() / 2.0
    mid_x = (all_verts[:, 0].max()+all_verts[:, 0].min()) * 0.5
    mid_y = (all_verts[:, 1].max()+all_verts[:, 1].min()) * 0.5
    mid_z = (all_verts[:, 2].max()+all_verts[:, 2].min()) * 0.5
    
    # Create frames for the GIF
    frames = []
    
    for angle in tqdm(angles, desc=f"Rendering frames for {series_id}"):
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        
        # Ground truth
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_trisurf(true_verts[:, 0], true_verts[:, 1], true_verts[:, 2],
                       triangles=true_faces, color='red', alpha=0.5)
        ax1.set_title('Ground Truth')
        
        # Prediction
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_trisurf(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2],
                       triangles=pred_faces, color='blue', alpha=0.5)
        ax2.set_title('Prediction')
        
        # Set same scale and viewpoint for both
        for ax in [ax1, ax2]:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.view_init(elev=30, azim=angle)
            ax.set_axis_off()  # Hide axes for cleaner look
        
        plt.tight_layout()
        
        # Save figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # Add frame to list
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
    
    # Save as GIF
    if frames:
        frames[0].save(
            gif_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=0
        )
        print(f"Animation saved to {gif_path}")
    else:
        print("No frames were created, could not save GIF")
    
    return gif_path

def save_3d_volume(volume, save_path, filename):
    """Save a 3D volume as a NIfTI file"""
    nifti_img = nib.Nifti1Image(volume, np.eye(4))
    nib.save(nifti_img, save_path / filename)

def validate_model_3d(model, loader, device, save_dir=None, create_animation=False, max_samples=5, animation_duration=100):
    """Validate model and create 3D visualizations"""
    model.eval()
    
    metrics = {
        'dice': [],
        'sensitivity': [],
        'specificity': [],
    }
    
    # Create directory for visualizations
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        vis_dir = save_dir / '3d_visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Directory for NIfTI files
        vol_dir = save_dir / 'volumes'
        vol_dir.mkdir(exist_ok=True)
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating with 3D visualizations"):
            # Get data
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            series_id = batch['series_id']
            
            # Forward pass
            pred = model(ct)
            
            # Convert to binary predictions
            pred_binary = (pred > 0.5).float()
            
            # Process each sample in the batch
            for j in range(ct.shape[0]):
                if samples_processed >= max_samples:
                    break
                    
                # Get numpy arrays
                ct_np = ct[j, 0].cpu().numpy()  # Remove channel dimension
                mask_np = mask[j, 0].cpu().numpy()
                pred_np = pred_binary[j, 0].cpu().numpy()
                
                # Calculate Dice score
                dice = dice_score(mask_np, pred_np)
                metrics['dice'].append(dice)
                
                # Sensitivity (recall) and specificity
                true_positive = np.sum(mask_np * pred_np)
                true_negative = np.sum((1 - mask_np) * (1 - pred_np))
                false_positive = np.sum((1 - mask_np) * pred_np)
                false_negative = np.sum(mask_np * (1 - pred_np))
                
                sensitivity_val = true_positive / (true_positive + false_negative + 1e-5)
                specificity_val = true_negative / (true_negative + false_positive + 1e-5)
                
                metrics['sensitivity'].append(sensitivity_val)
                metrics['specificity'].append(specificity_val)
                
                # Create 3D visualization
                if save_dir:
                    print(f"\nCreating 3D visualization for {series_id[j]} (Dice: {dice:.4f})")
                    
                    # Save the 3D visualization
                    vis_path = create_3d_visualization(
                        ct_np, mask_np, pred_np, vis_dir, series_id[j]
                    )
                    print(f"3D visualization saved to {vis_path}")
                    
                    # Create animation if requested
                    if create_animation:
                        # Create GIF directly instead of saving individual frames
                        gif_path = create_3d_animation_gif(
                            ct_np, mask_np, pred_np, vis_dir, series_id[j], 
                            duration=animation_duration
                        )
                        print(f"Animation saved to {gif_path}")
                    
                    # Save NIfTI volumes
                    save_3d_volume(ct_np, vol_dir, f"{series_id[j]}_ct.nii.gz")
                    save_3d_volume(mask_np, vol_dir, f"{series_id[j]}_true_mask.nii.gz")
                    save_3d_volume(pred_np, vol_dir, f"{series_id[j]}_pred_mask.nii.gz")
                
                samples_processed += 1
                
            if samples_processed >= max_samples:
                break
    
    # Calculate and return average metrics
    results = {}
    for metric, values in metrics.items():
        results[metric] = np.mean(values)
        results[f"{metric}_std"] = np.std(values)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Create 3D visualizations of lung segmentation results')
    parser.add_argument('--data_dir', type=str, default='data/Luna16',
                      help='Path to LUNA16 dataset')
    parser.add_argument('--model_path', type=str, default='unet3d_models/best_model.pt',
                      help='Path to saved model')
    parser.add_argument('--save_dir', type=str, default='validation_3d_results',
                      help='Directory to save validation results')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='Number of samples to visualize')
    parser.add_argument('--create_animation', action='store_true',
                      help='Create animated rotation of 3D models')
    parser.add_argument('--animation_duration', type=int, default=100,
                      help='Duration of each animation frame in milliseconds')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Check if PIL is installed
    try:
        from PIL import Image
        print("PIL is installed. GIF creation is available.")
    except ImportError:
        print("WARNING: PIL is not installed. Install with: pip install Pillow")
        if args.create_animation:
            print("Animation requested but PIL not available. Installing PIL...")
            import subprocess
            subprocess.call([sys.executable, "-m", "pip", "install", "Pillow"])
            print("PIL installed.")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    data_dir = Path(args.data_dir)
    print(f"Loading dataset from: {data_dir.absolute()}")
    
    try:
        # Initialize dataset
        dataset = LungDataset(str(data_dir), str(data_dir))
        print(f"Successfully loaded dataset with {len(dataset)} samples")
        
        # Create dataloader with at most num_samples
        val_size = min(len(dataset), args.num_samples)
        val_dataset = Subset(dataset, range(val_size))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        # Load model
        model = UNet3D(n_channels=1, n_classes=1).to(device)
        
        print(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with Dice score: {checkpoint['dice_score']:.4f}")
        
        # Validate and create 3D visualizations
        results = validate_model_3d(
            model,
            val_loader,
            device,
            save_dir=args.save_dir,
            create_animation=args.create_animation,
            max_samples=args.num_samples,
            animation_duration=args.animation_duration
        )
        
        # Print results
        print("\n3D Visualization Results:")
        print(f"Dice Score: {results['dice']:.4f} ± {results['dice_std']:.4f}")
        print(f"Sensitivity: {results['sensitivity']:.4f} ± {results['sensitivity_std']:.4f}")
        print(f"Specificity: {results['specificity']:.4f} ± {results['specificity_std']:.4f}")
        
        # Save metrics to CSV
        save_dir = Path(args.save_dir)
        with open(save_dir / 'metrics_3d.csv', 'w') as f:
            f.write('Metric,Value,StdDev\n')
            for metric in ['dice', 'sensitivity', 'specificity']:
                f.write(f"{metric},{results[metric]:.6f},{results[f'{metric}_std']:.6f}\n")
        
        print(f"\nResults saved to {save_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()