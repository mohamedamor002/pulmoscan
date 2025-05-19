import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import json
from torchinfo import summary as model_summary
import matplotlib.pyplot as plt
import random

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Import local modules
from unet3d import UNet3D
from dataset import NoduleDataset

class NoduleTransforms:
    """Custom transforms for 3D nodule data"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, data_tuple):
        """Apply transforms to both volume and mask.
        
        Args:
            data_tuple: Tuple of (volume, mask) as numpy arrays or tensors
            
        Returns:
            Tuple of transformed (volume, mask) as torch tensors
        """
        volume, mask = data_tuple
        
        # Convert to NumPy arrays if needed
        if isinstance(volume, torch.Tensor):
            volume = volume.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            
        # Ensure 3D (remove channel dimension if present)
        if volume.ndim == 4 and volume.shape[0] == 1:
            volume = volume[0]  # Remove channel dimension
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]  # Remove channel dimension
        
        # Start with fresh copies to avoid stride issues
        volume = volume.copy()
        mask = mask.copy()
            
        # Apply augmentations - manually implementing each to avoid stride issues
        # Random horizontal flip (along X)
        if np.random.random() < self.p:
            volume = volume[:, :, ::-1].copy()  # Manual flip + copy
            mask = mask[:, :, ::-1].copy()      # Manual flip + copy
        
        # Random vertical flip (along Y)
        if np.random.random() < self.p:
            volume = volume[:, ::-1, :].copy()  # Manual flip + copy
            mask = mask[:, ::-1, :].copy()      # Manual flip + copy
        
        # Random z-flip
        if np.random.random() < self.p:
            volume = volume[::-1, :, :].copy()  # Manual flip + copy
            mask = mask[::-1, :, :].copy()      # Manual flip + copy
        
        # Random rotation (90 degrees in x-y plane)
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)  # 1, 2, or 3 (90, 180, 270 degrees)
            # Manually implement rotation without using np.rot90
            if k == 1:
                volume = np.transpose(volume, (0, 2, 1))[:, ::-1, :].copy()
                mask = np.transpose(mask, (0, 2, 1))[:, ::-1, :].copy()
            elif k == 2:
                volume = volume[:, ::-1, ::-1].copy()
                mask = mask[:, ::-1, ::-1].copy()
            elif k == 3:
                volume = np.transpose(volume, (0, 2, 1))[:, :, ::-1].copy()
                mask = np.transpose(mask, (0, 2, 1))[:, :, ::-1].copy()
        
        # Random intensity scaling (volume only)
        if np.random.random() < self.p:
            scale = 0.9 + 0.2 * np.random.random()  # Scale between 0.9 and 1.1
            volume = (volume * scale).clip(0, 1).copy()
            
        # Fully defensive conversion to tensors 
        # Using torch.tensor guarantees a copy with C-contiguous memory
        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return volume_tensor, mask_tensor

def calculate_detection_metrics(pred_coords, gt_coords, volume_shape):
    """
    Calculate detection accuracy metrics
    
    Args:
        pred_coords: Predicted normalized [z,y,x,r] coordinates, tensor of shape [batch_size, 4]
        gt_coords: Ground truth normalized [z,y,x,r] coordinates, tensor of shape [batch_size, 4]
        volume_shape: Shape of the volume (D,H,W) to denormalize coordinates
        
    Returns:
        dict: Dictionary of detection metrics
    """
    # Convert to numpy for easier calculation
    pred_coords_np = pred_coords.detach().cpu().numpy()
    gt_coords_np = gt_coords.detach().cpu().numpy()
    
    # Extract depth, height, width for denormalization
    D, H, W = volume_shape
    
    # Denormalize coordinates to get actual voxel positions
    batch_size = pred_coords_np.shape[0]
    
    # Initialize metrics
    center_distances = []
    radius_errors = []
    detection_success = []
    
    for i in range(batch_size):
        # Get normalized coordinates
        pred_z, pred_y, pred_x, pred_r = pred_coords_np[i]
        gt_z, gt_y, gt_x, gt_r = gt_coords_np[i]
        
        # Denormalize to voxel space
        pred_z_voxel, pred_y_voxel, pred_x_voxel = pred_z * D, pred_y * H, pred_x * W
        gt_z_voxel, gt_y_voxel, gt_x_voxel = gt_z * D, gt_y * H, gt_x * W
        
        # Denormalize radius
        min_dim = min(D, H, W)
        pred_r_voxel = pred_r * min_dim
        gt_r_voxel = gt_r * min_dim
        
        # Calculate Euclidean distance between centers
        center_dist = np.sqrt(
            (pred_z_voxel - gt_z_voxel)**2 + 
            (pred_y_voxel - gt_y_voxel)**2 + 
            (pred_x_voxel - gt_x_voxel)**2
        )
        
        # Calculate absolute radius error
        radius_error = abs(pred_r_voxel - gt_r_voxel)
        
        # Detection is successful if predicted center is within ground truth radius
        is_detection_successful = center_dist <= gt_r_voxel
        
        # Store metrics
        center_distances.append(center_dist)
        radius_errors.append(radius_error)
        detection_success.append(float(is_detection_successful))
    
    # Calculate average metrics
    avg_center_distance = np.mean(center_distances)
    avg_radius_error = np.mean(radius_errors)
    detection_success_rate = np.mean(detection_success)
    
    return {
        'center_distance': avg_center_distance,  # Lower is better
        'radius_error': avg_radius_error,        # Lower is better
        'detection_rate': detection_success_rate  # Higher is better (0-1)
    }

def calculate_nodule_detection_accuracy(pred_coords, target, volume_shape):
    """
    Calculate detection accuracy by checking if predicted coordinates fall within the nodule mask
    
    Args:
        pred_coords: Predicted normalized [z,y,x,r] coordinates, tensor of shape [batch_size, 4]
        target: Ground truth segmentation mask
        volume_shape: Shape of the volume (D, H, W) 
        
    Returns:
        detection_accuracy: Percentage of correctly detected nodules (0-100%)
    """
    try:
        # Convert to cpu and numpy for easier manipulation
        pred_coords_cpu = pred_coords.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        
        # Determine target shape dimensions
        if len(target_cpu.shape) == 5:  # [B, C, D, H, W]
            batch_size, _, D, H, W = target_cpu.shape
        elif len(target_cpu.shape) == 4:  # [B, D, H, W]
            batch_size, D, H, W = target_cpu.shape
        else:
            print(f"Unsupported target shape: {target_cpu.shape}")
            return 0.0
            
        if batch_size == 0:
            return 0.0
            
        # Expand volume_shape to match dimensions if needed
        if len(volume_shape) < 3:
            print(f"Warning: volume_shape {volume_shape} has fewer than 3 dimensions")
            if len(target_cpu.shape) >= 4:
                if len(target_cpu.shape) == 5:
                    volume_shape = (D, H, W)
                else:
                    volume_shape = target_cpu.shape[1:]
                
        # Track correct detections
        correct_detections = 0
        
        # Process each sample in batch
        for i in range(batch_size):
            if pred_coords_cpu.shape[1] >= 4:
                # Get predicted normalized coordinates
                z_norm = pred_coords_cpu[i, 0]
                y_norm = pred_coords_cpu[i, 1]
                x_norm = pred_coords_cpu[i, 2]
                
                # Ensure values are in [0,1] range (apply sigmoid if needed)
                z_norm = max(0, min(1, z_norm))
                y_norm = max(0, min(1, y_norm))
                x_norm = max(0, min(1, x_norm))
                
                # Convert normalized coordinates [0,1] to voxel coordinates
                z_voxel = min(max(int(z_norm * D), 0), D-1)
                y_voxel = min(max(int(y_norm * H), 0), H-1)
                x_voxel = min(max(int(x_norm * W), 0), W-1)
                
                # Check if this point hits a nodule in the target mask
                if len(target_cpu.shape) == 5:  # [B, C, D, H, W]
                    mask_value = target_cpu[i, 0, z_voxel, y_voxel, x_voxel]
                else:  # [B, D, H, W]
                    mask_value = target_cpu[i, z_voxel, y_voxel, x_voxel]
                
                # Consider hit if value > 0.5 (binary mask threshold)
                if mask_value > 0.5:
                    correct_detections += 1
        
        # Calculate accuracy
        accuracy = (correct_detections / batch_size) * 100
        return accuracy
        
    except Exception as e:
        print(f"Error in detection accuracy calculation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def train_preloaded(model, preloaded_data, criterion, optimizer, device, logger):
    """Train the model on preloaded data (detection only)"""
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    with tqdm(preloaded_data, desc='Training Detection') as pbar:
        for batch_idx, batch in enumerate(pbar):
            try:
                # Extract data and detection target
                if len(batch) == 3:  # (data, mask, det_target)
                    data, _, det_target = batch
                else:  # If it only contains (data, target), use target as det_target for compatibility
                    data, det_target = batch
                
                data, det_target = data.to(device), det_target.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass - get both outputs but only use detection
                mask_pred, det_pred = model(data)
                loss = criterion(det_pred, det_target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                pbar.set_postfix({'mse': f'{loss.item():.4f}'})
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
    
    if batch_count > 0:
        epoch_loss /= batch_count
    logger.info(f'Training Detection MSE: {epoch_loss:.4f}')
    return epoch_loss

def evaluate_preloaded(model, preloaded_data, criterion, device, logger):
    """Evaluate the model on preloaded validation data"""
    model.eval()
    val_loss = 0
    val_detection_accuracy = 0.0
    batch_count = 0
    
    with torch.no_grad():
        with tqdm(preloaded_data, desc='Validation') as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Unpack batch data safely
                    if len(batch) == 3:  # (data, target, det_target)
                        data, target, det_target = batch
                    else:  # (data, target)
                        data, target = batch
                        det_target = target  # Use target as det_target for compatibility
                    
                    # Move to device
                    data = data.to(device)
                    det_target = det_target.to(device)
                    
                    # Forward pass - get both outputs but only use detection
                    _, det_output = model(data)
                    
                    # Calculate detection loss
                    loss = criterion(det_output, det_target)
                    
                    # Calculate detection accuracy using target mask
                    try:
                        detection_accuracy = calculate_nodule_detection_accuracy(
                            det_output, target, 
                            volume_shape=target.shape[2:] if len(target.shape) > 3 else target.shape[1:])
                    except Exception as e:
                        logger.warning(f"Detection accuracy calculation failed: {e}")
                        detection_accuracy = 0.0
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_detection_accuracy += detection_accuracy
                    batch_count += 1
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'det_acc': f'{detection_accuracy:.2f}%'
                    })
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
    
    if batch_count > 0:
        val_loss /= batch_count
        val_detection_accuracy /= batch_count
    
    logger.info(f'Validation Loss: {val_loss:.4f}')
    logger.info(f'Nodule Detection Accuracy: {val_detection_accuracy:.2f}%')
    return val_loss, val_detection_accuracy

def preload_data(dataset, loader, logger):
    """Preload all data into memory to avoid disk bottlenecks during training."""
    logger.info("Preloading data to avoid disk I/O during training...")
    preloaded_data = []
    
    with tqdm(loader, desc='Preloading') as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Check the length of the batch tuple
            if len(batch) == 3:  # If it contains (data, target, det_target)
                data, target, det_target = batch
                preloaded_data.append((data, target, det_target))
            else:  # If it only contains (data, target)
                data, target = batch
                preloaded_data.append((data, target))
            
            pbar.set_postfix({'loaded': f'{batch_idx+1}/{len(loader)}'})
    
    logger.info(f"Preloaded {len(preloaded_data)} batches")
    return preloaded_data

def save_model_architecture(model, save_dir):
    """Save the model architecture summary"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate model summary
    model_info = model_summary(model, input_size=(1, 1, 64, 64, 64), verbose=0)
    
    # Save as text
    with open(save_dir / 'model_architecture.txt', 'w', encoding='utf-8') as f:
        f.write(str(model_info))
    
    # Save parameter count summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    params_summary = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Approximate size in MB (4 bytes per float32 param)
    }
    
    with open(save_dir / 'model_params.json', 'w', encoding='utf-8') as f:
        json.dump(params_summary, f, indent=4)
    
    print(f"Model summary saved to {save_dir}")
    print(f"Total trainable parameters: {trainable_params:,}")
    
    return model_info, params_summary

def visualize_detection_results(model, data_samples, device, save_dir):
    """
    Visualize detection results on a sample of scans with simplified view
    
    Args:
        model: Trained model
        data_samples: List of (data, target) tuples
        device: Device to run model on
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, sample in enumerate(data_samples):
            try:
                # Get data and ground truth
                if len(sample) == 3:  # (data, target, det_target)
                    data, target, _ = sample
                else:  # (data, target)
                    data, target = sample
                
                # Move to device
                data = data.to(device)
                
                # Forward pass
                _, det_output = model(data)
                
                # Convert to numpy for visualization
                volume = data.cpu().numpy()[0, 0]  # (D, H, W)
                mask = target.cpu().numpy()
                if len(mask.shape) > 3:
                    mask = mask[0, 0] if len(mask.shape) == 5 else mask[0]  # (D, H, W)
                
                pred_coords = det_output.cpu().numpy()[0]  # (4,) -> z, y, x, r
                
                # Get dimensions
                D, H, W = volume.shape
                
                # Normalize coordinates to voxel space
                z_norm, y_norm, x_norm, r_norm = pred_coords
                z_voxel = int(min(max(z_norm * D, 0), D-1))
                y_voxel = int(min(max(y_norm * H, 0), H-1))
                x_voxel = int(min(max(x_norm * W, 0), W-1))
                
                # Create simplified visualizations for 3 orthogonal slices through the predicted center
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Axial view (x-y plane at z)
                axes[0].imshow(volume[z_voxel], cmap='gray')
                
                # Add marker for prediction center (red X)
                axes[0].scatter(x_voxel, y_voxel, c='r', marker='x', s=100, linewidths=2)
                
                # For this slice, show ground truth mask as overlay instead of circle
                if 0 <= z_voxel < mask.shape[0]:
                    mask_slice = mask[z_voxel]
                    # Only draw contour if there are non-zero values in this slice
                    if np.any(mask_slice > 0.5):
                        axes[0].contour(mask_slice, levels=[0.5], colors='g', linewidths=2)
                
                axes[0].set_title(f'Axial Slice (z={z_voxel})')
                
                # Coronal view (x-z plane at y)
                axes[1].imshow(volume[:, y_voxel, :].T, cmap='gray')
                
                # Add marker for prediction center (red X)
                axes[1].scatter(z_voxel, x_voxel, c='r', marker='x', s=100, linewidths=2)
                
                # For this slice, show ground truth mask as overlay
                if 0 <= y_voxel < mask.shape[1]:
                    mask_slice = mask[:, y_voxel, :].T
                    # Only draw contour if there are non-zero values in this slice
                    if np.any(mask_slice > 0.5):
                        axes[1].contour(mask_slice, levels=[0.5], colors='g', linewidths=2)
                
                axes[1].set_title(f'Coronal Slice (y={y_voxel})')
                
                # Sagittal view (y-z plane at x)
                axes[2].imshow(volume[:, :, x_voxel].T, cmap='gray')
                
                # Add marker for prediction center (red X)
                axes[2].scatter(z_voxel, y_voxel, c='r', marker='x', s=100, linewidths=2)
                
                # For this slice, show ground truth mask as overlay
                if 0 <= x_voxel < mask.shape[2]:
                    mask_slice = mask[:, :, x_voxel].T
                    # Only draw contour if there are non-zero values in this slice
                    if np.any(mask_slice > 0.5):
                        axes[2].contour(mask_slice, levels=[0.5], colors='g', linewidths=2)
                
                axes[2].set_title(f'Sagittal Slice (x={x_voxel})')
                
                # Add a legend
                legend_elements = [
                    plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='r', markersize=10, label='Predicted Center'),
                    plt.Line2D([0], [0], color='g', lw=2, label='Ground Truth Boundary')
                ]
                axes[0].legend(handles=legend_elements, loc='lower right')
                
                plt.tight_layout()
                plt.savefig(save_dir / f"detection_result_{i+1}.png")
                plt.close()
                
            except Exception as e:
                print(f"Error visualizing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

def main():
    # Enable deterministic training for reproducibility
    torch.backends.cudnn.deterministic = False  # Disable for better performance
    torch.backends.cudnn.benchmark = True  # Enable autotuner
    
    # Set up logging
    log_dir = Path("../../logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PyTorch training...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset with correct paths
    logger.info("Creating dataset from preprocessed annotations...")
    # root folder with CT scans
    data_dir = Path("data/Luna16").resolve()
    # use the processed CSV in this script's directory
    annotations_file = Path(__file__).parent / "processed_annotations.csv"
    
    # Create transforms
    transforms = NoduleTransforms(p=0.5)
    
    # Create dataset with smaller cache size
    dataset = NoduleDataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        patch_size=(64, 64, 64),
        transform=transforms
    )
    
    # Use all available samples
    max_samples = len(dataset)  # Load all scans
    print(f"Loading all {max_samples} scans")

    # Create sampler for limiting dataset size initially
    class LimitedSampler(torch.utils.data.Sampler):
        def __init__(self, dataset, max_samples):
            self.dataset = dataset
            self.max_samples = min(max_samples, len(dataset))
            self.indices = torch.randperm(len(dataset))[:self.max_samples].tolist()
        
        def __iter__(self):
            return iter(self.indices)
        
        def __len__(self):
            return self.max_samples

    # Update the train/val split ratio
    train_size = int(0.9 * max_samples)  # 90% for training
    val_size = max_samples - train_size   # 10% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    # Create data loaders with specified batch size and sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Minimum batch size to reduce memory
        shuffle=False,  # Using sampler instead
        sampler=LimitedSampler(train_dataset, max_samples=train_size),  # Use all training samples
        num_workers=0,  # No workers to avoid multiprocessing
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Minimum batch size
        shuffle=False,
        sampler=LimitedSampler(val_dataset, val_size),  # Use all validation samples
        num_workers=0,
        pin_memory=True
    )
    
    # Create model - use standard UNet3D with 1 input channel and 1 output channel for segmentation
    # The detection outputs come from a separate path in the UNet3D class
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    
    # Initialize model weights properly
    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Save model architecture summary
    metrics_dir = Path(__file__).parent / "model_metrics"
    save_model_architecture(model, metrics_dir)
    
    # Loss function for detection only
    criterion = nn.MSELoss()
    
    # Lower learning rate and adjusted optimizer parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-5,  # Lower learning rate
        weight_decay=1e-4,  # Increased weight decay for better regularization
        betas=(0.9, 0.999)  # Default Adam betas
    )
    
    # More patient learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,  # Larger reduction in learning rate
        patience=5,  # More patience
        verbose=True,
        min_lr=1e-7  # Minimum learning rate
    )
    
    # Training parameters
    num_epochs = 10  # REDUCED FROM 200 TO 10
    # Save under models/ next to this script
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "best_model_detection_only.pth"
    
    # Early stopping parameters
    early_stopping_patience = 5
    no_improvement_count = 0
    best_val_detection_accuracy = 0.0
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    # Preload data to avoid disk I/O during training
    train_data = preload_data(train_dataset, train_loader, logger)
    val_data   = preload_data(val_dataset, val_loader, logger)

    # Prepare to record detection accuracy per epoch
    train_loss_history = []
    val_acc_history = []

    try:
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Detection-only training
            train_detection_mse = train_preloaded(
                model, train_data, criterion, optimizer, device, logger
            )

            # Validation 
            val_loss, val_detection_accuracy = evaluate_preloaded(
                model, val_data, criterion, device, logger
            )
            scheduler.step(val_loss)
            
            # Record metrics
            train_loss_history.append(train_detection_mse)
            val_acc_history.append(val_detection_accuracy)
            
            # Save best model with error handling
            if val_detection_accuracy > best_val_detection_accuracy:
                best_val_detection_accuracy = val_detection_accuracy
                no_improvement_count = 0
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_detection_mse': train_detection_mse,
                        'val_loss': val_loss,
                        'val_detection_accuracy': val_detection_accuracy
                    }, best_model_path)
                    logger.info(f"Saved best model with validation detection accuracy: {val_detection_accuracy:.4f}%")
                except Exception as e:
                    logger.error(f"Failed to save best model: {e}")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user; plotting partial results.")
    finally:
        # Plot training metrics
        try:
            # Create a figure with two y-axes
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            
            epochs_list = list(range(1, len(train_loss_history) + 1))
            
            # Plot training loss on left axis
            ax1.plot(epochs_list, train_loss_history, 'b-', label='Training Loss (MSE)')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Training Loss', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Plot validation accuracy on right axis
            ax2.plot(epochs_list, val_acc_history, 'r-', label='Validation Accuracy (%)')
            ax2.set_ylabel('Detection Accuracy (%)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            plt.title('Training Loss vs Validation Accuracy')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(metrics_dir / "training_metrics.png")
            plt.close()
            logger.info(f"Saved training_metrics.png")
        except Exception as e:
            logger.warning(f"Failed to plot metrics: {e}")

        # Final timing and summary
        training_time = time.time() - start_time
        logger.info(f"Training ended after {len(train_loss_history)} epochs in {training_time/60:.2f} minutes")
        logger.info(f"Best validation Detection Accuracy: {best_val_detection_accuracy:.2f}%")
        
        # Try loading the best model, but use current model if loading fails
        logger.info("Loading best model for visualization...")
        best_loaded = False
        
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                best_epoch = checkpoint['epoch']
                best_accuracy = checkpoint['val_detection_accuracy']
                logger.info(f"Loaded best model from epoch {best_epoch+1} with accuracy {best_accuracy:.2f}%")
                best_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load best model: {e}. Using current model instead.")
        
        if not best_loaded:
            logger.info("Using current model state for visualizations.")
        
        # Create visualization directory
        vis_dir = Path(__file__).parent / "visualizations"
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Select 5 random samples from validation set for visualization
        logger.info("Selecting 5 random samples for visualization...")
        val_samples = random.sample(val_data, min(5, len(val_data)))
        
        # Visualize detection results
        logger.info("Generating visualizations...")
        visualize_detection_results(model, val_samples, device, vis_dir)
        logger.info(f"Saved visualizations to {vis_dir}")

if __name__ == "__main__":
    main() 
