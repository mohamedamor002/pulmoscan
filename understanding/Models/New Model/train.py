import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torchvision import transforms
import logging
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import torch.nn.functional as F
import json
from torchinfo import summary as model_summary
import matplotlib.pyplot as plt

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

class CombinedBCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss for better segmentation"""
    def __init__(self, bce_weight=0.7, dice_weight=0.3, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

def dice_score(pred, target, smooth=1e-5):
    """Calculate Dice score for evaluation"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    batch_count = 0
    
    with tqdm(dataloader, desc='Training') as pbar:
        for batch_idx, (data, target, det_target) in enumerate(pbar):
            try:
                data       = data.to(device)
                target     = target.to(device)
                det_target = det_target.to(device)
                
                optimizer.zero_grad()
                
                # Standard forward pass without mixed precision
                mask_pred, det_pred = model(data)
                loss_seg = criterion(mask_pred, target)
                loss_reg = nn.MSELoss()(det_pred, det_target)
                loss = loss_seg + loss_reg
                dice = dice_score(mask_pred, target)
                
                # Handle NaN or inf values
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Warning: NaN or inf loss detected in batch {batch_idx}. Skipping.")
                    continue
                
                # Standard backward pass and optimization
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_dice += dice.item()
                batch_count += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice.item():.4f}'
                })
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
    
    if batch_count > 0:
        epoch_loss /= batch_count
        epoch_dice /= batch_count
    
    logger.info(f'Training Loss: {epoch_loss:.4f}, Dice Score: {epoch_dice:.4f}')
    return epoch_loss, epoch_dice

def evaluate(model, dataloader, criterion, device, logger):
    """Evaluate the model on the validation set"""
    model.eval()
    val_loss = 0
    val_dice = 0
    val_detection_accuracy = 0.0  # New metric
    batch_count = 0
    
    with torch.no_grad():
        with tqdm(dataloader, desc='Validation') as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Handle different batch formats
                    if len(batch) == 3:  # If it contains (data, target, det_target)
                        data, target, det_target = batch
                        data = data.to(device)
                        target = target.to(device)
                        det_target = det_target.to(device)
                        
                        # Forward pass
                        mask_output, det_output = model(data)
                        
                        # Calculate losses
                        loss_seg = criterion(mask_output, target)
                        loss_reg = nn.MSELoss()(det_output, det_target)
                        loss = loss_seg + loss_reg
                        
                        # Calculate detection accuracy
                        detection_accuracy = calculate_nodule_detection_accuracy(
                            det_output, target, volume_shape=target.shape[1:])
                    else:  # If it only contains (data, target)
                        data, target = batch
                        data = data.to(device)
                        target = target.to(device)
                        
                        # Forward pass
                        mask_output, det_output = model(data)
                        
                        # Only use segmentation loss
                        loss = criterion(mask_output, target)
                        
                        # Calculate detection accuracy using target as stand-in for detection target
                        detection_accuracy = calculate_nodule_detection_accuracy(
                            det_output, target, volume_shape=target.shape[1:])
                    
                    # Calculate dice score
                    dice = dice_score(mask_output, target)
                    
                    val_loss += loss.item()
                    val_dice += dice.item()
                    val_detection_accuracy += detection_accuracy
                    batch_count += 1
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'dice': f'{dice.item():.4f}'
                    })
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
    
    if batch_count > 0:
        val_loss /= batch_count
        val_dice /= batch_count
        val_detection_accuracy /= batch_count
    
    logger.info(f'Validation Loss: {val_loss:.4f}, Dice Score: {val_dice:.4f}')
    logger.info(f'Nodule Detection Accuracy: {val_detection_accuracy:.2f}%')
    return val_loss, val_dice, val_detection_accuracy

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

def train_preloaded(model, preloaded_data, criterion, optimizer, device, logger):
    """Train the model on preloaded data (detection only)"""
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    with tqdm(preloaded_data, desc='Training Detection') as pbar:
        for batch_idx, batch in enumerate(pbar):
            try:
                # only look at batches with a detection target
                if len(batch) != 3:
                    continue
                # batch = (data, mask, det_target)
                data, _, det_target = batch
                data, det_target = data.to(device), det_target.to(device)
                
                optimizer.zero_grad()
                
                # forward & detection MSE
                _, det_pred = model(data)
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

def calculate_nodule_detection_accuracy(pred_coords, target, volume_shape, distance_threshold=None):
    """
    Calculate detection accuracy by checking if predicted coordinates fall within the nodule mask
    
    Args:
        pred_coords: Predicted normalized [z,y,x,r] coordinates, tensor of shape [batch_size, 4]
        target: Ground truth segmentation mask
        volume_shape: Shape of the volume (D, H, W) 
        distance_threshold: Optional parameter, not used
        
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
                    
                # Debug output for first few samples
                if i < 5:
                    print(f"Sample {i}: Coords [{z_norm:.2f}, {y_norm:.2f}, {x_norm:.2f}] -> [{z_voxel}, {y_voxel}, {x_voxel}], Mask value: {mask_value:.2f}")
        
        # Calculate accuracy
        accuracy = (correct_detections / batch_size) * 100
        return accuracy
        
    except Exception as e:
        print(f"Error in detection accuracy calculation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def evaluate_preloaded(model, preloaded_data, criterion, device, logger):
    """Evaluate the model on preloaded validation data"""
    model.eval()
    val_loss = 0
    val_dice = 0
    val_detection_accuracy = 0.0
    batch_count = 0
    
    with torch.no_grad():
        with tqdm(preloaded_data, desc='Validation') as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Unpack batch data safely
                    if len(batch) == 3:  # (data, target, det_target)
                        data, target, _ = batch  # Ignore det_target for simplicity
                    else:  # (data, target)
                        data, target = batch
                    
                    # Move to device
                    data = data.to(device)
                    target = target.to(device)
                    
                    # Forward pass
                    mask_output, det_output = model(data)
                    
                    # Calculate segmentation loss
                    loss = criterion(mask_output, target)
                    dice = dice_score(mask_output, target)
                    
                    # Calculate detection accuracy - without using det_target
                    try:
                        detection_accuracy = calculate_nodule_detection_accuracy(
                            det_output, target, volume_shape=target.shape[2:] if len(target.shape) > 3 else target.shape[1:])
                    except Exception as e:
                        logger.warning(f"Detection accuracy calculation failed: {e}")
                        detection_accuracy = 0.0
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_dice += dice.item()
                    val_detection_accuracy += detection_accuracy
                    batch_count += 1
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'dice': f'{dice.item():.4f}'
                    })
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
    
    if batch_count > 0:
        val_loss /= batch_count
        val_dice /= batch_count
        val_detection_accuracy /= batch_count
    
    logger.info(f'Validation Loss: {val_loss:.4f}, Dice Score: {val_dice:.4f}')
    logger.info(f'Nodule Detection Accuracy: {val_detection_accuracy:.2f}%')
    return val_loss, val_dice, val_detection_accuracy

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
    data_dir = "C:/Users/PC/Downloads/Project/data/Luna16"
    annotations_file = "C:/Users/PC/Downloads/Project/data/Luna16/processed_annotations.csv"
    
    # Create transforms
    transforms = NoduleTransforms(p=0.5)
    
    # Disable mixed precision for now - simplify training
    # scaler = GradScaler()
    
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
    
    # Create model with correct parameters
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
    
    # Loss functions
    seg_criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)  # still used for eval
    det_criterion = nn.MSELoss()  # detection-only training
    
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
        patience=15,  # More patience
        verbose=True,
        min_lr=1e-7  # Minimum learning rate
    )
    
    # Training parameters
    num_epochs = 200  # More epochs
    # Save under models/ next to this script
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "best_model.pth"
    
    # Early stopping parameters
    early_stopping_patience = 25
    no_improvement_count = 0
    best_val_detection_accuracy = 0.0
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    # Preload data to avoid disk I/O during training
    train_data = preload_data(train_dataset, train_loader, logger)
    val_data   = preload_data(val_dataset, val_loader, logger)

    # Prepare to record detection accuracy per epoch
    train_acc_history = []
    val_acc_history   = []

    try:
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Detection-only training
            train_detection_mse = train_preloaded(
                model, train_data, det_criterion, optimizer, device, logger
            )

            # Full eval
            val_loss, val_dice, val_detection_accuracy = evaluate_preloaded(
                model, val_data, seg_criterion, device, logger
            )
            scheduler.step(val_loss)
            
            # Record detection accuracy on train & val sets
            _, _, train_detection_accuracy = evaluate_preloaded(
                model, train_data, seg_criterion, device, logger
            )
            train_acc_history.append(train_detection_accuracy)
            val_acc_history.append(val_detection_accuracy)
            
            if val_dice > best_val_detection_accuracy:
                best_val_detection_accuracy = val_dice
                no_improvement_count = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_detection_mse': train_detection_mse,
                    'val_loss':   val_loss,
                    'val_dice':   val_dice,
                    'val_detection_accuracy': val_detection_accuracy
                }, best_model_path)
                logger.info(f"Saved best model with validation Dice score: {val_dice:.4f}")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = models_dir / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_detection_mse': train_detection_mse,
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'val_detection_accuracy': val_detection_accuracy
                }, checkpoint_path)
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user; plotting partial results.")
    finally:
        # Plot detection accuracy curves over epochs
        try:
            plt.figure(figsize=(8,6))
            epochs_list = list(range(1, len(train_acc_history) + 1))
            plt.plot(epochs_list, train_acc_history, label="Train Det Acc")
            plt.plot(epochs_list, val_acc_history,   label="Val   Det Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Detection Accuracy (%)")
            plt.title("Detection Accuracy Over Epochs")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(metrics_dir / "accuracy_over_epochs.png")
            plt.close()
            logger.info(f"Saved accuracy_over_epochs.png with {len(epochs_list)} points")
        except Exception as e:
            logger.warning(f"Failed to plot accuracy curves: {e}")

        # Final timing and summary
        training_time = time.time() - start_time
        logger.info(f"Training ended after {len(train_acc_history)} epochs in {training_time/60:.2f} minutes")
        logger.info(f"Best validation Detection Accuracy: {best_val_detection_accuracy:.2f}%")

if __name__ == "__main__":
    main() 