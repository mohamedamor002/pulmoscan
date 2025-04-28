import os
import sys
import argparse
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
import pandas as pd
import SimpleITK as sitk

from unet3d import UNet3D

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

class NoduleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotations_file, patch_size=(64, 64, 64), transform=None):
        """
        Dataset for loading nodule patches from processed annotations
        
        Args:
            data_dir: Path to Luna16 dataset directory
            annotations_file: Path to processed annotations CSV file
            patch_size: Size of patches to extract (z, y, x)
            transform: Optional transform to apply to samples
        """
        self.data_dir = Path(data_dir)
        self.annotations = pd.read_csv(annotations_file)
        self.patch_size = patch_size
        self.transform = transform
        
        # Filter out any out-of-bounds annotations
        self.annotations = self.annotations[self.annotations['out_of_bounds'] == False]
        logger.info(f"Loaded {len(self.annotations)} valid nodule annotations")
        
        # Analyze nodule size distribution
        self.annotations['nodule_volume'] = (4/3) * np.pi * (self.annotations['diameter_mm']/2)**3
        mean_volume = self.annotations['nodule_volume'].mean()
        median_volume = self.annotations['nodule_volume'].median()
        logger.info(f"Nodule volume statistics - Mean: {mean_volume:.2f} mm³, Median: {median_volume:.2f} mm³")
        
        # Create a balanced sampling based on nodule size
        self.annotations['size_category'] = pd.qcut(self.annotations['diameter_mm'], 3, labels=['small', 'medium', 'large'])
        logger.info(f"Size distribution: {self.annotations['size_category'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get annotation info
        annotation = self.annotations.iloc[idx]
        series_uid = annotation['seriesuid']
        
        # Extract nodule information
        center_x = int(annotation['voxel_x'])
        center_y = int(annotation['voxel_y'])
        center_z = int(annotation['voxel_z'])
        diameter_mm = float(annotation['diameter_mm'])
        
        # Find and load the CT scan
        ct_path = None
        for subset in range(10):
            potential_path = self.data_dir / f'subset{subset}' / f'{series_uid}.mhd'
            if potential_path.exists():
                ct_path = potential_path
                break
        
        if ct_path is None:
            # If file not found, return empty patch and mask
            logger.warning(f"CT scan not found for series {series_uid}")
            volume = np.zeros(self.patch_size, dtype=np.float32)
            mask = np.zeros(self.patch_size, dtype=np.float32)
            if self.transform:
                volume, mask = self.transform((volume, mask))
            return volume, mask
        
        # Load the CT scan
        try:
            ct_image = sitk.ReadImage(str(ct_path))
            volume = sitk.GetArrayFromImage(ct_image)
            spacing = ct_image.GetSpacing()
        except Exception as e:
            logger.error(f"Error loading CT scan {ct_path}: {e}")
            volume = np.zeros(self.patch_size, dtype=np.float32)
            mask = np.zeros(self.patch_size, dtype=np.float32)
            if self.transform:
                volume, mask = self.transform((volume, mask))
            return volume, mask
        
        # Calculate patch boundaries
        half_size = [p // 2 for p in self.patch_size]
        
        # Extract coordinates
        z_start = max(0, center_z - half_size[0])
        z_end = min(volume.shape[0], center_z + half_size[0])
        y_start = max(0, center_y - half_size[1])
        y_end = min(volume.shape[1], center_y + half_size[1])
        x_start = max(0, center_x - half_size[2])
        x_end = min(volume.shape[2], center_x + half_size[2])
        
        # Extract the patch
        patch = np.zeros(self.patch_size, dtype=np.float32)
        extracted = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Create a zero-filled patch of the correct size
        patch = np.zeros(self.patch_size, dtype=np.float32)
        
        # Copy the extracted region into the patch
        z_size = min(z_end - z_start, self.patch_size[0])
        y_size = min(y_end - y_start, self.patch_size[1])
        x_size = min(x_end - x_start, self.patch_size[2])
        
        patch[:z_size, :y_size, :x_size] = extracted[:z_size, :y_size, :x_size]
        
        # Normalize to [0, 1]
        patch = np.clip(patch, -1000, 400)  # Clip to reasonable HU range
        patch = (patch - patch.min()) / (max(patch.max() - patch.min(), 1e-8))
        
        # Create nodule mask (spherical)
        mask = np.zeros(self.patch_size, dtype=np.float32)
        
        # Adjust the center for the patch
        center_patch_z = center_z - z_start
        center_patch_y = center_y - y_start
        center_patch_x = center_x - x_start
        
        # Calculate radius in voxels
        radius_mm = diameter_mm / 2.0
        radius_voxels = int(radius_mm / min(spacing))
        
        # Create coordinate grids
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(self.patch_size[0]),
            np.arange(self.patch_size[1]),
            np.arange(self.patch_size[2]), 
            indexing='ij'
        )
        
        # Calculate distances
        distances = np.sqrt(
            ((z_grid - center_patch_z) * spacing[2]) ** 2 +
            ((y_grid - center_patch_y) * spacing[1]) ** 2 +
            ((x_grid - center_patch_x) * spacing[0]) ** 2
        )
        
        # Create spherical mask
        mask[distances <= radius_mm] = 1.0
        
        # Apply transforms if specified
        if self.transform:
            patch, mask = self.transform((patch, mask))
        else:
            # Add channel dimension
            patch = torch.from_numpy(patch).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return patch, mask

class NoduleTransforms:
    """Custom transforms for 3D nodule data"""
    def __init__(self, p=0.7):  # Increased probability for more aggressive augmentation
        self.p = p
        
    def __call__(self, data_tuple):
        """Apply transforms to both volume and mask."""
        volume, mask = data_tuple
        
        # Make sure inputs are numpy arrays
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # Apply random transforms
        # Random horizontal flip
        if np.random.random() < self.p:
            volume = np.flip(volume, axis=2).copy()  # Use copy() to ensure contiguous array
            mask = np.flip(mask, axis=2).copy()
        
        # Random vertical flip
        if np.random.random() < self.p:
            volume = np.flip(volume, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # Random z-flip
        if np.random.random() < self.p:
            volume = np.flip(volume, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        
        # Random rotation
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)  # 1, 2, or 3 times 90 degrees
            volume = np.rot90(volume, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        
        # Additional augmentations
        # Random intensity variations
        if np.random.random() < self.p:
            # Add random noise
            noise_level = np.random.uniform(0, 0.05)
            volume = volume + np.random.normal(0, noise_level, volume.shape)
            volume = np.clip(volume, 0, 1)  # Ensure values stay in [0,1]
        
        # Random gamma adjustment
        if np.random.random() < self.p:
            gamma = np.random.uniform(0.8, 1.2)
            volume = np.power(volume, gamma)
            
        # Random translation
        if np.random.random() < self.p:
            shift_z = np.random.randint(-3, 4)
            shift_y = np.random.randint(-3, 4)
            shift_x = np.random.randint(-3, 4)
            
            # Create shifted arrays
            shifted_vol = np.zeros_like(volume)
            shifted_mask = np.zeros_like(mask)
            
            # Calculate source and destination slices
            src_z_start = max(0, -shift_z)
            src_z_end = min(volume.shape[0], volume.shape[0] - shift_z)
            dst_z_start = max(0, shift_z)
            dst_z_end = min(volume.shape[0], volume.shape[0] + shift_z)
            
            src_y_start = max(0, -shift_y)
            src_y_end = min(volume.shape[1], volume.shape[1] - shift_y)
            dst_y_start = max(0, shift_y)
            dst_y_end = min(volume.shape[1], volume.shape[1] + shift_y)
            
            src_x_start = max(0, -shift_x)
            src_x_end = min(volume.shape[2], volume.shape[2] - shift_x)
            dst_x_start = max(0, shift_x)
            dst_x_end = min(volume.shape[2], volume.shape[2] + shift_x)
            
            # Apply translation
            shifted_vol[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                volume[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
            
            shifted_mask[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                mask[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
            
            volume = shifted_vol
            mask = shifted_mask
        
        # Ensure arrays are contiguous before converting to tensors
        if not volume.flags.c_contiguous:
            volume = np.ascontiguousarray(volume)
        if not mask.flags.c_contiguous:
            mask = np.ascontiguousarray(mask)
            
        # Convert to torch tensors with channel dimension
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return volume_tensor, mask_tensor

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, weight=0.9):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight  # Increased weight for positive class to address class imbalance
        
    def forward(self, pred, target):
        # Flatten the tensors
        pred_flat = torch.sigmoid(pred).view(-1)
        target_flat = target.view(-1)
        
        # Calculate weighted intersection and union
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum() 
        target_sum = target_flat.sum()
        
        # Apply higher weight to positive class (nodules)
        weighted_intersection = intersection * self.weight
        weighted_union = pred_sum + target_sum * self.weight
        
        # Calculate weighted Dice coefficient
        dice = (2. * weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        return 1.0 - dice

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate Dice score for monitoring
        with torch.no_grad():
            pred = torch.sigmoid(output) > 0.5
            intersection = (pred * target).sum().item()
            union = pred.sum().item() + target.sum().item()
            dice = (2. * intersection) / (union + 1e-5)
            epoch_dice += dice
        
        # Update metrics
        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})
    
    avg_loss = epoch_loss / len(dataloader)
    avg_dice = epoch_dice / len(dataloader)
    return avg_loss, avg_dice

def validate(model, dataloader, criterion, device):
    """Validate the model on the validation set"""
    model.eval()
    val_loss = 0
    dice_score = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validation"):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Calculate Dice score for evaluation
            pred = torch.sigmoid(output) > 0.5
            intersection = (pred * target).sum().item()
            union = pred.sum().item() + target.sum().item()
            dice = (2. * intersection) / (union + 1e-5)
            
            # Update metrics
            val_loss += loss.item()
            dice_score += dice
    
    avg_loss = val_loss / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    return avg_loss, avg_dice

def main():
    parser = argparse.ArgumentParser(description='Train Nodule Segmentation Model')
    parser.add_argument('--data_dir', type=str, default='data/Luna16', help='Path to Luna16 dataset')
    parser.add_argument('--annotations', type=str, default='src/Nodules segmentation/processed_annotations.csv', 
                        help='Path to processed annotations CSV file')
    parser.add_argument('--output_dir', type=str, default='src/Nodules segmentation/models', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='src/Nodules segmentation/logs', 
                        help='Directory to save training logs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64], 
                        help='Patch size (z, y, x)')
    parser.add_argument('--limit_samples', type=int, default=0, 
                        help='Limit the number of samples for training (0 = use all samples)')
    parser.add_argument('--dice_weight', type=float, default=0.9, 
                        help='Weight for positive class in Dice loss (higher values emphasize nodule detection)')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use attention gates in UNet3D')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up file logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.log_dir) / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log training parameters
    logger.info(f"Training parameters: {args}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    transform = NoduleTransforms(p=0.7)
    dataset = NoduleDataset(
        data_dir=args.data_dir,
        annotations_file=args.annotations,
        patch_size=tuple(args.patch_size),
        transform=transform
    )
    
    # Limit samples if specified
    if args.limit_samples > 0 and args.limit_samples < len(dataset):
        logger.info(f"Limiting dataset to {args.limit_samples} samples (from {len(dataset)} total)")
        total_indices = list(range(len(dataset)))
        np.random.shuffle(total_indices)
        limited_indices = total_indices[:args.limit_samples]
        dataset = torch.utils.data.Subset(dataset, limited_indices)
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create enhanced UNet3D model with attention gates
    model = UNet3D(n_channels=1, n_classes=1, use_attention=args.use_attention)
    model.to(device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Create loss function and optimizer with weight decay
    criterion = DiceLoss(weight=args.dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Gradient scaler for mixed precision (if using GPU)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Train the model
    best_dice = 0.0
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train with mixed precision if on GPU
        if scaler:
            train_loss, train_dice = train_epoch_mixed_precision(model, train_loader, criterion, optimizer, device, scaler)
        else:
            train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation Dice score
        scheduler.step(val_dice)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model based on validation Dice score
        if val_dice > best_dice:
            best_dice = val_dice
            model_path = Path(args.output_dir) / f"best_model_dice_{timestamp}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }, model_path)
            logger.info(f"Saved best model (Dice: {val_dice:.4f}) to {model_path}")
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = Path(args.output_dir) / f"model_epoch_{epoch+1}_{timestamp}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }, model_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")

def train_epoch_mixed_precision(model, dataloader, criterion, optimizer, device, scaler):
    """Train the model for one epoch using mixed precision"""
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass and optimize with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate Dice score for monitoring
        with torch.no_grad():
            pred = torch.sigmoid(output) > 0.5
            intersection = (pred * target).sum().item()
            union = pred.sum().item() + target.sum().item()
            dice = (2. * intersection) / (union + 1e-5)
            epoch_dice += dice
        
        # Update metrics
        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})
    
    avg_loss = epoch_loss / len(dataloader)
    avg_dice = epoch_dice / len(dataloader)
    return avg_loss, avg_dice

if __name__ == "__main__":
    main() 