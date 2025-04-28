import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from torch.cuda.amp import GradScaler, autocast
import gc  # Import garbage collector

# Local imports
from vnet import VNet
from dataset import LungDataset

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid inside Dice loss

        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice  # because we minimize loss

class CombinedLoss(nn.Module):
    """Combined Dice and BCE Loss"""
    def __init__(self, dice_weight=0.7):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()  # Updated here
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (self.dice_weight * self.dice(pred, target) +
                (1 - self.dice_weight) * self.bce(pred, target))

def dice_score(pred, target, smooth=1e-5):
    """Calculate Dice score for binary segmentation"""
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    epoch_loss = 0
    
    with tqdm(loader, desc="Training") as pbar:
        for batch in pbar:
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                pred = model(ct)
                loss = criterion(pred, mask)
            
            if scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Clean up GPU memory
            del ct, mask, pred, loss
            torch.cuda.empty_cache()
            gc.collect()
    
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            
            pred = model(ct)
            loss = criterion(pred, mask)
            
            pred_binary = (pred > 0.5).float()
            dice = dice_score(pred_binary, mask)
            dice_scores.append(dice.item())
            
            val_loss += loss.item()
            
            # Clean up GPU memory
            del ct, mask, pred, pred_binary, loss, dice
            torch.cuda.empty_cache()
            gc.collect()
    
    return val_loss / len(loader), np.mean(dice_scores)

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train V-Net for lung segmentation')
        parser.add_argument('--batch_size', type=int, default=2,
                          help='Batch size for training (default: 2)')
        parser.add_argument('--epochs', type=int, default=200,
                          help='Number of epochs to train (default: 200)')
        parser.add_argument('--lr', type=float, default=1e-5,
                          help='Learning rate (default: 1e-5)')
        parser.add_argument('--device', type=str, default='cuda',
                          help='Device to use (cuda or cpu)')
        parser.add_argument('--amp', action='store_true',
                          help='Enable mixed precision training')
        args = parser.parse_args()
    
    # Setup paths and device
    BASE_DIR = Path(__file__).parent.parent  # src/Lungs segmentation
    DATA_DIR = BASE_DIR.parent.parent / "data" / "Luna16"  # ../../../data/Luna16
    MODEL_SAVE_DIR = BASE_DIR / "Vnet" / "models"
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models will be saved to: {MODEL_SAVE_DIR}")

    # Create datasets
    dataset = LungDataset(DATA_DIR, DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=max(1, args.batch_size // 2),
        shuffle=True, num_workers=2, pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=max(1, args.batch_size // 2),
        shuffle=False, num_workers=1, pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model and training components
    model = VNet(in_channels=1, out_channels=1).to(device)
    criterion = CombinedLoss(dice_weight=0.7)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
    scaler = GradScaler() if args.amp else None
    
    # Training variables
    best_dice = 0
    early_stop_counter = 0
    train_losses, val_losses, dice_scores = [], [], []
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Training phase
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validation phase
        val_loss, dice = validate(model, val_loader, criterion, device)
        scheduler.step(dice)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        dice_scores.append(dice)
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {dice:.4f}')
        
        # Save best model
        if dice > best_dice:
            best_dice = dice
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'dice': dice,
                'args': vars(args)
            }, MODEL_SAVE_DIR / 'vnet_best_model.pth')
            print(f'✅ Saved new best model with Dice: {dice:.4f}')
        else:
            early_stop_counter += 1
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'dice': dice,
            'args': vars(args)
        }, MODEL_SAVE_DIR / 'vnet_latest_model.pth')
        
        # Early stopping
        if early_stop_counter >= 15:
            print("🛑 Early stopping triggered!")
            break
    
    # Save training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(dice_scores, label='Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.savefig(MODEL_SAVE_DIR / 'vnet_training_history.png')
    plt.close()
    print(f"📊 Training curves saved to {MODEL_SAVE_DIR / 'vnet_training_history.png'}")

if __name__ == '__main__':
    main()