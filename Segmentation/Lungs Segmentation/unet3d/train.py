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

from unet3d import UNet3D
from dataset import LungDataset

def dice_score(y_true, y_pred):
    """Calculate Dice score for binary segmentation"""
    smooth = 1e-5
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    
    with tqdm(loader, desc="Training") as pbar:
        for batch in pbar:
            # Get data
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            pred = model(ct)
            loss = criterion(pred, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            # Get data
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            pred = model(ct)
            loss = criterion(pred, mask)
            
            # Calculate Dice score
            pred_binary = (pred > 0.5).float()
            dice = dice_score(mask.cpu().numpy().flatten(), 
                           pred_binary.cpu().numpy().flatten())
            dice_scores.append(dice)
            
            val_loss += loss.item()
    
    return val_loss / len(loader), np.mean(dice_scores)

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train 3D UNet for lung segmentation')
        parser.add_argument('--data_dir', type=str, default='data/Luna16',
                          help='Path to LUNA16 dataset')
        parser.add_argument('--batch_size', type=int, default=2,
                          help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=100,
                          help='Number of epochs to train')
        parser.add_argument('--lr', type=float, default=1e-4,
                          help='Learning rate')
        parser.add_argument('--device', type=str, default='cuda',
                          help='Device to use (cuda or cpu)')
        parser.add_argument('--save_dir', type=str, default='src/Lungs segmentation/checkpoints',
                          help='Directory to save checkpoints')
        args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Create dataset
    dataset = LungDataset(args.data_dir, args.data_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4)
    
    # Create model
    model = UNet3D(n_channels=1, n_classes=1).to(device)
    
    # Create loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    best_dice = 0
    train_losses = []
    val_losses = []
    dice_scores = []
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, dice = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        dice_scores.append(dice)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Dice Score: {dice:.4f}')
        
        # Save best model
        if dice > best_dice:
            best_dice = dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'dice_score': dice
            }, save_dir / 'best_model.pt')
            print(f'Saved best model with Dice score: {dice:.4f}')
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'dice_score': dice
        }, save_dir / 'latest_model.pt')
    
    # Plot training history
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
    plt.ylabel('Dice Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png')
    plt.close()

if __name__ == '__main__':
    main() 