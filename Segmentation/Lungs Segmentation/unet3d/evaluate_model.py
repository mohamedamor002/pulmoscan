import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import LungDataset
from unet3d import UNet3D
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_model(model_path, device):
    """Load the trained model"""
    model = UNet3D(n_channels=1, n_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader, device, save_dir):
    """Evaluate model on validation set and save visualizations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            if i >= 5:  # Only evaluate first 5 cases
                break
                
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            series_id = batch['series_id'][0]
            
            # Get model prediction
            pred = model(ct)
            pred = torch.sigmoid(pred)
            
            # Convert to numpy for visualization
            ct_np = ct.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()
            
            # Save middle slice visualization
            mid_slice = ct_np.shape[0] // 2
            
            plt.figure(figsize=(15, 5))
            
            # Original CT
            plt.subplot(131)
            plt.imshow(ct_np[mid_slice], cmap='gray')
            plt.title('CT Scan')
            plt.axis('off')
            
            # Ground Truth
            plt.subplot(132)
            plt.imshow(mask_np[mid_slice], cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Prediction
            plt.subplot(133)
            plt.imshow(pred_np[mid_slice], cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            
            plt.suptitle(f'Case {series_id}')
            plt.savefig(save_dir / f'case_{series_id}.png')
            plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = LungDataset(
        data_dir='data/Luna16',
        mask_dir='data/Luna16',
        target_size=(128, 128, 128)
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = load_model('src/Lungs segmentation/checkpoints/best_model.pt', device)
    
    # Evaluate
    evaluate_model(model, val_loader, device, 'src/Lungs segmentation/evaluation_results')

if __name__ == '__main__':
    main() 