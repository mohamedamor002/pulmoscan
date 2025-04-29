#!/usr/bin/env python
"""
Visualize model metrics and performance summary for the nodule detection model.
"""

import argparse
import logging
from pathlib import Path
import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchinfo import summary

from train import UNet3D, NoduleDataset, NoduleTransforms
from train import CombinedBCEDiceLoss, evaluate_preloaded, calculate_nodule_detection_accuracy

def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_training_history(checkpoint_path, output_dir):
    """
    Extract training metrics from checkpoint files in the same directory as the best model.
    
    Args:
        checkpoint_path: Path to the best model checkpoint
        output_dir: Directory to save visualizations
    """
    checkpoint_dir = checkpoint_path.parent
    
    # Try to find checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    checkpoint_files.sort(key=lambda f: int(f.stem.split('_')[-1]))
    
    if not checkpoint_files:
        print("No checkpoint files found for training history")
        return None
    
    # Extract metrics from checkpoints
    epochs = []
    train_losses = []
    val_losses = []
    val_dices = []
    val_detection_accs = []
    checkpoints_data = []  # Store full checkpoint data for detailed table
    
    for cp_file in checkpoint_files:
        try:
            cp = torch.load(cp_file, map_location='cpu', weights_only=True)
            if isinstance(cp, dict) and 'epoch' in cp:
                epochs.append(cp['epoch'] + 1)  # +1 because epochs are 0-indexed
                
                # Store all available metrics
                checkpoint_metrics = {
                    'epoch': cp['epoch'] + 1,
                    'filename': cp_file.name
                }
                
                # Extract common metrics
                for key in ['train_loss', 'val_loss', 'train_dice', 'val_dice', 
                           'val_detection_accuracy', 'train_detection_mse']:
                    if key in cp:
                        checkpoint_metrics[key] = cp[key]
                        
                        # Also add to our plotting arrays
                        if key == 'train_loss':
                            train_losses.append(cp[key])
                        elif key == 'val_loss':
                            val_losses.append(cp[key])
                        elif key == 'val_dice':
                            val_dices.append(cp[key])
                        elif key == 'val_detection_accuracy':
                            val_detection_accs.append(cp[key])
                
                # Append to our detailed data
                checkpoints_data.append(checkpoint_metrics)
        except Exception as e:
            print(f"Error loading checkpoint {cp_file}: {e}")
    
    # Create a table of checkpoint data
    if checkpoints_data:
        # Create a DataFrame for easier table generation
        df = pd.DataFrame(checkpoints_data)
        
        # Save to CSV
        csv_path = output_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
        
        # Generate HTML table
        html_table = df.to_html(index=False, border=1, classes='dataframe')
        with open(output_dir / 'training_history_table.html', 'w', encoding='utf-8') as f:
            f.write(f"""
            <html>
            <head>
                <style>
                    .dataframe {{
                        border-collapse: collapse;
                        margin: 25px 0;
                        font-size: 0.9em;
                        font-family: sans-serif;
                        min-width: 400px;
                        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                    }}
                    .dataframe thead tr {{
                        background-color: #009879;
                        color: #ffffff;
                        text-align: left;
                    }}
                    .dataframe th,
                    .dataframe td {{
                        padding: 12px 15px;
                    }}
                    .dataframe tbody tr {{
                        border-bottom: 1px solid #dddddd;
                    }}
                    .dataframe tbody tr:nth-of-type(even) {{
                        background-color: #f3f3f3;
                    }}
                    .dataframe tbody tr:last-of-type {{
                        border-bottom: 2px solid #009879;
                    }}
                </style>
            </head>
            <body>
                <h1>Epoch-by-Epoch Training History</h1>
                {html_table}
            </body>
            </html>
            """)
    
    # If we have history data, create plots
    if epochs:
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot training/validation loss
        if train_losses and val_losses:
            axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
            axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot validation dice score
        if val_dices:
            axes[1].plot(epochs, val_dices, 'g-', label='Validation Dice')
            axes[1].set_ylabel('Dice Score')
            axes[1].set_title('Validation Dice Score')
            axes[1].legend()
            axes[1].grid(True)
        
        # Plot validation detection accuracy
        if val_detection_accs:
            axes[2].plot(epochs, val_detection_accs, 'm-', label='Detection Accuracy')
            axes[2].set_ylabel('Accuracy (%)')
            axes[2].set_title('Validation Detection Accuracy')
            axes[2].legend()
            axes[2].grid(True)
        
        # Set common x-axis label
        axes[2].set_xlabel('Epoch')
        
        # Save the figure
        history_path = output_dir / 'training_history.png'
        plt.tight_layout()
        plt.savefig(history_path, dpi=150)
        plt.close()
        
        # Return history data
        return {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dices': val_dices,
            'val_detection_accs': val_detection_accs,
            'history_path': history_path,
            'epochs_table_path': output_dir / 'training_history_table.html',
            'epochs_csv_path': output_dir / 'training_history.csv',
            'checkpoints_data': checkpoints_data
        }
    
    return None

def test_model_performance(model, dataset, device, num_samples=50, output_dir=None):
    """
    Test model performance on a subset of samples and generate visualizations.
    
    Args:
        model: The UNet3D model
        dataset: NoduleDataset
        device: torch device
        num_samples: Number of samples to test
        output_dir: Directory to save visualizations
    """
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Track metrics
    dice_scores = []
    detection_accs = []
    gt_coords = []
    pred_coords = []
    radius_errors = []
    
    # Process each sample
    for idx in indices:
        # Get data
        volume, mask, det_target = dataset[idx]
        
        # Move to device
        volume = volume.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            mask_pred, det_pred = model(volume)
        
        # Calculate dice score
        pred_sigmoid = torch.sigmoid(mask_pred)
        pred_binary = (pred_sigmoid > 0.5).float()
        intersection = (pred_binary * mask).sum().item()
        dice = (2. * intersection) / (pred_binary.sum().item() + mask.sum().item() + 1e-8)
        
        # Calculate detection accuracy
        det_acc = calculate_nodule_detection_accuracy(
            det_pred, mask, volume_shape=volume.shape[2:]
        )
        
        # Store metrics
        dice_scores.append(dice)
        detection_accs.append(det_acc)
        
        # Store coordinates for analysis
        gt = det_target.cpu().numpy()
        pred = det_pred.squeeze(0).cpu().numpy()
        gt_coords.append(gt[:3])  # z, y, x
        pred_coords.append(pred[:3])  # z, y, x
        
        # Calculate radius error
        radius_errors.append(abs(gt[3] - pred[3]))
    
    # Create visualizations if output_dir is provided
    if output_dir:
        # Performance metrics summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Dice score histogram
        axes[0, 0].hist(dice_scores, bins=10, alpha=0.7, color='blue')
        axes[0, 0].set_title('Dice Score Distribution')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(dice_scores), color='r', linestyle='--', 
                         label=f'Mean: {np.mean(dice_scores):.3f}')
        axes[0, 0].legend()
        
        # Detection accuracy histogram
        axes[0, 1].hist(detection_accs, bins=10, alpha=0.7, color='green')
        axes[0, 1].set_title('Detection Accuracy Distribution')
        axes[0, 1].set_xlabel('Detection Accuracy (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(detection_accs), color='r', linestyle='--',
                         label=f'Mean: {np.mean(detection_accs):.1f}%')
        axes[0, 1].legend()
        
        # Coordinate error scatter plot (z vs distance error)
        gt_coords = np.array(gt_coords)
        pred_coords = np.array(pred_coords)
        coord_errors = np.sqrt(np.sum((gt_coords - pred_coords)**2, axis=1))
        
        axes[1, 0].scatter(gt_coords[:, 0], coord_errors, alpha=0.7)
        axes[1, 0].set_title('Coordinate Error vs Z-position')
        axes[1, 0].set_xlabel('Z-position (normalized)')
        axes[1, 0].set_ylabel('Coordinate Error (Euclidean)')
        
        # Radius error histogram
        axes[1, 1].hist(radius_errors, bins=10, alpha=0.7, color='purple')
        axes[1, 1].set_title('Nodule Radius Error Distribution')
        axes[1, 1].set_xlabel('Radius Error (normalized)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(radius_errors), color='r', linestyle='--',
                         label=f'Mean: {np.mean(radius_errors):.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        perf_path = output_dir / 'performance_metrics.png'
        plt.savefig(perf_path, dpi=150)
        plt.close()
        
        # Create summary report
        report = {
            'dice_score': {
                'mean': float(np.mean(dice_scores)),
                'median': float(np.median(dice_scores)),
                'std': float(np.std(dice_scores)),
                'min': float(np.min(dice_scores)),
                'max': float(np.max(dice_scores))
            },
            'detection_accuracy': {
                'mean': float(np.mean(detection_accs)),
                'median': float(np.median(detection_accs)),
                'std': float(np.std(detection_accs)),
                'min': float(np.min(detection_accs)),
                'max': float(np.max(detection_accs))
            },
            'coordinate_error': {
                'mean': float(np.mean(coord_errors)),
                'median': float(np.median(coord_errors)),
                'std': float(np.std(coord_errors))
            },
            'radius_error': {
                'mean': float(np.mean(radius_errors)),
                'median': float(np.median(radius_errors)),
                'std': float(np.std(radius_errors))
            }
        }
        
        with open(output_dir / 'performance_summary.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return {
            'dice_scores': dice_scores,
            'detection_accs': detection_accs,
            'coord_errors': coord_errors.tolist(),
            'radius_errors': radius_errors,
            'performance_path': perf_path,
            'summary': report
        }
    
    return {
        'dice_scores': dice_scores,
        'detection_accs': detection_accs
    }

def main():
    parser = argparse.ArgumentParser(description="Visualize model metrics and performance")
    parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", "-d", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output-dir", "-o", type=str, default="model_metrics", help="Output directory")
    parser.add_argument("--test-samples", "-t", type=int, default=50, help="Number of samples to test")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("metrics")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving metrics to {output_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    checkpoint_path = Path(args.model_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Generate model summary and save to file
    model_info = summary(model, input_size=(1, 1, 64, 64, 64), verbose=0)
    with open(output_dir / 'model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(str(model_info))
    
    logger.info(f"Model Parameters: {count_parameters(model):,}")
    
    # Extract training history from checkpoints
    logger.info("Extracting training history...")
    history = extract_training_history(checkpoint_path, output_dir)
    if history:
        logger.info(f"Training history visualization saved to {history['history_path']}")
    else:
        logger.warning("Could not extract training history from checkpoints")
    
    # Create dataset for testing
    logger.info("Creating test dataset...")
    dataset = NoduleDataset(
        data_dir=args.data_dir,
        annotations_file=Path(os.path.dirname(os.path.abspath(__file__))) / "processed_annotations.csv",
        patch_size=(64, 64, 64),
        transform=NoduleTransforms(p=0.0)  # No augmentation for testing
    )
    
    # Test model performance
    logger.info(f"Testing model on {args.test_samples} random samples...")
    performance = test_model_performance(
        model, dataset, device, num_samples=args.test_samples, output_dir=output_dir
    )
    
    # Generate HTML report with all metrics
    with open(output_dir / 'report.html', 'w', encoding='utf-8') as f:
        f.write(f"""
        <html>
        <head>
            <title>Nodule Detection Model Metrics</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric-box {{ 
                    background: #f5f5f5; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 10px;
                    min-width: 200px;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
                img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Nodule Detection Model Performance Report</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Model Architecture</h2>
            <p>Total Parameters: {count_parameters(model):,}</p>
            <pre>{model_info}</pre>
            
            <h2>Performance Summary</h2>
            <div class="metrics">
                <div class="metric-box">
                    <h3>Dice Score</h3>
                    <div class="metric-value">{performance['summary']['dice_score']['mean']:.3f}</div>
                    <p>Min: {performance['summary']['dice_score']['min']:.3f}, 
                       Max: {performance['summary']['dice_score']['max']:.3f}</p>
                </div>
                
                <div class="metric-box">
                    <h3>Detection Accuracy</h3>
                    <div class="metric-value">{performance['summary']['detection_accuracy']['mean']:.1f}%</div>
                    <p>Min: {performance['summary']['detection_accuracy']['min']:.1f}%, 
                       Max: {performance['summary']['detection_accuracy']['max']:.1f}%</p>
                </div>
                
                <div class="metric-box">
                    <h3>Coordinate Error</h3>
                    <div class="metric-value">{performance['summary']['coordinate_error']['mean']:.3f}</div>
                    <p>Median: {performance['summary']['coordinate_error']['median']:.3f}</p>
                </div>
                
                <div class="metric-box">
                    <h3>Radius Error</h3>
                    <div class="metric-value">{performance['summary']['radius_error']['mean']:.3f}</div>
                    <p>Median: {performance['summary']['radius_error']['median']:.3f}</p>
                </div>
            </div>
            
            <h2>Performance Metrics Visualization</h2>
            <img src="performance_metrics.png" alt="Performance Metrics">
            
            {"<h2>Training History</h2><img src='training_history.png' alt='Training History'>" if history else ""}
            
            <h2>Detailed Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
                <tr>
                    <td>Dice Score</td>
                    <td>{performance['summary']['dice_score']['mean']:.3f}</td>
                    <td>{performance['summary']['dice_score']['median']:.3f}</td>
                    <td>{performance['summary']['dice_score']['std']:.3f}</td>
                    <td>{performance['summary']['dice_score']['min']:.3f}</td>
                    <td>{performance['summary']['dice_score']['max']:.3f}</td>
                </tr>
                <tr>
                    <td>Detection Accuracy (%)</td>
                    <td>{performance['summary']['detection_accuracy']['mean']:.1f}</td>
                    <td>{performance['summary']['detection_accuracy']['median']:.1f}</td>
                    <td>{performance['summary']['detection_accuracy']['std']:.1f}</td>
                    <td>{performance['summary']['detection_accuracy']['min']:.1f}</td>
                    <td>{performance['summary']['detection_accuracy']['max']:.1f}</td>
                </tr>
                <tr>
                    <td>Coordinate Error</td>
                    <td>{performance['summary']['coordinate_error']['mean']:.3f}</td>
                    <td>{performance['summary']['coordinate_error']['median']:.3f}</td>
                    <td>{performance['summary']['coordinate_error']['std']:.3f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Radius Error</td>
                    <td>{performance['summary']['radius_error']['mean']:.3f}</td>
                    <td>{performance['summary']['radius_error']['median']:.3f}</td>
                    <td>{performance['summary']['radius_error']['std']:.3f}</td>
                    <td>{min(performance['radius_errors']):.3f}</td>
                    <td>{max(performance['radius_errors']):.3f}</td>
                </tr>
            </table>
        </body>
        </html>
        """)
    
    logger.info(f"Report generated at {output_dir / 'report.html'}")

if __name__ == "__main__":
    main()