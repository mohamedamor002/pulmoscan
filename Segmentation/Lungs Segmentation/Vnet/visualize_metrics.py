#!/usr/bin/env python3
import os
import time
import json
import argparse
import torch
from dataset import LungDataset
from vnet import VNet

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'vnet_best_model.pth')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'model_metrics')
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create metrics directory if it doesn't exist


def dice_coefficient(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice.item()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VNet metrics on Lung CT scans")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CT scan .mhd files')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing lung mask .mhd files')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the model weights')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to run')
    parser.add_argument('--output_path', type=str, default=os.path.join(OUTPUT_DIR, 'metrics.json'), help='Path to save metrics JSON')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for computation')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    model = VNet(in_channels=1, out_channels=1)
    model.to(device)
    model.eval()

    # Load checkpoint and properly unwrap it
    print(f"Loading model from {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        print("Found model_state_dict in checkpoint")
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        print("Found state_dict in checkpoint")
        state_dict = ckpt['state_dict']
    else:
        print("Using checkpoint directly as state_dict")
        state_dict = ckpt
    model.load_state_dict(state_dict)

    dataset = LungDataset(data_dir=args.data_dir, mask_dir=args.mask_dir)
    sample_count = min(len(dataset), args.num_samples)
    print(f"Found {len(dataset)} samples, processing {sample_count}")

    dice_scores = []
    runtimes = []

    for idx in range(sample_count):
        sample = dataset[idx]
        ct = sample['ct'].unsqueeze(0).to(device)
        mask = sample['mask'].to(device)
        with torch.no_grad():
            start = time.time()
            output = model(ct)
            runtime = time.time() - start
        runtimes.append(runtime)

        prob = torch.sigmoid(output)
        pred = (prob > 0.5).float()
        dice = dice_coefficient(pred, mask)
        dice_scores.append(dice)

        print(f"Sample {idx+1}/{sample_count}: Dice={dice:.4f}, Runtime={runtime:.4f}s")

    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0
    model_size_bytes = os.path.getsize(args.model_path)
    model_size_mb = model_size_bytes / (1024**2)
    num_params = sum(p.numel() for p in model.parameters())

    metrics = {
        'dice_scores': dice_scores,
        'avg_dice': sum(dice_scores) / len(dice_scores) if dice_scores else 0.0,
        'avg_runtime': avg_runtime,
        'model_size_mb': model_size_mb,
        'num_params': num_params
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nAverage dice score: {metrics['avg_dice']:.4f}")
    print(f"Average runtime per CT scan: {avg_runtime:.4f}s")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Number of parameters: {num_params}")
    print(f"Metrics saved to {args.output_path}")

if __name__ == '__main__':
    main()