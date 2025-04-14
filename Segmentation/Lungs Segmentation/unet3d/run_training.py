import os
from pathlib import Path
import torch
import sys

from train import main

if __name__ == '__main__':
    # Set default parameters
    data_dir = 'data/Luna16'
    batch_size = 2
    epochs = 100
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints'
    
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)
    
    # Create argument list
    sys.argv = [sys.argv[0],  # Script name
                '--data_dir', data_dir,
                '--batch_size', str(batch_size),
                '--epochs', str(epochs),
                '--lr', str(learning_rate),
                '--device', device,
                '--save_dir', save_dir]
    
    # Run training
    main() 