"""
Setup script to create the expected directory structure for the lung nodule detection project.
This script creates the necessary directories for the models and copies any required files.
"""

import os
import sys
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure for the models."""
    # Get the parent directory (project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\Segmentation\pipeline
    root_dir = os.path.dirname(parent_dir)  # D:\Segmentation
    
    print(f"Current directory: {current_dir}")
    print(f"Pipeline directory: {parent_dir}")
    print(f"Root directory: {root_dir}")
    
    # Required directory structure
    required_dirs = [
        os.path.join(root_dir, 'src'),
        os.path.join(root_dir, 'src', 'Lungs segmentation'),
        os.path.join(root_dir, 'src', 'Nodules detection'),
        os.path.join(root_dir, 'src', 'Nodules segmentation'),
        os.path.join(root_dir, 'src', 'Lungs segmentation', 'checkpoints'),
        os.path.join(root_dir, 'src', 'Nodules detection', 'models'),
        os.path.join(root_dir, 'src', 'Nodules segmentation', 'models'),
    ]
    
    # Create the directories
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"Directory already exists: {directory}")

    # Check for combined_pipeline.py in various locations
    combined_pipeline_src = os.path.join(parent_dir, 'combined_pipeline.py')
    
    # First, check if it already exists in the pipeline directory
    if not os.path.exists(combined_pipeline_src):
        # Check if it's in the root directory
        root_pipeline_path = os.path.join(root_dir, 'combined_pipeline.py')
        if os.path.exists(root_pipeline_path):
            print(f"Found combined_pipeline.py at: {root_pipeline_path}")
            
            # No need to copy if it's already in the correct location
            print(f"The file is already in the correct location")
            combined_pipeline_src = root_pipeline_path
        else:
            # Search for it in the project
            print("\nSearching for combined_pipeline.py in the project...")
            found = False
            for root, dirs, files in os.walk(root_dir):
                if 'combined_pipeline.py' in files and root != parent_dir:
                    found = True
                    found_path = os.path.join(root, 'combined_pipeline.py')
                    print(f"Found combined_pipeline.py at: {found_path}")
                    
                    # No need to copy if it's already in the root directory
                    if os.path.dirname(found_path) == root_dir:
                        print(f"The file is already in the correct location")
                        combined_pipeline_src = found_path
                    else:
                        # Copy to root directory instead (user's preferred location)
                        root_pipeline_path = os.path.join(root_dir, 'combined_pipeline.py')
                        print(f"Copying combined_pipeline.py to: {root_pipeline_path}")
                        shutil.copy(found_path, root_pipeline_path)
                        combined_pipeline_src = root_pipeline_path
                    break
            
            if not found:
                print("Could not find combined_pipeline.py anywhere in the project")
                print("Please make sure the file exists before running the application")
    else:
        print(f"combined_pipeline.py already exists at: {combined_pipeline_src}")

    # Create placeholder unet3d.py files if they don't exist
    placeholder_text = """
# This is a placeholder UNet3D model file
# Please replace with the actual implementation

import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, input_channels=None, output_channels=None, use_attention=False):
        super(UNet3D, self).__init__()
        
        # Use the parameters however they are passed
        self.in_channels = n_channels if n_channels is not None else input_channels if input_channels is not None else 1
        self.out_channels = n_classes if n_classes is not None else output_channels if output_channels is not None else 1
        
        print(f"Initialized UNet3D with in_channels={self.in_channels}, out_channels={self.out_channels}")
        
        # Simple placeholder architecture
        self.encoder = nn.Sequential(
            nn.Conv3d(self.in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 2, stride=2),
            nn.Conv3d(16, self.out_channels, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out
"""

    unet_files = [
        os.path.join(root_dir, 'src', 'Lungs segmentation', 'unet3d.py'),
        os.path.join(root_dir, 'src', 'Nodules detection', 'unet3d.py'),
        os.path.join(root_dir, 'src', 'Nodules segmentation', 'unet3d.py'),
    ]
    
    for unet_file in unet_files:
        if not os.path.exists(unet_file):
            print(f"Creating placeholder UNet3D file: {unet_file}")
            with open(unet_file, 'w') as f:
                f.write(placeholder_text)
        else:
            print(f"UNet3D file already exists: {unet_file}")
    
    # Create placeholder model files if they don't exist
    model_files = [
        os.path.join(root_dir, 'src', 'Lungs segmentation', 'checkpoints', 'best_model.pt'),
        os.path.join(root_dir, 'src', 'Nodules detection', 'models', 'best_model_detection_only.pt'),
        os.path.join(root_dir, 'src', 'Nodules segmentation', 'models', 'best_model.pt'),
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"Note: Model file {model_file} does not exist.")
            print(f"  You will need to provide this file for the pipeline to work correctly.")
            print(f"  A placeholder can be created, but it will not be functional for actual predictions.")
            
            # Ask if user wants to create a placeholder
            response = input(f"Create a placeholder for {os.path.basename(model_file)}? (y/n): ")
            if response.lower() == 'y':
                print(f"Creating placeholder model file: {model_file}")
                # Create a dummy PyTorch model and save it
                dummy_model = torch.nn.Sequential(
                    torch.nn.Conv3d(1, 16, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv3d(16, 1, 1)
                )
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                torch.save(dummy_model.state_dict(), model_file)
                print(f"Created placeholder model file")
        else:
            print(f"Model file already exists: {model_file}")
    
    print("\nSetup complete! Directory structure has been created.")
    print("Note: If you created placeholder models, these will not produce meaningful results.")
    print("You should replace them with the actual trained models for production use.")

if __name__ == "__main__":
    try:
        import torch
        create_directory_structure()
    except ImportError:
        print("Error: PyTorch is required to create placeholder model files.")
        print("Please install PyTorch first: pip install torch")
        sys.exit(1) 