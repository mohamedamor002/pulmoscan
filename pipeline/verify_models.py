"""
Verification script for the lung nodule detection models and dependencies.
This script checks if all required model files and directories exist.
"""

import os
import sys
from pathlib import Path

def verify_structure():
    """Verify that all required files and directories exist."""
    # Get the parent directory (project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\Segmentation\pipeline
    root_dir = os.path.dirname(parent_dir)  # D:\Segmentation
    
    print(f"Current directory: {current_dir}")
    print(f"Pipeline directory: {parent_dir}")
    print(f"Root directory: {root_dir}")
    
    # Required directories
    required_dirs = [
        os.path.join(root_dir, 'src'),
        os.path.join(root_dir, 'src', 'Lungs segmentation'),
        os.path.join(root_dir, 'src', 'Nodules detection'),
        os.path.join(root_dir, 'src', 'Nodules segmentation'),
        os.path.join(root_dir, 'src', 'Lungs segmentation', 'checkpoints'),
        os.path.join(root_dir, 'src', 'Nodules detection', 'models'),
        os.path.join(root_dir, 'src', 'Nodules segmentation', 'models'),
    ]
    
    # Required model files
    required_files = [
        os.path.join(root_dir, 'src', 'Lungs segmentation', 'unet3d.py'),
        os.path.join(root_dir, 'src', 'Nodules detection', 'unet3d.py'),
        os.path.join(root_dir, 'src', 'Nodules segmentation', 'unet3d.py'),
        os.path.join(root_dir, 'src', 'Lungs segmentation', 'checkpoints', 'best_model.pt'),
        os.path.join(root_dir, 'src', 'Nodules detection', 'models', 'best_model_detection_only.pt'),
        os.path.join(root_dir, 'src', 'Nodules segmentation', 'models', 'best_model.pt'),
        os.path.join(parent_dir, 'combined_pipeline.py'),
    ]
    
    # Check for pipeline file specifically
    pipeline_file = os.path.join(parent_dir, 'combined_pipeline.py')
    if not os.path.exists(pipeline_file):
        print(f"Combined pipeline file not found at expected location: {pipeline_file}")
        print("Checking alternative location...")
        
        # Check if it's directly in the root directory
        alt_pipeline_file = os.path.join(root_dir, 'combined_pipeline.py')
        if os.path.exists(alt_pipeline_file):
            print(f"Found pipeline file at: {alt_pipeline_file}")
            # Replace the entry in required_files
            for i, file_path in enumerate(required_files):
                if file_path == pipeline_file:
                    required_files[i] = alt_pipeline_file
                    break
            pipeline_file = alt_pipeline_file
    
    # Check directories
    print("\nChecking required directories:")
    missing_dirs = []
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} (MISSING)")
            missing_dirs.append(directory)
    
    # Check files
    print("\nChecking required files:")
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (MISSING)")
            missing_files.append(file_path)
    
    # Search for the combined_pipeline.py file if it's missing
    if pipeline_file in missing_files:
        print("\nSearching for combined_pipeline.py in the project...")
        found = False
        for root, dirs, files in os.walk(root_dir):
            if 'combined_pipeline.py' in files:
                found = True
                found_path = os.path.join(root, 'combined_pipeline.py')
                print(f"Found combined_pipeline.py at: {found_path}")
                break
        
        if not found:
            print("Could not find combined_pipeline.py anywhere in the project")
    
    # Summary
    print("\nSummary:")
    if missing_dirs:
        print(f"Missing {len(missing_dirs)} directories:")
        for directory in missing_dirs:
            print(f"  - {directory}")
    else:
        print("All required directories exist.")
    
    if missing_files:
        print(f"Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    else:
        print("All required files exist.")
    
    if not missing_dirs and not missing_files:
        print("\nAll required files and directories exist! The models should work correctly.")
        return True
    else:
        print("\nSome required files or directories are missing.")
        print("Please ensure that the project structure is correct and all model files are present.")
        return False

if __name__ == "__main__":
    verify_structure() 