import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random

def load_ct_scan(filepath):
    """Load a CT scan and return as a SimpleITK image"""
    try:
        return sitk.ReadImage(str(filepath))
    except Exception as e:
        print(f"Error loading scan {filepath}: {e}")
        return None

def extract_patch(image, voxel_coord, patch_size=(64, 64, 64)):
    """Extract a patch around the nodule location"""
    # Get image size
    size = image.GetSize()
    
    # Calculate patch boundaries
    half_size = [p // 2 for p in patch_size]
    
    # Ensure coordinates are within bounds
    start_x = max(0, int(voxel_coord[0] - half_size[0]))
    start_y = max(0, int(voxel_coord[1] - half_size[1]))
    start_z = max(0, int(voxel_coord[2] - half_size[2]))
    
    # Adjust size to stay within image bounds
    size_x = min(patch_size[0], size[0] - start_x)
    size_y = min(patch_size[1], size[1] - start_y)
    size_z = min(patch_size[2], size[2] - start_z)
    
    # Extract the region using SimpleITK
    extract = sitk.RegionOfInterestImageFilter()
    extract.SetSize([size_x, size_y, size_z])
    extract.SetIndex([start_x, start_y, start_z])
    
    patch = extract.Execute(image)
    return patch, [start_x, start_y, start_z]

def create_nodule_mask(patch_array, center_in_patch, diameter_mm, spacing):
    """Create a binary mask for the nodule"""
    # Create an empty mask volume (note: SimpleITK uses [x,y,z] but numpy uses [z,y,x])
    shape = patch_array.shape  # This is in [z,y,x] order for numpy arrays
    mask = np.zeros(shape, dtype=np.float32)
    
    # Calculate radius in mm
    radius_mm = diameter_mm / 2.0
    
    # Generate coordinates grid (in [z,y,x] order for numpy indexing)
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(shape[0]), 
        np.arange(shape[1]), 
        np.arange(shape[2]), 
        indexing='ij'
    )
    
    # Center coordinates for the patch (adjusted for patch extraction)
    center_z, center_y, center_x = center_in_patch
    
    # Calculate Euclidean distance in mm (accounting for spacing)
    distance = np.sqrt(
        ((x_grid - center_x) * spacing[0]) ** 2 +
        ((y_grid - center_y) * spacing[1]) ** 2 +
        ((z_grid - center_z) * spacing[2]) ** 2
    )
    
    # Create spherical mask
    mask[distance <= radius_mm] = 1.0
    
    return mask

def visualize_nodule(ct_array, mask_array, original_spacing, save_path=None, show=False):
    """Visualize a nodule patch with mask overlay in three planes"""
    # Normalize the CT image for better visualization
    ct_min, ct_max = np.min(ct_array), np.max(ct_array)
    ct_normalized = (ct_array - ct_min) / (ct_max - ct_min)
    
    # Get the middle slices for each dimension
    z_mid, y_mid, x_mid = np.array(ct_array.shape) // 2
    
    # Create a figure with subplots for axial, coronal, and sagittal views
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Axial view (Z plane)
    axes[0, 0].imshow(ct_normalized[z_mid], cmap='gray')
    axes[0, 0].set_title(f'Axial CT (Z={z_mid})')
    axes[0, 0].axis('off')
    
    # Coronal view (Y plane)
    axes[0, 1].imshow(ct_normalized[:, y_mid, :], cmap='gray')
    axes[0, 1].set_title(f'Coronal CT (Y={y_mid})')
    axes[0, 1].axis('off')
    
    # Sagittal view (X plane)
    axes[0, 2].imshow(ct_normalized[:, :, x_mid], cmap='gray')
    axes[0, 2].set_title(f'Sagittal CT (X={x_mid})')
    axes[0, 2].axis('off')
    
    # Axial view with mask overlay
    axes[1, 0].imshow(ct_normalized[z_mid], cmap='gray')
    mask_overlay = np.ma.masked_where(mask_array[z_mid] < 0.5, mask_array[z_mid])
    axes[1, 0].imshow(mask_overlay, cmap='hot', alpha=0.5)
    axes[1, 0].set_title('Axial with Nodule Overlay')
    axes[1, 0].axis('off')
    
    # Coronal view with mask overlay
    axes[1, 1].imshow(ct_normalized[:, y_mid, :], cmap='gray')
    mask_overlay = np.ma.masked_where(mask_array[:, y_mid, :] < 0.5, mask_array[:, y_mid, :])
    axes[1, 1].imshow(mask_overlay, cmap='hot', alpha=0.5)
    axes[1, 1].set_title('Coronal with Nodule Overlay')
    axes[1, 1].axis('off')
    
    # Sagittal view with mask overlay
    axes[1, 2].imshow(ct_normalized[:, :, x_mid], cmap='gray')
    mask_overlay = np.ma.masked_where(mask_array[:, :, x_mid] < 0.5, mask_array[:, :, x_mid])
    axes[1, 2].imshow(mask_overlay, cmap='hot', alpha=0.5)
    axes[1, 2].set_title('Sagittal with Nodule Overlay')
    axes[1, 2].axis('off')
    
    # Add spacing information and other details
    plt.figtext(0.01, 0.01, f'Spacing (mm): {original_spacing[0]:.2f} x {original_spacing[1]:.2f} x {original_spacing[2]:.2f}')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def main():
    # Setup paths - Use direct paths to avoid issues
    current_dir = Path().absolute()
    root_dir = current_dir.parents[1] if 'Nodules segmentation' in str(current_dir) else current_dir
    
    data_dir = root_dir / "data" / "Luna16"
    annotations_file = current_dir / "processed_annotations.csv"
    output_dir = current_dir / "Visualizations" / "Annotations Samples"
    
    print(f"Current directory: {current_dir}")
    print(f"Using annotations file: {annotations_file}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed annotations
    print(f"Loading annotations from {annotations_file}")
    annotations = pd.read_csv(annotations_file)
    print(f"Loaded {len(annotations)} annotations")
    
    # Randomly select series with nodules
    n_samples = 10  # Number of samples to visualize
    series_uids = annotations['seriesuid'].unique()
    selected_series = random.sample(list(series_uids), min(n_samples, len(series_uids)))
    
    print(f"Selected {len(selected_series)} series for visualization")
    
    # Process each selected series
    for i, series_uid in enumerate(selected_series):
        print(f"Processing series {i+1}/{len(selected_series)}: {series_uid}")
        
        # Get annotations for this series
        series_annotations = annotations[annotations['seriesuid'] == series_uid]
        
        # Find the MHD file
        mhd_file = None
        for subset in range(10):
            potential_path = data_dir / f'subset{subset}' / f'{series_uid}.mhd'
            if potential_path.exists():
                mhd_file = potential_path
                break
        
        if mhd_file is None:
            print(f"Warning: MHD file not found for series {series_uid}, skipping")
            continue
        
        try:
            # Load the CT scan
            ct_image = load_ct_scan(mhd_file)
            if ct_image is None:
                continue
            
            # Get spacing
            spacing = ct_image.GetSpacing()
            
            # Get a sample of nodules from this series (max 3 per series)
            nodule_samples = series_annotations.sample(min(3, len(series_annotations)))
            
            for j, (_, nodule) in enumerate(nodule_samples.iterrows()):
                # Extract nodule information
                voxel_x = int(nodule['voxel_x'])
                voxel_y = int(nodule['voxel_y'])
                voxel_z = int(nodule['voxel_z'])
                voxel_coord = (voxel_x, voxel_y, voxel_z)
                diameter_mm = float(nodule['diameter_mm'])
                
                # Skip if out of bounds
                if nodule['out_of_bounds']:
                    print(f"  Skipping nodule {j+1} as it is out of bounds")
                    continue
                
                try:
                    # Extract patch around nodule
                    patch_size = (64, 64, 64)  # Desired size
                    ct_patch, patch_start = extract_patch(ct_image, voxel_coord, patch_size)
                    
                    # Convert to numpy for processing
                    patch_array = sitk.GetArrayFromImage(ct_patch)
                    
                    # Calculate nodule center within the patch
                    # Note: SimpleITK's coordinate system is (x,y,z) but numpy's is (z,y,x)
                    # So we need to flip the order for the mask generation
                    center_in_patch = (
                        voxel_z - patch_start[2],  # z
                        voxel_y - patch_start[1],  # y
                        voxel_x - patch_start[0]   # x
                    )
                    
                    # Create spherical mask
                    mask_array = create_nodule_mask(patch_array, center_in_patch, diameter_mm, spacing)
                    
                    # Create output filename
                    output_file = output_dir / f"{series_uid}_nodule_{j+1}_d{diameter_mm:.2f}mm.png"
                    
                    # Visualize and save
                    visualize_nodule(patch_array, mask_array, spacing, save_path=output_file)
                    print(f"  Saved nodule {j+1} visualization to {output_file}")
                    
                except Exception as e:
                    print(f"  Error processing nodule {j+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Error processing series {series_uid}: {e}")
    
    print(f"Completed visualization of {len(selected_series)} series")

if __name__ == "__main__":
    main() 