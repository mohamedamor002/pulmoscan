import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import random
import cv2

# Import our preprocessing module
from preprocessing import preprocess_scan, get_nodule_mask

class NoduleDataset(Dataset):
    def __init__(self, data_dir, annotations_file, patch_size=(64, 64, 64), transform=None):
        """
        Args:
            data_dir (str): Directory with CT scans
            annotations_file (str): Path to CSV file with nodule annotations
            patch_size (tuple): Size of patches to extract
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = Path(data_dir)
        self.annotations = pd.read_csv(annotations_file)
        self.patch_size = patch_size
        self.transform = transform
        
        # Create a list of (series_uid, nodule_idx) pairs
        self.samples = []
        for series_uid, group in self.annotations.groupby('seriesuid'):
            for i in range(len(group)):
                self.samples.append((series_uid, i))
    
    def __len__(self):
        return len(self.samples)
    
    def load_volume(self, series_uid):
        """Load CT volume for a given series UID."""
        # Find the CT scan file
        ct_path = None
        for subset in range(10):
            potential_path = self.data_dir / f'subset{subset}' / f'{series_uid}.mhd'
            if potential_path.exists():
                ct_path = potential_path
                break
        
        if ct_path is None:
            raise FileNotFoundError(f"CT scan not found for series {series_uid}")
        
        # Use our enhanced preprocessing function
        ct_array, mask_array, metadata = preprocess_scan(str(ct_path), mask_path=None)
        
        # Get spacing and origin
        spacing = metadata['original_spacing']  # This is still in (x,y,z) order
        
        # Get original origin - we need to read the image for this
        ct_image = sitk.ReadImage(str(ct_path))
        origin = ct_image.GetOrigin()
        
        # Print debug info about the scan
        print(f"Series UID: {series_uid}")
        print(f"CT Array shape: {ct_array.shape}")
        print(f"CT Spacing: {spacing}")
        print(f"CT Origin: {origin}")
        print(f"CT Direction: {ct_image.GetDirection()}")
        
        # Note: ct_array is already normalized to [-1, 1] by preprocess_scan
        # If a different normalization is needed, adjust here
        
        return ct_array, spacing, origin
    
    def create_nodule_mask(self, volume_shape, annotations, spacing, origin):
        """Create binary mask for nodules."""
        mask = np.zeros(volume_shape, dtype=np.float32)
        
        # SimpleITK returns spacing and origin in (x,y,z) order
        # But the array from GetArrayFromImage is in (z,y,x) order
        print(f"Volume shape: {volume_shape} - this is in (z,y,x) order")
        print(f"Spacing: {spacing} - this is in (x,y,z) order")
        print(f"Origin: {origin} - this is in (x,y,z) order")
        
        for _, row in annotations.iterrows():
            # Get nodule center in world coordinates (mm)
            world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
            
            # SimpleITK uses array indexing as [z][y][x] but world coordinates as (x,y,z)
            # We need to convert world coordinates to ITK physical point to index correctly
            # Create temporary image to use SimpleITK's built-in coordinate transformation
            reference_image = sitk.Image(
                int(volume_shape[2]), int(volume_shape[1]), int(volume_shape[0]), 
                sitk.sitkUInt8
            )
            reference_image.SetSpacing(spacing)
            reference_image.SetOrigin(origin)
            
            # Use SimpleITK's TransformPhysicalPointToIndex for accurate world->voxel conversion
            try:
                physical_point = (float(world_coord[0]), float(world_coord[1]), float(world_coord[2]))
                index = reference_image.TransformPhysicalPointToIndex(physical_point)
                # Convert to (z,y,x) order for numpy indexing
                center_voxel = np.array([index[2], index[1], index[0]])
            except Exception as e:
                print(f"Error converting world to voxel coordinates: {e}")
                print(f"  World coord: {world_coord}, Origin: {origin}, Spacing: {spacing}")
                continue
            
            # Get nodule radius in voxels (using the smallest spacing for spherical shape)
            radius_voxel = int(row['diameter_mm'] / (2 * min(spacing)))
            
            # Check if the nodule center is within the volume
            if not (0 <= center_voxel[0] < volume_shape[0] and 
                   0 <= center_voxel[1] < volume_shape[1] and 
                   0 <= center_voxel[2] < volume_shape[2]):
                print(f"Warning: Nodule center {center_voxel} is outside volume shape {volume_shape}")
                print(f"  World coord: {world_coord}, Origin: {origin}, Spacing: {spacing}")
                print(f"  Series UID: {row['seriesuid']}, Nodule ID: {row.get('nodule_id', 'unknown')}")
                continue
            
            # Create a sphere for the nodule
            z, y, x = np.ogrid[-radius_voxel:radius_voxel+1,
                              -radius_voxel:radius_voxel+1,
                              -radius_voxel:radius_voxel+1]
            sphere = (x*x + y*y + z*z) <= (radius_voxel*radius_voxel)
            
            # Calculate boundaries for placing the sphere
            z_start = max(0, int(center_voxel[0]) - radius_voxel)
            z_end = min(volume_shape[0], int(center_voxel[0]) + radius_voxel + 1)
            y_start = max(0, int(center_voxel[1]) - radius_voxel)
            y_end = min(volume_shape[1], int(center_voxel[1]) + radius_voxel + 1)
            x_start = max(0, int(center_voxel[2]) - radius_voxel)
            x_end = min(volume_shape[2], int(center_voxel[2]) + radius_voxel + 1)
            
            # Calculate sphere boundaries
            sz_start = max(0, radius_voxel - int(center_voxel[0]))
            sz_end = min(2*radius_voxel+1, 2*radius_voxel+1 - (int(center_voxel[0]) + radius_voxel + 1 - volume_shape[0]))
            sy_start = max(0, radius_voxel - int(center_voxel[1]))
            sy_end = min(2*radius_voxel+1, 2*radius_voxel+1 - (int(center_voxel[1]) + radius_voxel + 1 - volume_shape[1]))
            sx_start = max(0, radius_voxel - int(center_voxel[2]))
            sx_end = min(2*radius_voxel+1, 2*radius_voxel+1 - (int(center_voxel[2]) + radius_voxel + 1 - volume_shape[2]))
            
            # Ensure valid sphere region
            if (sz_end <= sz_start or sy_end <= sy_start or sx_end <= sx_start or
                z_end <= z_start or y_end <= y_start or x_end <= x_start):
                print(f"Warning: Invalid sphere region for nodule at {center_voxel}")
                continue
            
            try:
                sphere_region = sphere[sz_start:sz_end, sy_start:sy_end, sx_start:sx_end]
                if sphere_region.size > 0:  # Only place sphere if it has valid size
                    mask[z_start:z_end, y_start:y_end, x_start:x_end] = np.maximum(
                        mask[z_start:z_end, y_start:y_end, x_start:x_end],
                        sphere_region
                    )
                    print(f"Successfully placed nodule at {center_voxel} with radius {radius_voxel}")
            except ValueError as e:
                print(f"Warning: Could not place nodule at {center_voxel}: {e}")
                continue
        
        return mask
    
    def extract_patch(self, volume, mask, center):
        """Extract a patch around the nodule center."""
        z, y, x = center
        dz, dy, dx = [s//2 for s in self.patch_size]
        
        # Calculate patch boundaries
        z_start = max(0, z - dz)
        z_end = min(volume.shape[0], z + dz)
        y_start = max(0, y - dy)
        y_end = min(volume.shape[1], y + dy)
        x_start = max(0, x - dx)
        x_end = min(volume.shape[2], x + dx)
        
        # Extract patches
        volume_patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        mask_patch = mask[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if volume_patch.shape != self.patch_size:
            padded_volume = np.zeros(self.patch_size, dtype=volume.dtype)
            padded_mask = np.zeros(self.patch_size, dtype=mask.dtype)
            
            z_size = min(volume_patch.shape[0], self.patch_size[0])
            y_size = min(volume_patch.shape[1], self.patch_size[1])
            x_size = min(volume_patch.shape[2], self.patch_size[2])
            
            padded_volume[:z_size, :y_size, :x_size] = volume_patch[:z_size, :y_size, :x_size]
            padded_mask[:z_size, :y_size, :x_size] = mask_patch[:z_size, :y_size, :x_size]
            
            volume_patch = padded_volume
            mask_patch = padded_mask
        
        return volume_patch, mask_patch
    
    def get_nodule_centers(self, annotations, spacing, origin):
        """Get nodule centers in voxel coordinates."""
        centers = []
        for _, row in annotations.iterrows():
            # Get nodule center in world coordinates (mm)
            world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
            
            # Create temporary image to use SimpleITK's built-in coordinate transformation
            # Use the first annotated volume shape as reference (this is just for getting indices)
            reference_shape = (512, 512, 512)  # Reasonable default for CT
            reference_image = sitk.Image(
                int(reference_shape[2]), int(reference_shape[1]), int(reference_shape[0]), 
                sitk.sitkUInt8
            )
            reference_image.SetSpacing(spacing)
            reference_image.SetOrigin(origin)
            
            # Use SimpleITK's TransformPhysicalPointToIndex for accurate world->voxel conversion
            try:
                physical_point = (float(world_coord[0]), float(world_coord[1]), float(world_coord[2]))
                index = reference_image.TransformPhysicalPointToIndex(physical_point)
                # Convert to (z,y,x) order for numpy indexing
                center_voxel = np.array([index[2], index[1], index[0]])
                centers.append(center_voxel.astype(int))
            except Exception as e:
                print(f"Error converting world to voxel coordinates: {e}")
                print(f"  World coord: {world_coord}, Origin: {origin}, Spacing: {spacing}")
                print(f"  Series UID: {row['seriesuid']}, Nodule ID: {row.get('nodule_id', 'unknown')}")
        
        return centers
    
    def __getitem__(self, idx):
        series_uid, nodule_idx = self.samples[idx]
        annotations = self.annotations[self.annotations['seriesuid'] == series_uid]
        
        # Load CT volume and create mask
        ct_array, spacing, origin = self.load_volume(series_uid)
        mask = self.create_nodule_mask(ct_array.shape, annotations, spacing, origin)
        
        # Get nodule centers in voxel coordinates
        centers = self.get_nodule_centers(annotations, spacing, origin)
        center = centers[nodule_idx]
        
        # Extract patch for the selected nodule
        volume_patch, mask_patch = self.extract_patch(ct_array, mask, center)
        
        # Convert to tensors
        volume_patch = torch.FloatTensor(volume_patch).unsqueeze(0)  # Add channel dimension
        mask_patch = torch.FloatTensor(mask_patch).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            volume_patch = self.transform(volume_patch)
        
        return volume_patch, mask_patch

    def get_random_negative_center(self, volume_shape: Tuple[int, int, int], 
                                 nodule_centers: List[Tuple[int, int, int]], 
                                 min_distance: int = 32) -> Tuple[int, int, int]:
        """Get a random center for a negative sample, away from nodules"""
        while True:
            x = random.randint(0, volume_shape[0]-1)
            y = random.randint(0, volume_shape[1]-1)
            z = random.randint(0, volume_shape[2]-1)
            
            # Check if this point is far enough from all nodules
            is_far = True
            for center in nodule_centers:
                distance = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                if distance < min_distance:
                    is_far = False
                    break
            
            if is_far:
                return (x, y, z) 