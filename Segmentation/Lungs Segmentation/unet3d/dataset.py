import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional

class LungDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 mask_dir: str,
                 transform: Optional[callable] = None,
                 target_size: Tuple[int, int, int] = (128, 128, 128)):
        """
        Dataset for lung CT scans and their segmentation masks
        
        Args:
            data_dir: Directory containing CT scans
            mask_dir: Directory containing lung masks
            transform: Optional transform to be applied on a sample
            target_size: Target size for resizing the volumes
        """
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir) / "seg-lungs-LUNA16" / "seg-lungs-LUNA16"
        self.transform = transform
        self.target_size = target_size
        
        # Get all CT scan files
        self.ct_files = []
        for subset in range(10):
            subset_dir = self.data_dir / f"subset{subset}"
            if subset_dir.exists():
                self.ct_files.extend(list(subset_dir.glob("*.mhd")))
                
        print(f"Found {len(self.ct_files)} CT scans")
        
    def __len__(self):
        return len(self.ct_files)
    
    def load_volume(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a CT scan and return the image array and spacing"""
        itk_image = sitk.ReadImage(str(filepath))
        image_array = sitk.GetArrayFromImage(itk_image)
        spacing = np.array(itk_image.GetSpacing())
        return image_array, spacing
    
    def load_mask(self, series_id: str) -> np.ndarray:
        """Load the corresponding lung mask"""
        mask_path = self.mask_dir / f"{series_id}.mhd"
        if not mask_path.exists():
            raise FileNotFoundError(f"No mask found for series {series_id}")
        
        itk_mask = sitk.ReadImage(str(mask_path))
        mask_array = sitk.GetArrayFromImage(itk_mask)
        return mask_array
    
    def preprocess_ct(self, ct_array: np.ndarray) -> np.ndarray:
        """Preprocess CT scan"""
        # Clip to reasonable HU range
        ct_array = np.clip(ct_array, -1000, 400)
        
        # Normalize to [0, 1]
        ct_array = (ct_array - (-1000)) / (400 - (-1000))
        ct_array = np.clip(ct_array, 0, 1)
        
        return ct_array
    
    def preprocess_mask(self, mask_array: np.ndarray) -> np.ndarray:
        """Preprocess mask"""
        # Ensure binary mask
        mask_array = (mask_array > 0).astype(np.float32)
        return mask_array
    
    def resize_volume(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize a 3D volume to target shape"""
        # Convert numpy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(volume)
        
        # Calculate resize factors
        original_size = sitk_image.GetSize()
        scale_factors = [float(t)/float(o) for t, o in zip(target_shape, original_size)]
        
        # Create resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_shape)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetOutputSpacing([o/s for o, s in zip(sitk_image.GetSpacing(), scale_factors)])
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        # Perform resampling
        resampled_image = resampler.Execute(sitk_image)
        
        # Convert back to numpy array
        resampled_array = sitk.GetArrayFromImage(resampled_image)
        return resampled_array
    
    def __getitem__(self, idx):
        # Load CT scan
        ct_path = self.ct_files[idx]
        series_id = ct_path.stem
        ct_array, spacing = self.load_volume(ct_path)
        
        # Load mask
        mask_array = self.load_mask(series_id)
        
        # Preprocess CT and mask
        ct_array = self.preprocess_ct(ct_array)
        mask_array = self.preprocess_mask(mask_array)
        
        # Resize both CT and mask to target size
        ct_array = self.resize_volume(ct_array, self.target_size)
        mask_array = self.resize_volume(mask_array, self.target_size)
        
        # Ensure values are in [0, 1] after resizing
        ct_array = np.clip(ct_array, 0, 1)
        mask_array = np.clip(mask_array, 0, 1)
        
        # Convert to tensors
        ct_tensor = torch.from_numpy(ct_array).float().unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            ct_tensor = self.transform(ct_tensor)
            mask_tensor = self.transform(mask_tensor)
        
        return {
            'ct': ct_tensor,
            'mask': mask_tensor,
            'series_id': series_id,
            'spacing': spacing
        } 