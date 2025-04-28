import os
import sys
import pandas as pd
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

def read_mhd_metadata(file_path):
    """
    Read metadata from MHD file
    
    Args:
        file_path: Path to MHD file
        
    Returns:
        dict: Metadata dictionary containing:
            - origin: Origin of the image (x,y,z)
            - spacing: Spacing of the image (x,y,z)
            - size: Size of the image (x,y,z)
            - transform_matrix: Transformation matrix
    """
    itk_image = sitk.ReadImage(str(file_path))
    metadata = {
        'origin': np.array(itk_image.GetOrigin()),
        'spacing': np.array(itk_image.GetSpacing()),
        'size': np.array(itk_image.GetSize()),
        'transform_matrix': np.array(itk_image.GetDirection()).reshape(3, 3),
        'orientation': itk_image.GetMetaData('AnatomicalOrientation') if itk_image.HasMetaDataKey('AnatomicalOrientation') else 'RAI'
    }
    return metadata

def world_to_voxel(world_coord, origin, spacing, transform_matrix=None):
    """
    Convert world coordinates to voxel coordinates
    
    Args:
        world_coord: World coordinates (x,y,z)
        origin: Origin of the image (x,y,z)
        spacing: Spacing of the image (x,y,z)
        transform_matrix: Transformation matrix (3x3)
        
    Returns:
        tuple: Voxel coordinates as (x,y,z) for image indexing
    """
    # Apply transform matrix if provided
    if transform_matrix is not None and not np.array_equal(transform_matrix, np.eye(3)):
        # Inverse the matrix to transform from world to image coordinates
        inv_matrix = np.linalg.inv(transform_matrix)
        world_coord_centered = world_coord - origin
        transformed_coord = np.matmul(inv_matrix, world_coord_centered)
        voxel_coord = transformed_coord / spacing
    else:
        # Direct conversion when transform is identity matrix
        voxel_coord = (world_coord - origin) / spacing
    
    # Round to nearest integer and convert to int
    voxel_coord = np.round(voxel_coord).astype(int)
    
    # Return as (x, y, z) for image indexing
    return tuple(voxel_coord)

def check_voxel_in_bounds(voxel_coord, image_size):
    """
    Check if voxel coordinates are within image bounds
    
    Args:
        voxel_coord: Voxel coordinates (x,y,z)
        image_size: Image size (x,y,z)
        
    Returns:
        bool: True if coordinates are within bounds, False otherwise
    """
    return (0 <= voxel_coord[0] < image_size[0] and 
            0 <= voxel_coord[1] < image_size[1] and 
            0 <= voxel_coord[2] < image_size[2])

def main():
    # Setup paths
    project_root = Path().absolute()
    data_dir = project_root / "data" / "Luna16"
    annotations_file = data_dir / "annotations.csv"
    output_file = project_root / "src" / "Nodules segmentation" / "processed_annotations.csv"
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}")
    annotations_df = pd.read_csv(annotations_file)
    print(f"Loaded {len(annotations_df)} annotations")
    
    # Initialize result dataframe
    result_columns = [
        'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
        'voxel_x', 'voxel_y', 'voxel_z', 'out_of_bounds',
        'spacing_x', 'spacing_y', 'spacing_z', 
        'origin_x', 'origin_y', 'origin_z',
        'image_size_x', 'image_size_y', 'image_size_z'
    ]
    result_df = pd.DataFrame(columns=result_columns)
    
    # Process each unique series
    series_uids = annotations_df['seriesuid'].unique()
    print(f"Processing {len(series_uids)} unique series")
    
    # Cache for metadata to avoid repeated reading
    metadata_cache = {}
    
    # Iterate through each annotation
    processed_rows = []
    
    for _, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Processing annotations"):
        series_uid = row['seriesuid']
        
        # Get metadata
        if series_uid not in metadata_cache:
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
            
            # Read metadata
            try:
                metadata_cache[series_uid] = read_mhd_metadata(mhd_file)
            except Exception as e:
                print(f"Error reading metadata for {series_uid}: {e}")
                continue
        
        # Get cached metadata
        metadata = metadata_cache[series_uid]
        
        # Get world coordinates
        world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
        
        # Convert to voxel coordinates
        try:
            voxel_coord = world_to_voxel(
                world_coord, 
                metadata['origin'], 
                metadata['spacing'], 
                metadata['transform_matrix']
            )
            
            # Check if voxel coordinates are within bounds
            out_of_bounds = not check_voxel_in_bounds(voxel_coord, metadata['size'])
            
            # Create result row
            result_row = {
                'seriesuid': series_uid,
                'coordX': row['coordX'],
                'coordY': row['coordY'],
                'coordZ': row['coordZ'],
                'diameter_mm': row['diameter_mm'],
                'voxel_x': voxel_coord[0],
                'voxel_y': voxel_coord[1],
                'voxel_z': voxel_coord[2],
                'out_of_bounds': out_of_bounds,
                'spacing_x': metadata['spacing'][0],
                'spacing_y': metadata['spacing'][1],
                'spacing_z': metadata['spacing'][2],
                'origin_x': metadata['origin'][0],
                'origin_y': metadata['origin'][1],
                'origin_z': metadata['origin'][2],
                'image_size_x': metadata['size'][0],
                'image_size_y': metadata['size'][1],
                'image_size_z': metadata['size'][2]
            }
            
            processed_rows.append(result_row)
            
        except Exception as e:
            print(f"Error processing annotation {series_uid}: {e}")
    
    # Create result dataframe
    result_df = pd.DataFrame(processed_rows)
    
    # Save result
    result_df.to_csv(output_file, index=False)
    print(f"Saved processed annotations to {output_file}")
    
    # Print statistics
    total_annotations = len(annotations_df)
    processed_annotations = len(result_df)
    out_of_bounds = result_df['out_of_bounds'].sum()
    
    print(f"Processed {processed_annotations}/{total_annotations} annotations ({processed_annotations/total_annotations*100:.2f}%)")
    print(f"Out of bounds annotations: {out_of_bounds}/{processed_annotations} ({out_of_bounds/processed_annotations*100:.2f}%)")

if __name__ == "__main__":
    main() 