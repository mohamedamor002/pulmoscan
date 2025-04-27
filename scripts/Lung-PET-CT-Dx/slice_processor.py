#!/usr/bin/env python3
"""
slice_processor.py

This script processes DICOM slices with tumor annotations to create
a standardized dataset for machine learning:
- Reads slice_annotation_index.csv
- For each entry, loads DICOM image and XML annotation
- Crops around tumor with margin
- Resizes to target size
- Normalizes HU values to [0,1]
- Optionally applies filters (gradient, LBP, wavelet, or combined)
- Saves as numpy arrays organized by cancer type and patient ID
- Supports parallel processing for improved performance

Usage:
    python slice_processor.py -i DATASET_ROOT -c CSV_FILE -o OUTPUT_DIR [-m MARGIN] [--size SIZE] [--filters FILTER1 FILTER2 ...]

Options:
    -i, --input     Dataset root directory (required)
    -c, --csv       Path to slice_annotation_index.csv (required)
    -o, --output    Output directory for processed slices (required)
    -m, --margin    Margin around tumor in pixels (default: 20)
    --size          Target size for resized images (default: 224)
    --filters       Apply filters to enhance features (choices: gradient, lbp, wavelet, combined)
"""

import argparse
import logging
import pandas as pd
import pydicom
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path
import concurrent.futures
from tqdm import tqdm  # For nicer progress bars
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dicom(dicom_path):
    """Load DICOM file and convert to HU values."""
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.int16)

    # Convert to Hounsfield Units (HU)
    intercept = ds.RescaleIntercept if "RescaleIntercept" in ds else 0
    slope = ds.RescaleSlope if "RescaleSlope" in ds else 1
    image = image * slope + intercept

    return image


def load_bbox(xml_path):
    """Load bounding box coordinates from XML annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find bounding box coordinates
    bndbox = root.find(".//bndbox")
    if bndbox is None:
        return None

    xmin = int(float(bndbox.find("xmin").text))
    ymin = int(float(bndbox.find("ymin").text))
    xmax = int(float(bndbox.find("xmax").text))
    ymax = int(float(bndbox.find("ymax").text))

    return xmin, ymin, xmax - xmin, ymax - ymin  # x, y, width, height


def crop_and_resize(image, bbox, margin, target_size):
    """Crop around tumor with margin and resize to target_size."""
    x, y, w, h = bbox

    # Add margin while staying within image bounds
    x_min = max(x - margin, 0)
    y_min = max(y - margin, 0)
    x_max = min(x + w + margin, image.shape[1])
    y_max = min(y + h + margin, image.shape[0])

    # Crop and resize
    crop = image[y_min:y_max, x_min:x_max]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    return resized


def normalize_hu(image, min_hu=-1000, max_hu=400):
    """Clip HU values and normalize to [0,1] range."""
    image = np.clip(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)  # Scale to [0,1]
    return image


def apply_filters(image, filters=None):
    """Apply image filters to enhance features.

    Args:
        image: Input image (normalized to [0,1])
        filters: List of filters to apply ["gradient", "lbp", "wavelet"]
                If None, return original image only

    Returns:
        Dictionary of filtered images {"original": img, "filter_name": filtered_img, ...}
    """
    if filters is None:
        return {"original": image}

    results = {"original": image}

    # Convert to uint8 for certain filters
    img_uint8 = (image * 255).astype(np.uint8)

    if "gradient" in filters:
        # Gradient magnitude filter - highlights edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        # Normalize gradient to [0,1]
        gradient = (gradient - gradient.min()) / (
            gradient.max() - gradient.min() + 1e-8
        )
        results["gradient"] = gradient

    if "lbp" in filters:
        # Local Binary Pattern for texture analysis
        from skimage.feature import local_binary_pattern

        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_uint8, n_points, radius, method="uniform")
        # Normalize LBP to [0,1]
        lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)
        results["lbp"] = lbp

    if "wavelet" in filters:
        # Create a combined wavelet result for consistent name handling
        try:
            import pywt

            # Use SWT instead of DWT which doesn't downsample
            coeffs2 = pywt.swt2(image, "coif1", level=1)[0]
            LL, (LH, HL, HH) = coeffs2

            # No resizing needed as SWT preserves size
            channel3 = (HH - HH.min()) / (HH.max() - HH.min() + 1e-8)
        except ImportError:
            logger.error(
                "PyWavelets not installed. Please install with 'pip install pywavelets'"
            )
            raise ImportError("Missing required dependency: pywavelets")

    if "combined" in filters:
        # Create a 3-channel image combining original + gradient + wavelet
        # Channel 1: Original image
        channel1 = image.copy()

        # Channel 2: Gradient magnitude
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        channel2 = np.sqrt(sobelx**2 + sobely**2)
        channel2 = (channel2 - channel2.min()) / (
            channel2.max() - channel2.min() + 1e-8
        )

        # Channel 3: Wavelet HH component (diagonal details)
        try:
            import pywt

            # Use SWT instead of DWT which doesn't downsample
            coeffs2 = pywt.swt2(image, "coif1", level=1)[0]
            LL, (LH, HL, HH) = coeffs2

            # No resizing needed as SWT preserves size
            channel3 = (HH - HH.min()) / (HH.max() - HH.min() + 1e-8)
        except ImportError:
            logger.warning("PyWavelets not installed, using LBP as third channel")
            from skimage.feature import local_binary_pattern

            img_uint8 = (image * 255).astype(np.uint8)
            radius = 3
            n_points = 8 * radius
            channel3 = local_binary_pattern(
                img_uint8, n_points, radius, method="uniform"
            )
            channel3 = (channel3 - channel3.min()) / (
                channel3.max() - channel3.min() + 1e-8
            )

        # Stack the channels to create a 3D array
        combined = np.stack([channel1, channel2, channel3], axis=2)
        results["combined"] = combined

    return results


def process_slice(args):
    """Process a single slice (for parallel execution)"""
    idx, row, dicom_root, anno_root, filter_dirs, args_filters, margin, size = args

    try:
        # Construct full paths
        dicom_path = dicom_root / row["DICOM_File"]
        xml_path = anno_root / row["XML_Path"]

        # Skip if files don't exist
        if not dicom_path.is_file():
            return False, f"DICOM file not found: {dicom_path}"

        if not xml_path.is_file():
            return False, f"XML file not found: {xml_path}"

        # Load DICOM image
        image = load_dicom(str(dicom_path))

        # Load bounding box
        bbox = load_bbox(str(xml_path))
        if bbox is None:
            return False, f"No bounding box found in {xml_path}"

        # Crop, resize, normalize
        target_size = (size, size)
        processed = crop_and_resize(image, bbox, margin, target_size)
        processed = normalize_hu(processed)

        # Patient info for directory structure
        patient_id = row["Patient_ID"]
        cancer_type = row["Cancer_Type"]
        uid = row["SOPInstanceUID"]

        if args_filters:
            # Apply each filter separately and save in its own directory
            for filter_name in args_filters:
                # Apply specific filter
                filter_result = apply_filters(processed, [filter_name])
                filtered_img = filter_result[filter_name]

                # Save in filter-specific directory structure
                save_dir = filter_dirs[filter_name] / cancer_type / patient_id
                save_dir.mkdir(exist_ok=True, parents=True)

                save_path = save_dir / f"{uid}.npy"
                # Fix by adding explicit dtype conversion
                # filtered_img = filtered_img.astype(np.float32)  # Use float32 for consistency
                np.save(save_path, filtered_img)
        else:
            # Save unfiltered image in 'normal' directory
            save_dir = filter_dirs["normal"] / cancer_type / patient_id
            save_dir.mkdir(exist_ok=True, parents=True)

            save_path = save_dir / f"{uid}.npy"
            # Fix for unfiltered images too
            # processed = processed.astype(np.float32)  # Use float32 for consistency
            np.save(save_path, processed)

        return True, None

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Process DICOM slices with tumor annotations for machine learning."
    )
    parser.add_argument("-i", "--input", required=True, help="Dataset root directory")
    parser.add_argument(
        "-c", "--csv", required=True, help="Path to slice_annotation_index.csv"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for processed slices"
    )
    parser.add_argument(
        "-m",
        "--margin",
        type=int,
        default=20,
        help="Margin around tumor in pixels (default: 20)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target size for resized images (default: 224)",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=["gradient", "lbp", "wavelet", "combined"],
        help="Apply filters to enhance image features",
    )
    args = parser.parse_args()

    dataset_root = Path(args.input)
    csv_file = Path(args.csv)
    output_dir = Path(args.output)

    # Define paths based on dataset structure
    dicom_root = dataset_root / "Lung-PET-CT-Dx" / "Lung-PET-CT-Dx"
    anno_root = dataset_root / "Annotation" / "Annotation"

    # Validate paths
    if not csv_file.is_file():
        logger.error(f"CSV index not found: {csv_file}")
        return

    if not dicom_root.is_dir():
        logger.error(f"DICOM root directory not found: {dicom_root}")
        return

    if not anno_root.is_dir():
        logger.error(f"Annotation directory not found: {anno_root}")
        return

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create filter-specific directories
    filter_dirs = {}
    if args.filters:
        for filter_name in args.filters:
            filter_dir = output_dir / filter_name
            filter_dir.mkdir(exist_ok=True, parents=True)
            filter_dirs[filter_name] = filter_dir
    else:
        # If no filters specified, use 'normal' directory
        normal_dir = output_dir / "normal"
        normal_dir.mkdir(exist_ok=True, parents=True)
        filter_dirs["normal"] = normal_dir

    # Load CSV
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} entries from {csv_file}")

    # Filter for CT modality (we don't want PET slices)
    df_ct = df[df["Modality"] == "CT"]
    logger.info(f"Processing {len(df_ct)} CT slices")

    # Determine number of workers based on CPU cores
    max_workers = min(32, os.cpu_count() - 1)  # threads - 1
    logger.info(f"Using {max_workers} worker threads")

    # Prepare arguments for all slices
    process_args = [
        (
            idx,
            row,
            dicom_root,
            anno_root,
            filter_dirs,
            args.filters,
            args.margin,
            args.size,
        )
        for idx, row in df_ct.iterrows()
    ]

    # Process slices in parallel
    processed_count = 0
    error_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm for a progress bar
        futures = list(
            tqdm(
                executor.map(process_slice, process_args),
                total=len(process_args),
                desc="Processing slices",
            )
        )

        # Process results
        for success, msg in futures:
            if success:
                processed_count += 1
            else:
                error_count += 1
                logger.error(f"Error: {msg}")

    logger.info(f"Done! Successfully processed {processed_count} slices.")
    logger.info(f"Encountered errors in {error_count} slices.")
    logger.info(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
