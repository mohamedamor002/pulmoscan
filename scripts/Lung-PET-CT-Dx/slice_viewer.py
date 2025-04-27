#!/usr/bin/env python3
"""
slice_viewer.py

A simple utility to visualize processed tumor slices from the output directory
created by the slice_processor.py script.

Usage:
    python slice_viewer.py -i /path/to/processed/slices/dir [-n NUM_SAMPLES] [-s RANDOM_SEED]

Options:
    -i, --input     Directory containing processed slices (required)
    -n, --num       Number of sample slices to display (default: 9)
    -s, --seed      Random seed for reproducibility (default: 42)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_sample_slices(base_dir, num_samples=4, random_seed=42):
    """
    Randomly select sample slices from the processed directory.

    Returns a list of tuples (slice_path, cancer_type, patient_id)
    """
    random.seed(random_seed)
    base_dir = Path(base_dir)

    # Find all cancer type directories
    cancer_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not cancer_dirs:
        logger.error(f"No cancer type directories found in {base_dir}")
        return []

    samples = []
    attempt_count = 0
    max_attempts = 100  # Avoid infinite loop

    while len(samples) < num_samples and attempt_count < max_attempts:
        # Select a random cancer type
        cancer_dir = random.choice(cancer_dirs)
        cancer_type = cancer_dir.name

        # Select a random patient in this cancer type
        patient_dirs = [d for d in cancer_dir.iterdir() if d.is_dir()]
        if not patient_dirs:
            attempt_count += 1
            continue

        patient_dir = random.choice(patient_dirs)
        patient_id = patient_dir.name

        # Select a random slice for this patient
        slice_files = list(patient_dir.glob("*.npy"))
        if not slice_files:
            attempt_count += 1
            continue

        slice_file = random.choice(slice_files)

        # Add to samples if not already there
        sample_info = (slice_file, cancer_type, patient_id)
        if sample_info not in samples:
            samples.append(sample_info)

        attempt_count += 1

    return samples


def visualize_slices(samples):
    """
    Display the sample slices in a grid.
    """
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(samples))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Flatten axes array if needed
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (slice_path, cancer_type, patient_id) in enumerate(samples):
        if i >= len(axes):
            break

        # Load the numpy array
        try:
            img = np.load(slice_path)

            # Display the image
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"{cancer_type}\nPatient: {patient_id}")
            axes[i].axis("off")

            # Extract UID from filename
            uid = slice_path.stem
            axes[i].text(
                5,
                20,
                f"UID: ...{uid[-8:]}",
                color="white",
                fontsize=8,
                backgroundcolor="black",
            )

        except Exception as e:
            logger.error(f"Error displaying {slice_path}: {e}")
            axes[i].text(
                0.5,
                0.5,
                f"Error loading\n{slice_path.name}",
                ha="center",
                va="center",
                color="red",
            )
            axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.suptitle("Sample Processed Tumor Slices", fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.92)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize processed lung tumor CT slices."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Directory containing processed slices"
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=9,
        help="Number of sample slices to display (default: 9)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Get sample slices
    logger.info(f"Looking for processed slices in {input_dir}")
    samples = get_sample_slices(input_dir, args.num, args.seed)

    if not samples:
        logger.error("No suitable samples found.")
        return

    logger.info(f"Found {len(samples)} samples to visualize")

    # Visualize
    fig = visualize_slices(samples)
    plt.show()

    # Option to save the visualization
    save = input("Save this visualization? (y/n): ").lower().strip()
    if save == "y":
        output_path = Path(input_dir) / "slice_visualization.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
