#!/usr/bin/env python3
"""
dicom_cache_builder.py

Recursively scans a given DICOM root directory and caches every DICOM file’s
full pydicom Dataset JSON (via ds.to_json_dict()) in a single JSON file.
Each entry is keyed by the DICOM file’s name.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from pydicom.filereader import dcmread


def main():
    parser = argparse.ArgumentParser(
        description="Build a JSON cache of all DICOM files under a directory"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Root directory containing DICOM files"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to write the JSON cache file"
    )
    args = parser.parse_args()

    dicom_root = Path(args.input)
    if not dicom_root.is_dir():
        logger.error(f"Input directory not found: {dicom_root}")
        sys.exit(1)

    cache_file = Path(args.output)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Log progress: find and process files ---
    dcm_files = list(dicom_root.rglob("*.dcm"))
    logger.info(f"Found {len(dcm_files)} DICOM files under {dicom_root}")

    cache = {}
    for idx, dcm_path in enumerate(dcm_files, start=1):
        logger.info(f"[{idx}/{len(dcm_files)}] Processing: {dcm_path.name}")
        try:
            ds = dcmread(str(dcm_path), stop_before_pixels=True, force=True)
            cache[dcm_path.name] = ds.to_json_dict()
        except Exception as e:
            logger.warning(f"Failed to read {dcm_path}: {e}")

    # --- Save and report ---
    cache_file.write_text(json.dumps(cache, indent=2))
    logger.info(f"Cached {len(cache)} / {len(dcm_files)} DICOM files to {cache_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    main()
