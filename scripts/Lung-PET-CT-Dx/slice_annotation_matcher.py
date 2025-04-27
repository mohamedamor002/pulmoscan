#!/usr/bin/env python3
"""
This script processes DICOM and XML annotation files to generate an index CSV pairing
each DICOM slice (by SOPInstanceUID) with its corresponding XML annotation.
"""

import argparse
import logging
import sys
import pandas as pd
import pydicom
from pathlib import Path
import csv

# ----- Configure lightweight logging -----
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_cancer_type(patient_id: str) -> str:
    mapping = {
        "A": "Adenocarcinoma",
        "B": "Small Cell Carcinoma",
        "E": "Large Cell Carcinoma",
        "G": "Squamous Cell Carcinoma",
    }
    for letter, ctype in mapping.items():
        if letter in patient_id:
            return ctype
    raise ValueError(f"Unrecognized cancer type for patient ID: {patient_id}")


def get_modality(sop_class_name: str) -> str:
    """Converts a SOP Class UID descriptive name to a modality string.

    Returns:
        "CT" for "CT Image Storage"
        "PT" for "Positron Emission Tomography Image Storage"
        "PET" for "Secondary Capture Image Storage"

    Raises:
        ValueError: If the SOP Class UID descriptive name is unrecognized.
    """
    if sop_class_name == "CT Image Storage":
        return "CT"
    elif sop_class_name == "Positron Emission Tomography Image Storage":
        return "PT"
    elif sop_class_name == "Secondary Capture Image Storage":
        return "PET"
    else:
        raise ValueError(f"Unrecognized SOP Class '{sop_class_name}'")


def build_uid_map(dicom_dir: Path) -> dict:
    """Builds and returns a UID-to-(DICOM file, modality) mapping for the given directory.

    For each DICOM file in dicom_dir, the modality is determined by converting the
    SOP Class UID name using get_modality(). If the SOP Class is unrecognized,
    get_modality() will raise a ValueError, which is caught and logged as an error.
    """
    uid_map = {}
    for dcm_file in dicom_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True, force=True)
            if hasattr(ds, "SOPInstanceUID"):
                sop_class_name = ds.SOPClassUID.name
                modality = get_modality(sop_class_name)  # may raise ValueError
                uid_map[ds.SOPInstanceUID] = (dcm_file, modality)
        except ValueError as ve:
            logger.error(f"Error in file {dcm_file}: {ve}")
        except Exception as e:
            logger.debug(f"Error reading {dcm_file}: {e}")
            continue
    return uid_map


def get_relative_to(file: Path, base: Path) -> str:
    """Returns a relative path string to a provided base directory, or the original path."""
    try:
        return str(file.relative_to(base))
    except ValueError:
        return str(file)


def get_scan_id(dcm_file: Path, patient_id: str) -> str:
    """
    Given a DICOM file path and a patient ID, returns the nested scan ID.
    For example, if the file path is:
      ./dataset/Lung-PET-CT-Dx/Lung-PET-CT-Dx/Lung_Dx-A0001/04-04-2007-NA-Chest-07990/2.000000-5mm-40805/1-01.dcm
    and patient_id is "A0001", this function returns:
      "04-04-2007-NA-Chest-07990/2.000000-5mm-40805"
    """
    # Use the parent directory of the DICOM file.
    parent_parts = dcm_file.parent.parts
    idx = None
    for i, part in enumerate(parent_parts):
        if patient_id in part:
            idx = i
            break
    if idx is None or idx >= len(parent_parts) - 1:
        return ""
    return "/".join(parent_parts[idx + 1 :])


# ----- Main script -----
def main():
    parser = argparse.ArgumentParser(
        description="Match DICOM slices to XML annotations and write slice_annotation_index.csv"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Root directory of the dataset (replaces dataset_root)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Directory where slice_annotation_index.csv will be written",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    if not output_dir.is_dir():
        logger.error(f"Output directory not found: {output_dir}")
        sys.exit(1)

    # Override global roots
    global dataset_root, metadata_csv, anno_root, dicom_root
    dataset_root = input_dir
    metadata_csv = dataset_root / "metadata.csv"
    anno_root = dataset_root / "Annotation" / "Annotation"
    dicom_root = dataset_root / "Lung-PET-CT-Dx" / "Lung-PET-CT-Dx"

    # Validate existence of required paths
    if not metadata_csv.is_file():
        logger.error(f"Metadata CSV not found at: {metadata_csv}")
        sys.exit(1)
    if not anno_root.is_dir():
        logger.error(f"Annotation root not found at: {anno_root}")
        sys.exit(1)
    if not dicom_root.is_dir():
        logger.error(f"DICOM root not found at: {dicom_root}")
        sys.exit(1)

    logger.info("Starting DICOM/XML pairing process...")

    # Read metadata CSV and filter for CT (non-PET) entries.
    metadata = pd.read_csv(metadata_csv)
    metadata["File Location"] = metadata["File Location"].str.replace("\\\\", "/")

    # Derive a cleaned patient ID from the "File Location" field.
    def derive_patient_id(file_loc):
        cleaned = file_loc.replace("\\", "/").lstrip("./")
        parts = Path(cleaned).parts
        if parts and parts[0] == "Lung-PET-CT-Dx" and len(parts) > 1:
            return parts[1]
        elif parts:
            return parts[0]
        return ""

    metadata["Cleaned_ID"] = metadata["File Location"].apply(derive_patient_id)
    metadata["Cleaned_ID"] = metadata["Cleaned_ID"].apply(
        lambda pid: pid.replace("Lung_Dx-", "") if pid.startswith("Lung_Dx-") else pid
    )

    # Group the metadata by Cleaned_ID (i.e. by patient)
    patient_groups = metadata.groupby("Cleaned_ID")
    logger.info(f"Found {len(patient_groups)} unique patients in metadata.")

    records = []
    # Create a cache to store DICOM mappings per directory to avoid re-reading.
    dicom_cache = {}

    # Iterate over each patient group.
    for cleaned_id, group in patient_groups:
        logger.info(f"Processing patient {cleaned_id} with {len(group)} record(s).")
        try:
            cancer_type = get_cancer_type(cleaned_id)
        except ValueError as e:
            logger.error(f"Error processing patient {cleaned_id}: {e}")
            continue

        # The annotation folder is assumed to be under anno_root/cleaned_id.
        patient_anno_dir = anno_root / cleaned_id
        if not patient_anno_dir.exists():
            logger.info(
                f"No annotation folder for patient {cleaned_id} at {patient_anno_dir}, skipping this patient."
            )
            continue

        xml_files = list(patient_anno_dir.rglob("*.xml"))
        if not xml_files:
            logger.info(
                f"No XML files found in {patient_anno_dir} for patient {cleaned_id}."
            )
            continue

        patient_pair_count = 0

        # For each DICOM file record in this patient group (from the CSV)
        for idx, row in group.iterrows():
            file_loc = (
                row["File Location"]
                .replace("\\", "/")
                .lstrip("./")
                .replace("Lung-PET-CT-Dx/", "")
            )
            dicom_dir = dicom_root / file_loc
            if not dicom_dir.exists():
                logger.info(
                    f"Directory not found: {dicom_dir} for patient {cleaned_id}, skipping this record."
                )
                continue

            # Build a UID-to-DICOM mapping if not already cached.
            if dicom_dir not in dicom_cache:
                dicom_cache[dicom_dir] = build_uid_map(dicom_dir)

            # For each XML file available for this patient, try to find a matching DICOM file in this dicom_dir.
            for xml_file in xml_files:
                uid = xml_file.stem  # using XML stem as the SOPInstanceUID
                entry = dicom_cache[dicom_dir].get(uid)
                if not entry:
                    continue
                matching_dcm, file_modality = entry

                records.append(
                    {
                        "XML_Path": get_relative_to(xml_file, anno_root),
                        "DICOM_File": get_relative_to(matching_dcm, dicom_root),
                        "Patient_ID": cleaned_id,
                        "Cancer_Type": cancer_type,
                        "SOPInstanceUID": uid,
                        "Modality": file_modality,
                        "Scan_ID": get_scan_id(matching_dcm, cleaned_id),
                    }
                )
                patient_pair_count += 1

        logger.info(
            f"Finished processing patient {cleaned_id}. Found {patient_pair_count} pair(s) across {len(group)} file record(s)."
        )

    # At the end, write CSV into output_dir:
    output_csv = output_dir / "slice_annotation_index.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "XML_Path",
                "DICOM_File",
                "Patient_ID",
                "Cancer_Type",
                "SOPInstanceUID",
                "Modality",
                "Scan_ID",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    logger.info(f"Saved {len(records)} records to {output_csv}")

    logger.info("Pairing process complete.")


if __name__ == "__main__":
    main()
