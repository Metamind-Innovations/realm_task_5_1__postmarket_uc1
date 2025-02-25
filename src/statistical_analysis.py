import json
import os
from typing import List, Dict

import numpy as np
import pydicom


def load_dicom_series(directory: str) -> List[pydicom.dataset.FileDataset]:
    """Load all DICOM files from a directory.

    Args:
        directory (str): Path to directory containing DICOM files

    Returns:
        List[pydicom.dataset.FileDataset]: List of DICOM files sorted by instance number
    """

    dicom_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".dcm"):
            file_path = os.path.join(directory, filename)
            dicom_files.append(pydicom.dcmread(file_path))

    return sorted(dicom_files, key=lambda x: float(x.InstanceNumber))


def missing_slice_detection(
    dicom_series: List[pydicom.dataset.FileDataset],
    gap_threshold: float = 1.5,
) -> List[float]:
    """Check for missing slices in the DICOM series.

    Args:
        dicom_series (List[pydicom.dataset.FileDataset]): List of DICOM files sorted by instance number
        gap_threshold (float, optional): Factor multiplied by mean spacing to detect gaps.
            Values > 1.0 indicate gaps larger than normal spacing. Defaults to 1.5.

    Returns:
        List[float]: Indices of slices after which gaps were detected
    """

    slice_positions = [float(dcm.SliceLocation) for dcm in dicom_series]
    slice_spacing = np.diff(slice_positions)
    mean_spacing = np.mean(np.abs(slice_spacing))  # Take absolute mean

    gaps = np.where(np.abs(slice_spacing) > gap_threshold * mean_spacing)[0]

    return gaps.tolist()


def dicom_header_consistency(dicom_series: List[pydicom.dataset.FileDataset]) -> Dict:
    """
    Check DICOM headers for completeness and consistency across the series.

    Args:
        dicom_series (List[pydicom.dataset.FileDataset]): List of DICOM files sorted by instance number

    Returns:
        Dict containing:
        - 'missing_headers': Dict[str, List[int]] - Headers missing in specific slices
        - 'inconsistent_headers': Dict[str, Dict] - Headers with unexpected variations
    """

    essential_headers = [
        # Identification (should be consistent)
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        # Geometry (some variation expected)
        "SliceLocation",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        # Image characteristics (should be consistent)
        "BitsAllocated",
        "BitsStored",
    ]

    # Headers that should be consistent across all slices
    consistent_headers = [
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "ImageOrientationPatient",
        "BitsAllocated",
        "BitsStored",
    ]

    dicom_header_consistency_results = {
        "missing_headers": {},
        "inconsistent_headers": {},
    }

    # Check for missing headers
    for header in essential_headers:
        missing_in_slices = []
        for idx, dcm in enumerate(dicom_series):
            if not hasattr(dcm, header):
                missing_in_slices.append(idx)
        if missing_in_slices:
            dicom_header_consistency_results["missing_headers"][header] = (
                missing_in_slices
            )

    # Check for consistency where expected
    for header in consistent_headers:
        if header in dicom_header_consistency_results["missing_headers"]:
            continue

        values = [getattr(dcm, header) for dcm in dicom_series if hasattr(dcm, header)]

        if not values:
            continue

        if header == "ImageOrientationPatient":
            # Convert numpy arrays to tuples for comparison
            values = [tuple(val) for val in values]

        if len(set(str(v) for v in values)) > 1:
            dicom_header_consistency_results["inconsistent_headers"][header] = {
                "unique_values": list(set(str(v) for v in values)),
                "message": f"Found {len(set(str(v) for v in values))} different values",
            }

    # Special check for ImagePositionPatient
    if (
        "ImagePositionPatient"
        not in dicom_header_consistency_results["missing_headers"]
    ):
        positions = [dcm.ImagePositionPatient for dcm in dicom_series]
        x_vals = [pos[0] for pos in positions]
        y_vals = [pos[1] for pos in positions]

        if len(set(x_vals)) > 1 or len(set(y_vals)) > 1:
            dicom_header_consistency_results["inconsistent_headers"][
                "ImagePositionPatient"
            ] = {
                "x_consistent": len(set(x_vals)) == 1,
                "y_consistent": len(set(y_vals)) == 1,
                "message": "X/Y values vary (might indicate gantry tilt or oblique acquisition)",
            }

    return dicom_header_consistency_results


def image_dimension_consistency(
    dicom_series: List[pydicom.dataset.FileDataset],
) -> Dict:
    """
    Check consistency of image dimensions and pixel spacing across the DICOM series.

    Args:
        dicom_series (List[pydicom.dataset.FileDataset]): List of DICOM files sorted by instance number

    Returns:
        Dict containing:
        - 'dimensions': Dict with dimension consistency info
        - 'pixel_spacing': Dict with pixel spacing consistency info
        - 'is_consistent': bool indicating if all checks passed
    """

    image_consistency_results = {
        "dimensions": {},
        "pixel_spacing": {},
        "is_consistent": True,
    }

    # Check image dimensions
    rows = set()
    columns = set()
    for dcm in dicom_series:
        rows.add(dcm.Rows)
        columns.add(dcm.Columns)

    image_consistency_results["dimensions"] = {
        "unique_rows": list(rows),
        "unique_columns": list(columns),
        "is_consistent": len(rows) == 1 and len(columns) == 1,
    }

    # Check pixel spacing
    pixel_spacing = []
    for dcm in dicom_series:
        if hasattr(dcm, "PixelSpacing"):
            pixel_spacing.append(tuple(dcm.PixelSpacing))

    unique_spacing = set(pixel_spacing)
    image_consistency_results["pixel_spacing"] = {
        "unique_values": list(unique_spacing),
        "is_consistent": len(unique_spacing) == 1 if pixel_spacing else False,
        "missing_in_slices": [
            idx
            for idx, dcm in enumerate(dicom_series)
            if not hasattr(dcm, "PixelSpacing")
        ],
    }

    image_consistency_results["is_consistent"] = (
        image_consistency_results["dimensions"]["is_consistent"]
        and image_consistency_results["pixel_spacing"]["is_consistent"]
    )

    return image_consistency_results


def do_statistical_analysis(
    dicom_directory: str,
    gap_threshold: float = 1.5,
    statistical_analysis_results_path: str = "statistical_analysis_results.json",
):
    dicom_series = load_dicom_series(dicom_directory)

    gaps = missing_slice_detection(dicom_series, gap_threshold)

    dicom_header_consistency_results = dicom_header_consistency(dicom_series)

    image_consistency_results = image_dimension_consistency(dicom_series)

    statistical_analysis_results = {
        "information": (
            "This statistical analysis checks for missing slices, DICOM header consistency, and image dimension consistency in a DICOM series. "
            "It helps ensure the integrity of the image data and the consistency of the DICOM headers across the series."
        ),
        "missing_slices": gaps,
        "dicom_header_consistency": dicom_header_consistency_results,
        "image_dimension_consistency": image_consistency_results,
    }

    with open(statistical_analysis_results_path, "w") as f:
        json.dump(statistical_analysis_results, f)
