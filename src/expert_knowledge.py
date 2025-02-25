# import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy import ndimage
from skimage import measure
from skimage.filters import threshold_otsu


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


def anatomical_feasibilty_check(
    dicom_series,
    min_valid_area_ratio: float = 0.01,
    max_valid_area_ratio: float = 0.50,
    darker_neighbors_threshold: float = 0.2,
    output_dir: str = "expert_knowledge_results",
) -> Tuple[List[int], List[int]]:
    """Check whether lungs appear within the patient's body. To test this, we use Otsu's thresholding method to segment
    the image and then verify that the areas corresponding to the lungs which are expected to appear darker are enclosed
    by lighter areas corresponding to the patient's body.

    Args:
        dicom_series: List of DICOM slices or image data
        min_valid_area_ratio (float): Minimum ratio of lung area to total image area (default: 0.01)
        max_valid_area_ratio (float): Maximum ratio of lung area to total image area (default: 0.50)
        darker_neighbors_threshold (float): Threshold for determining acceptable fraction of darker neighboring pixels (default: 0.2)
        output_dir (str): Directory to save the results

    Returns:
        tuple[list[int], list[int]]: Two lists containing indices of valid and invalid slices respectively
    """

    valid_slices = []
    invalid_slices = []

    for idx, slice_data in enumerate(dicom_series):
        filename = slice_data.filename
        filename = os.path.basename(filename)
        try:
            img = slice_data.pixel_array
        except AttributeError:
            img = slice_data.get("pixel_array") or slice_data.get("PixelData")
            if img is None:
                print(f"Warning: Could not extract pixel data from slice {filename}")
                invalid_slices.append(idx)
                continue

        # Normalization
        img = img.astype(float)
        if img.max() > 0:
            img = (img - img.min()) / (img.max() - img.min())

        # Otsu's thresholding
        thresh = threshold_otsu(img)
        binary = img < thresh  # Dark regions (potential lungs) are True

        # Label connected components in the binary image
        labeled_array, _ = measure.label(binary, return_num=True)
        regions = measure.regionprops(labeled_array)

        # Filter regions by size to identify potential lung regions
        total_area = img.shape[0] * img.shape[1]
        min_area = total_area * min_valid_area_ratio
        max_area = total_area * max_valid_area_ratio

        lung_candidates = [
            region for region in regions if min_area < region.area < max_area
        ]

        # Check if potential lung regions are enclosed by body
        valid_regions = []
        if len(lung_candidates) >= 1:
            for lung in lung_candidates:
                # First Validity Check - Check if potential lung region touches the edges of the image
                bbox = lung.bbox
                if (
                    (0 in bbox)
                    or (bbox[2] == img.shape[0])
                    or (bbox[3] == img.shape[1])
                ):
                    continue

                # Second Validity Check - Check if the lung region is enclosed by body
                region_mask = labeled_array == lung.label
                filled_region = ndimage.binary_fill_holes(region_mask)

                # Create a dilated version of the filled region to get outer neighboring pixels
                outer_region = ndimage.binary_dilation(filled_region) & ~filled_region
                outer_coords = np.where(outer_region)

                # Check if region is properly enclosed by lighter tissue
                darker_neighbors_ratio = np.mean(img[outer_coords] < thresh)
                if darker_neighbors_ratio > darker_neighbors_threshold:
                    continue

                valid_regions.append(lung)

        is_valid = len(valid_regions) > 0

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot original image
            ax1.imshow(img, cmap="gray")
            ax1.set_title(f"Original Slice {idx}")
            ax1.axis("off")

            # Plot image with valid regions
            visualization_mask = np.zeros_like(img, dtype=bool)
            ax2.imshow(img, cmap="gray")

            for lung in lung_candidates:
                region_mask = labeled_array == lung.label
                filled_region = ndimage.binary_fill_holes(region_mask)
                edges = filled_region ^ ndimage.binary_erosion(filled_region)

                if lung in valid_regions:
                    ax2.contour(edges, colors="green", linewidths=1)
                    visualization_mask |= filled_region

            ax2.imshow(visualization_mask, alpha=0.3, cmap="Greens")
            ax2.set_title(
                f"Valid Regions ({len(valid_regions)})\n{'Valid' if is_valid else 'Invalid'}"
            )
            ax2.axis("off")

            fig_path = os.path.join(
                output_dir, f"slice_{idx:03d}_{'valid' if is_valid else 'invalid'}.png"
            )
            plt.savefig(fig_path, bbox_inches="tight", dpi=300)
            plt.close()

        (valid_slices if is_valid else invalid_slices).append(filename)

    return valid_slices, invalid_slices


def do_expert_knowledge_check(
    dicom_directory: str,
    darker_neighbors_threshold: float = 0.2,
    min_valid_area_ratio: float = 0.01,
    max_valid_area_ratio: float = 0.50,
    expert_knowledge_results_path: str = "expert_knowledge_results.json",
    visualization: bool = False,
):
    dicom_series = load_dicom_series(dicom_directory)
    base_folder = os.path.basename(os.path.dirname(dicom_directory))

    if visualization:
        output_dir = os.path.join("expert_knowledge_output", base_folder)
    else:
        output_dir = None

    valid_idx, invalid_idx = anatomical_feasibilty_check(
        dicom_series,
        min_valid_area_ratio,
        max_valid_area_ratio,
        darker_neighbors_threshold,
        output_dir,
    )

    expert_knowledge_results = {
        "information": (
            "Lungs in humans are expected to be within the examined person's body. "
            "In CT scans, lung areas appear darker than the surrounding body tissues. "
            "Based on (Ntampakis, et al., 2024), Otsu's thresholding method can be used "
            "to make a distinction between lungs and the surrounding areas to verify that "
            "identified lung areas are surrounded by body tissue."
        ),
        "Valid Slices": valid_idx,
        "Invalid Slices": invalid_idx,
    }

    with open(expert_knowledge_results_path, "w") as f:
        json.dump(expert_knowledge_results, f)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Process DICOM images with anatomical feasibility checks."
#     )
#     parser.add_argument(
#         "--dicom-directory",
#         type=str,
#         default="sample_data",
#         help="Directory containing DICOM files",
#     )
#     parser.add_argument(
#         "--darker-neighbors-threshold",
#         type=float,
#         default=0.2,
#         help="Threshold for darker neighbor ratio check",
#     )
#     parser.add_argument(
#         "--min-valid-area-ratio",
#         type=float,
#         default=0.01,
#         help="Minimum valid area ratio for lung regions",
#     )
#     parser.add_argument(
#         "--max-valid-area-ratio",
#         type=float,
#         default=0.50,
#         help="Maximum valid area ratio for lung regions",
#     )
#     parser.add_argument(
#         "--expert-knowledge-results-path",
#         type=str,
#         default="expert_knowledge_results.json",
#         help="Output path for expert knowledge results JSON",
#     )
#     parser.add_argument(
#         "--visualization",
#         action="store_true",
#         help="Enable creation of visualization figures",
#     )
#     args = parser.parse_args()

#     dicom_directory = args.dicom_directory
#     darker_neighbors_threshold = args.darker_neighbors_threshold
#     min_valid_area_ratio = args.min_valid_area_ratio
#     max_valid_area_ratio = args.max_valid_area_ratio
#     expert_knowledge_results_path = args.expert_knowledge_results_path
#     dicom_series = load_dicom_series(dicom_directory)
#     base_folder = os.path.basename(os.path.dirname(dicom_directory))

#     if args.visualization:
#         output_dir = os.path.join("expert_knowledge_output", base_folder)
#     else:
#         output_dir = None

#     valid_idx, invalid_idx = anatomical_feasibilty_check(
#         dicom_series,
#         min_valid_area_ratio,
#         max_valid_area_ratio,
#         darker_neighbors_threshold,
#         output_dir,
#     )

#     expert_knowledge_results = {
#         "information": (
#             "Lungs in humans are expected to be within the examined person's body. "
#             "In CT scans, lung areas appear darker than the surrounding body tissues. "
#             "Based on (Ntampakis, et al., 2024), Otsu's thresholding method can be used "
#             "to make a distinction between lungs and the surrounding areas to verify that "
#             "identified lung areas are surrounded by body tissue."
#         ),
#         "Valid Slices": valid_idx,
#         "Invalid Slices": invalid_idx,
#     }

#     with open(expert_knowledge_results_path, "w") as f:
#         json.dump(expert_knowledge_results, f)


# if __name__ == "__main__":
#     main()
