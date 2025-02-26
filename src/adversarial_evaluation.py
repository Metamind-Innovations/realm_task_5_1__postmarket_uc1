import json
import os

from src.utils.TheDuneAI import SegmentationModel


def calculate_metric_differences(original_metrics, synthetic_metrics):
    """
    Calculate differences between metrics of original and synthetic slices.

    Args:
        original_metrics (dict): Metrics of original slices
        synthetic_metrics (dict): Metrics of synthetic slices

    Returns:
        dict: Differences between metrics
    """
    differences = {}

    for patient_id in original_metrics:
        if patient_id in synthetic_metrics:
            differences[patient_id] = {
                "original_dice_coefficient_score": original_metrics[patient_id],
                "synthetic_dice_coefficient_score": synthetic_metrics[patient_id],
                "difference": original_metrics[patient_id]
                - synthetic_metrics[patient_id],
            }

    differences["information"] = (
        "This adversarial evaluation compares segmentation results between original CT slices "
        "and synthetically generated ones. The procedure involves: "
        "1) Running lung segmentation on original CT data, "
        "2) Running lung segmentation on synthetic data, "
        "3) Computing differences in segmentation metrics between the two datasets to "
        "evaluate how well the synthetic data match the original data by comparing the performance "
        "of a SOTA segmentation model on the two datasets."
    )

    return differences


def do_adversarial_evaluation(
    model_path: str,
    original_data_path: str,
    synthetic_data_path: str,
    segmentation_threshold: float,
    visualization: bool = False,
    verbosity: bool = False,
):
    print(f"Visualization: {visualization}")

    if visualization:
        original_save_path = "adversarial_evaluation_original_output"
        synthetic_save_path = "adversarial_evaluation_synthetic_output"
    else:
        original_save_path = None
        synthetic_save_path = None

    original_model = SegmentationModel(
        model_path=model_path,
        data_path=original_data_path,
        segmentation_threshold=segmentation_threshold,
        output_path=original_save_path,
        verbosity=verbosity,
        json_path="adversarial_evaluation_original_metrics.json",
    )
    original_model.segment()

    synthetic_model = SegmentationModel(
        model_path=model_path,
        data_path=synthetic_data_path,
        segmentation_threshold=segmentation_threshold,
        output_path=synthetic_save_path,
        verbosity=verbosity,
        json_path="adversarial_evaluation_synthetic_metrics.json",
    )
    synthetic_model.segment()

    with open("adversarial_evaluation_original_metrics.json", "r") as f:
        original_metrics = json.load(f)

    with open("adversarial_evaluation_synthetic_metrics.json", "r") as f:
        synthetic_metrics = json.load(f)

    metric_differences = calculate_metric_differences(
        original_metrics, synthetic_metrics
    )

    with open("adversarial_evaluation_results.json", "w") as f:
        json.dump(metric_differences, f, indent=4)

    os.remove("adversarial_evaluation_original_metrics.json")
    os.remove("adversarial_evaluation_synthetic_metrics.json")
