import argparse
import json
import os

from src.adversarial_evaluation import do_adversarial_evaluation
from src.expert_knowledge import do_expert_knowledge_check
from src.statistical_analysis import do_statistical_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Perform post-market evaluation of the synthetic data"
    )
    # Expert Knowledge Arguments
    parser.add_argument(
        "--dicom-directory",
        type=str,
        default="sample_data",
        help="Directory containing DICOM files",
    )
    parser.add_argument(
        "--darker-neighbors-threshold",
        type=float,
        default=0.2,
        help="Threshold for darker neighbor ratio check",
    )
    parser.add_argument(
        "--min-valid-area-ratio",
        type=float,
        default=0.01,
        help="Minimum valid area ratio for lung regions",
    )
    parser.add_argument(
        "--max-valid-area-ratio",
        type=float,
        default=0.50,
        help="Maximum valid area ratio for lung regions",
    )
    parser.add_argument(
        "--expert-knowledge-results-path",
        type=str,
        default="expert_knowledge_results.json",
        help="Output path for expert knowledge results JSON",
    )
    parser.add_argument(
        "--visualization_expert_knowledge",
        action="store_true",
        help="Enable creation of visualization figures for expert knowledge",
    )
    # Statistical Analysis Arguments
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=1.5,
        help="Threshold for detecting gaps between slices",
    )
    parser.add_argument(
        "--statistical-analysis-results-path",
        type=str,
        default="statistical_analysis_results.json",
        help="Output path for statistical analysis results JSON",
    )
    # Adversarial Evaluation Arguments
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model directory containing model_v7.json and weights_v7.hdf5",
    )
    parser.add_argument(
        "--original-data-path", required=True, help="Path to original data directory"
    )
    parser.add_argument(
        "--synthetic-data-path", required=True, help="Path to synthetic data directory"
    )
    parser.add_argument(
        "--segmentation-threshold",
        required=True,
        help="Segmentation threshold for the model",
    )
    parser.add_argument(
        "--visualization-adversarial-evaluation",
        action="store_true",
        help="Enable creation of visualization figures for adversarial evaluation",
    )
    parser.add_argument(
        "--verbosity",
        action="store_true",
        help="Enable verbosity",
    )
    args = parser.parse_args()

    do_expert_knowledge_check(
        args.dicom_directory,
        args.darker_neighbors_threshold,
        args.min_valid_area_ratio,
        args.max_valid_area_ratio,
        args.expert_knowledge_results_path,
        args.visualization_expert_knowledge,
    )

    do_statistical_analysis(
        args.dicom_directory,
        args.gap_threshold,
        args.statistical_analysis_results_path,
    )

    do_adversarial_evaluation(
        args.model_path,
        args.original_data_path,
        args.synthetic_data_path,
        args.segmentation_threshold,
        args.visualization_adversarial_evaluation,
        args.verbosity,
    )

    with open(args.expert_knowledge_results_path, "r") as f:
        expert_knowledge_results = json.load(f)

    with open(args.statistical_analysis_results_path, "r") as f:
        statistical_analysis_results = json.load(f)

    with open("adversarial_evaluation_results.json", "r") as f:
        adversarial_evaluation_results = json.load(f)

    combined_results = {
        "expert_knowledge_evaluation": expert_knowledge_results,
        "statistical_analysis_evaluation": statistical_analysis_results,
        "adversarial_evaluation": adversarial_evaluation_results,
    }

    with open("post_market_evaluation_results.json", "w") as f:
        json.dump(combined_results, f, indent=4)

    os.remove(args.expert_knowledge_results_path)
    os.remove(args.statistical_analysis_results_path)
    os.remove("adversarial_evaluation_results.json")


if __name__ == "__main__":
    main()
