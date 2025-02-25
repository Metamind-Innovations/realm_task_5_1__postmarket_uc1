import kfp
from kfp.dsl import Dataset, Input, Model, Output


@kfp.dsl.component(
    base_image="python:3.8-slim",
    target_image="post-market-evaluation:v1",
    packages_to_install=[
        "keras==2.10.0",
        "kfp==2.5.0",
        "kfp_pipeline_spec==0.2.2",
        "matplotlib==3.7.5",
        "numpy==1.24.4",
        "opencv_python==4.11.0.86",
        "pandas==2.0.3",
        "plotly==6.0.0",
        "pydicom==2.4.4",
        "pyradiomics==3.1.0",
        "scikit_learn==1.3.2",
        "scipy==1.10.1",
        "seaborn==0.13.2",
        "SimpleITK==2.4.1",
        "scikit-learn",
        "scikit-image==0.21.0",
        "statsmodels==0.14.1",
        "tqdm==4.67.1",
        "tensorflow==2.10.0",
        "tensorflow-estimator==2.10.0",
        "tensorflow-io-gcs-filesystem==0.31.0",
    ],
)
def post_market_evaluation(
    dicom_directory: Input[Dataset],
    model_path: Input[Model],
    original_data_path: Input[Dataset],
    synthetic_data_path: Input[Dataset],
    expert_knowledge_results_path: Output[Dataset],
    statistical_analysis_results_path: Output[Dataset],
    # Expert Knowledge Parameters
    darker_neighbors_threshold: float = 0.2,
    min_valid_area_ratio: float = 0.01,
    max_valid_area_ratio: float = 0.50,
    visualization_expert_knowledge: bool = False,
    # Statistical Analysis Parameters
    gap_threshold: float = 1.5,
    # Adversarial Evaluation Parameters
    segmentation_threshold: float = 0.5,
    visualization_adversarial_evaluation: bool = False,
    verbosity: bool = False,
):
    """Perform post-market evaluation of synthetic data.

    Args:
        dicom_directory: Directory containing DICOM files
        model_path: Directory containing model_v7.json and weights files
        original_data_path: Directory containing original DICOM and nrrd files
        synthetic_data_path: Directory containing synthetic DICOM and nrrd files
        darker_neighbors_threshold: Threshold for darker neighbor ratio check
        min_valid_area_ratio: Minimum valid area ratio for lung regions
        max_valid_area_ratio: Maximum valid area ratio for lung regions
        visualization_expert_knowledge: Enable visualization for expert knowledge
        gap_threshold: Threshold for detecting gaps between slices
        segmentation_threshold: Segmentation threshold for the model
        visualization_adversarial_evaluation: Enable visualization for adversarial evaluation
        verbosity: Enable verbosity
    """
    import subprocess

    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "ffmpeg", "libsm6", "libxext6"])

    import json
    import os
    from src.adversarial_evaluation import do_adversarial_evaluation
    from src.expert_knowledge import do_expert_knowledge_check
    from src.statistical_analysis import do_statistical_analysis

    final_results_path = "post_market_evaluation_results.json"

    do_expert_knowledge_check(
        dicom_directory.path,
        darker_neighbors_threshold,
        min_valid_area_ratio,
        max_valid_area_ratio,
        expert_knowledge_results_path.path,
        visualization_expert_knowledge,
    )

    do_statistical_analysis(
        dicom_directory.path,
        gap_threshold,
        statistical_analysis_results_path.path,
    )

    do_adversarial_evaluation(
        model_path.path,
        original_data_path.path,
        synthetic_data_path.path,
        segmentation_threshold,
        visualization_adversarial_evaluation,
        verbosity,
    )

    with open(expert_knowledge_results_path.path, "r") as f:
        expert_knowledge_results = json.load(f)

    with open(statistical_analysis_results_path.path, "r") as f:
        statistical_analysis_results = json.load(f)

    with open("adversarial_evaluation_results.json", "r") as f:
        adversarial_evaluation_results = json.load(f)

    combined_results = {
        "expert_knowledge_evaluation": expert_knowledge_results,
        "statistical_analysis_evaluation": statistical_analysis_results,
        "adversarial_evaluation": adversarial_evaluation_results,
    }

    with open(final_results_path, "w") as f:
        json.dump(combined_results, f, indent=4)

    os.remove(expert_knowledge_results_path.path)
    os.remove(statistical_analysis_results_path.path)
    os.remove("adversarial_evaluation_results.json")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        post_market_evaluation, "post_market_evaluation_component.yaml"
    )
