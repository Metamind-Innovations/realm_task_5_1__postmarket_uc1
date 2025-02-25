import kfp
from kfp.dsl import Dataset, Input, Model, Output


@kfp.dsl.component(
    base_image="python:3.8-slim",
    target_image="post-market-evaluation:v1",
    packages_to_install=[
        "absl-py==2.1.0",
        "astunparse==1.6.3",
        "cachetools==5.5.1",
        "certifi==2025.1.31",
        "chardet==5.2.0",
        "charset-normalizer==3.4.1",
        "click==8.1.8",
        "contourpy==1.1.1",
        "cycler==0.12.1",
        "docopt==0.6.2",
        "docstring_parser==0.16",
        "fairlearn==0.12.0",
        "flatbuffers==25.2.10",
        "fonttools==4.56.0",
        "gast==0.4.0",
        "google-api-core==2.24.1",
        "google-auth==2.38.0",
        "google-auth-oauthlib==0.4.6",
        "google-cloud-core==2.4.2",
        "google-cloud-storage==2.19.0",
        "google-crc32c==1.5.0",
        "google-pasta==0.2.0",
        "google-resumable-media==2.7.2",
        "googleapis-common-protos==1.63.1",
        "grpcio==1.70.0",
        "h5py==3.11.0",
        "idna==3.10",
        "imageio==2.35.1",
        "imbalanced-learn==0.12.4",
        "importlib_metadata==8.5.0",
        "importlib_resources==6.4.5",
        "joblib==1.4.2",
        "keras==2.10.0",
        "Keras-Preprocessing==1.1.2",
        "kiwisolver==1.4.7",
        "kubernetes==26.1.0",
        "lazy_loader==0.4",
        "libclang==18.1.1",
        "Markdown==3.7",
        "MarkupSafe==2.1.5",
        "matplotlib==3.7.5",
        "narwhals==1.27.1",
        "networkx==3.1",
        "numpy==1.24.4",
        "oauthlib==3.2.2",
        "opencv-python==4.11.0.86",
        "opt_einsum==3.4.0",
        "packaging==24.2",
        "pandas==2.0.3",
        "patsy==1.0.1",
        "pillow==10.4.0",
        "plotly==6.0.0",
        "proto-plus==1.26.0",
        "protobuf==3.19.6",
        "pyasn1==0.6.1",
        "pyasn1_modules==0.4.1",
        "pydicom==2.4.4",
        "pykwalify==1.8.0",
        "pyparsing==3.1.4",
        "pyradiomics==3.1.0",
        "python-dateutil==2.9.0.post0",
        "pytz==2025.1",
        "PyWavelets==1.4.1",
        "PyYAML==6.0.2",
        "reportlab==4.3.0",
        "requests==2.32.3",
        "requests-oauthlib==2.0.0",
        "requests-toolbelt==0.10.1",
        "rsa==4.9",
        "ruamel.yaml==0.18.10",
        "ruamel.yaml.clib==0.2.8",
        "ruff==0.9.6",
        "scikit-fuzzy==0.5.0",
        "scikit-image==0.21.0",
        "scikit-learn==1.3.2",
        "scipy==1.10.1",
        "seaborn==0.13.2",
        "SimpleITK==2.4.1",
        "six==1.17.0",
        "statsmodels==0.14.1",
        "tabulate==0.9.0",
        "tensorboard==2.10.1",
        "tensorboard-data-server==0.6.1",
        "tensorboard-plugin-wit==1.8.1",
        "tensorflow==2.10.0",
        "tensorflow-estimator==2.10.0",
        "tensorflow-io-gcs-filesystem==0.34.0",
        "termcolor==2.4.0",
        "threadpoolctl==3.5.0",
        "tifffile==2023.7.10",
        "tqdm==4.67.1",
        "typing_extensions==4.12.2",
        "tzdata==2025.1",
        "urllib3==1.26.20",
        "websocket-client==1.8.0",
        "Werkzeug==3.0.6",
        "wrapt==1.17.2",
        "zipp==3.20.2",
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
    subprocess.run(["apt-get", "install", "-y", "libgl1-mesa-glx", "libglib2.0-0"])

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
