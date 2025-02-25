# REALM Task 5.1 Post-Market Evaluation

## General Task Description

Components developed in Task 5.1 focus on the post-market evaluation of synthetically generated medical data that are used in various medical applications such as lung tumor segmentation, pharmacogenomics, COPD prediction, etc.

The post-market evaluation is performed to ensure that the synthetic data generated is of high quality and is similar to the real data. To evaluate the quality of the synthetic data, examination along three main axes is performed:

1. **Expert Knowledge**: Evaluates the synthetic data based on domain-specific rules and medical knowledge to ensure anatomical correctness and clinical validity.
2. **Statistical Analysis**: Examines statistical and distributional properties of the synthetic data compared to ensure their validity from a statistical standpoint.
3. **Adversarial Evaluation**: Compares the performance of SOTA machine/deep learning models on the synthetic data with their performance on the real data to ensure that the two datasets (real and synthetic) yield comparable results.

## Use Case 1 Specific Description

This repository implements a comprehensive post-market evaluation pipeline for synthetic CT lung images, analyzing their quality and similarity to real data through three distinct evaluation approaches:

Key Components:

- **Expert Knowledge Evaluation**: Performs anatomical feasibility checks on CT slices using Otsu's thresholding to verify that lung areas appear within the patient's body and are properly enclosed by surrounding tissue. The analysis generates visual reports and identifies valid/invalid slices based on this anatomical criterion.

- **Statistical Analysis**: Conducts thorough DICOM series validation by:
  - Detecting missing or inconsistent slice spacing
  - Verifying DICOM header consistency across series
  - Checking image dimension and pixel spacing consistency
  - Generating a detailed report of any anomalies or inconsistencies

- **Adversarial Evaluation**: Compares segmentation performance between real and synthetic CT data using:
  - A SOTA lung segmentation model
  - Dice coefficient scoring for quantitative comparison
  - Detailed difference analysis between original and synthetic results
  - Optional visualization of segmentation outputs

## Prerequisites

1. **Python version must be 3.8.x**
2. Create a virtual environment and install the dependencies using the `requirements.txt` file.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Unzip `weights_v7.zip` in order to create the `weights_v7.hdf5` file.
4. Both DICOM files and nrrf files are required to be present in the data folder for the real and synthetic data respectively.

## Generation of Post-Market Evaluation Report

This can be achieved using the `post_market_evaluation.py` script.

Example usage using the provided sample data:

```bash
python3 post_market_evaluation.py --dicom-directory ./sample_data/synthetic_slices \
                                  --visualization_expert_knowledge \
                                  --model-path ./model_files \
                                  --original-data-path ./sample_data/original_slices \
                                  --synthetic-data-path ./sample_data/synthetic_slices \
                                  --segmentation-threshold 0.5 \
                                  --visualization-adversarial-evaluation \
                                  --verbosity
```

## Docker Instructions

### Create the image

To create the image we navigate to the root folder of the repo and execute: `docker build -t post-market-evaluation:v1 .`

### Run the image

To run the created image run: `docker run post-market-evaluation:v1`. Default values can be found in the `Dockerfile`.

## Kubeflow Components for Post-Market Evaluation of Lung Segmentation Synthetic Data

kubeflow_components directory contains Kubeflow pipeline components for the post-market evaluation pipeline.

### Components

#### Post Market Evaluation Component

The main component that performs the post-market evaluation of the synthetic data.

#### Usage

Compile the component generating the post_market_evaluation_component.yaml:
```
python post_market_evaluation_component.py
```

## 📜 License & Usage

All rights reserved by MetaMinds Innovations. 