FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "python", "post_market_evaluation.py", \
             "--dicom-directory", "./sample_data/synthetic_slices/dicom_images", \
             "--darker-neighbors-threshold", "0.2", \
             "--min-valid-area-ratio", "0.01", \
             "--max-valid-area-ratio", "0.5", \
             "--expert-knowledge-results-path", "./expert_knowledge_results.json", \
             "--visualization_expert_knowledge", \
             "--gap-threshold", "1.5", \
             "--statistical-analysis-results-path", "./statistical_analysis_results.json", \
             "--model-path", "./model_files", \
             "--original-data-path", "./sample_data/original_slices", \
             "--synthetic-data-path", "./sample_data/synthetic_slices", \
             "--segmentation-threshold", "0.5", \
             "--visualization-adversarial-evaluation", \
             "--verbosity" ]