FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "python", "post_market_evaluation.py", \
             "--dicom-directory", "./sample_data/synthetic_slices", \
             "--visualization_expert_knowledge", \
             "--model-path", "./model_files", \
             "--original-data-path", "./sample_data/original_slices", \
             "--synthetic-data-path", "./sample_data/synthetic_slices", \
             "--segmentation-threshold", "0.5", \
             "--visualization-adversarial-evaluation", \
             "--verbosity" ]