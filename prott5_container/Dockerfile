FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN pip3 install --upgrade pip

# Set cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DOWNLOAD_TIMEOUT=300

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the ProtT5 model during build
RUN python3 -c "from transformers import T5EncoderModel, T5Tokenizer; \
    T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc'); \
    T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')"

# Copy application files
WORKDIR /app
COPY prott5_main.py .
COPY prott5_chunks.py .
COPY run_prott5.sh .
COPY prott5_embedder.py .
COPY process_embeddings_gpu.py .
COPY normalize_embeddings.py .
COPY retrieve_terms.py .
COPY ontology.py .
COPY config.yaml .

# Make shell script executable
RUN chmod +x run_prott5.sh

ENTRYPOINT ["python3", "prott5_main.py"]
