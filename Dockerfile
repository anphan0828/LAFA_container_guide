FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python scripts and config files
COPY blast_chunks_optimized.py .
COPY blast_main.py .
COPY retrieve_terms.py .
COPY ontology.py .
COPY config.yaml .
COPY run_blast.sh .

# Make the bash script executable
RUN chmod +x run_blast.sh

# Set environment variable for thread count
ENV NUM_THREADS=8

ENTRYPOINT ["python3", "blast_main.py"]
