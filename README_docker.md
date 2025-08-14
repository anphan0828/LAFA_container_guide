# Docker Container Guide for Protein Function Prediction Methods

This guide explains how to dockerize a protein function prediction method to participate in LAFA: Longitudinal Assessment of Functional Annotation. A protein function prediction method on LAFA predicts Gene Ontology (GO) terms for provided protein IDs and protein sequences, similar to CAFA competition. The prediction task will be repeated every time there is a new ground truth (released every two months by UniProt).

## Container Requirements

### 1. Input Requirements
- **Required**: Protein sequences in FASTA format and/or protein IDs
- **Optional**: Additional parameters via command-line arguments (e.g., `go.obo` file, GO annotations in GAF format)

### 2. Output Requirements
- **Format**: 3-column TSV file (can be gzipped), with no header
- **Columns**: Typically `Query_ID`, `GO_Term`, `Score` (do not include column names in the output file)

Example output file:
```
P12345      GO:0005737  0.123
P12345      GO:0016020  0.123
P67890      GO:0003824  0.123
```

The file can be optionally gzipped for space efficiency.

### 3. Container Structure
- **Entry point**: Non-interactive execution with wrapper script. This means that your container should execute a default wrapper script which runs your method from start to end without additional input (aside from the initial arguments that are provided to the container).
- **Dependencies**: All required libraries pre-installed
- **Models**: Pre-downloaded during build (if applicable)

## Directory Structure

```
method_container/
├── Dockerfile
├── requirements.txt
├── method_main.py          # Main entry point script
├── config.yaml           # Configuration file
├── [method_specific_files]
└── README.md   # Includes command to run container
```

## Dockerfile Template

```dockerfile
FROM python:3.10  # Adjust base image as needed

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set cache/model directories (if needed)
# ENV MODEL_CACHE=/app/.cache/models
# RUN mkdir -p /app/.cache/models

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build (if applicable)
# RUN python3 -c "import your_model_library; your_model_library.download_model()"

# Copy application files
COPY method_main.py .
COPY [other_files] .

# Make scripts executable (if using bash scripts)
# RUN chmod +x run_method.sh

# Set environment variable for thread count (if applicable)
# ENV NUM_THREADS=8

# Set entry point - should accept standard arguments
ENTRYPOINT ["python3", "method_main.py"]
```

## Entry Point Script (method_main.py)

Your main script should:

1. **Parse arguments**: Handle required and optional parameters
2. **Validate inputs**: Check file existence and formats
3. **Execute pipeline**: Orchestrate your method's workflow
4. **Generate output**: Produce standardized 3-column TSV

Example of `method_main.py`

```python
#!/usr/bin/env python3
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Your method description')
    
    # Required arguments
    parser.add_argument('--query_file', '-q', required=True,
                        help='FASTA file containing query sequences')
    parser.add_argument('--train_sequences', required=True,
                        help='FASTA file containing training sequences')
    parser.add_argument('--annot_file', '-a', required=True,
                        help='Annotation file (.gaf/.dat/.npz)')
    parser.add_argument('--graph', required=True,
                        help='GO ontology file (.obo)')
    parser.add_argument('--output_file', '-o', required=True,
                        help='Output predictions file')
    
    # Optional arguments
    parser.add_argument('--num_threads', type=int, default=8,
                        help='Number of threads')
    
    args = parser.parse_args()
    
    # Validate inputs
    for file_path in [args.query_file, args.train_sequences, args.annot_file, args.graph]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # Execute your method pipeline
    try:
        run_your_method(args) # example wrapper function for method
        print(f"Prediction completed successfully: {args.output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

## Data Mounting
Data mounting allows the method to bind directories on your computer to the directories inside the container. These directories can be local data folders (during container testing), or LAFA-hosted repositories (when the method is added to LAFA). Note that large data should not be included in the Docker container of your method. If your method needs access to intermediate or external data (e.g., trained model weights, protein embeddings), please contact us.

Docker data mount documentation: https://docs.docker.com/engine/storage/bind-mounts/
### Input Mounts
```bash
# Mounting `local_folder/data` directory to the `data/` directory within the container ("ro" stands for readonly)
-v /local_folder/data:/data:ro 
```

### Output Mounts
```bash
# Mount `local_folder/output` directory to the `output/` directory within the container ("rw" stands for readwrite)
-v /local_folder/output:/output:rw
```

### Docker Run Example (with data mounting)
```bash
docker run \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output:rw \
    your-method-container \
    --query_file /data/queries.fasta \
    --train_sequences /data/training.fasta \
    --annot_file /data/annotations.gaf \
    --graph /data/go-basic.obo \
    --output_file /output/predictions.tsv \
    --num_threads 16
```

You can ignore data mounting during testing by skipping the "-v" arguments in the `docker run` command.


## Build and Test

You will need Docker to build and test your containers. Steps to [install Docker on Linux](https://docs.docker.com/desktop/setup/install/linux/). If you have problem installing Docker on your local machine, please first create a Dockerfile and generate all necessary scripts, then contact us.

### Building
```bash
docker build -t your-method-container .
```

### Testing
```bash
# Test with sample data
docker run --rm \
    -v $(pwd)/test_data:/data:ro \
    -v $(pwd)/test_output:/output:rw \
    your-method-container \
    --query_file /data/test_queries.fasta \
    --train_sequences /data/test_training.fasta \
    --annot_file /data/test_annotations.gaf \
    --graph /data/go-basic.obo \
    --output_file /output/test_predictions.tsv
```

### Publishing

You will need a Dockerhub account to push your containerized method to Dockerhub. After creating an account, you can push your container to Dockerhub.

```bash
docker tag your_method yourusername/method_name:v1
docker push yourusername/method_name:v1
```