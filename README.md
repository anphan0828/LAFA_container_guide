# Docker Container Guide for Protein Function Prediction Methods

This guide explains how to dockerize a protein function prediction method to participate in LAFA: Longitudinal Assessment of Functional Annotation. A protein function prediction method on LAFA predicts Gene Ontology (GO) terms for provided protein IDs and protein sequences, similar to CAFA competition. The prediction task will be repeated every time there is a new ground truth (released every two months by UniProt).

**Docker Basics**: Docker container is a way to package your entire application (code, dependencies, environment) into a portable container that runs consistently anywhere. This ensures your method can run seamlessly on LAFA's infrastructure.

We'll use our **ProtT5 container** (`prott5_container/` in this repository) as a concrete example throughout this guide to illustrate best practices.

## Container Requirements

### 1. Input Requirements
- **Required**: Protein sequences in FASTA format and/or protein IDs
- **Provided** (more details can be found in the [Data Storage on HuggingFace](#data-storage-on-huggingface) section):
  -   Training protein sequences (FASTA format)
  -   Test protein sequences (FASTA format)
  -   GO graph structure (OBO format)
  -   Training labels/terms (TSV format, 3 columns `EntryID`, `term`, `aspect`)
  -   Training taxonomy (TSV format no header, 2 columns `EntryID`,`TaxonID`)
  -   Raw GOA file containing all annotations (GAF format, gzipped)
  -   BLAST hits (TSV format, 6 columns `qseqid`, `sseqid`, `evalue`, `length`, `pident`, `nident`). BLAST self-hits are included.
  -   AlphaFoldDB structures (PDB format, stored in TAR archives)
- **Optional**: Additional parameters via command-line arguments (e.g., GO evidence code configuration)

Please make sure that your container accepts input files in the provided format. If you need additional files to be provided, let us know.

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
method_container/                    # ProtT5 example: prott5_container/
├── Dockerfile                       # Container definition
├── requirements.txt                 # Python dependencies
├── method_main.py                   # Main entry script (prott5_main.py)
├── config.yaml                      # Configuration file
├── [method_specific_files]          # e.g., prott5_embedder.py, run_prott5.sh
└── README.md                        # Usage instructions
```

## Dockerfile Template

```dockerfile
FROM python:3.10  # Adjust base image as needed
                  # ProtT5 uses: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 for GPU support

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory as /app inside the container
WORKDIR /app

# (if applicable) Set cache/model directories 
# ProtT5 example sets HuggingFace cache directories:
# ENV HF_HOME=/app/.cache/huggingface
# ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
# RUN mkdir -p /app/.cache/huggingface

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (if applicable) Pre-download models during build 
# ProtT5 pre-downloads the transformer model to avoid download delays during runtime:
# RUN python3 -c "from transformers import T5EncoderModel, T5Tokenizer; \
#     T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc'); \
#     T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')"

# Copy application files
COPY method_main.py .
COPY [other_files] .

# (if applicable) Make bash scripts executable 
# RUN chmod +x run_method.sh
# ProtT5 example: RUN chmod +x run_prott5.sh

# (if applicable) Set environment variables 
# ENV NUM_THREADS=8

# Set entry point - should accept standard arguments
ENTRYPOINT ["python3", "method_main.py"]
# ProtT5: ENTRYPOINT ["python3", "prott5_main.py"]
```

**Key Docker concepts**:
- **FROM**: Specifies the base operating system and tools (like Ubuntu + Python)
- **RUN**: Executes commands during the build process (installing packages, downloading models)
- **COPY**: Copies files from your computer into the container
- **ENV**: Sets environment variables that your program can use
- **ENTRYPOINT**: Defines what command runs when the container starts

## Entry Point Script (method_main.py)

Your main script should handle all the standard arguments and orchestrate your method's workflow. Here's a template based on the **ProtT5 example**:

The ProtT5 main wrapper script `prott5_main.py`:
1. **Argument parsing**: Handles both required and optional parameters
2. **Input validation**: Checks that input files exist before processing
3. **Pipeline orchestration**: Executes Part 1 (embedding generation) then Part 2 (prediction)
4. **Error handling**: Provides clear error messages and proper exit codes

Recommended template for `method_main.py`:
```python
#!/usr/bin/env python3
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Your method description')
    
    # Required arguments (standard LAFA interface)
    parser.add_argument('--query_file', '-q', required=True,
                        help='FASTA file containing query sequences (.fasta)')
    parser.add_argument('--train_sequences', required=True,
                        help='FASTA file containing training sequences (.fasta)')
    parser.add_argument('--annot_file', '-a', required=True,
                        help='Annotation file (.gaf)')
    parser.add_argument('--graph', required=True,
                        help='GO ontology file (.obo)')
    parser.add_argument('--output_file', '-o', required=True,
                        help='Output predictions file')
    
    # Optional arguments (method-specific)
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

Data mounting allows your container to access files on your computer (or LAFA's servers) without copying them into the container. This helps reduce the size of the container image (that will be pushed to Dockerhub). Large data should not be included in the Docker container of your method. If your method needs access to intermediate or external data (e.g., trained model weights, protein embeddings), please contact us.

Docker data mount documentation: https://docs.docker.com/engine/storage/bind-mounts/

### ProtT5 Example: Docker Run with Data Mounting
```bash
# ProtT5 container run command
docker run --rm \
    -v /path/to/data:/app/data:ro \
    -v /path/to/output:/app/output:rw \
    prott5_predictor \
    --query_file /app/data/test_sequences.fasta \
    --train_sequences /app/data/train_sequences.fasta \
    --annot_file /app/data/train_terms.tsv \
    --graph /app/data/go-basic.obo \
    --output_baseline /app/output/prott5_predictions.tsv.gz
```

**Mount Explanation**:
- `/path/to/data:/app/data:ro` - Your `path/to/data` folder becomes `/app/data` inside container (read-only)
- `/path/to/output:/app/output:rw` - Your `path/to/output` folder becomes `/app/output` inside container (read-write)

You can ignore data mounting during testing by skipping the "-v" arguments in the `docker run` command.


## Build and Test

You will need Docker to build and test your containers. Steps to [install Docker on Linux](https://docs.docker.com/desktop/setup/install/linux/). If you have problem installing Docker on your local machine, please first create a Dockerfile and generate all necessary scripts, then contact us.

For Linux users: after installation, you might need follow this [guide](https://docs.docker.com/engine/install/linux-postinstall/) to run Docker containers as a non-root user.

### Building
```bash
# Navigate to your method directory, then build the container
cd /path/to/your/method_container/
docker build -t your-method-container-name .

# ProtT5 example:
# cd prott5_container/
# docker build -t prott5_predictor .
```

**Build Process**: Docker reads your Dockerfile line by line, downloads dependencies, copies files, and creates a ready-to-run container.

### Testing
You can use the files in `test_data` folder to test your containerized method. We recommend testing with data mounting so that your method can work seamlessly once it participates in LAFA.  

```bash
# Test with sample data (basic run without data mounting)
docker run --rm \
    your-method-container \
    --query_file test_sequences.fasta \
    --train_sequences train_sequences.fasta \
    --annot_file train_terms.tsv \
    --graph go-basic.obo \
    --output_file test_predictions.tsv
```
You can also specify memory usage and CPU/GPU option (if applicable to your method). The following example is recommended if you have 16GB RAM. However, since ProtT5 embeddings are expensive to generate, limitting to 16GB RAM might cause some proteins to not be processed.

```bash
# Test with data mounting and memory/CPU/GPU usage (recommended)
docker run --rm \
    --memory=8g --memory-swap=12g \
    --cpus=4 --gpus all \ 
    -v /path/to/test_data:/app/data:ro \
    -v /path/to/test_output:/app/output:rw \
    your-method-container \
    --query_file /app/data/test_sequences.fasta \
    --train_sequences /app/data/train_sequences.fasta \
    --annot_file /app/data/train_terms.tsv \
    --graph /app/data/go-basic.obo \
    --output_file /app/output/test_predictions.tsv.gz \
    --num_threads 4
```

**Docker Flags Explained**:
- `--rm`: Automatically remove container when it finishes (cleanup)
- `-v`: Mount directories (as explained in Data Mounting section)
- (Optional)`--gpus all`: Give container access to all GPUs (for methods like ProtT5, requires more steps not listed here to make GPU accessible to the container)

### Publishing

You will need a DockerHub account to push your containerized method to DockerHub. After creating an account:

```bash
# Tag your container with your DockerHub username
docker tag your_method yourusername/method_name:v1

# Push to DockerHub (makes it publicly available)
docker push yourusername/method_name:v1

# ProtT5 example:
# docker tag prott5_predictor myusername/prott5_predictor:v1
# docker push myusername/prott5_predictor:v1
```

## Data storage on HuggingFace
Required data for input are stored in a public repository on HuggingFace. To download this data and use, you can either install HuggingFace Command Line Interface (CLI) [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli#download-a-dataset-or-a-space) or load the data directly into Python.


To download the whole dataset using HuggingFace CLI, use:
```bash
hf download anphan0828/lafa --repo-type dataset --local-dir="your_local_dir" 
```
(updated Oct 2025) The public LAFA repository on HuggingFace contains:
- AlphaFoldDB_v6 folder: PDB files of almost all SwissProt proteins (550,122 PDB files), version 6 (obtained from AlphaFoldDB FTP site). This folder contains 12 gzipped tar archives, each containing 50,000 PDB files. The list of proteins contained in each tar archive can be found in `tarball_proteins.json` file in the same directory. Example file name for protein `A0A009IHW8`: "AF-A0A009IHW8-F1-model_v6.pdb.gz"
- JunOct folder: all data files collected from UniProt release 2025_03 (released in June 2025) needed for your protein function prediction method. This directory includes:
  1. GO structure: `go-basic-20250601.obo`
  2. Training sequences: `train_sequences.fasta` includes sequences of all experimentally-annotated proteins
  3. Training terms: `train_terms.tsv` includes GO annotations of experimentally annotated proteins
  4. Test sequences: `test_sequences.fasta` includes sequences of all target proteins that we are asking predictions for. For this release, the test set includes every protein in SwissProt.
  5. BLAST hits: `blast_results.tsv` includes BLAST hits of every sequence in test set against the training set. BLAST format is as follows: `BLAST_FORMAT="6 qseqid sseqid evalue length pident nident"`. Self-hits are included.
  6. Additional files: `train_taxonomy.tsv` (taxonomy ID for every protein in the training set), `goa_uniprot_sprot.gaf.226.gz` (all GO annotations of SwissProt proteins). 
 
Note that the file names might be standardized in later releases, so input files should be passed into container as command-line arguments instead of hard-coded paths inside your script.
