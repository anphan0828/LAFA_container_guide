# BLAST Container for Protein Function Prediction

This Docker container implements a BLAST-based protein function prediction pipeline. It runs NCBI BLAST to find sequence similarities and then uses those results to predict GO term annotations.

## Contents

- `blast_main.py` - Main orchestration script
- `blast_chunks.py` - BLAST-based prediction algorithm (using chunk strategy)
- `run_blast.sh` - Modified BLAST execution script (works with Docker instead of HPC)
- `retrieve_terms.py` - GO term retrieval utilities
- `ontology.py` - Ontology processing utilities
- `config.yaml` - GO evidence code configuration
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition

## Building the Container

```bash
# From the blast_container directory
docker build -t blast_predictor .
```

## Running the Container

### Basic Usage

```bash
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  blast_predictor \
  --annot_file /app/data/annotations.gaf.gz \
  --query_file /app/data/test_sequences.fasta \
  --train_sequences /app/data/train_sequences.fasta \
  --graph /app/data/go-basic.obo \
  --output_baseline /app/output/blast_predictions.tsv.gz
```

### Required Arguments

- `--annot_file`: Path to annotation file (.gaf/.dat or .npz sparse matrix)
- `--query_file`: FASTA file containing query sequences to predict
- `--train_sequences`: FASTA file containing training sequences (for BLAST database)
- `--graph`: Path to GO ontology file (.obo format)
- `--output_baseline`: Path to output predictions file

### Optional Arguments

- `--indices`: Path to term indices file (required when using .npz annotation files)
- `--add_graph`: Path to additional OBO file for temporal filtering
- `--use_rscore`: Use R-score instead of sequence identity for weighting
- `--keep_self_hits`: Keep self-hits in BLAST results
- `--num_threads`: Number of threads for BLAST (default: 8)

### Example with Real Data

```bash
docker run --rm \
  -v /home/user/cafa6:/app/data \
  -v /home/user/cafa6/output:/app/output \
  blast_predictor \
  --annot_file /app/data/goa_uniprot_filtered.gaf.gz \
  --query_file /app/data/test_superset_all.fasta \
  --train_sequences /app/data/train_sequences.fasta \
  --graph /app/data/go-basic.obo \
  --output_baseline /app/output/blast_predictions.tsv.gz \
  --num_threads 16
```

## How It Works

1. **BLAST Search**: The container first runs BLAST to find sequence similarities between query sequences and training sequences
2. **Annotation Loading**: Loads GO term annotations from the provided annotation file
3. **Prediction**: Uses BLAST hits and their annotations to generate GO term predictions for query sequences
4. **Output**: Saves predictions in TSV format with columns: [protein_id, go_term, score]

## BLAST Backend

The container can use either:
- System-installed BLAST+ tools (if available in the container). This container is set up to install BLAST+ tools inside the container.
<!-- - NCBI BLAST Docker container (ncbi/blast:latest) - requires Docker socket access

For Docker-in-Docker usage, you may need to mount the Docker socket:

```bash
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  blast_predictor \
  [arguments...]
``` -->

## Performance Notes

- BLAST is computationally intensive; adjust `--num_threads` based on available CPU cores
- Memory usage depends on the size of training sequences and number of query sequences
- For large datasets, consider chunking the query sequences

## Dependencies

- Python 3.10
- BioPython for sequence parsing
- NetworkX and obonet for ontology processing
- SciPy for sparse matrix operations
- Pandas for data manipulation
- NCBI BLAST+ tools for sequence similarity search
