# ProtT5 Container for Protein Function Prediction

This Docker container implements a ProtT5-based protein function prediction pipeline. It uses ProtT5 transformer embeddings to compute sequence similarities and then uses those similarities to predict GO term annotations.

## Contents

- `prott5_main.py` - Main orchestration script
- `run_prott5.sh` - Part 1 wrapper of ProtT5 execution script
- `prott5_embedder.py` - ProtT5 embedding generation script for database set and query set
- `process_embeddings_gpu.py` - Similarity calculation from embeddings
- `normalize_embeddings.py` - Similarity score normalization
- `prott5_chunks.py` - Part 2 wrapper of ProtT5-based prediction algorithm (using chunk strategy)
- `retrieve_terms.py` - GO term retrieval from gzipped GAF file
- `ontology.py` - Ontology processing utilities
- `config.yaml` - GO evidence code configuration
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition

## How It Works
`prott5_main.py` orchestrates two parts of the method. 

Part 1 (wrapped in `run_prott5.sh` script) includes steps 1-3 below:
1. **ProtT5 Embeddings**: The container first generates ProtT5 transformer embeddings for both query and database (training) sequences
2. **Similarity Calculation**: Computes euclidean distances between each query to every database embedding
3. **Score Normalization**: Normalizes similarity scores to percentages, which will be used as confidence score to transfer GO annotations from database sequences to query sequences

Part 2 (wrapped in `prott5_chunks.py` script) includes steps 4-6:

4. **Annotation Loading**: Loads GO term annotations from the provided annotation file
5. **Prediction**: Uses similarity scores and annotations to generate GO term predictions for query sequences
6. **Output**: Saves predictions in TSV format with columns (without header): [protein_id, go_term, score]


## Building the Container
The container should be built after all scripts have been generated/collected. 

```bash
# From the prott5_container directory
docker build -t prott5_predictor .
```

## Running the Container

### Basic Usage with Test Data

```bash
docker run --rm \
  -v /path/to/test_data:/app/data \
  -v /path/to/test_output:/app/output \
  prott5_predictor \
  --annot_file /app/data/annotations.gaf.gz \
  --query_file /app/data/test_sequences.fasta \
  --train_sequences /app/data/train_sequences.fasta \
  --graph /app/data/go-basic.obo \
  --output_baseline /app/output/prott5_predictions.tsv.gz
```

### Required Arguments

- `--annot_file`: Path to annotation file (.gaf/.dat or .npz sparse matrix)
- `--query_file`: FASTA file containing query sequences to predict
- `--train_sequences`: FASTA file containing training sequences (for similarity database)
- `--graph`: Path to GO ontology file (.obo format)
- `--output_baseline`: Path to output predictions file

### Optional Arguments

- `--indices`: Path to term indices file (required when using .npz annotation files)
- `--add_graph`: Path to additional OBO file for temporal filtering
- `--model_dir`: Path to HuggingFace model cache directory (default: /app/.cache/huggingface/)
- `--keep_self_hits`: Keep self-hits in similarity results
- `--num_threads`: Number of threads (default: 8)

### Running with Model Caching

For better performance and to avoid re-downloading models, you can create a local directory for caching model and then mount this cache directory to the container `/app/.cache`:

```bash
# Create a local cache directory
mkdir -p /home/user/.cache/huggingface

# Run with mounted cache
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  -v /home/user/.cache:/app/.cache \
  prott5_predictor \
  [arguments...]
```

## GPU Requirements

- **Recommended**: TBD
- **Minimum**: TBD
- **CPU fallback**: Will run on CPU but significantly slower

## Model Information

- **Default Model**: `Rostlab/prot_t5_xl_half_uniref50-enc`
- **Model Size**: ~2.3GB
- **Cache Location**: `/app/.cache/huggingface/` (mountable)
- **First Build**: Model will be downloaded automatically from HuggingFace

## Dependencies

- **CUDA**: 11.8+ (for GPU support)
- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support
- **Transformers**: 4.21+ (HuggingFace)
- **BioPython**: 1.85 (or sequence parsing)
- **h5py**: For embedding storage
- **scikit-learn**: For similarity calculations
- **NetworkX and obonet**: For ontology processing
- **SciPy**: For sparse matrix operations
- **Pandas**: For data manipulation
