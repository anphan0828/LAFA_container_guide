# ProtT5 Container for Protein Function Prediction

This Docker container implements a ProtT5-based protein function prediction pipeline. It uses ProtT5 transformer embeddings to compute sequence similarities and then uses those similarities to predict GO term annotations.

## Contents

- `prott5_main.py` - Main orchestration script
- `prott5_chunks.py` - ProtT5-based prediction algorithm (using chunk strategy)
- `run_prott5.sh` - Modified ProtT5 execution script (works in container instead of HPC)
- `prott5_embedder.py` - ProtT5 embedding generation script
- `process_embeddings_gpu.py` - Similarity calculation from embeddings
- `normalize_embeddings.py` - Similarity score normalization
- `retrieve_terms.py` - GO term retrieval utilities
- `ontology.py` - Ontology processing utilities
- `config.yaml` - GO evidence code configuration
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition

## Building the Container

```bash
# From the prott5_container directory
docker build -t prott5_predictor .
```

**Note**: This container requires GPU support for optimal performance. Build with NVIDIA Docker runtime.

## Running the Container

### Basic Usage

```bash
docker run --gpus all --rm \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  -v /path/to/cache:/app/.cache \
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

### Example with Real Data

```bash
docker run --gpus all --rm \
  -v /home/user/cafa6:/app/data \
  -v /home/user/cafa6/output:/app/output \
  -v /home/user/.cache:/app/.cache \
  prott5_predictor \
  --annot_file /app/data/goa_uniprot_filtered.gaf.gz \
  --query_file /app/data/test_superset_all.fasta \
  --train_sequences /app/data/train_sequences.fasta \
  --graph /app/data/go-basic.obo \
  --output_baseline /app/output/prott5_predictions.tsv.gz \
  --num_threads 16
```

### Running with Model Caching

For better performance and to avoid re-downloading models:

```bash
# Create a persistent cache directory
mkdir -p /home/user/.cache/huggingface

# Run with mounted cache
docker run --gpus all --rm \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  -v /home/user/.cache:/app/.cache \
  prott5_predictor \
  [arguments...]
```

## How It Works

1. **ProtT5 Embeddings**: The container first generates ProtT5 transformer embeddings for both query and training sequences
2. **Similarity Calculation**: Computes euclidean distances between query and training embeddings
3. **Score Normalization**: Normalizes similarity scores to percentages
4. **Annotation Loading**: Loads GO term annotations from the provided annotation file
5. **Prediction**: Uses similarity scores and annotations to generate GO term predictions for query sequences
6. **Output**: Saves predictions in TSV format with columns: [protein_id, go_term, score]

## GPU Requirements

- **Recommended**: NVIDIA GPU with at least 8GB VRAM
- **Minimum**: NVIDIA GPU with 4GB VRAM (may require smaller batch sizes)
- **CPU fallback**: Will run on CPU but significantly slower

## Model Information

- **Default Model**: `Rostlab/prot_t5_xl_half_uniref50-enc`
- **Model Size**: ~2.3GB
- **Cache Location**: `/app/.cache/huggingface/` (mountable)
- **First Run**: Model will be downloaded automatically from HuggingFace

## Performance Notes

- **GPU Memory**: Adjust batch sizes if you encounter out-of-memory errors
- **Model Caching**: Mount a persistent cache directory to avoid re-downloading models
- **Embedding Storage**: Temporary embeddings are stored in HDF5 format during processing
- **Parallelization**: Uses multi-threading for similarity calculations
- **Memory Usage**: Can be memory-intensive for large protein sets

## Environment Variables

- `HF_CACHE`: HuggingFace model cache directory (default: `/app/.cache/huggingface/`)
- `NUM_THREADS`: Number of threads for processing (default: 8)
- `CUDA_VISIBLE_DEVICES`: GPU device selection (if multiple GPUs)

## Troubleshooting

### GPU Issues
```bash
# Check GPU availability
docker run --gpus all --rm nvidia/cuda:11.8-base nvidia-smi

# Run without GPU (CPU mode)
docker run --rm prott5_predictor [arguments...]
```

### Memory Issues
- Reduce `--num_threads` value
- Process smaller batches of sequences
- Ensure sufficient system RAM (recommend 16GB+)

### Model Download Issues
- Ensure internet connectivity
- Check HuggingFace Hub status
- Verify mounted cache directory permissions

## Dependencies

- **CUDA**: 11.8+ (for GPU support)
- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support
- **Transformers**: 4.21+ (HuggingFace)
- **BioPython**: For sequence parsing
- **h5py**: For embedding storage
- **scikit-learn**: For similarity calculations
- **NetworkX and obonet**: For ontology processing
- **SciPy**: For sparse matrix operations
- **Pandas**: For data manipulation

## Comparison with BLAST Container

| Feature | BLAST Container | ProtT5 Container |
|---------|----------------|------------------|
| **Method** | Sequence alignment | Transformer embeddings |
| **Speed** | Fast | Slower (embedding generation) |
| **Accuracy** | Good for similar sequences | Better for distant homologs |
| **GPU Requirement** | No | Yes (recommended) |
| **Memory Usage** | Moderate | High |
| **Model Size** | None | ~2.3GB |
