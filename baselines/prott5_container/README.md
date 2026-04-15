# ProtT5 Container for Protein Function Prediction

This container implements a ProtT5-based protein function prediction baseline for CAFA-style evaluation.

The active pipeline is:
1. Generate mean-pooled ProtT5 embeddings for the query proteins.
2. Generate mean-pooled ProtT5 embeddings for the training proteins.
3. Compute top-k nearest training neighbors for each query in embedding space using Euclidean distance.
4. Retain only those top-k raw distances per query.
5. Normalize the retained distances into percentage similarity scores using the maximum distance across the retained rows.
6. Transfer GO terms from all annotated ProtT5 hits to each query.
7. Keep each GO term score as the maximum transferred score across hits, matching the BLAST container transfer rule.

Repeated query-subject rows are deduplicated before transfer, keeping the strongest ProtT5 similarity for that pair.

## Active Files

- `prott5_main.py`: container entrypoint and pipeline orchestration
- `run_prott5.sh`: wrapper that generates embeddings and similarity tables
- `prott5_embedder.py`: ProtT5 embedding generation for FASTA inputs
- `process_embeddings_gpu_optimized.py`: nearest-neighbor search and distance normalization
- `prott5_gpu.py`: active annotation-transfer predictor
- `retrieve_terms.py`: annotation extraction utilities
- `ontology.py`: annotation matrix construction utilities
- `config.yaml`: GO evidence-code groups
- `requirements2.txt`: active Docker dependency list
- `Dockerfile`: container definition

## Cleanup Candidates

- `prott5_chunks.py`: older CPU-side prediction path; no longer imported by `prott5_main.py` or copied by the Dockerfile
- `requirements.txt`: not used by the Dockerfile; `requirements2.txt` is the active dependency file

## Build

```bash
docker build -t prott5_predictor democafa/baselines/prott5_container
```

## Run

### Generate ProtT5 similarities inside the container

```bash
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  -v /path/to/cache:/app/.cache \
  prott5_predictor \
  --annot_file /app/data/train_terms.tsv \
  --query_file /app/data/test_sequences.fasta \
  --train_sequences /app/data/train_sequences.fasta \
  --graph /app/data/go-basic.obo \
  --output_baseline /app/output/prott5_predictions.tsv.gz
```

### Reuse an existing ProtT5 similarity table

```bash
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  prott5_predictor \
  --annot_file /app/data/train_terms.tsv \
  --query_file /app/data/test_sequences.fasta \
  --graph /app/data/go-basic.obo \
  --prott5_results /app/data/prott5_results_norm.tsv \
  --output_baseline /app/output/prott5_predictions.tsv.gz
```

## Arguments

### Required

- `--annot_file`: training annotations as `.tsv`, `.gaf`, `.dat`, or propagated sparse `.npz`
- `--query_file`: query proteins as FASTA or one ID per line text file
- `--graph`: GO ontology `.obo`
- `--output_baseline`: output prediction table

### Optional

- `--train_sequences`: training FASTA used to generate ProtT5 embedding neighbors; required unless `--prott5_results` already exists
- `--prott5_results`: reuse a precomputed normalized ProtT5 similarity table
- `--indices`: required with `.npz` annotations
- `--add_graph`: reserved additional ontology input
- `--model_dir`: HuggingFace cache directory; default is `$HF_CACHE` or `/app/.cache/huggingface/`
- `--keep_self_hits`: keep self-hits instead of removing exact query-vs-subject accession matches
- `--num_threads`: passed through to the embedding wrapper script; default `8`
- `--top_k`: number of nearest neighbors retained per query before normalization; default `3`
- `--n_terms`: cap the number of output terms per query

## Similarity File Format

The active predictor expects the normalized ProtT5 results table produced by `run_prott5.sh`, with header columns equivalent to:

```text
Query ID    DB ID    e-val    Length    Similarity    N-ident
```

The transfer step uses the `Similarity` column as the score source and writes the final prediction file without a header:

```text
<query_protein_id>    <go_term>    <score>
```

## Notes

- The active shipped prediction path is GPU-based: `prott5_main.py` calls `prott5_gpu.py` directly.
- The Docker image uses `prott5_gpu.py` for annotation transfer, not `prott5_chunks.py`.
- `process_embeddings_gpu_optimized.py` currently performs GPU nearest-neighbor search, truncation to `top_k`, and post-truncation normalization in the same script.
- The image pre-downloads `Rostlab/prot_t5_xl_half_uniref50-enc` during build.
- Mounting a persistent HuggingFace cache directory avoids repeated model downloads.
