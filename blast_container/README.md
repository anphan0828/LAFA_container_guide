# BLAST Container for Protein Function Prediction

This container implements a BLAST-based protein function prediction baseline for CAFA-style evaluation.

The active pipeline is:
1. Build a BLAST database from the training FASTA file.
2. Run `blastp` from each query protein against the training database and write a BLAST tabular file.
3. For each query, transfer GO terms from all annotated BLAST hits.
4. Score each transferred term by the hit similarity:
   - default: `pident / 100`
   - with `--use_rscore`: `min(-log10(evalue) + 2, 500)`
5. If a term is transferred by multiple hits, keep the maximum transferred score for that term.

Repeated alignments for the same query-subject pair are deduplicated before transfer, keeping the strongest scoring hit for that pair.

## Active Files

- `blast_main.py`: container entrypoint and pipeline orchestration
- `blast_chunks_optimized.py`: active multi-hit annotation transfer predictor
- `run_blast.sh`: BLAST database creation and `blastp` execution
- `retrieve_terms.py`: annotation extraction utilities
- `ontology.py`: annotation matrix construction utilities
- `config.yaml`: GO evidence-code groups
- `Dockerfile`: container definition
- `requirements.txt`: Python dependencies

## Legacy File Kept for Now

- `blast_chunks.py`: older multi-hit implementation; no longer used by the container entrypoint or Dockerfile

## Build

```bash
docker build -t blast_predictor democafa/baselines/blast_container
```

## Run

### Generate BLAST results inside the container

```bash
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  blast_predictor \
  --annot_file /app/data/train_terms.tsv \
  --query_file /app/data/test_sequences.fasta \
  --train_sequences /app/data/train_sequences.fasta \
  --train_taxonomy /app/data/train_taxonomy.tsv \
  --graph /app/data/go-basic.obo \
  --output_baseline /app/output/blast_predictions.tsv.gz
```

### Reuse an existing BLAST table

```bash
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/output \
  blast_predictor \
  --annot_file /app/data/train_terms.tsv \
  --query_file /app/data/test_sequences.fasta \
  --graph /app/data/go-basic.obo \
  --blast_results /app/data/blast_results.tsv \
  --output_baseline /app/output/blast_predictions.tsv.gz
```

## Arguments

### Required

- `--annot_file`: training annotations as `.tsv`, `.gaf`, `.dat`, or propagated sparse `.npz`
- `--query_file`: query proteins as FASTA or one ID per line text file
- `--graph`: GO ontology `.obo`
- `--output_baseline`: output prediction table

### Optional

- `--blast_results`: reuse a precomputed BLAST result file; skips the BLAST search step
- `--train_sequences`: training FASTA used to build the BLAST database; required unless `--blast_results` already exists
- `--train_taxonomy`: optional taxonomy map passed to `makeblastdb -taxid_map`
- `--indices`: required with `.npz` annotations
- `--add_graph`: reserved additional ontology input
- `--use_rscore`: score transferred terms with R-score instead of percent identity
- `--keep_self_hits`: keep self-hits instead of removing exact query-vs-subject accession matches
- `--n_terms`: cap the number of output terms per query
- `--num_threads`: threads for BLAST and CPU-side prediction; default `4`
- `--batch_size`: query batch size used by `blast_chunks_optimized.py`; default `1000`

## Output

The output is a tab-separated file with no header:

```text
<query_protein_id>    <go_term>    <score>
```

Each query receives GO terms aggregated across its annotated BLAST hits, with the final score for each term equal to the maximum transferred score across those hits.

## Notes

- The Docker image sets `NUM_THREADS=4` by default.
- `run_blast.sh` also defaults to `4` threads when `--threads` is omitted.
- BLAST database files are created next to the provided training FASTA file.
- The container uses the system-installed `ncbi-blast+` tools from the image.
