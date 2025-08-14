#!/bin/bash

# ProtT5 script for Docker container environment
# This script runs ProtT5 embeddings and similarity calculations within a container

set -e  # Exit on any error

# Set default values
QUERY_FILE=""
DATABASE_FILE=""
MODEL_DIR=${HF_CACHE:-"/app/.cache/huggingface/"}
OUTPUT_FILE=""
NUM_THREADS=${NUM_THREADS:-8}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query|-q)
            QUERY_FILE="$2"
            shift 2
            ;;
        --database|-d)
            DATABASE_FILE="$2"
            shift 2
            ;;
        --model|-m)
            MODEL_DIR="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$QUERY_FILE" || -z "$DATABASE_FILE" || -z "$OUTPUT_FILE" ]]; then
    echo "Usage: $0 --query <query_fasta> --database <database_fasta> --output <output_file> [--model <model_dir>] [--threads <num_threads>]"
    echo "  --query: FASTA file containing query sequences"
    echo "  --database: FASTA file containing database sequences"
    echo "  --output: Output file for similarity results"
    echo "  --model: Directory for HuggingFace model cache (default: \$HF_CACHE or /app/.cache/huggingface/)"
    echo "  --threads: Number of threads to use (default: 8)"
    exit 1
fi

# Check if input files exist
if [[ ! -f "$QUERY_FILE" ]]; then
    echo "Error: Query file '$QUERY_FILE' not found"
    exit 1
fi

if [[ ! -f "$DATABASE_FILE" ]]; then
    echo "Error: Database file '$DATABASE_FILE' not found"
    exit 1
fi

echo "Starting ProtT5 analysis..."
echo "Query file: $QUERY_FILE"
echo "Database file: $DATABASE_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Model cache directory: $MODEL_DIR"
echo "Threads: $NUM_THREADS"

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for intermediate files
TEMP_DIR=$(mktemp -d -p "$OUTPUT_DIR")
echo "Using temporary directory: $TEMP_DIR"

# Define intermediate file paths
EVALSET_EMBEDDINGS="$TEMP_DIR/evalset_embeddings.h5"
DBSET_EMBEDDINGS="$TEMP_DIR/dbset_embeddings.h5"

# Set environment variables
export HF_HOME="$MODEL_DIR"
export NUM_THREADS="$NUM_THREADS"

echo "Step 1: Generating ProtT5 embeddings for query set..."
python3 prott5_embedder.py --input "$QUERY_FILE" --output "$EVALSET_EMBEDDINGS" --per_protein 1 --model "$MODEL_DIR"

echo "Step 2: Generating ProtT5 embeddings for database set..."
python3 prott5_embedder.py --input "$DATABASE_FILE" --output "$DBSET_EMBEDDINGS" --per_protein 1 --model "$MODEL_DIR"

echo "Step 3: Computing similarity matrix and processing embeddings..."
python3 process_embeddings_gpu.py "$EVALSET_EMBEDDINGS" "$DBSET_EMBEDDINGS" "$OUTPUT_FILE"

echo "Step 4: Normalizing similarity scores..."
python3 normalize_embeddings.py "$OUTPUT_FILE"

# # Generate normalized output filename
OUTPUT_NORM=${OUTPUT_FILE/.tsv/_norm.tsv}

# echo "Step 5: Cleaning up format (removing UniProt prefixes)..."
# # Remove sp| and tr| prefixes and clean up IDs
# sed -ri 's/sp\|//g' "$OUTPUT_NORM"
# sed -ri 's/tr\|//g' "$OUTPUT_NORM"
# sed -ri 's/\|[^\t]*\t/\t/g' "$OUTPUT_NORM"

# Clean up intermediate files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Verify output files were created
if [[ -f "$OUTPUT_FILE" ]]; then
    RESULT_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo "ProtT5 analysis completed successfully!"
    echo "Raw results written to: $OUTPUT_FILE"
    echo "Normalized results written to: $OUTPUT_NORM"
    echo "Number of similarity results: $RESULT_COUNT"
else
    echo "Error: Output file was not created"
    exit 1
fi
