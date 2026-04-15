#!/bin/bash

# BLAST helper for the container environment
# Builds a local BLAST database and runs blastp inside the image

set -e  # Exit on any error

# Set default values
QUERY_FILE=""
DB_FASTA=""
DB_TAXID=""
OUTPUT_FILE=""
NUM_THREADS=${NUM_THREADS:-4}
BLAST_FORMAT="6 qseqid sseqid evalue length pident nident"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query)
            QUERY_FILE="$2"
            shift 2
            ;;
        --database)
            DB_FASTA="$2"
            shift 2
            ;;
        --taxid)
            DB_TAXID="$2"
            shift 2
            ;;
        --output)
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
if [[ -z "$QUERY_FILE" || -z "$DB_FASTA" || -z "$OUTPUT_FILE" ]]; then
    echo "Usage: $0 --query <query_fasta> --database <database_fasta> --output <output_file> [--taxid <taxid_map>] [--threads <num_threads>]"
    echo "  --query: FASTA file containing query sequences"
    echo "  --database: FASTA file to use as BLAST database"
    echo "  --taxid: Optional taxonomy ID map for the BLAST database"
    echo "  --output: Output file for BLAST results"
    echo "  --threads: Number of threads to use (default: 4)"
    exit 1
fi

# Check if input files exist
if [[ ! -f "$QUERY_FILE" ]]; then
    echo "Error: Query file '$QUERY_FILE' not found"
    exit 1
fi

if [[ ! -f "$DB_FASTA" ]]; then
    echo "Error: Database file '$DB_FASTA' not found"
    exit 1
fi

if [[ -n "$DB_TAXID" && ! -f "$DB_TAXID" ]]; then
    echo "Error: Taxonomy ID file '$DB_TAXID' not found"
    exit 1
fi

echo "Starting BLAST analysis..."
echo "Query file: $QUERY_FILE"
echo "Database file: $DB_FASTA"
if [[ -n "$DB_TAXID" ]]; then
    echo "Taxonomy ID file: $DB_TAXID"
else
    echo "Taxonomy ID file: not provided"
fi
echo "Output file: $OUTPUT_FILE"
echo "Threads: $NUM_THREADS"

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Check if BLAST tools are available (either via system installation or Docker)
if command -v makeblastdb >/dev/null 2>&1 && command -v blastp >/dev/null 2>&1; then
    echo "Using system BLAST installation..."
    
    # Build BLAST database
    echo "Building BLAST database..."
    MAKEBLASTDB_CMD=(
        makeblastdb
        -in "$DB_FASTA"
        -parse_seqids
        -blastdb_version 5
        -title 'blast_db'
        -dbtype prot
    )
    if [[ -n "$DB_TAXID" ]]; then
        MAKEBLASTDB_CMD+=(-taxid_map "$DB_TAXID")
    fi
    "${MAKEBLASTDB_CMD[@]}"
    
    # Run BLAST search
    echo "Running BLAST search..."
    blastp -query "$QUERY_FILE" -db "$DB_FASTA" -outfmt "$BLAST_FORMAT" -mt_mode 1 -num_threads "$NUM_THREADS" -out "$OUTPUT_FILE"

else
    echo "Error: Neither BLAST tools nor Docker are available"
    echo "Please install BLAST+ tools or Docker to run this script"
    exit 1
fi

# Verify output file was created
if [[ -f "$OUTPUT_FILE" ]]; then
    RESULT_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo "BLAST analysis completed successfully!"
    echo "Results written to: $OUTPUT_FILE"
    echo "Number of hits found: $RESULT_COUNT"
else
    echo "Error: Output file was not created"
    exit 1
fi
