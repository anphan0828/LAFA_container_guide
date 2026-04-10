#!/usr/bin/env python3
"""
BLAST Container Main Script

This script orchestrates the BLAST-based protein function prediction pipeline:
1. Runs BLAST to find sequence similarities
2. Transfers annotations from BLAST hits and keeps the maximum score per term

Usage:
    python blast_main.py --annot_file <annotations> --query_file <queries> --graph <ontology> --output_baseline <output>
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_blast_search(query_file, train_sequences, train_taxonomy, blast_results, num_threads=4):
    """
    Run BLAST search using the run_blast.sh script
    
    Args:
        query_file: Path to query sequences FASTA file
        train_sequences: Path to training sequences FASTA file (used as database)
        train_taxonomy: Optional taxonomy map for makeblastdb -taxid_map
        blast_results: Path to output BLAST results file
        num_threads: Number of threads to use for BLAST
    """
    
    # Get the directory of this script to find run_blast.sh
    script_dir = Path(__file__).parent
    blast_script = script_dir / "run_blast.sh"
    
    if not blast_script.exists():
        raise FileNotFoundError(f"BLAST script not found: {blast_script}")
    
    # Commented out because no permission to do this, bash script should already be executable
    # os.chmod(blast_script, 0o755)
    
    print("Running BLAST search...")
    print(f"Query: {query_file}")
    print(f"Database: {train_sequences}")
    print(f"Taxonomy ID file: {train_taxonomy}")
    print(f"Output: {blast_results}")
    
    # Run the BLAST script
    cmd = [
        "bash", str(blast_script),
        "--query", query_file,
        "--database", train_sequences,
        "--output", blast_results,
        "--threads", str(num_threads)
    ]
    if train_taxonomy:
        cmd.extend(["--taxid", train_taxonomy])
    
    if not os.path.exists(blast_results):
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("BLAST search completed successfully")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"BLAST search failed with exit code {e.returncode}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            raise
    else:
        print(f"BLAST results already exist at {blast_results}. Skipping BLAST search.")


def main():
    parser = argparse.ArgumentParser(description='BLAST-based protein function prediction container')
    
    # Required arguments
    parser.add_argument('--annot_file', '-a', required=True,
                        help='Path to annotation file (.gaf or .dat) or sparse matrix (.npz)')
    parser.add_argument('--query_file', '-q', required=True,
                        help='FASTA file containing query sequences')
    parser.add_argument('--graph', required=True,
                        help='Path to GO ontology file (.obo)')
    parser.add_argument('--output_baseline', '-o', required=True,
                        help='Path to output predictions file')
    
    # Optional arguments
    parser.add_argument('--blast_results', default=None,
                        help='Path to precomputed BLAST results file (if available)')
    parser.add_argument('--indices', '-i', default=None,
                        help='Path to term indices file (required for .npz annotation files)')
    parser.add_argument('--add_graph', default=None,
                        help='Path to additional OBO file for temporal filtering')
    parser.add_argument('--train_sequences', default=None,
                        help='Path to training sequences FASTA file (required unless --blast_results already exists)')
    parser.add_argument('--train_taxonomy', default=None,
                        help='Path to training taxonomy mapping file (for BLAST database)')
    parser.add_argument('--use_rscore', action='store_true',
                        help='Use R-score instead of sequence identity for weighting')
    parser.add_argument('--keep_self_hits', action='store_true',
                        help='Keep self-hits in BLAST results')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads for BLAST and CPU prediction (default: 4)')
    parser.add_argument('--n_terms', '-n', type=int, required=False,
                        help='Upper limit for number of terms per target', default=None)
    
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error('--num_threads must be a positive integer')
    
    # Validate input files
    if not os.path.exists(args.annot_file):
        print(f"Error: Annotation file not found: {args.annot_file}")
        sys.exit(1)
    
    if not os.path.exists(args.query_file):
        print(f"Error: Query file not found: {args.query_file}")
        sys.exit(1)
        
    if not os.path.exists(args.graph):
        print(f"Error: Ontology file not found: {args.graph}")
        sys.exit(1)

    if args.train_sequences and not os.path.exists(args.train_sequences):
        print(f"Error: Training sequences file not found: {args.train_sequences}")
        sys.exit(1)
    if args.train_taxonomy and not os.path.exists(args.train_taxonomy):
        print(f"Error: Training taxonomy file not found: {args.train_taxonomy}")
        sys.exit(1)

    output_dir = os.path.dirname(args.output_baseline) or '.'
    blast_results = args.blast_results or os.path.join(output_dir, 'blast_results.tsv')
    need_blast_search = not (args.blast_results and os.path.exists(args.blast_results))

    if need_blast_search and not args.train_sequences:
        print("Error: --train_sequences is required unless --blast_results points to an existing BLAST table")
        sys.exit(1)
        
    try:
        # Step 1: Run BLAST search
        if need_blast_search:
            print("Step 1: Running BLAST search...")
            run_blast_search(
                query_file=args.query_file,
                train_sequences=args.train_sequences,
                train_taxonomy=args.train_taxonomy,
                blast_results=blast_results,
                num_threads=args.num_threads
            )
        else:
            print(f"Step 1: Reusing precomputed BLAST results at {blast_results}")
        
        # Step 2: Run blast_chunks.py for predictions
        print("Step 2: Generating predictions from BLAST results...")
        
        # Import and run blast_chunks
        from blast_chunks_optimized import blast_predict
        
        blast_predict(
            annot_file=args.annot_file,
            query_file=args.query_file,
            indices=args.indices,
            graph=args.graph,
            add_graph=args.add_graph,
            blast_results=blast_results,
            output_baseline=args.output_baseline,
            config_path=os.path.join(os.path.dirname(__file__), 'config.yaml'),
            keep_self_hits=args.keep_self_hits,
            use_rscore=args.use_rscore,
            n_terms=args.n_terms,
            num_threads=args.num_threads,
        )
        
        print("Prediction pipeline completed successfully!")
        print(f"Results saved to: {args.output_baseline}")
        
    except Exception as e:
        print(f"Error in prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # finally:
    #     # Clean up temporary BLAST results file
    #     if os.path.exists(blast_results):
    #         os.unlink(blast_results)


if __name__ == '__main__':
    main()
