#!/usr/bin/env python3
"""
ProtT5 Container Main Script

This script orchestrates the ProtT5-based protein function prediction pipeline:
1. Runs ProtT5 to generate embeddings and compute similarities
2. Uses similarity results to generate GO term predictions

Usage:
    python prott5_main.py --annot_file <annotations> --query_file <queries> --train_sequences <training> --graph <ontology> --output_baseline <output>
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_prott5_analysis(query_file, train_sequences, prott5_results, model_dir=None, num_threads=8):
    """
    Run ProtT5 analysis using the run_prott5.sh script
    
    Args:
        query_file: Path to query sequences FASTA file
        train_sequences: Path to training sequences FASTA file
        prott5_results: Path to output similarity results file
        model_dir: Path to HuggingFace model cache directory
        num_threads: Number of threads to use
    """
    
    # Get the directory of this script to find run_prott5.sh
    script_dir = Path(__file__).parent
    prott5_script = script_dir / "run_prott5.sh"
    
    if not prott5_script.exists():
        raise FileNotFoundError(f"ProtT5 script not found: {prott5_script}")
    
    # Make sure the script is executable
    # os.chmod(prott5_script, 0o755)
    
    print(f"Running ProtT5 analysis...")
    print(f"Query: {query_file}")
    print(f"Database: {train_sequences}")
    print(f"Output: {prott5_results}")
    
    # Build command arguments
    cmd = [
        str(prott5_script),
        "--query", query_file,
        "--database", train_sequences,
        "--output", prott5_results,
        "--threads", str(num_threads)
    ]
    
    if model_dir:
        cmd.extend(["--model", model_dir])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("ProtT5 analysis completed successfully")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ProtT5 analysis failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description='ProtT5-based protein function prediction container')
    
    # Required arguments
    parser.add_argument('--annot_file', '-a', required=True,
                        help='Path to annotation file (.gaf or .dat) or sparse matrix (.npz)')
    parser.add_argument('--query_file', '-q', required=True,
                        help='FASTA file containing query sequences')
    parser.add_argument('--train_sequences', required=True,
                        help='FASTA file containing training sequences')
    parser.add_argument('--graph', required=True,
                        help='Path to GO ontology file (.obo)')
    parser.add_argument('--output_baseline', '-o', required=True,
                        help='Path to output predictions file')
    
    # Optional arguments
    parser.add_argument('--indices', '-i', default=None,
                        help='Path to term indices file (required for .npz annotation files)')
    parser.add_argument('--add_graph', default=None,
                        help='Path to additional OBO file for temporal filtering')
    parser.add_argument('--model_dir', default=None,
                        help='Path to HuggingFace model cache directory (default: $HF_CACHE or /app/.cache/huggingface/)')
    parser.add_argument('--keep_self_hits', action='store_true',
                        help='Keep self-hits in ProtT5 results')
    parser.add_argument('--num_threads', type=int, default=8,
                        help='Number of threads (default: 8)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.annot_file):
        print(f"Error: Annotation file not found: {args.annot_file}")
        sys.exit(1)
    
    if not os.path.exists(args.query_file):
        print(f"Error: Query file not found: {args.query_file}")
        sys.exit(1)
        
    if not os.path.exists(args.train_sequences):
        print(f"Error: Training sequences file not found: {args.train_sequences}")
        sys.exit(1)
        
    if not os.path.exists(args.graph):
        print(f"Error: Ontology file not found: {args.graph}")
        sys.exit(1)
    
    # Set model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.environ.get('HF_CACHE', '/app/.cache/huggingface/')
    
    # Create file for ProtT5 results
    # Use normalized output since that's what prott5_chunks expects
    prott5_results_norm = f"{os.path.dirname(args.output_baseline)}/prott5_results_norm.tsv"

    # The raw results file (before normalization)
    prott5_results_raw = prott5_results_norm.replace('_norm.tsv', '.tsv')
    
    try:
        # Step 1: Run ProtT5 analysis
        print("Step 1: Running ProtT5 embedding analysis...")
        run_prott5_analysis(
            query_file=args.query_file,
            train_sequences=args.train_sequences,
            prott5_results=prott5_results_raw,
            model_dir=model_dir,
            num_threads=args.num_threads
        )
        
        # Check if normalized file was created
        if not os.path.exists(prott5_results_norm):
            print(f"Error: Normalized ProtT5 results file not found: {prott5_results_norm}")
            sys.exit(1)
        
        # Step 2: Run prott5_chunks.py for predictions
        print("Step 2: Generating predictions from ProtT5 similarity results...")
        
        # Import and run prott5_chunks
        from prott5_chunks import prott5_predict
        
        prott5_predict(
            annot_file=args.annot_file,
            query_file=args.query_file,
            indices=args.indices,
            graph=args.graph,
            add_graph=args.add_graph,
            prott5_results=prott5_results_norm,
            output_baseline=args.output_baseline,
            config_path=os.path.join(os.path.dirname(__file__), 'config.yaml'),
            keep_self_hits=args.keep_self_hits
        )
        
        print(f"ProtT5 prediction pipeline completed successfully!")
        print(f"Results saved to: {args.output_baseline}")
        
    except Exception as e:
        print(f"Error in ProtT5 prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # finally:
    #     # Clean up temporary ProtT5 results files
    #     for temp_file in [prott5_results_raw, prott5_results_norm]:
    #         if os.path.exists(temp_file):
    #             os.unlink(temp_file)


if __name__ == '__main__':
    main()
