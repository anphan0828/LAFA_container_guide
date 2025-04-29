#!/usr/bin/env python3
"""
Naive predictor implementation for CAFA2 evaluation.
Makes predictions based on term frequency in training data.

This is a direct port of the MATLAB pfp_naive.m function.
"""

import sys
import numpy as np
from scipy import sparse
from typing import Dict
import pandas as pd
import pickle as cp
import dask.dataframe as dd
from Bio import SeqIO
import argparse   


def naive_predict(annotations: sparse.csr_matrix, query_file: str, indices: str, output_baseline) -> pd.DataFrame:
    """
    Make predictions based on term frequencies in training annotations.
    
    Args:
        query_file: FASTA file or text file containing query IDs
        annotations: Training annotations sparse matrix (n_sequences x n_terms)
        term_indices: Path to the term indices file of the sparse matrix
        
    Returns:
        DataFrame containing prediction scores for each query sequence and term.
        
    Note:
        Assigns the same term frequency-based scores to all query sequences.
    """
    # TODO: accept input of raw GAF file and propagate annotations
    # Load annotation matrix
    annotation_mat = sparse.load_npz(annotations)
    # Load protein and term indices
    with open(indices, 'rb') as f:
        proteins, terms = cp.load(f)
    
    # For naive baseline predictor, only use terms, don't need training proteins    
    query_ids = []
    if query_file.endswith('.fasta'):
        print("Reading query IDs from FASTA file")
        with open(query_file, 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                entry_id = record.id.split("|")[1] if "|" in record.id else record.id
                query_ids.append(entry_id)
    elif query_file.endswith('.txt'):
        print("Reading query IDs from text file")
        with open(query_file, 'r') as handle:
            query_ids = [line.strip() for line in handle]
    else:
        print("Please provide a fasta file or a text file with query IDs")
        sys.exit(1)
        
    # Calculate term frequencies from annotation_mat
    term_frequencies = np.array(annotation_mat.sum(axis=0)).flatten() # sum along rows (within columns), not affected by number of columns (terms)
    # Normalize scores by maximum frequency
    max_freq = np.max(term_frequencies)
    if max_freq > 0:
        normalized_scores = term_frequencies / max_freq
    else:
        normalized_scores = term_frequencies
        
        
    # Create output matrix with same scores for all queries
    n_queries = len(query_ids)
    n_terms = annotation_mat.shape[1]
    scores = sparse.lil_matrix((n_queries, n_terms), dtype=np.float32)
    for i in range(n_queries):
        scores[i] = normalized_scores # TODO: this can be optimized further
    scores = scores.tocsr()
        
    # Turn sparse matrix into dataframe format
    scores_df = pd.DataFrame(scores.toarray(), index=query_ids)
    scores_df.columns = terms
    melted_df = scores_df.reset_index().melt(id_vars = 'index',var_name = 'term', value_name = 'value')
    melted_df = melted_df.rename(columns ={'index':'EntryID'})
    melted_df = melted_df[melted_df['value'] > 0] 
    print(melted_df.head())
    
    # Create Dask DataFrame
    # TODO: speed this step up
    dask_df = dd.from_pandas(melted_df, npartitions=16)
    dask_df.to_csv(output_baseline, sep="\t", index=False, header=False, single_file = True, compression='gzip')
    return 


def parse_args(argv):
    parser = argparse.ArgumentParser(description='UniProt non-experimental annotations baseline')
    parser.add_argument('--annot_matrix', '-a', 
                        help='Path to the propagated annotation sparse matrix', required=True)
    parser.add_argument('--indices', '-i', 
                        help='Path to the term & protein indices file', required=True)
    parser.add_argument('--query_file', '-q', 
                        help='FASTA file or text file containing query IDs', required=True)
    parser.add_argument('--output_baseline', '-o', 
                        help='Path to the output file', required=True)
    return parser.parse_args(argv)    


def main():
    args = parse_args(sys.argv[1:])
    naive_predict(
        annotations=args.annot_matrix,
        query_file=args.query_file,
        indices=args.indices,
        output_baseline=args.output_baseline
    )
    
    
if __name__ == '__main__':
    main()