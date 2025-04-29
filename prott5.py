#!/usr/bin/env python3
"""
Embedding-based predictor implementation.
Makes predictions based on ProtT5 embedding similarity.

"""

import sys
import numpy as np
import pandas as pd
from scipy import sparse
import pickle as cp
from Bio import SeqIO
from typing import Dict
import dask.dataframe as dd
import argparse


# def calculate_rscore(evalue: float, max_rscore: float = 500.0) -> float:
#     """
#     Calculate R-score from BLAST E-value.
#     R-score = -log(E-value) + 2, capped at max_rscore
    
#     Args:
#         evalue: BLAST E-value
#         max_rscore: Maximum allowed R-score (default: 500)
        
#     Returns:
#         R-score value
#     """
#     if evalue == 0:
#         return max_rscore
#     rscore = -np.log10(evalue) + 2
#     return min(rscore, max_rscore)


def prott5_predict(annotations: sparse.csr_matrix,
                  query_file: str, 
                  indices: str,
                  prott5_results: pd.DataFrame,
                  output_baseline: str,
                  keep_self_hits: bool = False) -> pd.DataFrame:
    """
    Make predictions for query sequences based on ProtT5 embedding similarity hits.
    
    Args:
        query_file: Path to list of query sequence IDs (.fasta or .txt)
        prott5_results: Path to ProtT5 results file
        annotations: Training annotations sparse matrix (n_sequences x n_terms)
        indices: Path to the term & protein indices file
        keep_self_hits: Whether to keep self-hits (True) or remove them (False)
                   
    Note:
        For each query sequence, finds ProtT5 hits in training data,
        weights their annotations by sequence similarity, and takes
        the maximum score per term across all hits.
    """
    # Load annotation matrix
    annotation_mat = sparse.load_npz(annotations)
    # Load protein and term indices
    with open(indices, 'rb') as f:
        proteins, terms = cp.load(f)
    
    # For ProtT5 baseline predictor, use both proteins and terms indices     
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
    
    
    n_queries = len(query_ids)
    n_terms = len(terms)
    scores = sparse.lil_matrix((n_queries, n_terms), dtype=np.float32)
    
    # Load prott5 results
    prott5_df = pd.read_csv(prott5_results, sep='\t', header=0, 
                           names=['qseqid', 'sseqid', 'evalue', 'length', 'similarity', 'nident'])
    
    # Remove self-hits
    prott5_df['qseqid_acc'] = prott5_df['qseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
    prott5_df['sseqid_acc'] = prott5_df['sseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
    if not keep_self_hits:
        prott5_df = prott5_df[prott5_df['sseqid_acc'] != prott5_df['qseqid']]
    
    proteins_with_hits = set(prott5_df['qseqid_acc'])
    # Process each query sequence
    for i, qid in enumerate(query_ids):
        if qid not in proteins_with_hits:
            continue
            
        hits = prott5_df[prott5_df['qseqid_acc'] == qid]
        hits = hits.sort_values('evalue', ascending=True)
        hits_unique = hits.drop_duplicates('sseqid_acc', keep='first').reset_index(drop=True) # keep only the first hit for each sseqid_acc result (lowest evalue)
        # hit_sseqids = set(hits['sseqid_acc'].unique())
        
        # Calculate weights for hits, make sure the order is retained
        weights = {}
        for _, hit in hits_unique.iterrows():
            # if use_rscore:
            #     weights[hit['qseqid']] = calculate_rscore(hit['evalue'])
            # else:
            weights[hit['sseqid_acc']] = hit['similarity']/100.0
            
        # Get annotations for hit sequences and weight them
        query_scores = sparse.lil_matrix((len(weights), n_terms), dtype=np.float32)
        for num_hit_seq, (hit_seq, weight) in enumerate(weights.items()):
            # check if sseqid has annotations
            if hit_seq in proteins:
                hit_annots = annotation_mat[proteins[hit_seq]].multiply(weight) # assign same weight (similarity) to all terms from this hit
                query_scores[num_hit_seq] = hit_annots
        query_scores = query_scores.tocsr()
        max_scores = query_scores.max(axis=0)
                
        assert len(np.unique(max_scores.data)) <= len(weights), f"More unique similarity values than number of hit sequences"                
        scores[i] = max_scores
        
    # Add row and column indices
    # Turn sparse matrix into dataframe format
    scores_df = pd.DataFrame(scores.toarray(), index=query_ids)
    scores_df.columns = terms
    melted_df = scores_df.reset_index().melt(id_vars = 'index',var_name = 'term', value_name = 'value')
    melted_df = melted_df.rename(columns ={'index':'EntryID'})
    melted_df = melted_df[melted_df['value'] > 0]
    print(melted_df.head())
    # Create Dask DataFrame
    dask_df = dd.from_pandas(melted_df, npartitions=16)
    dask_df.to_csv(output_baseline, sep="\t", index=False, header=False, single_file = True, compression='gzip')
    # Write the DataFrame to a gzipped TSV file using the Dask function
    # write_dask_dataframe_to_gzipped_tsv(dask_df, output_baseline) # this function is defined in utils/write_dask.py
    return 

    
def parse_args(argv):
    parser = argparse.ArgumentParser(description='ProtT5 embeddings baseline')
    parser.add_argument('--annot_matrix', '-a', 
                        help='Path to the propagated annotation sparse matrix', required=True)
    parser.add_argument('--indices', '-i', 
                        help='Path to the term & protein indices file', required=True)
    parser.add_argument('--query_file', '-q', 
                        help='FASTA file or text file containing query IDs', required=True)
    parser.add_argument('--prott5_results', '-p',
                        help='Path to the ProtT5 embedding results file', required=True)
    parser.add_argument('--output_baseline', '-o', 
                        help='Path to the output file', required=True)
    return parser.parse_args(argv)    


def main():
    args = parse_args(sys.argv[1:])
    prott5_predict(
        annotations=args.annot_matrix,
        query_file=args.query_file,
        indices=args.indices,
        prott5_results=args.prott5_results,
        output_baseline=args.output_baseline
    )
    
    
if __name__ == '__main__':
    main()