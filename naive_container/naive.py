#!/usr/bin/env python3
"""
Naive predictor implementation for CAFA2 evaluation.
Makes predictions based on term frequency in training data.

This is a direct port of the MATLAB pfp_naive.m function.
"""

import obonet
import os
import sys
import numpy as np
from scipy import sparse
from typing import Dict
import pandas as pd
import pickle as cp
from Bio import SeqIO
import argparse   
from retrieve_terms import wrapper_retrieve_terms
from ontology import sparse_matrix_and_indices, fetch_aspect

def naive_predict(annot_file, query_file: str, indices, graph, add_graph, output_baseline) -> pd.DataFrame:
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
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    GO_CODES = config.get('go_codes', {})
    
    # Load annotation matrix
    if '.gaf' in annot_file or '.dat' in annot_file:
        if not graph:
            print("Please provide a graph file for GAF or DAT input")
            sys.exit(1)
        print("Loading annotations from GAF or DAT file")
        wrapper_retrieve_terms(
            annot_file=annot_file,
            go_codes=GO_CODES,
            selected_go_codes='Experimental,IC,TAS', # only use non-experimental terms
            graph=graph,
            # add_graph=add_graph, # maybe don't need to filter comparable with t-1
            output_tsv=f'{os.path.dirname(output_baseline)}/naive_terms.tsv' # just a temporary file
        )
        terms_df = pd.read_csv(f'{os.path.dirname(output_baseline)}/naive_terms.tsv', sep='\t', header=0, names=['EntryID', 'term', 'aspect'])
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
        os.remove(f'{os.path.dirname(output_baseline)}/naive_terms.tsv')
    elif '.tsv' in annot_file:
        terms_df = pd.read_csv(annot_file, sep='\t', header=0)
        if terms_df.shape[1] != 3:
            print("Invalid TSV format. Expected 3 columns: EntryID, term, aspect.")
            sys.exit(1)
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
    elif annot_file.endswith('.npz'):
        if not indices:
            print("Please provide a term indices file for matrix input")
            sys.exit(1)
        annotation_mat = sparse.load_npz(annot_file)
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
    # max_freq = np.max(term_frequencies)
    
    # Fix 2025-07-17: normalize by maximum count within aspects
    roots = {'P': 'GO:0008150', 'C': 'GO:0005575', 'F': 'GO:0003674'}
    max_freqs = dict()
    for aspect, root in roots.items():
        if root in terms:
            max_freqs[aspect] = term_frequencies[terms[root]]
    
    if len(max_freqs) == 0: # when terms were not propagated
        aspect_terms_df = terms_df.groupby('aspect')['EntryID'].apply(set).apply(len).reset_index(name='count')
        max_freqs = {aspect: count for aspect, count in zip(aspect_terms_df['aspect'], aspect_terms_df['count'])}
    # if max_freq > 0:
    #     normalized_scores = term_frequencies / max_freq
    # else:
    #     normalized_scores = term_frequencies
    assert len(max_freqs) == 3, "Expected three aspects for max frequencies: P, C, F"
    print(f"Max frequencies per aspect: {max_freqs}")
    # Get terms by aspect
    subontologies = {aspect: fetch_aspect(obonet.read_obo(graph), roots[aspect]) for aspect in roots}
    aspect_terms = {aspect: set(subontologies[aspect].nodes()) for aspect in subontologies}
    
    # Normalize scores by maximum frequency for each aspect
    normalized_scores = np.zeros_like(term_frequencies, dtype=np.float32)
    for aspect, max_freq in max_freqs.items():
        aspect_indices = [i for t,i in terms.items() if t in aspect_terms[aspect]]
        normalized_scores[aspect_indices] = term_frequencies[aspect_indices] / max_freq
        
    ## RAM-INTENSIVE PART ##    
    # # Create output matrix with same scores for all queries
    # n_queries = len(query_ids)
    # n_terms = annotation_mat.shape[1]
    # scores = sparse.lil_matrix((n_queries, n_terms), dtype=np.float32)
    # for i in range(n_queries):
    #     scores[i] = normalized_scores
    # scores = scores.tocsr()
        
    # # Turn sparse matrix into dataframe format
    # scores_df = pd.DataFrame(scores.toarray(), index=query_ids)
    # scores_df.columns = terms
    # melted_df = scores_df.reset_index().melt(id_vars = 'index',var_name = 'term', value_name = 'value')
    # melted_df = melted_df.rename(columns ={'index':'EntryID'})
    # melted_df = melted_df[melted_df['value'] > 0] 
    # print(melted_df.head())
    
    # # Create Dask DataFrame
    # dask_df = dd.from_pandas(melted_df, npartitions=16)
    # dask_df.to_csv(output_baseline, sep="\t", index=False, header=False, single_file = True, compression='gzip')
    
    ## RAM-EFFICIENT SOLUTION ##
    import csv
    import gzip

    output_dir = os.path.dirname(output_baseline)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 'wt' mode for writing text to a gzipped file
        open_func = gzip.open if output_baseline.endswith('.gz') else open
        with open_func(output_baseline, 'wt', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            # writer.writerow(["EntryID", "term", "value"]) # Header row

            for query_id in query_ids:
                # We iterate through the terms list by index to match scores
                for term_index, term in enumerate(terms):
                    score = normalized_scores[term_index]
                    if score > 0:
                        writer.writerow([query_id, term, score])
        print(f"Predictions for {len(set(query_ids))} proteins written to {output_baseline}")

    except Exception as e:
        print(f"Error writing output file {output_baseline}: {e}")
        sys.exit(1) 
    return 


def parse_args(argv):
    parser = argparse.ArgumentParser(description='UniProt non-experimental annotations baseline')
    parser.add_argument('--annot_file', '-a', 
                        help='Path to the propagated annotation sparse matrix or a .gaf or .dat file', required=True)
    parser.add_argument('--indices', '-i', 
                        help='Path to the term & protein indices file', required=False, default=None)
    parser.add_argument('--graph', help='Path to OBO file for GO graph', required=False, default=None)
    parser.add_argument('--add_graph',help='Path to additional OBO file for GO graph at a later time point', required=False, default=None)
    parser.add_argument('--query_file', '-q', 
                        help='FASTA file or text file containing query IDs', required=True)
    parser.add_argument('--output_baseline', '-o', 
                        help='Path to the output file', required=True)
    return parser.parse_args(argv)    


def main():
    args = parse_args(sys.argv[1:])
    naive_predict(
        annot_file=args.annot_file,
        query_file=args.query_file,
        indices=args.indices,
        graph=args.graph,
        add_graph=args.add_graph,
        output_baseline=args.output_baseline
    )
    
    
if __name__ == '__main__':
    main()
