#!/usr/bin/env python3
"""
BLAST-based predictor implementation for CAFA2 evaluation.
Makes predictions based on BLAST sequence similarity.

This is a direct port of the MATLAB pfp_blast.m function.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import pickle as cp
from Bio import SeqIO
# from typing import Dict
# import dask.dataframe as dd
import argparse
from democafa.datacollection.retrieve_terms import wrapper_retrieve_terms
from democafa.utils.ontology import sparse_matrix_and_indices
from democafa.config import GO_CODES

def calculate_rscore(evalue: float, max_rscore: float = 500.0) -> float:
    """
    Calculate R-score from BLAST E-value.
    R-score = -log(E-value) + 2, capped at max_rscore
    
    Args:
        evalue: BLAST E-value
        max_rscore: Maximum allowed R-score (default: 500)
        
    Returns:
        R-score value
    """
    if evalue == 0:
        return max_rscore
    rscore = -np.log10(evalue) + 2
    return min(rscore, max_rscore)

def process_single_query(task_data):
    # start=time.time()
    qid, hits, worker_static_data = task_data
    (annotation_mat, proteins, terms, use_rscore) = worker_static_data
    results_for_query = [] # List to store [qid, term, score] for this query
    term_names = list(terms.keys())
    
    # hits = blast_df[blast_df['qseqid_acc'] == qid]
    # if hits.empty:
        # return results_for_query
    hits = hits.sort_values('evalue', ascending=True)
    hits_unique = hits.drop_duplicates('sseqid_acc', keep='first').reset_index(drop=True) # keep only the first hit for each sseqid_acc result (lowest evalue)
    
    # Calculate weights for hits, make sure the order is retained
    weights = {}
    for _, hit in hits_unique.iterrows():
        if hit['sseqid_acc'] not in proteins:
            continue
        if use_rscore:
            weights[hit['sseqid_acc']] = calculate_rscore(hit['evalue'])
        else:
            weights[hit['sseqid_acc']] = hit['pident']/100.0
    # current_query_max_scores = np.zeros(len(terms), dtype=np.float32) # Initialize all zeros vector for current query
    # Convert terms to numpy array for indexing
    # Get annotations for hit sequences and weight them
    hit_idx = np.array([proteins[hit_seq] for hit_seq in weights.keys() if hit_seq in proteins]) # Get indices of hit sequences that have annotations
    hit_annots = annotation_mat[hit_idx, :]
    weighted_matrix = hit_annots.multiply(sparse.csr_matrix(list(weights.values())).T) 
    max_scores = weighted_matrix.max(axis=0) 
    # weighted_matrix2=[]
    # for num_hit_seq, (hit_seq, weight) in enumerate(weights.items()):
    #     if hit_seq in proteins:
    #         hit_annots = annotation_mat.getrow(proteins[hit_seq]) # getrow from protein idx
    #         weighted_matrix2.append(hit_annots.multiply(weight)) # assign same weight (pident or rscore) to all terms from this hit
            # for term_idx, value in zip(hit_annots.indices, hit_annots.data):
            #     weighted_value = value * weight
            #     current_query_max_scores[term_idx] = max(
            #         current_query_max_scores[term_idx], weighted_value) # always a 1 x n_terms vector at a time                           )
    # nonzeroes = current_query_max_scores[current_query_max_scores > 0]
    # assert len(np.unique(nonzeroes.data)) <= len(weights), f"More unique similarity values than number of hit sequences"
    # Collect the results for this query
    # Iterate through the max scores and add non-zero ones to the list
    # stacked_annots = sparse.vstack(weighted_matrix, format='csr')
    # max_scores = stacked_annots.max(axis=0)
    for term_index, score in zip(max_scores.col, max_scores.data):
        if score > 0:
            term = term_names[term_index] # Get the term ID
            results_for_query.append([qid, term, score])
    # end=time.time()
    # print(end-start)
    return results_for_query


def blast_predict(annotations,
                  query_file: str, 
                  indices,
                  graph, add_graph,
                  blast_results: pd.DataFrame,
                  output_baseline: str,
                  keep_self_hits: bool = False,
                  use_rscore: bool = False) -> pd.DataFrame:
    """
    Make predictions for query sequences based on BLAST hits.
    
    Args:
        query_file: Path to list of query sequence IDs (.fasta or .txt)
        blast_results: Path to BLAST results file
        annotations: Training annotations sparse matrix (n_sequences x n_terms)
        indices: Path to the term & protein indices file
        use_rscore: Whether to use R-score (True) or sequence identity (False)
                   for weighting hits
        keep_self_hits: Whether to keep self-hits (True) or remove them (False)
                    
    Note:
        For each query sequence, finds BLAST hits in training data,
        weights their annotations by sequence similarity, and takes
        the maximum score per term across all hits.
    """
    # Load annotation matrix
    if '.gaf' in annotations or '.dat' in annotations:
        if not graph:
            print("Please provide a graph file for GAF or DAT input")
            sys.exit(1)
        print("Loading annotations from GAF or DAT file")
        wrapper_retrieve_terms(
        annot_file=annotations,
        filetype='dat' if '.dat' in annotations else 'gaf',
        go_codes=GO_CODES,
        selected_go_codes='Experimental,IC,TAS', # only use non-experimental terms
        graph=graph,
        add_graph=add_graph, # maybe don't need to filter comparable with t-1
        output_tsv=f'{os.path.dirname(output_baseline)}/nonexp_terms.tsv' # just a temporary file
        )
        terms_df = pd.read_csv(f'{os.path.dirname(output_baseline)}/nonexp_terms.tsv', sep='\t', header=0, names=['EntryID', 'term', 'aspect'])
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
        os.remove(f'{os.path.dirname(output_baseline)}/nonexp_terms.tsv')
    elif annotations.endswith('.npz'):
        if not indices:
            print("Please provide a term indices file for matrix input")
            sys.exit(1)
        annotation_mat = sparse.load_npz(annotations)
        with open(indices, 'rb') as f:
            proteins, terms = cp.load(f)
    
    # For BLAST baseline predictor, use both proteins and terms indices     
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
    term_names = list(terms.keys())
    # scores = sparse.lil_matrix((n_queries, n_terms), dtype=np.float32)
    
    # Load BLAST results
    blast_df = pd.read_csv(blast_results, sep='\t', header=None, 
                           names=['qseqid', 'sseqid', 'evalue', 'length', 'pident', 'nident'])
    
    
    
    # Remove self-hits, qseqid is from test set, sseqid is from training set (blast db)
    blast_df['qseqid_acc'] = blast_df['qseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
    blast_df['sseqid_acc'] = blast_df['sseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
    if not keep_self_hits:
        blast_df = blast_df[blast_df['sseqid_acc'] != blast_df['qseqid_acc']]
    
    proteins_with_hits = set(blast_df['qseqid_acc'])
    print(f"Processing {len(proteins_with_hits)} query sequences and writing to {output_baseline}")
    
    ## RAM-INTENSIVE PART ##  
    # # Process each query sequence
    # for i, qid in enumerate(query_ids):
    #     if qid not in proteins_with_hits:
    #         continue
            
    #     hits = blast_df[blast_df['qseqid_acc'] == qid]
    #     hits = hits.sort_values('evalue', ascending=True)
    #     hits_unique = hits.drop_duplicates('sseqid_acc', keep='first').reset_index(drop=True) # keep only the first hit for each sseqid_acc result (lowest evalue)
    #     # hit_sseqids = set(hits['sseqid_acc'].unique())
        
    #     # Calculate weights for hits, make sure the order is retained
    #     weights = {}
    #     for _, hit in hits_unique.iterrows():
    #         if use_rscore:
    #             weights[hit['sseqid_acc']] = calculate_rscore(hit['evalue'])
    #         else:
    #             weights[hit['sseqid_acc']] = hit['pident']/100.0
            
    #     # Get annotations for hit sequences and weight them
    #     query_scores = sparse.lil_matrix((len(weights), n_terms), dtype=np.float32) # matrix of shape (n_sseqid, n_terms)
    #     for num_hit_seq, (hit_seq, weight) in enumerate(weights.items()):
    #         # check if sseqid has annotations
    #         if hit_seq in proteins:
    #             hit_annots = annotation_mat[proteins[hit_seq]].multiply(weight) # assign same weight (pident) to all terms from this hit
    #             query_scores[num_hit_seq] = hit_annots
    #     query_scores = query_scores.tocsr()
    #     max_scores = query_scores.max(axis=0)
                
    #     assert len(np.unique(max_scores.data)) <= len(weights), f"More unique similarity values than number of hit sequences"
    #     scores[i] = max_scores # score vector for query sequence qid, containing max values by each column (term) of hit sequences
    
      
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
    
    import multiprocessing
    from tqdm import tqdm
    worker_static_data = (annotation_mat, proteins, terms, use_rscore)
    blast_by_protein = {}
    for qid in query_ids:
        if qid not in proteins_with_hits:
            continue
        blast_by_protein[qid] = blast_df[blast_df['qseqid_acc'] == qid] # takes a long time
    num_processes = int(os.environ['SLURM_CPUS_PER_TASK']) -1 
    # def worker_wrapper(task_tuple):
    #     qid, (blast_df, annotation_mat, proteins, terms, use_rscore) = task_tuple
    #     return process_single_query(qid, blast_df, annotation_mat, proteins, terms, use_rscore)
    
    
    # MORE RAM-EFFICIENT: still slow ##
    import csv
    import gzip
    
    output_dir = os.path.dirname(output_baseline)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:    
        with gzip.open(output_baseline, 'wt', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            # writer.writerow(["EntryID", "term", "value"]) # Header row
            tasks = [(qid, blast_by_protein[qid], worker_static_data) for qid in query_ids]
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                # imap_unordered returns results as they finish, order is not guaranteed
                results_iterator = pool.imap_unordered(process_single_query, tasks)
            
                for result_list in tqdm(results_iterator, total=len(tasks), desc="Writing predictions"):
                    for row in result_list:
                         writer.writerow(row)
            # Getting max scores for each query then write to output file
            # for i, qid in enumerate(query_ids):
            #     if qid not in proteins_with_hits:
            #         continue
                    
                # hits = blast_df[blast_df['qseqid_acc'] == qid]
                # if hits.empty:
                #     continue
                # hits = hits.sort_values('evalue', ascending=True)
                # hits_unique = hits.drop_duplicates('sseqid_acc', keep='first').reset_index(drop=True) # keep only the first hit for each sseqid_acc result (lowest evalue)
                
                # # Calculate weights for hits, make sure the order is retained
                # weights = {}
                # for _, hit in hits_unique.iterrows():
                #     if use_rscore:
                #         weights[hit['sseqid_acc']] = calculate_rscore(hit['evalue'])
                #     else:
                #         weights[hit['sseqid_acc']] = hit['pident']/100.0
                                
                # current_query_max_scores = np.zeros(n_terms, dtype=np.float32) # Initialize all zeros vector for current query
                # # Convert terms to numpy array for indexing
                # # Get annotations for hit sequences and weight them
                # for num_hit_seq, (hit_seq, weight) in enumerate(weights.items()):
                #     if hit_seq in proteins:
                #         hit_annots = annotation_mat.getrow(proteins[hit_seq]) # getrow from protein idx
                #         for term_idx, value in zip(hit_annots.indices, hit_annots.data):
                #             weighted_value = value * weight
                #             current_query_max_scores[term_idx] = max(
                #                 current_query_max_scores[term_idx], weighted_value) # always a 1 x n_terms vector at a time                           )
                
                # Write to file right after each query is processed
                # nonzeroes = current_query_max_scores[current_query_max_scores > 0]
                # assert len(np.unique(nonzeroes.data)) <= len(weights), f"More unique similarity values than number of hit sequences"
                # for term_idx, value in enumerate(current_query_max_scores):
                #     if value > 0:
                #         term = term_names[term_idx] # Get the term ID from the index
                #         writer.writerow([qid, term, value])
                
    except Exception as e:
        print(f"Error writing output file {output_baseline}: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) 
    
    return 

    
def parse_args(argv):
    parser = argparse.ArgumentParser(description='BLAST sequence similarity baseline')
    parser.add_argument('--annotations', '-a', 
                        help='Path to the propagated annotation sparse matrix or a .gaf or .dat file', required=True)
    parser.add_argument('--indices', '-i', 
                        help='Path to the term & protein indices file', required=False, default=None)
    parser.add_argument('--graph', help='Path to OBO file for GO graph', required=False, default=None)
    parser.add_argument('--add_graph',help='Path to additional OBO file for GO graph at a later time point', required=False, default=None)
    parser.add_argument('--query_file', '-q', 
                        help='FASTA file or text file containing query IDs', required=True)
    parser.add_argument('--blast_results', '-b',
                        help='Path to the BLAST results file', required=True)
    parser.add_argument('--output_baseline', '-o', 
                        help='Path to the output file', required=True)
    return parser.parse_args(argv)    


def main():
    args = parse_args(sys.argv[1:])
    blast_predict(
        annotations=args.annotations,
        query_file=args.query_file,
        indices=args.indices,
        graph=args.graph,
        add_graph=args.add_graph,
        blast_results=args.blast_results,
        output_baseline=args.output_baseline
    )
    
    
if __name__ == '__main__':
    main()