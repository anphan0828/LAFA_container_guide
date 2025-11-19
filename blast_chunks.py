#!/usr/bin/env python3
"""
BLAST-based predictor implementation optimized for large-scale processing.
Makes predictions based on BLAST sequence similarity.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import pickle as cp
from Bio import SeqIO
import argparse
import multiprocessing
from tqdm import tqdm
import csv
import gzip
import tempfile
from democafa.datacollection.retrieve_terms import wrapper_retrieve_terms
from democafa.utils.ontology import sparse_matrix_and_indices
from democafa.config import GO_CODES


def calculate_rscore(evalue: float, max_rscore: float = 500.0) -> float:
    """Calculate R-score from BLAST E-value."""
    if evalue == 0:
        return max_rscore
    rscore = -np.log10(evalue) + 2
    return min(rscore, max_rscore)


def process_query_chunk(task_data):
    query_chunk, worker_static_data = task_data
    (blast_groups, annotation_mat, proteins, terms, use_rscore, n_terms) = worker_static_data
    results = []
    term_names = list(terms.keys())
    
    # Process each query in the chunk
    for qid in query_chunk:
        if qid not in blast_groups:
            continue
        
        hits = blast_groups[qid]
        
        if len(hits) == 0:
            continue
            
        # Sort and deduplicate hits
        hits = hits.sort_values('evalue', ascending=True)
        hits_unique = hits.drop_duplicates('sseqid_acc', keep='first')
        
        # Calculate weights for hits
        weights = {}
        for _, hit in hits_unique.iterrows():
            if hit['sseqid_acc'] not in proteins:
                continue
            weights[hit['sseqid_acc']] = calculate_rscore(hit['evalue']) if use_rscore else hit['pident']/100.0
        
        if not weights:
            continue
            
        # Get annotations for hit sequences and weight them
        hit_idx = [proteins[hit_seq] for hit_seq in weights.keys() if hit_seq in proteins]
        if not hit_idx:
            continue
            
        hit_idx = np.array(hit_idx)
        weight_values = np.array([weights[hit_seq] for hit_seq in weights.keys() if hit_seq in proteins])
        
        # Get annotations and multiply by weights
        hit_annots = annotation_mat[hit_idx, :]
        weighted_matrix = hit_annots.multiply(sparse.csr_matrix(weight_values).T) 
        max_scores = weighted_matrix.max(axis=0) 
        
        # if n_terms is not None and max_scores.nnz > n_terms:
        #     # Convert sparse matrix to dense array for processing
        #     scores_array = max_scores.toarray().flatten()
            
        #     # Find indices of top n_terms scores
        #     top_indices = np.argpartition(scores_array, -n_terms)[-n_terms:]
            
        #     # Keep only top scores (filter out the rest)
        #     top_scores = scores_array[top_indices]
            
        #     # Create results only for top n_terms
        #     for i, term_index in enumerate(top_indices):
        #         score = top_scores[i]
        #         if score > 0:
        #             term = term_names[term_index]
        #             results.append([qid, term, score])
        # else:        
        #     # Collect non-zero scores
        #     for term_index, score in zip(max_scores.col, max_scores.data):
        #         if score > 0:
        #             term = term_names[term_index]
        #             results.append([qid, term, score])
        
        # Convert to COO for efficient iteration
        max_scores_coo = max_scores.tocoo()
        
        # Collect all non-zero scores
        term_score_pairs = [(term_index, score) 
                           for term_index, score in zip(max_scores_coo.col, max_scores_coo.data)
                           if score > 0]
        
        # Apply n_terms limit if specified
        if n_terms is not None and len(term_score_pairs) > n_terms:
            # Sort by score descending and keep top n_terms
            term_score_pairs.sort(key=lambda x: x[1], reverse=True)
            term_score_pairs = term_score_pairs[:n_terms]
        for term_index, score in term_score_pairs:
            term = term_names[term_index]
            results.append([qid, term, score])
            
    return results


def blast_predict(annot_file, query_file, indices, graph, add_graph,
                 blast_results, output_baseline, keep_self_hits=False, use_rscore=False, n_terms=None):
    """Make predictions for query sequences based on BLAST hits."""
    # Load annotation matrix from appropriate source
    if '.gaf' in annot_file or '.dat' in annot_file:
        if not graph:
            print("Please provide a graph file for GAF or DAT input")
            sys.exit(1)
        print("Loading annotations from GAF or DAT file")
        wrapper_retrieve_terms(
            annot_file=annot_file,
            selected_go_codes='Experimental,IC,TAS', # only use non-experimental terms
            graph=graph,
            # add_graph=add_graph, # maybe don't need to filter comparable with t-1
            output_tsv=f'{os.path.dirname(output_baseline)}/blast_terms.tsv' # just a temporary file
        )
        terms_df = pd.read_csv(
            f'{os.path.dirname(output_baseline)}/blast_terms.tsv', 
            sep='\t', header=0, 
            names=['EntryID', 'term', 'aspect']
        )
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
        os.remove(f'{os.path.dirname(output_baseline)}/blast_terms.tsv')
    elif '.tsv' in annot_file:
        print("Loading annotations from TSV file")
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
    else:
        print("Invalid annotation file format")
        sys.exit(1)
        
    # Load query IDs
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
    
    print(f"Loaded {len(query_ids)} query IDs")
    
    # Load BLAST results
    print(f"Loading BLAST results from {blast_results}")
    blast_df = pd.read_csv(
        blast_results, 
        sep='\t', 
        header=None,
        names=['qseqid', 'sseqid', 'evalue', 'length', 'pident', 'nident']
    )
    
    # Process BLAST data
    blast_df['qseqid_acc'] = blast_df['qseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
    blast_df['sseqid_acc'] = blast_df['sseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
    
    # Remove self-hits if needed
    if not keep_self_hits:
        blast_df = blast_df[blast_df['sseqid_acc'] != blast_df['qseqid_acc']]
    
    # Create a subset of query IDs that have hits
    proteins_with_hits = set(blast_df['qseqid_acc'])
    effective_queries = [qid for qid in query_ids if qid in proteins_with_hits]
    
    print(f"Processing {len(effective_queries)} query sequences with BLAST hits")
    
    # Pre-group the blast results by query ID
    print("Grouping BLAST results by query ID...")
    blast_groups = dict(tuple(blast_df.groupby('qseqid_acc')))
    
    # Determine number of processes
    # num_processes = min(int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count())) - 1, 16)
    # num_processes = max(1, num_processes)
    num_processes = 8
    print(f"Using {num_processes} processes for parallel computation")
    
    # Create temporary directory for output chunks
    temp_dir = tempfile.mkdtemp(dir=os.path.dirname(output_baseline))
    
    # Prepare worker data
    worker_static_data = (blast_groups, annotation_mat, proteins, terms, use_rscore, n_terms)
    
    # Create chunks of query IDs for better memory management
    chunk_size = max(1, len(effective_queries) // (num_processes * 10))
    query_chunks = [effective_queries[i:i+chunk_size] for i in range(0, len(effective_queries), chunk_size)]
    tasks = [(chunk, worker_static_data) for chunk in query_chunks]
    
    print(f"Split processing into {len(tasks)} chunks of approximately {chunk_size} queries each")
    
    # Process chunks in parallel and write results to temporary files
    temp_files = []
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(tasks), desc="Processing query chunks") as pbar:
                for i, results in enumerate(pool.imap_unordered(process_query_chunk, tasks)):
                    if results:
                        # Write chunk results to temporary file
                        temp_file = os.path.join(temp_dir, f"chunk_{i}.tsv")
                        with open(temp_file, 'w', newline='') as f:
                            writer = csv.writer(f, delimiter='\t')
                            writer.writerows(results)
                        temp_files.append(temp_file)
                    pbar.update(1)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_baseline)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Merge temporary files into final gzipped output
        print(f"Merging results into final output file: {output_baseline}")
        open_func = gzip.open if output_baseline.endswith('.gz') else open
        with open_func(output_baseline, 'wt', newline='') as outfile:
            for temp_file in temp_files:
                with open(temp_file, 'r') as infile:
                    outfile.write(infile.read())
                # os.remove(temp_file)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temporary files and directory
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    
    print("Prediction completed successfully")
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(description='BLAST sequence similarity baseline')
    parser.add_argument('--annot_file', '-a', 
                        help='Path to the propagated annotation sparse matrix or a .gaf or .dat file', required=True)
    parser.add_argument('--indices', '-i', 
                        help='Path to the term & protein indices file', required=False, default=None)
    parser.add_argument('--graph', help='Path to OBO file for GO graph', required=False, default=None)
    parser.add_argument('--add_graph',help='Path to additional OBO file for GO graph at a later time point', 
                        required=False, default=None)
    parser.add_argument('--query_file', '-q', 
                        help='FASTA file or text file containing query IDs', required=True)
    parser.add_argument('--blast_results', '-b',
                        help='Path to the BLAST results file', required=True)
    parser.add_argument('--output_baseline', '-o', 
                        help='Path to the output file', required=True)
    parser.add_argument('--use_rscore', action='store_true',
                        help='Use R-score instead of sequence identity for weighting')
    parser.add_argument('--keep_self_hits', action='store_true',
                        help='Keep self-hits in BLAST results')
    parser.add_argument('--n_terms', '-n', type=int, required=False,
                        help='Upper limit for number of terms per target', default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    blast_predict(
        annot_file=args.annot_file,
        query_file=args.query_file,
        indices=args.indices,
        graph=args.graph,
        add_graph=args.add_graph,
        blast_results=args.blast_results,
        output_baseline=args.output_baseline,
        keep_self_hits=args.keep_self_hits,
        use_rscore=args.use_rscore,
        n_terms=args.n_terms
    )


if __name__ == '__main__':
    main()