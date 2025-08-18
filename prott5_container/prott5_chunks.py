#!/usr/bin/env python3
"""
ProtT5-based predictor implementation optimized for large-scale processing.
Makes predictions based on ProtT5 embeddings similarity.
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
from retrieve_terms import wrapper_retrieve_terms
from ontology import sparse_matrix_and_indices


def process_query_chunk(task_data):
    """Process a chunk of query IDs efficiently."""
    query_chunk, worker_static_data = task_data
    (prott5_groups, annotation_mat, proteins, terms) = worker_static_data
    results = []
    term_names = list(terms.keys())
    
    # Process each query in the chunk
    for qid in query_chunk:
        # Skip if no hits for this query
        if qid not in prott5_groups:
            continue
        
        # Get prott5 hits for this query
        hits = prott5_groups[qid]
        
        # Early exit if no hits
        if len(hits) == 0:
            continue
            
        # Sort and deduplicate hits
        hits = hits.sort_values('evalue', ascending=True)
        hits_unique = hits.drop_duplicates('sseqid_acc', keep='first').reset_index(drop=True) 
        
        # Calculate weights for hits
        weights = {}
        for _, hit in hits_unique.iterrows():
            if hit['sseqid_acc'] not in proteins:
                continue
            weights[hit['sseqid_acc']] = hit['similarity']/100.0
        
        # Skip if no valid weights
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
        
        # Collect non-zero scores
        for term_index, score in zip(max_scores.col, max_scores.data):
            if score > 0:
                term = term_names[term_index]
                results.append([qid, term, score])
    
    return results


def prott5_predict(annot_file, query_file, indices, graph, add_graph,
                 prott5_results, output_baseline,  config_path=None, keep_self_hits=False):
    """Make predictions for query sequences based on prott5 hits."""
    
    if '.gaf' in annot_file or '.dat' in annot_file:
        if not graph:
            print("Please provide a graph file for GAF or DAT input")
            sys.exit(1)
        print("Loading annotations from GAF or DAT file")
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        GO_CODES = config.get('go_codes', {})
        wrapper_retrieve_terms(
            annot_file=annot_file,
            go_codes=GO_CODES,
            selected_go_codes='Experimental,IC,TAS', # only use non-experimental terms
            graph=graph,
            # add_graph=add_graph, # maybe don't need to filter comparable with t-1
            output_tsv=f'{os.path.dirname(output_baseline)}/prott5_terms.tsv' # just a temporary file
        )
        terms_df = pd.read_csv(
            f'{os.path.dirname(output_baseline)}/prott5_terms.tsv', 
            sep='\t', header=0, 
            names=['EntryID', 'term', 'aspect']
        )
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
        os.remove(f'{os.path.dirname(output_baseline)}/prott5_terms.tsv')
    elif annot_file.endswith('.npz'):
        if not indices:
            print("Please provide a term indices file for matrix input")
            sys.exit(1)
        annotation_mat = sparse.load_npz(annot_file)
        with open(indices, 'rb') as f:
            proteins, terms = cp.load(f)
    
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
    
    # Load prott5 results
    print(f"Loading prott5 results from {prott5_results}")
    prott5_df = pd.read_csv(
        prott5_results, 
        sep='\t', 
        header=0,
        names=['qseqid', 'sseqid', 'evalue', 'length', 'similarity', 'nident']
    )
    
    # Process prott5 data
    prott5_df['qseqid_acc'] = prott5_df['qseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
    prott5_df['sseqid_acc'] = prott5_df['sseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
    if not keep_self_hits:
        prott5_df = prott5_df[prott5_df['sseqid_acc'] != prott5_df['qseqid_acc']]
    
    # Create a subset of query IDs that have hits
    proteins_with_hits = set(prott5_df['qseqid_acc'])
    effective_queries = [qid for qid in query_ids if qid in proteins_with_hits]
    
    print(f"Processing {len(effective_queries)} query sequences with prott5 hits")
    
    # Pre-group the prott5 results by query ID
    print("Grouping prott5 results by query ID...")
    prott5_groups = dict(tuple(prott5_df.groupby('qseqid_acc')))
    
    # Determine number of processes
    num_processes = min(int(os.environ.get('NUM_THREADS', multiprocessing.cpu_count())), 16)
    num_processes = max(1, num_processes)
    print(f"Using {num_processes} processes for parallel computation")
    
    # Create temporary directory for output chunks
    temp_dir = tempfile.mkdtemp(dir=os.path.dirname(output_baseline))
    
    # Prepare worker data
    worker_static_data = (prott5_groups, annotation_mat, proteins, terms)
    
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
        with gzip.open(output_baseline, 'wt', newline='') as outfile:
            for temp_file in temp_files:
                with open(temp_file, 'r') as infile:
                    outfile.write(infile.read())
    
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
    parser = argparse.ArgumentParser(description='ProtT5 embeddings similarity baseline')
    parser.add_argument('--annot_file', '-a', 
                        help='Path to the propagated annotation sparse matrix or a .gaf or .dat file', required=True)
    parser.add_argument('--indices', '-i', 
                        help='Path to the term & protein indices file', required=False, default=None)
    parser.add_argument('--graph', help='Path to OBO file for GO graph', required=False, default=None)
    parser.add_argument('--add_graph',help='Path to additional OBO file for GO graph at a later time point', 
                        required=False, default=None)
    parser.add_argument('--query_file', '-q', 
                        help='FASTA file or text file containing query IDs', required=True)
    parser.add_argument('--prott5_results', '-b',
                        help='Path to the prott5 results file', required=True)
    parser.add_argument('--output_baseline', '-o', 
                        help='Path to the output file', required=True)
    parser.add_argument('--keep_self_hits', action='store_true',
                        help='Keep self-hits in prott5 results')
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    prott5_predict(
        annot_file=args.annot_file,
        query_file=args.query_file,
        indices=args.indices,
        graph=args.graph,
        add_graph=args.add_graph,
        prott5_results=args.prott5_results,
        output_baseline=args.output_baseline,
        config_path=os.path.join(os.path.dirname(__file__), 'config.yaml'),
        keep_self_hits=args.keep_self_hits,
    )


if __name__ == '__main__':
    main()
