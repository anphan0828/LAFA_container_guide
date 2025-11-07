#!/usr/bin/env python3
"""
GPU-accelerated ProtT5-based predictor implementation.
Uses CuPy for GPU acceleration of sparse matrix operations.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import pickle as cp
from Bio import SeqIO
import argparse
import csv
import gzip
from tqdm import tqdm
from retrieve_terms import wrapper_retrieve_terms
from ontology import sparse_matrix_and_indices

try:
    import cupy as cp_gpu
    import cupyx.scipy.sparse as cp_sparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: CuPy not available. GPU acceleration will not be used.")


def process_query_batch_gpu(query_batch_indices, prott5_df_subset, 
                            annotation_mat_gpu, proteins, terms, 
                            query_id_to_name):
    """
    Process a batch of queries on GPU.
    
    Args:
        query_batch_indices: List of query indices in this batch
        prott5_df_subset: DataFrame subset containing only hits for this batch
        annotation_mat_gpu: Annotation matrix on GPU (CuPy sparse matrix)
        proteins: Dict mapping protein ID to matrix row index
        terms: Dict mapping term to matrix column index
        query_id_to_name: Dict mapping index back to query ID
    
    Returns:
        List of [query_id, term, score] predictions
    """
    results = []
    term_names = list(terms.keys())
    
    # Group hits by query within this batch
    grouped = prott5_df_subset.groupby('qseqid_acc')
    
    for qid_idx in query_batch_indices:
        qid = query_id_to_name[qid_idx]
        
        # Skip if no hits for this query
        if qid not in grouped.groups:
            continue
        
        hits = grouped.get_group(qid)
        
        # Early exit if no hits
        if len(hits) == 0:
            continue
        
        # Sort and deduplicate hits
        hits = hits.sort_values('evalue', ascending=True)
        hits_unique = hits.drop_duplicates('sseqid_acc', keep='first').reset_index(drop=True)
        
        # Calculate weights for hits
        weights_list = []
        hit_indices_list = []
        
        for _, hit in hits_unique.iterrows():
            if hit['sseqid_acc'] not in proteins:
                continue
            hit_indices_list.append(proteins[hit['sseqid_acc']])
            weights_list.append(hit['similarity'] / 100.0)
        
        # Skip if no valid weights
        if not weights_list:
            continue
        
        # Convert to GPU arrays
        hit_idx_gpu = cp_gpu.array(hit_indices_list, dtype=cp_gpu.int32)
        weights_gpu = cp_gpu.array(weights_list, dtype=cp_gpu.float32)
        
        # Get annotations for hit sequences (sparse matrix slicing on GPU)
        hit_annots = annotation_mat_gpu[hit_idx_gpu, :]
        
        # Weight the annotations: multiply each row by its weight
        # For sparse matrix, we need to multiply via broadcasting
        weights_column = cp_sparse.csr_matrix(weights_gpu.reshape(-1, 1))
        weighted_matrix = hit_annots.multiply(weights_column)
        
        # Get maximum score for each term across all hits
        max_scores = weighted_matrix.max(axis=0)
        
        # Convert back to CPU for result collection
        # max_scores is a sparse matrix with shape (1, n_terms)
        max_scores_coo = max_scores.tocoo()
        col_indices = cp_gpu.asnumpy(max_scores_coo.col)
        score_values = cp_gpu.asnumpy(max_scores_coo.data)
        
        # Collect non-zero scores
        for term_index, score in zip(col_indices, score_values):
            if score > 0:
                term = term_names[term_index]
                results.append([qid, term, float(score)])
    
    return results


def prott5_predict_gpu(annot_file, query_file, indices, graph, add_graph,
                      prott5_results, output_baseline, config_path=None, 
                      keep_self_hits=False, batch_size=1000, device_id=0):
    """
    GPU-accelerated prediction for query sequences based on ProtT5 hits.
    
    Args:
        annot_file: Path to annotation file (TSV, GAF, DAT, or NPZ)
        query_file: FASTA or text file with query IDs
        indices: Path to indices file (for NPZ input)
        graph: Path to OBO file for GO graph
        add_graph: Path to additional OBO file
        prott5_results: Path to ProtT5 similarity results
        output_baseline: Path to output file
        config_path: Path to config file
        keep_self_hits: Whether to keep self-hits
        batch_size: Number of queries to process per batch
        device_id: GPU device ID to use
    """
    
    if not GPU_AVAILABLE:
        print("ERROR: CuPy is not available. Please install CuPy for GPU acceleration.")
        print("Install with: pip install cupy-cuda11x  (replace 11x with your CUDA version)")
        sys.exit(1)
    
    # Set GPU device
    cp_gpu.cuda.Device(device_id).use()
    print(f"Using GPU device {device_id}: {cp_gpu.cuda.Device(device_id).compute_capability}")
    
    # Load annotations
    print("Loading annotations...")
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
            selected_go_codes='Experimental,IC,TAS',
            graph=graph,
            output_tsv=f'{os.path.dirname(output_baseline)}/prott5_terms.tsv'
        )
        terms_df = pd.read_csv(
            f'{os.path.dirname(output_baseline)}/prott5_terms.tsv',
            sep='\t', header=0,
            names=['EntryID', 'term', 'aspect']
        )
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
        os.remove(f'{os.path.dirname(output_baseline)}/prott5_terms.tsv')
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
    
    print(f"Annotation matrix shape: {annotation_mat.shape}")
    print(f"Number of proteins: {len(proteins)}")
    print(f"Number of terms: {len(terms)}")
    
    # Convert annotation matrix to GPU (CuPy sparse matrix)
    print("Transferring annotation matrix to GPU...")
    annotation_mat_gpu = cp_sparse.csr_matrix(annotation_mat, dtype=cp_gpu.float32)
    
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
    
    # Load ProtT5 results in chunks to manage memory
    print(f"Loading ProtT5 results from {prott5_results}")
    chunksize = 1_000_000  # Process 1M rows at a time
    
    # First pass: get all unique query IDs that have hits
    print("First pass: identifying queries with hits...")
    queries_with_hits = set()
    for chunk in pd.read_csv(prott5_results, sep='\t', header=0,
                             names=['qseqid', 'sseqid', 'evalue', 'length', 'similarity', 'nident'],
                             chunksize=chunksize):
        chunk['qseqid_acc'] = chunk['qseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
        queries_with_hits.update(chunk['qseqid_acc'].unique())
    
    effective_queries = [qid for qid in query_ids if qid in queries_with_hits]
    print(f"Found {len(effective_queries)} query sequences with ProtT5 hits")
    
    # Create query index mapping
    query_name_to_idx = {qid: idx for idx, qid in enumerate(effective_queries)}
    query_idx_to_name = {idx: qid for qid, idx in query_name_to_idx.items()}
    
    # Create batches of query indices
    num_queries = len(effective_queries)
    num_batches = (num_queries + batch_size - 1) // batch_size
    print(f"Processing {num_queries} queries in {num_batches} batches of size ~{batch_size}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_baseline)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open output file
    open_func = gzip.open if output_baseline.endswith('.gz') else open
    
    with open_func(output_baseline, 'wt', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        
        # Process each batch
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_queries)
            batch_query_names = effective_queries[batch_start:batch_end]
            batch_query_indices = list(range(batch_start, batch_end))
            
            # Load only the ProtT5 hits for queries in this batch
            batch_prott5_data = []
            for chunk in pd.read_csv(prott5_results, sep='\t', header=0,
                                    names=['qseqid', 'sseqid', 'evalue', 'length', 'similarity', 'nident'],
                                    chunksize=chunksize):
                chunk['qseqid_acc'] = chunk['qseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
                chunk['sseqid_acc'] = chunk['sseqid'].apply(lambda x: x.split('|')[1] if "|" in x else x)
                
                # Filter self-hits if needed
                if not keep_self_hits:
                    chunk = chunk[chunk['sseqid_acc'] != chunk['qseqid_acc']]
                
                # Keep only hits for queries in this batch
                chunk_subset = chunk[chunk['qseqid_acc'].isin(batch_query_names)]
                if len(chunk_subset) > 0:
                    batch_prott5_data.append(chunk_subset)
            
            if not batch_prott5_data:
                continue
            
            # Combine all chunks for this batch
            batch_prott5_df = pd.concat(batch_prott5_data, ignore_index=True)
            
            # Process this batch on GPU
            batch_results = process_query_batch_gpu(
                batch_query_indices,
                batch_prott5_df,
                annotation_mat_gpu,
                proteins,
                terms,
                query_idx_to_name
            )
            
            # Write results immediately
            writer.writerows(batch_results)
            
            # Clear memory
            del batch_prott5_data
            del batch_prott5_df
    
    print(f"Prediction completed successfully. Results written to {output_baseline}")
    
    # Print GPU memory usage
    mempool = cp_gpu.get_default_memory_pool()
    print(f"GPU memory used: {mempool.used_bytes() / 1e9:.2f} GB")
    print(f"GPU memory total: {mempool.total_bytes() / 1e9:.2f} GB")


def parse_args(argv):
    parser = argparse.ArgumentParser(description='GPU-accelerated ProtT5 embeddings similarity baseline')
    parser.add_argument('--annot_file', '-a',
                       help='Path to the propagated annotation sparse matrix or a .gaf or .dat file',
                       required=True)
    parser.add_argument('--indices', '-i',
                       help='Path to the term & protein indices file',
                       required=False, default=None)
    parser.add_argument('--graph',
                       help='Path to OBO file for GO graph',
                       required=False, default=None)
    parser.add_argument('--add_graph',
                       help='Path to additional OBO file for GO graph at a later time point',
                       required=False, default=None)
    parser.add_argument('--query_file', '-q',
                       help='FASTA file or text file containing query IDs',
                       required=True)
    parser.add_argument('--prott5_results', '-b',
                       help='Path to the ProtT5 results file',
                       required=True)
    parser.add_argument('--output_baseline', '-o',
                       help='Path to the output file',
                       required=True)
    parser.add_argument('--keep_self_hits',
                       action='store_true',
                       help='Keep self-hits in ProtT5 results')
    parser.add_argument('--batch_size',
                       type=int,
                       default=1000,
                       help='Number of queries to process per batch (default: 1000)')
    parser.add_argument('--device_id',
                       type=int,
                       default=0,
                       help='GPU device ID to use (default: 0)')
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    prott5_predict_gpu(
        annot_file=args.annot_file,
        query_file=args.query_file,
        indices=args.indices,
        graph=args.graph,
        add_graph=args.add_graph,
        prott5_results=args.prott5_results,
        output_baseline=args.output_baseline,
        config_path=os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml'),
        keep_self_hits=args.keep_self_hits,
        batch_size=args.batch_size,
        device_id=args.device_id,
    )


if __name__ == '__main__':
    main()
