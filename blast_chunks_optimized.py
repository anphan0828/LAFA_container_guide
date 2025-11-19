#!/usr/bin/env python3
"""BLAST-based predictor with single-pass I/O and optional GPU aggregation."""

import os
import sys
import csv
import gzip
import shutil
import tempfile
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
import pickle as cp
from Bio import SeqIO
from tqdm import tqdm

from retrieve_terms import wrapper_retrieve_terms
from ontology import sparse_matrix_and_indices

try:
    import cupy as cp_gpu
    import cupyx.scipy.sparse as cp_sparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def calculate_rscore_array(evalues, max_rscore=500.0):
    """Vectorized R-score calculation compatible with numpy arrays."""

    evalues = np.asarray(evalues, dtype=np.float64)
    rscores = np.full(evalues.shape, max_rscore, dtype=np.float32)
    mask = evalues > 0
    if np.any(mask):
        with np.errstate(divide='ignore'):
            scores = -np.log10(evalues[mask]) + 2
        rscores[mask] = np.minimum(scores, max_rscore)
    return rscores


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def partition_blast_hits_single_pass(blast_results, query_set, query_to_bucket,
                                     keep_self_hits, chunksize, num_buckets):
    """Stream the BLAST table once, bucketizing hits per query."""

    temp_dir = tempfile.mkdtemp(prefix='blast_hits_')
    bucket_paths = [os.path.join(temp_dir, f'bucket_{i}.tsv') for i in range(num_buckets)]
    bucket_has_header = [False] * num_buckets
    bucket_query_sets = [set() for _ in range(num_buckets)]
    queries_with_hits = set()

    dtype_map = {'evalue': np.float64, 'pident': np.float32}

    for chunk in pd.read_csv(
        blast_results,
        sep='\t',
        header=None,
        names=['qseqid', 'sseqid', 'evalue', 'length', 'pident', 'nident'],
        usecols=['qseqid', 'sseqid', 'evalue', 'pident'],
        chunksize=chunksize,
        dtype=dtype_map,
    ):
        q_parts = chunk['qseqid'].str.split('|', n=2, expand=True)
        chunk['qseqid_acc'] = q_parts[1].fillna(q_parts[0])
        chunk = chunk[chunk['qseqid_acc'].isin(query_set)]
        if chunk.empty:
            continue

        s_parts = chunk['sseqid'].str.split('|', n=2, expand=True)
        chunk['sseqid_acc'] = s_parts[1].fillna(s_parts[0])
        if not keep_self_hits:
            chunk = chunk[chunk['sseqid_acc'] != chunk['qseqid_acc']]
        if chunk.empty:
            continue

        chunk['bucket'] = chunk['qseqid_acc'].map(query_to_bucket)

        for bucket_idx, bucket_df in chunk.groupby('bucket'):
            if bucket_df.empty:
                continue
            bucket_df = bucket_df[['qseqid_acc', 'sseqid_acc', 'evalue', 'pident']]
            mode = 'w' if not bucket_has_header[bucket_idx] else 'a'
            bucket_df.to_csv(
                bucket_paths[bucket_idx],
                sep='\t',
                header=not bucket_has_header[bucket_idx],
                index=False,
                mode=mode,
            )
            bucket_has_header[bucket_idx] = True
            qids = bucket_df['qseqid_acc'].unique()
            bucket_query_sets[bucket_idx].update(qids)
            queries_with_hits.update(qids)

    return temp_dir, bucket_paths, bucket_query_sets, queries_with_hits


def prepare_batch_arrays(batch_hits_df, proteins, batch_query_to_local, use_rscore):
    """Return per-hit arrays for GPU/CPU aggregation."""

    if batch_hits_df.empty:
        return None

    batch_hits_df = batch_hits_df.sort_values(
        by=['qseqid_acc', 'evalue'], ascending=[True, True], kind='mergesort'
    )
    batch_hits_df = batch_hits_df.drop_duplicates(
        subset=['qseqid_acc', 'sseqid_acc'], keep='first'
    )

    batch_hits_df['protein_idx'] = batch_hits_df['sseqid_acc'].map(proteins)
    batch_hits_df = batch_hits_df.dropna(subset=['protein_idx'])
    if batch_hits_df.empty:
        return None

    batch_hits_df['query_local_idx'] = batch_hits_df['qseqid_acc'].map(batch_query_to_local)
    batch_hits_df = batch_hits_df.dropna(subset=['query_local_idx'])
    if batch_hits_df.empty:
        return None

    query_idx = batch_hits_df['query_local_idx'].to_numpy(dtype=np.int32, copy=False)
    protein_idx = batch_hits_df['protein_idx'].to_numpy(dtype=np.int32, copy=False)

    if use_rscore:
        weights = calculate_rscore_array(batch_hits_df['evalue'].to_numpy(dtype=np.float64, copy=False))
    else:
        weights = batch_hits_df['pident'].to_numpy(dtype=np.float32, copy=False) / 100.0

    valid_mask = weights > 0
    if not np.any(valid_mask):
        return None

    return (
        query_idx[valid_mask],
        protein_idx[valid_mask],
        weights[valid_mask].astype(np.float32, copy=False),
    )


def collect_batch_results(batch_query_names, batch_scores, term_names, n_terms):
    """Convert dense batch scores into prediction rows."""

    results = []
    for local_idx, qid in enumerate(batch_query_names):
        row = batch_scores[local_idx]
        nz_terms = np.flatnonzero(row)
        if nz_terms.size == 0:
            continue
        if n_terms is not None and nz_terms.size > n_terms:
            top_idx = nz_terms[np.argsort(row[nz_terms])[-n_terms:]]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        else:
            top_idx = nz_terms
        for term_idx in top_idx:
            results.append([qid, term_names[term_idx], float(row[term_idx])])
    return results


def process_query_batch_cpu(batch_query_names, batch_hits_df, annotation_mat,
                            proteins, term_names, use_rscore, n_terms):
    batch_query_to_local = {qid: idx for idx, qid in enumerate(batch_query_names)}
    prepared = prepare_batch_arrays(batch_hits_df, proteins, batch_query_to_local, use_rscore)
    if prepared is None:
        return []
    query_idx, protein_idx, weights = prepared

    hit_annots = annotation_mat[protein_idx, :].tocoo()
    if hit_annots.nnz == 0:
        return []

    weighted_data = hit_annots.data * weights[hit_annots.row]
    term_indices = hit_annots.col
    query_assignments = query_idx[hit_annots.row]

    batch_scores = np.zeros((len(batch_query_names), annotation_mat.shape[1]), dtype=np.float32)
    np.maximum.at(batch_scores, (query_assignments, term_indices), weighted_data)

    return collect_batch_results(batch_query_names, batch_scores, term_names, n_terms)


def process_query_batch_gpu(batch_query_names, batch_hits_df, annotation_mat_gpu,
                            proteins, term_names, use_rscore, n_terms):
    batch_query_to_local = {qid: idx for idx, qid in enumerate(batch_query_names)}
    prepared = prepare_batch_arrays(batch_hits_df, proteins, batch_query_to_local, use_rscore)
    if prepared is None:
        return []
    query_idx, protein_idx, weights = prepared

    query_idx_gpu = cp_gpu.asarray(query_idx)
    protein_idx_gpu = cp_gpu.asarray(protein_idx)
    weights_gpu = cp_gpu.asarray(weights)

    hit_annots = annotation_mat_gpu[protein_idx_gpu, :]
    if hit_annots.nnz == 0:
        return []
    hit_annots = hit_annots.tocoo()

    weighted_data = hit_annots.data * weights_gpu[hit_annots.row]
    term_indices = hit_annots.col
    query_assignments = query_idx_gpu[hit_annots.row]

    batch_scores = cp_gpu.zeros((len(batch_query_names), annotation_mat_gpu.shape[1]), dtype=cp_gpu.float32)
    cp_gpu.maximum.at(batch_scores, (query_assignments, term_indices), weighted_data)

    batch_scores_cpu = cp_gpu.asnumpy(batch_scores)
    return collect_batch_results(batch_query_names, batch_scores_cpu, term_names, n_terms)


def blast_predict(annot_file, query_file, indices, graph, add_graph,
                 blast_results, output_baseline, config_path=None, keep_self_hits=False,
                 use_rscore=False, n_terms=None, batch_size=1000):
    """Make predictions for query sequences based on BLAST hits."""

    if '.gaf' in annot_file or '.dat' in annot_file:
        if not graph:
            print("Please provide a graph file for GAF or DAT input")
            sys.exit(1)
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        GO_CODES = config.get('go_codes', {})
        wrapper_retrieve_terms(
            annot_file=annot_file,
            go_codes=GO_CODES,
            selected_go_codes='Experimental,IC,TAS',
            graph=graph,
            output_tsv=f'{os.path.dirname(output_baseline)}/blast_terms.tsv'
        )
        terms_df = pd.read_csv(
            f'{os.path.dirname(output_baseline)}/blast_terms.tsv',
            sep='\t', header=0,
            names=['EntryID', 'term', 'aspect']
        )
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
        os.remove(f'{os.path.dirname(output_baseline)}/blast_terms.tsv')
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
    else:
        print("Invalid annotation file format")
        sys.exit(1)

    print(f"Annotation matrix shape: {annotation_mat.shape}")
    print(f"Number of proteins: {len(proteins)} | Number of terms: {len(terms)}")

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

    query_set = set(query_ids)
    query_order = {qid: idx for idx, qid in enumerate(query_ids)}
    chunksize = 1_000_000

    estimated_buckets = max(1, len(query_ids) // max(batch_size, 1)) * 4
    num_buckets = min(4096, max(1, estimated_buckets))
    query_to_bucket = {qid: hash(qid) % num_buckets for qid in query_ids}

    print(f"Partitioning BLAST hits from {blast_results} into {num_buckets} buckets (single pass)...")
    temp_dir, bucket_paths, bucket_query_sets, queries_with_hits = partition_blast_hits_single_pass(
        blast_results=blast_results,
        query_set=query_set,
        query_to_bucket=query_to_bucket,
        keep_self_hits=keep_self_hits,
        chunksize=chunksize,
        num_buckets=num_buckets,
    )

    effective_queries = [qid for qid in query_ids if qid in queries_with_hits]
    print(f"Processing {len(effective_queries)} query sequences with BLAST hits")
    if not effective_queries:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("No BLAST hits available for the provided queries.")
        return

    term_names = [None] * len(terms)
    for term, idx in terms.items():
        term_names[idx] = term

    annotation_mat_gpu = None
    if GPU_AVAILABLE:
        print("Transferring annotation matrix to GPU...")
        annotation_mat_gpu = cp_sparse.csr_matrix(annotation_mat, dtype=cp_gpu.float32)
    else:
        print("GPU not available, proceeding with CPU computation.")

    output_dir = os.path.dirname(output_baseline)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    open_func = gzip.open if output_baseline.endswith('.gz') else open
    total_batches = sum(
        (len(bucket_query_sets[idx]) + batch_size - 1) // batch_size
        for idx in range(num_buckets)
    )
    progress = tqdm(total=total_batches, desc="Processing batches") if total_batches else None

    try:
        with open_func(output_baseline, 'wt', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t')

            for bucket_idx, bucket_path in enumerate(bucket_paths):
                if not os.path.exists(bucket_path):
                    continue
                bucket_queries = bucket_query_sets[bucket_idx]
                if not bucket_queries:
                    continue
                bucket_df = pd.read_csv(
                    bucket_path,
                    sep='\t',
                    dtype={'evalue': np.float64, 'pident': np.float32},
                )

                sorted_queries = sorted(
                    bucket_queries,
                    key=lambda qid: query_order.get(qid, sys.maxsize)
                )

                for batch_query_names in chunk_list(sorted_queries, batch_size):
                    batch_hits_df = bucket_df[bucket_df['qseqid_acc'].isin(batch_query_names)]
                    if batch_hits_df.empty:
                        if progress:
                            progress.update(1)
                        continue

                    if GPU_AVAILABLE and annotation_mat_gpu is not None:
                        batch_results = process_query_batch_gpu(
                            batch_query_names,
                            batch_hits_df,
                            annotation_mat_gpu,
                            proteins,
                            term_names,
                            use_rscore,
                            n_terms,
                        )
                    else:
                        batch_results = process_query_batch_cpu(
                            batch_query_names,
                            batch_hits_df,
                            annotation_mat,
                            proteins,
                            term_names,
                            use_rscore,
                            n_terms,
                        )

                    if batch_results:
                        writer.writerows(batch_results)

                    if progress:
                        progress.update(1)

                del bucket_df
    finally:
        if progress:
            progress.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Prediction completed successfully")
    if GPU_AVAILABLE and annotation_mat_gpu is not None:
        mempool = cp_gpu.get_default_memory_pool()
        print(f"GPU memory used: {mempool.used_bytes() / 1e9:.2f} GB")
        print(f"GPU memory total: {mempool.total_bytes() / 1e9:.2f} GB")


def parse_args(argv):
    parser = argparse.ArgumentParser(description='BLAST sequence similarity baseline')
    parser.add_argument('--annot_file', '-a',
                        help='Path to the propagated annotation sparse matrix or a .gaf or .dat file',
                        required=True)
    parser.add_argument('--indices', '-i',
                        help='Path to the term & protein indices file',
                        required=False, default=None)
    parser.add_argument('--graph', help='Path to OBO file for GO graph',
                        required=False, default=None)
    parser.add_argument('--add_graph', help='Path to additional OBO file',
                        required=False, default=None)
    parser.add_argument('--query_file', '-q',
                        help='FASTA file or text file containing query IDs',
                        required=True)
    parser.add_argument('--blast_results', '-b',
                        help='Path to the BLAST results file',
                        required=True)
    parser.add_argument('--output_baseline', '-o',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--use_rscore', action='store_true',
                        help='Use R-score instead of sequence identity for weighting')
    parser.add_argument('--keep_self_hits', action='store_true',
                        help='Keep self-hits in BLAST results')
    parser.add_argument('--n_terms', '-n', type=int, required=False,
                        help='Upper limit for number of terms per target', default=None)
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Number of queries to process per batch (default: 1000)')
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
        config_path=os.path.join(os.path.dirname(__file__), 'config.yaml'),
        keep_self_hits=args.keep_self_hits,
        use_rscore=args.use_rscore,
        n_terms=args.n_terms,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
