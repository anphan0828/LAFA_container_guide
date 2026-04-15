import argparse
import h5py
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import warnings
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)

def normalize_embeddings_chunked(input_file, chunk_size=100000):
    """
    Normalize embedding similarities using chunked processing to handle large files
    without OOM errors. Processes 53GB+ files efficiently.
    """
    print(f"Starting normalization of {input_file}")
    print("Phase 1: Finding global maximum similarity...")
    
    # Phase 1: Find global maximum similarity value across all chunks
    global_max = float('-inf')
    total_rows = 0
    
    # Read in chunks to find global max without loading entire file
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, sep='\t', chunksize=chunk_size)):
        chunk_max = chunk['Similarity'].max()
        if chunk_max > global_max:
            global_max = chunk_max
        total_rows += len(chunk)
        
        if chunk_num % 100 == 0:
            print(f"  Processed {chunk_num * chunk_size:,} rows, current global max: {global_max:.6f}")
    
    print(f"Global maximum similarity: {global_max:.6f}")
    print(f"Total rows to process: {total_rows:,}")
    
    # Phase 2: Process chunks and write normalized results
    output_file = input_file.replace(".tsv", "_norm.tsv")
    print(f"Phase 2: Normalizing and writing to {output_file}")
    
    def normalize_similarity(x, x_max):
        """Normalize similarity: convert distance to percentage similarity"""
        normalized = (x_max - x) / x_max
        return round(normalized * 100, 2)
    
    def clean_protein_ids(df_chunk):
        """Remove UniProt prefixes and suffixes from protein IDs"""
        # Create a copy of relevant columns only
        result = df_chunk.iloc[:, 1:].copy()  # Skip first column (index)
        
        # Remove 'sp|' and 'tr|' prefixes, then remove everything after '|'
        result['DB ID'] = result['DB ID'].str.replace(r'^(sp\||tr\|)', '', regex=True)
        result['DB ID'] = result['DB ID'].str.replace(r'\|.*', '', regex=True)
        
        # Also clean Query ID if it has prefixes
        if 'Query ID' in result.columns:
            result['Query ID'] = result['Query ID'].str.replace(r'^(sp\||tr\|)', '', regex=True)
            result['Query ID'] = result['Query ID'].str.replace(r'\|.*', '', regex=True)
        
        return result
    
    # Process file in chunks
    processed_rows = 0
    first_chunk = True
    
    with open(output_file, 'w') as out_file:
        for chunk_num, chunk in enumerate(pd.read_csv(input_file, sep='\t', chunksize=chunk_size)):
            # Normalize similarity values
            chunk['Similarity'] = chunk['Similarity'].apply(lambda x: normalize_similarity(x, global_max))
            
            # Clean protein IDs
            cleaned_chunk = clean_protein_ids(chunk)
            
            # Write to output file
            if first_chunk:
                # Write header
                cleaned_chunk.to_csv(out_file, sep='\t', index=False, mode='w')
                first_chunk = False
            else:
                # Append without header
                cleaned_chunk.to_csv(out_file, sep='\t', index=False, mode='a', header=False)
            
            processed_rows += len(chunk)
            
            if chunk_num % 50 == 0:
                print(f"  Processed {processed_rows:,}/{total_rows:,} rows ({processed_rows/total_rows*100:.1f}%)")
    
    print(f"Normalization completed: {output_file}")
    print(f"Final stats: {processed_rows:,} rows processed")
    return output_file

# def process_embeddings_optimized(evalset_file, dbset_file, output_file, top_k=1000, batch_size=1000):
#     print(f"Started at: {time.ctime()}")
    
#     # Load database embeddings efficiently
#     print("Loading database embeddings...")
#     with h5py.File(dbset_file, 'r') as db_h5:
#         db_ids = list(db_h5.keys())
#         # Load as contiguous numpy array (much faster)
#         db_embeddings = np.array([db_h5[db_id][:] for db_id in tqdm(db_ids)])
    
#     print(f"Database shape: {db_embeddings.shape}")
    
#     actual_top_k = min(top_k, len(db_ids))
#     if actual_top_k < top_k:
#         print(f"Warning: Database only has {len(db_ids)} proteins, using top_k={actual_top_k} instead of {top_k}")
#         top_k = actual_top_k
#     # Process queries in batches
#     print("Processing query embeddings...")
    
#     results = []
#     def safe_top_k_selection(distances, k):
#         """Safely select top-k indices, handling cases where k > len(distances)"""
#         n = len(distances)
#         if n == 0:
#             return np.array([]), np.array([])
        
#         actual_k = min(k, n)
        
#         if actual_k == n:
#             # Return all indices sorted by distance
#             indices = np.argsort(distances)
#         else:
#             # Use argpartition for efficiency
#             indices = np.argpartition(distances, actual_k)[:actual_k]
        
#         return indices, distances[indices]
    
#     with h5py.File(evalset_file, 'r') as query_h5:
#         query_ids = list(query_h5.keys())
        
#         # Process in batches to manage memory
#         for batch_start in tqdm(range(0, len(query_ids), batch_size), desc="Query batches"):
#             batch_end = min(batch_start + batch_size, len(query_ids))
#             batch_query_ids = query_ids[batch_start:batch_end]
            
#             # Load batch of query embeddings
#             batch_queries = np.array([query_h5[qid][:] for qid in batch_query_ids])
            
#             # Compute all pairwise distances for this batch
#             distances = euclidean_distances(batch_queries, db_embeddings)
            
#             # Find top-k for each query in batch
#             for i, query_id in enumerate(batch_query_ids):
#                 query_distances = distances[i]
                
#                 top_k_indices, top_k_distances = safe_top_k_selection(query_distances, top_k)
#                 top_k_db_ids = [db_ids[idx] for idx in top_k_indices]
                
#                 # Create results for this query
#                 batch_results = pd.DataFrame({
#                     'Query ID': [query_id] * top_k,
#                     'DB ID': top_k_db_ids,
#                     'e-val': [0] * top_k,
#                     'Length': [0] * top_k,
#                     'Similarity': top_k_distances,
#                     'N-ident': [0] * top_k
#                 })
                
#                 results.append(batch_results)
    
#     # Combine all results
#     print("Combining results...")
#     final_results = pd.concat(results, ignore_index=True)
    
#     # Save results
#     print(f"Saving {len(final_results)} results to {output_file}")
#     final_results.to_csv(output_file, sep='\t', index=True)
    
#     print(f"Completed at: {time.ctime()}")



def process_embeddings_gpu(evalset_file, dbset_file, output_file, top_k=3, normalize=True):
    import cupy as cp  # GPU arrays
    from cuml.neighbors import NearestNeighbors  # GPU k-NN
    
    print("Loading database embeddings to GPU...")
    with h5py.File(dbset_file, 'r') as db_h5:
        db_ids = list(db_h5.keys())
        db_embeddings = cp.array([db_h5[db_id][:] for db_id in db_ids])
    
    # Truncation happens here: retain only the nearest top-k distances per query.
    actual_top_k = min(top_k, len(db_ids))
    if actual_top_k < top_k:
        print(f"Warning: Database only has {len(db_ids)} proteins, using top_k={actual_top_k} instead of {top_k}")
    
    # Use GPU k-NN for fast similarity search
    knn = NearestNeighbors(n_neighbors=actual_top_k, metric='euclidean')
    knn.fit(db_embeddings)
    
    results = []
    with h5py.File(evalset_file, 'r') as query_h5:
        query_ids = list(query_h5.keys())
        
        for query_id in tqdm(query_ids):
            query_embedding = cp.array(query_h5[query_id][:]).reshape(1, -1)
            
            # Find k nearest neighbors on GPU
            distances, indices = knn.kneighbors(query_embedding)
            
            # Convert back to CPU for DataFrame creation
            distances = distances.get().flatten()
            indices = indices.get().flatten()
            
            # Use actual number of results returned
            n_results = len(distances)
            
            batch_results = pd.DataFrame({
                'Query ID': [query_id] * n_results,
                'DB ID': [db_ids[idx] for idx in indices],
                'e-val': [0] * n_results,
                'Length': [0] * n_results,
                'Similarity': distances,
                'N-ident': [0] * n_results
            })
            
            results.append(batch_results)
    
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_csv(output_file, sep='\t', index=True)
    
    # Normalize only the retained top-k rows, so scores depend on the retained neighborhood distribution.
    if normalize:
        print("Starting automatic normalization...")
        normalized_file = normalize_embeddings_chunked(output_file)
        return normalized_file
    
    return output_file
    
def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Compute ProtT5 embedding distances, retain top-k neighbors, and optionally normalize them.'
    )
    parser.add_argument('evalset_file', help='HDF5 file containing query embeddings')
    parser.add_argument('dbset_file', help='HDF5 file containing database embeddings')
    parser.add_argument('output_file', help='Path to raw distance output TSV')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of nearest neighbors to retain per query before normalization (default: 3)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize the retained distances into percentage similarity scores')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.top_k < 1:
        raise SystemExit('--top_k must be a positive integer')

    result_file = process_embeddings_gpu(
        args.evalset_file,
        args.dbset_file,
        args.output_file,
        top_k=args.top_k,
        normalize=args.normalize,
    )

    print(f"Processing completed. Final output: {result_file}")
        
