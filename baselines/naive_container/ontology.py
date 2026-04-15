#!/usr/bin/env python3

import sys
import argparse
import logging
import obonet
import numpy as np
import pandas as pd
import networkx as nx
import pickle as cp
import time
from collections import Counter
from scipy.sparse import dok_matrix, save_npz, csr_matrix, vstack

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def clean_ontology_edges(ontology):
    """
    Remove all ontology edges except types "is_a" and "part_of" and ensure there are no inter-ontology edges
    :param ontology: Ontology stucture (networkx DiGraph or MultiDiGraph)
    """
    
    # keep only "is_a" and "part_of" edges (All the "regulates" edges are in BPO)
    remove_edges = [(i, j, k) for i, j, k in ontology.edges if not(k=="is_a" or k=="part_of")]
    
    ontology.remove_edges_from(remove_edges)
    
    # There should not be any cross-ontology edges, but we verify here
    crossont_edges = [(i, j, k) for i, j, k in ontology.edges if
                      ontology.nodes[i]['namespace']!= ontology.nodes[j]['namespace']]
    if len(crossont_edges)>0:
        ontology.remove_edges_from(crossont_edges)
    
    return ontology
    

def fetch_aspect(ontology, root:str):
    """
    Return a subgraph of an ontology starting at node <root>
    
    :param ontology: Ontology stucture (networkx DiGraph or MultiDiGraph)
    :param root: node name (GO term) to start subgraph
    """
    
    
    namespace = ontology.nodes[root]['namespace']
    aspect_nodes = [n for n,v in ontology.nodes(data=True) 
                    if v['namespace']==namespace]
    subont_ = ontology.subgraph(aspect_nodes)
    return subont_

def add_aspect_column(terms_df, subontologies):
    """
    Add aspect column to terms_df based on the subontologies provided.
    
    :param terms_df: pandas DataFrame of annotated terms (column names 'EntryID', 'term')
    :param subontologies: dict of ontology aspects (networkx DiGraphs or MultiDiGraphs)
    """
    
    # Create a mapping from term to aspect
    term_to_aspect = {}
    for aspect, subont in subontologies.items():
        for term in subont.nodes:
            term_to_aspect[term] = aspect
    
    # Map the aspect to each term in the DataFrame
    terms_df['aspect'] = terms_df['term'].map(term_to_aspect)
    
    return terms_df


def propagate_terms(terms_df, subontologies):
    """
    Propagate terms in DataFrame terms_df abbording to the structure in subontologies.
    If terms were already propagated with the same graph, the returned dataframe will be equivalent to the input
    
    :param terms_df: pandas DataFrame of annotated terms (column names 'EntryID', 'term' 'aspect')
    :param subontologies: dict of ontology aspects (networkx DiGraphs or MultiDiGraphs)
    """
    
    # Look up ancestors ahead of time for efficiency
    subont_terms = {aspect: set(terms_df[terms_df.aspect==aspect].term.values) for aspect in subontologies.keys()}
    ancestor_lookup = {aspect:{t: nx.descendants(subont,t) for t in subont_terms[aspect]
                             if t in subont} for aspect, subont in subontologies.items()}
    
    propagated_terms = []
    for (protein, aspect), entry_df in terms_df.groupby(['EntryID', 'aspect']):
        protein_terms = set().union(*[list(ancestor_lookup[aspect][t])+[t] for t in set(entry_df.term.values)])
    
        propagated_terms += [{'EntryID': protein, 'term': t, 'aspect': aspect} for t in protein_terms]
    
    return pd.DataFrame(propagated_terms)


def replace_alternate_GO_terms(df, ontology_graph):
    """
    Replace alternate GO terms with their main GO terms using vectorized operations.
    
    Args:
        df: pandas DataFrame containing GO terms in 'term' column
        graph: path to the ontology graph file (obo)
        
    Returns:
        DataFrame with updated GO terms where alternate IDs are replaced with main IDs
    """
    # Create a dictionary to map alternate GO terms to main GO terms
    alt_id_to_id = {}
    for id_, data in ontology_graph.nodes(data=True):
        alt_ids = data.get('alt_id')
        if alt_ids:
            for alt_id in alt_ids:
                alt_id_to_id[alt_id] = id_
    
    replaced = df['term'].isin(alt_id_to_id.keys())
    if replaced.any():
        df.loc[replaced, 'term'] = df.loc[replaced, 'term'].map(alt_id_to_id)
    return df


def filter_terms_given_obo(terms_df, current_graph, pivot_graph):
    """Remove terms on a future obo that is not in the chosen pivot graph"""
    
    # Propagate using current graph
    ontology_graph = clean_ontology_edges(obonet.read_obo(current_graph))
    roots = {'P': 'GO:0008150', 'C': 'GO:0005575', 'F': 'GO:0003674'}
    subontologies = {aspect: fetch_aspect(ontology_graph, roots[aspect]) for aspect in roots} 
    
    prop_terms_df = propagate_terms(terms_df, subontologies)
    before_length = len(prop_terms_df)
    
    # Compare with pivot graph, remove terms not in pivot graph
    ontology_graph = clean_ontology_edges(obonet.read_obo(pivot_graph))
    
    prop_terms_df = prop_terms_df[prop_terms_df['term'].isin(ontology_graph.nodes())].reset_index(drop=True)
    after_length = len(prop_terms_df)
    logger.info(f"Filtered terms from {before_length} to {after_length} using ontology graph {pivot_graph.split('/')[-1]}")
    
    # Propagate terms using the chosen pivot graph
    subontologies = {aspect: fetch_aspect(ontology_graph, roots[aspect]) for aspect in roots} 
    annotation_df = propagate_terms(prop_terms_df, subontologies)
    
    return annotation_df


def term_counts(terms_df, term_indices):
    """
    Count the number of instances of each term
    
    :param terms_df: pandas DataFrame of (propagated) annotated terms (column names 'EntryID', 'term', 'aspect')
    :param term_indices:
    """
    
    num_proteins = len(terms_df.groupby('EntryID'))
    S = dok_matrix((num_proteins+1, len(term_indices)), dtype=np.int32)
    S[-1,:] = 1  # dummy protein
    
    for i, (protein, protdf) in enumerate(terms_df.groupby('EntryID')):
        row_count = {term_indices[t]:c for t,c in Counter(protdf['term']).items()}
        for col, count in row_count.items():
            S[i, col] = count
    
    # return S
    
    # this term_counts is currently dealing with all aspects at once
    # proteins_idx = {p:i for i,p in enumerate(terms_df['EntryID'].unique())}
    # term_indices = {t:i for i,t in enumerate(terms_df['term'].unique())}
    # protein_idx = {p:i for i,p in enumerate(sorted(terms_df['EntryID'].unique()))}
    # num_proteins = len(terms_df.groupby('EntryID'))
    # S = dok_matrix((len(num_proteins)+1, len(term_indices)), dtype=np.int32)
    # S[-1,:] = 1  # dummy protein
    
    # for protein, i in protein_idx.items():
    #     protdf = terms_df.groupby('EntryID').get_group(protein)
    #     protein_terms = [t for t in protdf['term'].values if t in term_indices]
    #     row_count = {term_indices[t]:c for t,c in Counter(protein_terms).items()}
    #     for col, count in row_count.items():
    #         S[i, col] = count
    
    S = S.astype(bool)
    return S


def calc_ia(term, count_matrix, ontology, terms_index):
    
    parents = nx.descendants_at_distance(ontology, term, 1)
    
    # count of proteins with term
    prots_with_term = count_matrix[:,terms_index[term]].sum()
    # count of proteins with all parents
    num_parents = len(parents)
    prots_with_parents = (count_matrix[:,[terms_index[p] for p in parents]].sum(1)==num_parents).sum()
    # avoid floating point errors by returning exactly zero
    if prots_with_term == prots_with_parents:
        return 0
    
    return -np.log2(prots_with_term/prots_with_parents)


def approach1_pivot_table(annotation_df):
    """First approach using pandas pivot_table"""
    start_time = time.time()
    
    matrix = pd.pivot_table(
        annotation_df,
        values='aspect',
        index='EntryID',
        columns='term',
        aggfunc='count',
        fill_value=0
    )
    protein_idx = {p:i for i,p in enumerate(matrix.index)}
    term_idx = {t: i for i,t in enumerate(matrix.columns)}
    matrix = matrix.astype(bool)
    matrix = dok_matrix(matrix.values)
    
    # # Add dummy protein
    # dummy_row = dok_matrix((1, matrix.shape[1]), dtype=bool)
    # dummy_row[0,:] = True
    # matrix = vstack([dummy_row, matrix])
    # matrix = matrix.todok()
    end_time = time.time()
    return matrix, protein_idx, term_idx, end_time - start_time


def approach2_term_counts(annotation_df):
    """Second approach using Counter and manual matrix construction"""
    start_time = time.time()
    
    term_indices = {t: i for i, t in enumerate(annotation_df['term'].unique())}
    num_proteins = len(annotation_df.groupby('EntryID'))
    # S = dok_matrix((num_proteins+1, len(term_indices)), dtype=np.int32)
    # S[-1,:] = 1  # dummy protein
    S = dok_matrix((num_proteins, len(term_indices)), dtype=np.int32)
    
    proteins = []
    for i, (protein, protdf) in enumerate(annotation_df.groupby('EntryID')):
        proteins.append(protein)
        row_count = {term_indices[t]:c for t,c in Counter(protdf['term']).items()}
        for col, count in row_count.items():
            S[i, col] = count
    
    protein_idx = {p:i for i,p in enumerate(proteins)}
    S = S.astype(bool)
    end_time = time.time()
    return S, protein_idx, term_indices, end_time - start_time


def sparse_matrix_and_indices(annotation_df):
    """Optimized approach using scipy's sparse matrix construction"""
    start_time = time.time()
    
    proteins = sorted(annotation_df['EntryID'].unique())
    terms = sorted(annotation_df['term'].unique())
    
    protein_idx = {p: i for i, p in enumerate(proteins)}
    term_indices = {t: i for i, t in enumerate(terms)}
    
    # Get rows and columns for sparse matrix
    rows = [protein_idx[p] for p in annotation_df['EntryID']]
    cols = [term_indices[t] for t in annotation_df['term']]
    
    # Create sparse matrix directly, from coordinates store in rows and cols
    data = np.ones(len(rows), dtype=bool)
    S = csr_matrix((data, (rows, cols)), 
                  shape=(len(protein_idx), len(term_indices)))
    
    # Convert to dok for comparison
    S_dok = S.todok()
    # dummy_row = dok_matrix((1, S.shape[1]), dtype=bool)
    # dummy_row[0,:] = True
    # S_dok = vstack([dummy_row, S_dok])
    # S_dok = S_dok.todok()
    
    end_time = time.time()
    return S_dok, protein_idx, term_indices, end_time - start_time
    
    
def propagate_and_ia(terms_file, graph, tsv_propagated, matrix_propagated, matrix_indices, output_tsv):
    # load ontology graph and get three subontologies
    ontology_graph = clean_ontology_edges(obonet.read_obo(graph))
    roots = {'P': 'GO:0008150', 'C': 'GO:0005575', 'F': 'GO:0003674'}
    subontologies = {aspect: fetch_aspect(ontology_graph, roots[aspect]) for aspect in roots} 
    
    # these terms should be propagated using the same ontology, otherwise IA may be negative
    annotation_df = pd.read_csv(terms_file, sep='\t')
    logger.info('Propagating Terms')
    annotation_df = propagate_terms(annotation_df, subontologies)
    
    if tsv_propagated:
        logger.info(f'Saving propagated terms to {tsv_propagated}')
        annotation_df.to_csv(tsv_propagated, sep='\t', index=False)
    ## Approach 1: Use pd.pivot_table (20sec for 900k annotations)
    # matrix1, pidx1, tidx1, time1 = approach1_pivot_table(annotation_df)
    
    ## Approach 2: use Counter in term_counts function (7sec for 900k annotations)       
    # matrix2, pidx2, tidx2, time2 = approach2_term_counts(annotation_df)
    
    ## Approach 3: create csr directly from coordinates
    matrix3, pidx3, tidx3, time3 = sparse_matrix_and_indices(annotation_df)
    
    if matrix_propagated and matrix_indices:
        logger.info(f'Saving to file {matrix_propagated}')
        save_npz(matrix_propagated, matrix3.tocsr())
        with open(matrix_indices, 'wb') as f:
            cp.dump((pidx3, tidx3), f)    
                    
    # # Count term instances with respect to aspect for IA calculation
    # logger.debug('Counting Terms')
    if output_tsv:
        aspect_counts = dict()
        aspect_terms = dict()
        term_idx = dict()
        # for aspect, subont in subontologies.items():
        #     aspect_terms[aspect] = sorted(subont.nodes)  # ensure same order
        #     term_idx[aspect] = {t:i for i,t in enumerate(aspect_terms[aspect])}
        #     # aspect_counts[aspect] = term_counts(annotation_df[annotation_df.aspect==aspect], term_idx[aspect])
        #     matrix = term_counts(annotation_df[annotation_df.aspect==aspect], term_idx[aspect])

        #     assert matrix.sum() == len(annotation_df[annotation_df.aspect==aspect]) + len(aspect_terms[aspect])
        #     aspect_counts[aspect] = matrix
            
        # Compute IA
        dummy_row = dok_matrix((1, matrix3.shape[1]), dtype=bool)
        dummy_row[0,:] = True
        matrix3 = vstack([dummy_row, matrix3])
        matrix3 = matrix3.todok()
        for aspect, subont in subontologies.items():
            aspect_terms[aspect] = sorted(subont.nodes)  # ensure same order
            # term_idx[aspect] = {t:i for i,t in enumerate(aspect_terms[aspect])}
            terms = set.intersection(set(aspect_terms[aspect]), set(tidx3.keys()))
            term_idx[aspect] = {t:i for i,t in enumerate(terms)}
            matrix = matrix3[:,[tidx3[t] for t in term_idx[aspect]]]
            # remove rows with all zeros (proteins that are not annotated in this aspect)
            matrix = matrix[matrix.sum(1).A.flatten()>0,:]
            assert matrix.sum() == len(annotation_df[annotation_df.aspect==aspect]) + len(terms)
            aspect_counts[aspect] = matrix
            
        # since we are indexing by column to compute IA, 
        # let's convert to Compressed Sparse Column format
        sp_matrix = {aspect:dok.tocsc() for aspect, dok in aspect_counts.items()}
        # TODO: new ia calculation is wrong, the roots have ia > 0 because proteins have to be within aspect
        logger.info('Computing Information Accretion')
        aspect_ia = {aspect: {t:0 for t in aspect_terms[aspect]} for aspect in aspect_terms.keys()}
        for aspect, subontology in subontologies.items():
            for term in term_idx[aspect].keys():
                aspect_ia[aspect][term] = calc_ia(term, sp_matrix[aspect], subontology, term_idx[aspect])
        
        ia_df = pd.concat([pd.DataFrame.from_dict(
            {'term':aspect_ia[aspect].keys(), 
            'ia': aspect_ia[aspect].values(), 
            'aspect': aspect}) for aspect in subontologies.keys()])
        
        # all counts should be non-negative
        assert ia_df['ia'].min() >= 0
        
        # Save to file
        logger.info(f'Saving to file {output_tsv}')
        ia_df[['term','ia']].to_csv(output_tsv, header=None, sep='\t', index=False)


