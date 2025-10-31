#!/usr/bin/env python3
"""
UniProt non-experimental annotations baseline
Makes predictions based on the non-experimental GO terms that the protein has
Requires helper scripts retrieve_terms.py and ontology.py
"""

from retrieve_terms import wrapper_retrieve_terms
import sys
import os
import argparse
import pandas as pd
from Bio import SeqIO
import gzip
# import dask.dataframe as dd


    
def create_predictions(terms_file, query_file, output_baseline):
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
    terms_df = pd.read_csv(terms_file, sep='\t', header=0, names=['EntryID', 'term', 'aspect'])
    terms_df = terms_df[terms_df['EntryID'].isin(query_ids)]
    del terms_df['aspect']
    terms_df['value'] = [1] * len(terms_df)
    
    open_func = gzip.open if output_baseline.endswith('.gz') else open
    with open_func(output_baseline, 'wt') as out_f:
        for idx_, row in terms_df.iterrows():
            out_f.write(f"{row['EntryID']}\t{row['term']}\t{row['value']}\n")
    # # Use dask to write to gzipped TSV
    # # terms_df.to_csv(output_baseline, sep='\t', index=False, header=False)
    # dask_df = dd.from_pandas(terms_df, npartitions=16)
    # if output_baseline.endswith('.gz'):
    #     dask_df.to_csv(output_baseline, sep="\t", index=False, header=False, single_file = True, compression='gzip')
    # elif output_baseline.endswith('.tsv'):
    #     dask_df.to_csv(output_baseline, sep="\t", index=False, header=False, single_file = True)
    print(f"Predictions for {len(set(terms_df['EntryID']))} proteins written to {output_baseline}")
    
    
def goa_nonexp_predict(annot_file, selected_go, graph, add_graph, query_file, output_baseline):
    # config_go_codes = GO_CODES
    # Reading config
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    GO_CODES = config.get('go_codes', {})
    # RAW_FILE_PATHS = config.get('raw_file_paths', {})
    
    wrapper_retrieve_terms(
        annot_file=annot_file,
        go_codes=GO_CODES,
        selected_go_codes=selected_go,
        graph=graph,
        output_tsv=f'{os.path.dirname(output_baseline)}/nonexp_terms.tsv' # just a temporary file
    )
    
    create_predictions(f'{os.path.dirname(output_baseline)}/nonexp_terms.tsv', query_file, output_baseline)
    os.remove(f'{os.path.dirname(output_baseline)}/nonexp_terms.tsv')
    
    
def parse_args(argv):
    parser = argparse.ArgumentParser(description='UniProt non-experimental annotations baseline')
    parser.add_argument('--annot_file',  help='Path to the UniProt GOA file', required=True)
    parser.add_argument('--selected_go',  help='Selected GO codes', required=True)
    parser.add_argument('--graph', help='Path to OBO file for GO graph', required=True)
    parser.add_argument('--add_graph',help='Path to additional OBO file for GO graph at a later time point', required=False, default=None)
    parser.add_argument('--query_file',  help='FASTA file or text file containing query IDs', required=True)
    parser.add_argument('--output_baseline',  help='Path to the output file', required=True)
    return parser.parse_args(argv)    


def main():
    args = parse_args(sys.argv[1:])
    goa_nonexp_predict(
        annot_file=args.annot_file,
        selected_go=args.selected_go,
        graph=args.graph,
        add_graph=args.add_graph,
        query_file=args.query_file,
        output_baseline=args.output_baseline
    )
    
if __name__ == '__main__':
    main()
