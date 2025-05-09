"""Batched genome query script.

Inputs:
    --query : query string.
    --input_fpath : text file of paths to genomes in fasta format (contigs).
    --outdir : directory to store output.
    --outfname : output filename.
    --batch_size : number of windows to process at once. Default 1000.
    --pad_left : number of bases to include before query match.
    --pad_right : number of bases to include after query match.
    --verbosity : verbosity level.
    --jax : use JAX acceleration.

Outputs:
    Writes results to an output file.
"""

import argparse
import sys
import os
import time
import numpy as np
import warnings

from .helpers import array_to_gene_seq, gene_seq_to_array, pad_matrix_for_batch_size
from .io import process_contig_file, get_contigs_matrix
from .core import search_matrix_for_query
from .jax.core import search_matrix_for_query as jax_search_matrix

GEN_NT_VAL = 4
PAD_VAL = 5


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', type=str, required=True)
    parser.add_argument('-i', '--input_fpath', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, default="out")
    parser.add_argument('-f', '--outfname', type=str, default="results.tsv")
    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-pl', '--pad_left', type=int, default=0)
    parser.add_argument('-pr', '--pad_right', type=int, default=0)
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--jax', action="store_true")
    return parser.parse_args(args)


def write_results(
        out_fpath, *, 
        genome_fpath, 
        query_string, 
        min_dist,
        nearest_idxs,
        location_on_contigs,
        contig_segments,
        time_elapsed,
):
    vals = [
        genome_fpath,
        query_string,
        min_dist,
        nearest_idxs,
        location_on_contigs,
        contig_segments,
        time_elapsed,
    ]
    line = "\t".join([str(v) for v in vals])
    with open(out_fpath, 'a') as f:
        f.write(line + "\n")


def main(args):
    input_fpath = args.input_fpath
    query_string = args.query
    outdir = args.outdir
    outfname = args.outfname
    batch_size = args.batch_size
    pad_left = args.pad_left
    pad_right = args.pad_right
    verbosity = args.verbosity
    use_jax = args.jax
    
    def printv(s, importance=1, **kwargs):
        if verbosity >= importance:
            print(s, **kwargs)

    os.makedirs(outdir, exist_ok=True)

    printv(f"Searching genome files listed in: {input_fpath}", 1)
    printv(f"Query (length {len(query_string)}): {query_string}", 1)
    query = gene_seq_to_array(query_string)

    # Read into a list the paths contained in the input file
    with open(input_fpath, 'r') as f:
        genome_filepaths = [line.strip() for line in f.readlines()]
    
    printv(f"Found {len(genome_filepaths)} genome files.", 1)

    outfpath = f"{outdir}/{outfname}"
    column_names = [
        "genome_fpath",
        "query_string",
        "min_distance",
        "nearest_idxs",
        "location_on_contigs",
        "contig_segments",
        "time_elapsed",
    ]
    with open(outfpath, 'w') as f:
        f.write("\t".join(column_names) + "\n")

    # Define the search function, with or without JAX acceleration
    if use_jax:
        def search_func(contigs, query):
            return jax_search_matrix(
                contigs, query,
                batch_size=batch_size, 
                pad_val=PAD_VAL,
            )
    else:
        def search_func(contigs, query):
            return search_matrix_for_query(
                contigs, query,
            )
    
    for genome_fpath in genome_filepaths:
        time0 = time.time()
        
        # Load contigs into a matrix from the contig file.
        printv("Loading genome file...", 1)
        contigs_list = process_contig_file(genome_fpath)
        contigs = get_contigs_matrix(contigs_list, pad_val=PAD_VAL)
        contigs = pad_matrix_for_batch_size(
            contigs, len(query_string), batch_size, pad_val=PAD_VAL, axis=1,
        )
        printv(f"Loaded {len(contigs_list)} contigs.", 1)
        printv(f"Contigs matrix shape: {contigs.shape}.", 1)
        
        # Perform search
        printv("Searching for query...", 1, flush=True)
        t0 = time.time()
        min_locs, min_dists = search_func(
            contigs, query, 
        )
        t1 = time.time()
        
        printv(f"Finished searching in {t1 - t0:.4g} sec", 1, flush=True)
        printv(f"Locations of nearest match:\n{min_locs}", 2)
        printv(f"Distances to nearest match:\n{min_dists}", 2)

        # Find close matches
        nearest_match_dist = np.min(min_dists)
        nearest_match_idxs = np.where(min_dists == nearest_match_dist)[0]
        printv(f"Nearest match distance: {nearest_match_dist}", 1)
        
        # if len(nearest_match_idxs) > 1:
        #     msg = "Found {} contigs with a near match (d={})".format(
        #         len(nearest_match_idxs), nearest_match_dist)
        #     warnings.warn(msg)
        
        for idx in nearest_match_idxs:
            printv("Found near match (d={}) at contig index {}".format(
                nearest_match_dist, idx
            ), 1)

        contig_segments = []
        location_on_contigs = []
        for c, loc in zip(contigs[nearest_match_idxs], min_locs[nearest_match_idxs]):
            result = array_to_gene_seq(c[loc:loc + len(query)])
            left_pad_arr = np.array(["_"] * pad_left)
            right_pad_arr = np.array(["_"] * pad_right)
            left_pad_start = max(loc - pad_left, 0)
            left_pad_stop = loc
            right_pad_start = loc + len(query)
            right_pad_stop = min(loc + len(query) + pad_right, len(c))
            left_pad_len = left_pad_stop - left_pad_start
            print(pad_left, left_pad_start, left_pad_stop, left_pad_len, left_pad_arr)
            right_pad_len = right_pad_stop - right_pad_start
            if left_pad_len > 0:
                left_pad_arr[-left_pad_len:] = array_to_gene_seq(
                    c[left_pad_start:left_pad_stop], dtype=str
                )
            if right_pad_len > 0:
                right_pad_arr[:right_pad_len] = array_to_gene_seq(
                    c[right_pad_start:right_pad_stop], dtype=str
                )
            left_pad_string = "".join(left_pad_arr)
            right_pad_string = "".join(right_pad_arr)
            contig_segments.append(
                left_pad_string.lower() \
                    + b"".join(result).decode() \
                    + right_pad_string .lower()
            )
            location_on_contigs.append(loc.item())
            
        time1 = time.time()
        printv(f"Time elapsed: {time1 - time0:.4f} sec", 1)

        write_results(
            outfpath,
            genome_fpath=genome_fpath,
            query_string=query_string,
            min_dist=nearest_match_dist,
            nearest_idxs=nearest_match_idxs.tolist(),
            location_on_contigs=location_on_contigs,
            contig_segments=contig_segments,
            time_elapsed=(time1 - time0),
        )


#######################
##  Main Entrypoint  ##
#######################

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
