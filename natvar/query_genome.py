"""Genome query script.

Inputs:
    --query : query string.
    --genome_fpath : genome file of contigs in fasta.gz format.
    --outdir : directory to store output.
    --outfname : output filename.
    --pad_left : number of bases to include before query match.
    --pad_right : number of bases to include after query match.
    --verbosity : verbosity level.
    --jax : use jax acceleration.
    --batch_size : number of windows to process at once, if using JAX.

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
# from .jax.core import search_matrix_for_query as jax_search_matrix
from .jax.core import static_search_matrix_batched as jax_search_matrix

GEN_NT_VAL = 4
PAD_VAL = 5

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', type=str, required=True)
    parser.add_argument('-i', '--genome_fpath', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, default="out")
    parser.add_argument('-f', '--outfname', type=str, default="query_results.tsv")
    parser.add_argument('-pl', '--pad_left', type=int, default=0)
    parser.add_argument('-pr', '--pad_right', type=int, default=0)
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--jax', action="store_true")
    parser.add_argument('--batch_size', type=int, default=800)
    return parser.parse_args(args)


def write_results(
        out_fpath, *, 
        genome_fpath, 
        query_string, 
        min_dist,
        nearest_idxs,
        contig_segments,
        location_on_contigs,
        time_elapsed,
):
    with open(out_fpath, 'w') as f:
        f.write(f"genome_fpath\t{genome_fpath}\n")
        f.write(f"query_string\t{query_string}\n")
        f.write(f"min_distance\t{min_dist}\n")
        f.write(f"nearest_idxs\t{nearest_idxs}\n")
        f.write(f"location_on_contigs\t{location_on_contigs}\n")
        f.write(f"contig_segments\t{contig_segments}\n")
        f.write(f"time_elapsed\t{time_elapsed}\n")


def main(args):
    genome_fpath = args.genome_fpath
    query_string = args.query
    outdir = args.outdir
    outfname = args.outfname
    pad_left = args.pad_left
    pad_right = args.pad_right
    verbosity = args.verbosity
    use_jax = args.jax
    batch_size = args.batch_size

    def printv(s, importance=1, **kwargs):
        if verbosity >= importance:
            print(s, **kwargs)

    os.makedirs(outdir, exist_ok=True)

    printv(f"Searching genome file: {genome_fpath}")
    printv(f"Query (length {len(query_string)}): {query_string}")

    time0 = time.time()
    
    # Load contigs into a matrix from the contig file.
    printv("Loading genome file...")
    contigs_list = process_contig_file(genome_fpath)
    contig_lengths = [len(c) for c in contigs_list]
    contigs = get_contigs_matrix(contigs_list, pad_val=PAD_VAL)
    contigs = pad_matrix_for_batch_size(
        contigs, len(query_string), batch_size, pad_val=PAD_VAL, axis=1,
    )
    printv(f"Loaded {len(contigs_list)} contigs.")
    printv(f"Max length contig: {np.max(contig_lengths)}", 2)
    printv(f"Shape of loaded contigs matrix: {contigs.shape}", 2)

    # Define the search function, with or without JAX acceleration
    if use_jax:
        def search_func(contigs, query):
            return jax_search_matrix(
                contigs, query,
                array_length=contigs.shape[1],
                query_length=len(query),
                batch_size=min(batch_size, contigs.shape[1]),
            )
    else:
        def search_func(contigs, query):
            return search_matrix_for_query(contigs, query)
    
    query = gene_seq_to_array(query_string)
    printv("Searching for query...")
    min_locs, min_dists = search_func(contigs, query)
    nearest_match_dist = np.min(min_dists)
    nearest_match_idxs = np.where(min_dists == nearest_match_dist)[0]

    printv("Locations of nearest match:", 2)
    printv(min_locs, 2)
    printv("Distances to nearest match:", 2)
    printv(min_dists, 2)
    printv(f"Nearest match distance: {nearest_match_dist}")
    if len(nearest_match_idxs) > 1:
        msg = "Found {} contigs with a near match (d={})".format(
            len(nearest_match_idxs), nearest_match_dist)
        warnings.warn(msg)
    
    for idx in nearest_match_idxs:
        printv("Found near match (d={}) at contig index {}".format(
            nearest_match_dist, idx
        ))
    
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
    time_elapsed = time1 - time0
    
    write_results(
        f"{outdir}/{outfname}",
        genome_fpath=genome_fpath,
        query_string=query_string,
        min_dist=nearest_match_dist,
        nearest_idxs=nearest_match_idxs,
        contig_segments=contig_segments,
        location_on_contigs=location_on_contigs,
        time_elapsed=time_elapsed,
    )


#######################
##  Main Entrypoint  ##
#######################

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
