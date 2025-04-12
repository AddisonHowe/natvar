"""Genome query script.

Inputs:
    --query : query string.
    --genome_fpath : genome file of contigs in fasta.gz format.
    --outdir : directory to store output.
    --outfname : output filename.
    --pad_left : number of bases to include before query match.
    --pad_right : number of bases to include after query match.
    --verbosity : verbosity level.

Outputs:
    Writes results to an output file
"""

import argparse
import sys
import os
import numpy as np
import warnings

from .helpers import array_to_gene_seq, gene_seq_to_array
from .io import process_contig_file, get_contigs_matrix
from .core import *


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', type=str, required=True)
    parser.add_argument('-i', '--genome_fpath', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, default="out")
    parser.add_argument('-f', '--outfname', type=str, default="query_results.tsv")
    parser.add_argument('-pl', '--pad_left', type=int, default=0)
    parser.add_argument('-pr', '--pad_right', type=int, default=0)
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    return parser.parse_args(args)


def printv(s, verbosity, importance=1, **kwargs):
    if verbosity >= importance:
        print(s, **kwargs)


def write_results(
        out_fpath, *, 
        genome_fpath, 
        query_string, 
        min_dist,
        nearest_idxs,
        contig_segments,
        location_on_contigs,
):
    with open(out_fpath, 'w') as f:
        f.write(f"genome_fpath\t{genome_fpath}\n")
        f.write(f"query\t{query_string}\n")
        f.write(f"min_distance\t{min_dist}\n")
        f.write(f"nearest_idxs\t{nearest_idxs}\n")
        f.write(f"location_on_contigs\t{location_on_contigs}\n")
        f.write(f"contig_segments\t{contig_segments}\n")


def main(args):
    genome_fpath = args.genome_fpath
    query_string = args.query
    outdir = args.outdir
    outfname = args.outfname
    pad_left = args.pad_left
    pad_right = args.pad_right
    verb = args.verbosity

    os.makedirs(outdir, exist_ok=True)

    printv(f"Searching genome file: {genome_fpath}", verb)
    printv(f"Query (length {len(query_string)}): {query_string}", verb)
    
    # Load contigs into a matrix from the contig file.
    printv("Loading genome file...", verb)
    contigs_list = process_contig_file(genome_fpath)
    contig_lengths = [len(c) for c in contigs_list]
    contigs = get_contigs_matrix(contigs_list, pad_val=4)
    printv(f"Loaded {len(contigs_list)} contigs.", verb, 1)
    printv(f"Max length contig: {np.max(contig_lengths)}", verb, 2)
    printv(f"Shape of loaded contigs matrix: {contigs.shape}", verb, 2)

    query = gene_seq_to_array(query_string)
    printv("Searching for query...", verb)
    min_locs, min_dists = search_matrix_for_query(contigs, query)
    printv("Locations of nearest match:", verb, 2)
    printv(min_locs, verb, 2)
    printv("Distances to nearest match:", verb, 2)
    printv(min_dists, verb, 2)

    nearest_match_dist = np.min(min_dists)
    nearest_match_idxs = np.where(min_dists == nearest_match_dist)[0]
    printv(f"Nearest match distance: {nearest_match_dist}", verb, 1)
    if len(nearest_match_idxs) > 1:
        msg = "Found {} contigs with a near match (d={})".format(
            len(nearest_match_idxs), nearest_match_dist)
        warnings.warn(msg)
    
    for idx in nearest_match_idxs:
        printv("Found near match (d={}) at contig index {}".format(
            nearest_match_dist, idx
        ), verb, 1)

    contig_segments = []
    location_on_contigs = []
    for c, loc in zip(contigs[nearest_match_idxs], min_locs[nearest_match_idxs]):
        start = max(loc - pad_left, 0)
        stop = min(loc + len(query) + pad_right, len(c))
        result_string = array_to_gene_seq(c[start:stop])
        contig_segments.append(b"".join(result_string).decode())
        location_on_contigs.append(loc)
        
    write_results(
        f"{outdir}/{outfname}",
        genome_fpath=genome_fpath,
        query_string=query_string,
        min_dist=nearest_match_dist,
        nearest_idxs=nearest_match_idxs,
        contig_segments=contig_segments,
        location_on_contigs=location_on_contigs,
    )


#######################
##  Main Entrypoint  ##
#######################

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
