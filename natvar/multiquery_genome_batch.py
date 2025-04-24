"""Batched genome multi-query script.

Inputs:
    --queries_fpath : text file with each row a query of constant length.
    --input_fpath : text file of paths to genomes in fasta format (contigs).
    --outdir : directory to store output.
    --outfname : output filename.
    --batch_size : number of windows to process at once. Default 1000.
    --pad_left : number of bases to include before query match.
    --pad_right : number of bases to include after query match.
    --verbosity : verbosity level.

Outputs:
    Writes results to an output file.
"""

import argparse
import sys
import os
import time
import numpy as np
import warnings

import jax
import jax.numpy as jnp
jax.config.update("jax_numpy_dtype_promotion", "strict")
import equinox as eqx

from .helpers import array_to_gene_seq, gene_seq_to_array
from .helpers import pad_matrix, pad_matrix_for_batch_size
from .io import process_contig_file, get_contigs_matrix
from .jax.core import static_multisearch_matrix_batched

GEN_NT_VAL = 4
PAD_VAL = 5


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--queries_fpath', type=str, required=True)
    parser.add_argument('-i', '--input_fpath', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, default="out")
    parser.add_argument('-f', '--outfname', type=str, default="results.tsv")
    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-pl', '--pad_left', type=int, default=0)
    parser.add_argument('-pr', '--pad_right', type=int, default=0)
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--jax_debug_max_traces', type=int, default=0)

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


def read_queries_file(queries_fpath):
    seq_strs = np.genfromtxt(queries_fpath, dtype=str)
    if seq_strs.ndim == 0:
        seq_strs = np.array([seq_strs])
    seqs = np.array([gene_seq_to_array(s) for s in seq_strs], dtype=np.uint8)
    return seqs, seq_strs


def read_genome_filepaths(input_fpath):
    """Read into a list the paths contained in the input file."""
    with open(input_fpath, 'r') as f:
        files = [line.strip() for line in f.readlines()]
    return files


def main(args):
    input_fpath = args.input_fpath
    queries_fpath = args.queries_fpath
    outdir = args.outdir
    outfname = args.outfname
    batch_size = args.batch_size
    pad_left = args.pad_left
    pad_right = args.pad_right
    verbosity = args.verbosity
    max_traces = args.jax_debug_max_traces
    
    def printv(s, importance=1, **kwargs):
        if verbosity >= importance:
            print(s, **kwargs)

    os.makedirs(outdir, exist_ok=True)

    printv(f"Searching genome files listed in: {input_fpath}", 1)
    printv(f"Queries file: {queries_fpath}", 1)
    
    queries, query_strings = read_queries_file(queries_fpath)
    genome_filepaths = read_genome_filepaths(input_fpath)
    
    printv(f"Found {len(genome_filepaths)} genome files.", 1)
    print(f"Searching for {len(queries)} queries.")

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

    
    if max_traces:
        jit_search = eqx.filter_jit(
            eqx.debug.assert_max_traces(max_traces=max_traces)(
                static_multisearch_matrix_batched
            )
        )
    else:
        jit_search = eqx.filter_jit(static_multisearch_matrix_batched)
        

    ncontigs = 0
    contig_length = 0
    matrix = jnp.zeros([ncontigs, contig_length], dtype=queries.dtype)
    query_length=queries.shape[1]
    repad_count1 = 0
    repad_count2 = 0
    repad_count3 = 0
    for genome_fpath in genome_filepaths:
        time0 = time.time()
        
        # Load contigs into a matrix from the contig file.
        printv("\tLoading genome file...", 1)
        contigs_list = process_contig_file(genome_fpath)
        contigs = get_contigs_matrix(contigs_list, pad_val=PAD_VAL)
        printv(f"\tLoaded {len(contigs_list)} contigs.", 1)
        printv(f"\tContigs matrix shape: {contigs.shape}.", 1)
        
        # Add padding if contigs are shorter than the query.
        # Should be unlikely, so issue a warning if so?
        if contigs.shape[1] < query_length:
            npad = query_length - contigs.shape[1]
            contigs = pad_matrix(contigs, npad, PAD_VAL, 1)
            contigs = pad_matrix_for_batch_size(contigs, batch_size, PAD_VAL, 1)
            contig_length = contigs.shape[1]
            warnings.warn("\tRepadding: Query is longer than matrix sequences!")
            repad_count1 += 1
        elif contigs.shape[1] > contig_length:
            contigs = pad_matrix_for_batch_size(contigs, batch_size, PAD_VAL, 1)
            contig_length = contigs.shape[1]
            printv("\tRepadding to accommodate increased matrix!", 2)
            repad_count2 += 1
        elif contigs.shape[1] < contig_length:
            npad = contig_length - contigs.shape[1]
            contigs = pad_matrix(contigs, npad, PAD_VAL, 1)
            contig_length = contigs.shape[1]
            repad_count3 += 1
        printv(f"\tContigs matrix shape post padding: {contigs.shape}.", 1)

        # Now, with new shape, adjust the matrix used for storage
        nrows, ncols = contigs.shape
        if nrows > matrix.shape[0] or ncols > matrix.shape[1]:
            newshape = (max(nrows, matrix.shape[0]), max(ncols, matrix.shape[1]))
            matrix = np.zeros(newshape, dtype=contigs.dtype)
        matrix[:] = PAD_VAL
        matrix[0:nrows,0:ncols] = contigs
        printv(f"\tStorage matrix shape: {matrix.shape}.", 2)
        
        # Perform search
        printv("\tSearching for query...", 1, flush=True)
        printv(f"\tcontig_length: {contig_length}", 3)
        printv(f"\tquery_length: {query_length}", 3)
        printv(f"\tbatch_size: {batch_size}", 3)
        t0 = time.time()
        min_locs_all, min_dists_all = jit_search(
            matrix, queries, 
            array_length=contig_length,
            query_length=query_length,
            batch_size=batch_size, 
        )
        t1 = time.time()
        search_time = t1 - t0
        min_locs_all = min_locs_all[:,0:nrows]
        min_dists_all = min_dists_all[:,0:nrows]

        printv(f"\tFinished searching in {search_time:.4g} sec", 1, flush=True)

        printv("\tWriting search results...", 3)
        t0 = time.time()
        for i in range(len(queries)):
            query = queries[i]
            query_string = query_strings[i]
            min_locs = min_locs_all[i]
            min_dists = min_dists_all[i]

            # Find close matches
            nearest_match_dist = np.min(min_dists)
            nearest_match_idxs = np.where(min_dists == nearest_match_dist)[0]
            printv(f"\t  Nearest match distance: {nearest_match_dist}", 3)
            
            # if len(nearest_match_idxs) > 1:
            #     msg = "Found {} contigs with a near match (d={})".format(
            #         len(nearest_match_idxs), nearest_match_dist)
            #     warnings.warn(msg)
            
            for idx in nearest_match_idxs:
                printv("\t  Found near match (d={}) at contig index {}".format(
                    nearest_match_dist, idx
                ), 3)

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
            
            write_results(
                outfpath,
                genome_fpath=genome_fpath,
                query_string=query_string,
                min_dist=nearest_match_dist,
                nearest_idxs=nearest_match_idxs.tolist(),
                location_on_contigs=location_on_contigs,
                contig_segments=contig_segments,
                time_elapsed=search_time,
            )
        t1 = time.time()
        printv(f"\tFinished writing in {t1 - t0:.4g} sec", 3, flush=True)
        time1 = time.time()
        printv(f"Time elapsed: {time1 - time0:.4f} sec", 1)



#######################
##  Main Entrypoint  ##
#######################

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
