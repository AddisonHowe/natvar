"""Batched genome multi-query script.

Inputs:
    --queries_fpath : text file with each row a query of constant length.
    --input_fpath : text file of paths to genomes in fasta format (contigs).
    --outdir : directory to store output.
    --outfname : output filename.
    --batch_size : number of windows to process at once. Default 1000.
    --pad_left : number of bases to include before query match.
    --pad_right : number of bases to include after query match.
    --nrows0 : number of rows with which to initialize the storage matrix. 
               Can be used to reduce number of recompilations. Default 0.
    --append : whether to append to the output file without overwrite.
    --verbosity : verbosity level.

Outputs:
    Writes results to an output file.

Verbosity levels:
    0: No output besides warnings.
    1: Minimal output with no loop prints.
    2: Minimal output with minimal loop prints.  <-- Default
    3: Additional loop prints.
    4: TBD
    5: Debugging statements.

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

V1, V2, V3, V4 = 1, 2, 3, 4
VDEBUG = 5  # verbosity level for debug prints


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--queries_fpath', type=str, required=True)
    parser.add_argument('-i', '--input_fpath', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, default="out")
    parser.add_argument('-f', '--outfname', type=str, default="results.tsv")
    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-pl', '--pad_left', type=int, default=0)
    parser.add_argument('-pr', '--pad_right', type=int, default=0)
    parser.add_argument('--nrows0', type=int, default=0)
    parser.add_argument('--append', action="store_true")
    parser.add_argument('--reset_rows_every', type=int, default=0)
    parser.add_argument('-v', '--verbosity', type=int, default=2)
    parser.add_argument('--jax_debug_max_traces', type=int, default=0)
    # parser.add_argument('--jax_debug_num_compiles', type=int, default=0)

    return parser.parse_args(args)


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
    nrows0 = args.nrows0
    append = args.append
    reset_rows_every = args.reset_rows_every
    verbosity = args.verbosity
    max_traces = args.jax_debug_max_traces
    
    def printv(s, importance=1, **kwargs):
        if verbosity >= importance:
            print(s, **kwargs)

    os.makedirs(outdir, exist_ok=True)

    printv(f"Searching genome files listed in: {input_fpath}", V1)
    printv(f"Queries file: {queries_fpath}", V1)
    
    # Load input data: queries and genomes
    queries, query_strings = read_queries_file(queries_fpath)
    genome_filepaths = read_genome_filepaths(input_fpath)
    
    printv(f"Found {len(genome_filepaths)} genome files.", V1)
    printv(f"Searching for {len(queries)} queries.", V1)
    printv(f"Using batch size: {batch_size}", V1)

    # Initialize the output file with a header
    outfpath = f"{outdir}/{outfname}"
    if append:
        printv(f"Appending to output file: {outfpath}", V1)
    else:
        printv(f"Creating output file: {outfpath}", V1)
        _write_header(outfpath)
    
    # JIT the multisearch function and apply a max_trace if specified
    if max_traces:
        jit_search = eqx.filter_jit(
            eqx.debug.assert_max_traces(max_traces=max_traces)(
                static_multisearch_matrix_batched
            )
        )
    else:
        jit_search = eqx.filter_jit(static_multisearch_matrix_batched)
        
    # Initialization prior to main loop
    ngenomes = len(genome_filepaths)
    ncontigs = nrows0
    contig_length = 0
    matrix = jnp.zeros([ncontigs, contig_length], dtype=queries.dtype)
    query_length=queries.shape[1]
    repad_count1 = 0
    repad_count2 = 0
    repad_count3 = 0
    row_multiplier = 1
    printv(f"Processing genomes...", V1)
    time00 = time.time()
    for genome_idx, genome_fpath in enumerate(genome_filepaths):
        if reset_rows_every and (genome_idx % reset_rows_every == 0):
            ncontigs = nrows0
            matrix = jnp.zeros([ncontigs, contig_length], dtype=queries.dtype)
            row_multiplier = 1
        time0 = time.time()
        
        # Load contigs into a matrix from the contig file.
        printv(f"  Processing genome file {genome_idx+1}/{ngenomes}", V2)
        contigs_list = process_contig_file(genome_fpath)
        contigs = get_contigs_matrix(contigs_list, pad_val=PAD_VAL)
        printv(f"\tLoaded {len(contigs_list)} contigs.", V3)
        printv(f"\tContigs matrix shape: {contigs.shape}.", V4)
        
        # Add padding if contigs are shorter than the query.
        # Should be unlikely, so issue a warning if so?
        contigs, contig_length, repad_counts = _pad_contigs(
            contigs, query_length, contig_length, batch_size, 
            [repad_count1, repad_count2, repad_count3], 
            printv,
        )
        repad_count1, repad_count2, repad_count3 = repad_counts
        printv(f"\tContigs matrix shape post padding: {contigs.shape}.", V4)

        # Now, with new shape, adjust the matrix used for storage
        matrix, nrows, ncols, row_multiplier = _adjust_storage_matrix(
            contigs, matrix, row_multiplier, PAD_VAL
        )
        printv(f"\tStorage matrix shape: {matrix.shape}.", V3)
        
        # Perform search
        printv("\tSearching for queries...", V2, flush=True)
        printv(f"\tcontig_length: {contig_length}", VDEBUG)
        printv(f"\tquery_length: {query_length}", VDEBUG)
        printv(f"\tbatch_size: {batch_size}", VDEBUG)
        t0 = time.time()
        min_locs_all, min_dists_all = jit_search(
            matrix, queries, 
            array_length=contig_length,
            query_length=query_length,
            batch_size=batch_size, 
        )
        t1 = time.time()
        search_time = t1 - t0
        printv(f"\tFinished searching in {search_time:.4g} sec", 2, flush=True)

        # Identify for each genome the location of minimums across the contigs
        min_locs_all = min_locs_all[:,0:nrows]
        min_dists_all = min_dists_all[:,0:nrows]

        printv("\tWriting search results...", V4)
        t0 = time.time()
        for i in range(len(queries)):
            query = queries[i]
            query_string = query_strings[i]
            min_locs = min_locs_all[i]
            min_dists = min_dists_all[i]

            # Find closest matches
            res = _find_closest_matches(
                query, contigs,
                min_locs, min_dists, 
                pad_left, pad_right, 
                printv
            )
            nearest_dist, nearest_idxs, loc_on_contigs, contig_segments = res
            
            # Write results to output file
            _write_results(
                outfpath,
                genome_fpath=genome_fpath,
                query_string=query_string,
                min_dist=nearest_dist,
                nearest_idxs=nearest_idxs.tolist(),
                location_on_contigs=loc_on_contigs,
                contig_segments=contig_segments,
                time_elapsed=search_time,
            )

        time1 = time.time()
        printv(f"\tFinished writing in {time1 - t0:.4g} sec", V4, flush=True)
        printv(f"  Time elapsed: {time1 - time0:.4g} sec", V2)
    
    printv("Processing complete!", V1)
    printv(f"Number of repaddings (matrix expansion): {repad_count2}", V1)
    printv(f"Number of repaddings (contig length < query): {repad_count1}", V1)
    printv(f"Total time elapsed: {time.time()-time00:.4g} sec", V1)


########################
##  Helper functions  ##
########################

def _write_header(outfpath):
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


def _write_results(
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


def _pad_contigs(
        contigs, query_length, contig_length, 
        batch_size, repad_counts,
        printv
):
    # Add padding if contigs are shorter than the query.
    # Should be unlikely, so issue a warning if so?
    repad_count1, repad_count2, repad_count3 = repad_counts
    if contigs.shape[1] < query_length:
        npad = query_length - contigs.shape[1]
        contigs = pad_matrix(contigs, npad, PAD_VAL, V1)
        contigs = pad_matrix_for_batch_size(
            contigs, query_length, batch_size, PAD_VAL, V1
        )
        contig_length = contigs.shape[1]
        warnings.warn("\tRepadding: Query is longer than matrix sequences!")
        repad_count1 += 1
    elif contigs.shape[1] > contig_length:
        contigs = pad_matrix_for_batch_size(
            contigs, query_length, batch_size, PAD_VAL, V1
        )
        contig_length = contigs.shape[1]
        printv("\tRepadding to accommodate increased matrix!", V3)
        repad_count2 += 1
    elif contigs.shape[1] < contig_length:
        npad = contig_length - contigs.shape[1]
        contigs = pad_matrix(contigs, npad, PAD_VAL, V1)
        contig_length = contigs.shape[1]
        repad_count3 += 1
    repad_counts = (repad_count1, repad_count2, repad_count3)
    return contigs, contig_length, repad_counts


def _adjust_storage_matrix(contigs, matrix, row_multiplier, pad_val):
    nrows, ncols = contigs.shape
    newshape = list(matrix.shape)
    if nrows > matrix.shape[0]:
        newshape[0] = int(row_multiplier * nrows)
        row_multiplier = 1.5 
    if ncols > matrix.shape[1]:
        newshape[1] = ncols
    if nrows > matrix.shape[0] or ncols > matrix.shape[1]:
        matrix = jnp.zeros(newshape, dtype=contigs.dtype)
    matrix = matrix.at[:].set(pad_val)
    matrix = matrix.at[0:nrows,0:ncols].set(contigs)
    return matrix, nrows, ncols, row_multiplier


def _find_closest_matches(
        query, contigs,
        min_locs, min_dists, 
        pad_left, pad_right, 
        printv,
):
    nearest_match_dist = np.min(min_dists)
    nearest_match_idxs = np.where(min_dists == nearest_match_dist)[0]
    printv(f"\t  Nearest match distance: {nearest_match_dist}", V4)

    for idx in nearest_match_idxs:
        printv("\t  Found near match (d={}) at contig index {}".format(
                nearest_match_dist, idx), V4)

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
    
    return (
        nearest_match_dist, 
        nearest_match_idxs, 
        location_on_contigs, 
        contig_segments, 
    )


#######################
##  Main Entrypoint  ##
#######################

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
