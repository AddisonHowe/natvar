"""Benchmarking for multiquery search function

pytest -s --benchmark tests/benchmarking/benchmark_multisearch.py

"""

import pytest
import os
import numpy as np
import timeit
import jax
import jax.numpy as jnp
import equinox as eqx

from ..conftest import DATDIR, TMPDIR

from natvar.io import get_contigs_matrix, process_contig_file
from natvar.jax.core import static_multisearch_matrix_batched
from natvar.multiquery_genome_batch import read_genome_filepaths, read_queries_file
from natvar.multiquery_genome_batch import _pad_contigs, _adjust_storage_matrix


#####################
##  Configuration  ##
#####################

DATDIR = f"{DATDIR}/benchmarking"
OUTDIR = f"{TMPDIR}/bm_multisearch"

os.makedirs(OUTDIR, exist_ok=True)

def write_results(
        func_name, run_name, niters, genome_fpaths, 
        all_warmup_times, 
        all_avg_times, all_std_times,
        all_matrix_shapes, 
        batch_size,
):
    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{run_name}_bs_{batch_size}.txt"
    
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")        
        f.write(f"num iterations: {niters}\n")
        f.write(f"batch_size: {batch_size}\n")
        for i in range(len(all_warmup_times)): 
            f.write(f"genome: {genome_fpaths[i]}\n")
            f.write(f"shape: {all_matrix_shapes[i]}\n")
            f.write(f"warmup time: {all_warmup_times[i]}\n")
            f.write(f"avg time: {all_avg_times[i]}\n")
            f.write(f"std time: {all_std_times[i]}\n")

##############################################################################
###########################   BEGIN BENCHMARKING   ###########################
##############################################################################

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "genomes_fpath, queries_fpath, name", [
    [f"{DATDIR}/genome_filelist1.txt", f"{DATDIR}/querylist1.txt", 'run1'],
])
@pytest.mark.parametrize("batch_size", [100, 200, 400, 800, 3200])
def test_static_multisearch_matrix_batched(
    genomes_fpath, queries_fpath, name, batch_size
):

    NITERS = 20
    PAD_VAL = 4
    NROWS0 = 1

    func_name = "static_multisearch_matrix_batched"
    func = eqx.filter_jit(static_multisearch_matrix_batched)
        
    genome_filepaths = read_genome_filepaths(genomes_fpath)
    queries, _ = read_queries_file(queries_fpath)

    # Initialization warmup
    # contigs_list = process_contig_file(genome_filepaths[0])
    # contigs = get_contigs_matrix(contigs_list, pad_val=PAD_VAL)
    # time0 = timeit.default_timer()
    # result = func(
    #     contigs, queries, 
    #     array_length=contigs.shape[1], 
    #     query_length=queries.shape[1], 
    #     batch_size=batch_size, 
    # )
    # jax.block_until_ready(result)
    # time1 = timeit.default_timer()
    # initialization_time = time1 - time0
    
    # Time each genome input
    all_avg_times = []
    all_std_times = []
    all_warmup_times = []
    all_matrix_shapes = []

    ngenomes = len(genome_filepaths)
    ncontigs = NROWS0
    contig_length = 0
    matrix = jnp.zeros([ncontigs, contig_length], dtype=queries.dtype)
    query_length=queries.shape[1]
    repad_count1 = 0
    repad_count2 = 0
    repad_count3 = 0
    row_multiplier = 1
    for genome_fpath in genome_filepaths:
        contigs_list = process_contig_file(genome_fpath)
        contigs = get_contigs_matrix(contigs_list, pad_val=PAD_VAL)
        contigs, contig_length, repad_counts = _pad_contigs(
            contigs, query_length, contig_length, batch_size, 
            [repad_count1, repad_count2, repad_count3], 
            lambda s, imp, **kwargs: None,
        )
        repad_count1, repad_count2, repad_count3 = repad_counts
        
        matrix, nrows, ncols, row_multiplier = _adjust_storage_matrix(
            contigs, matrix, row_multiplier,
        )
        matrix[:] = PAD_VAL
        matrix[0:nrows,0:ncols] = contigs

        # Warmup
        time0 = timeit.default_timer()
        result = func(
            matrix, queries, 
            array_length=contig_length,
            query_length=query_length,
            batch_size=batch_size, 
        )
        jax.block_until_ready(result)
        time1 = timeit.default_timer()
        warmup_time = time1 - time0
        
        times = []
        for _ in range(NITERS):
            time0 = timeit.default_timer()
            result = func(
                matrix, queries, 
                array_length=contig_length,
                query_length=query_length,
                batch_size=batch_size, 
            )
            time1 = timeit.default_timer()
            times.append(time1 - time0)
        all_avg_times.append(np.mean(times))
        all_std_times.append(np.std(times))
        all_warmup_times.append(warmup_time)
        all_matrix_shapes.append(matrix.shape)

    write_results(
        func_name, name, NITERS, genome_filepaths, 
        all_warmup_times, 
        all_avg_times, all_std_times,
        all_matrix_shapes, 
        batch_size,
    )
