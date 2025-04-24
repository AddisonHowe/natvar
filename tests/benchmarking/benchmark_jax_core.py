"""Benchmarking core functions with JAX acceleration

pytest -s --benchmark tests/benchmarking/benchmark_jax_core.py

"""

import pytest
import os
import numpy as np
import timeit
import jax
import equinox as eqx

from ..conftest import DATDIR, TMPDIR

from natvar.helpers import pad_matrix_for_batch_size
from natvar.io import get_contigs_matrix, process_contig_file, gene_seq_to_array
from natvar.jax.core import search_matrix_for_query, static_search_matrix
from natvar.jax.core import static_search_matrix_batched


#####################
##  Configuration  ##
#####################

DATDIR = f"{DATDIR}/benchmarking"
OUTDIR = f"{TMPDIR}/jax_benchmarking"

os.makedirs(OUTDIR, exist_ok=True)

def load_genome(fpath):
    contigs_list = process_contig_file(fpath)
    contigs = get_contigs_matrix(contigs_list, pad_val=4)
    return contigs

def write_results(
        func_name, run_name, niters, warmup_time, avg_time, std_time,
        batch_size,
):
    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{run_name}_bs_{batch_size}.txt"
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")
        f.write(f"total warmup time: {warmup_time}\n")
        f.write(f"num iterations: {niters}\n")
        f.write(f"avg time: {avg_time}\n")
        f.write(f"std time: {std_time}\n")
        f.write(f"batch_size: {batch_size}\n")


##############################################################################
###########################   BEGIN BENCHMARKING   ###########################
##############################################################################

@pytest.mark.skip
@pytest.mark.benchmark
@pytest.mark.parametrize("fname, name, query_length", [
    ['SAMD00000649.contigs.fa.gz', 'run1', 10],
])
@pytest.mark.parametrize("batch_size", [100, 200, 400, 800])
def test_search_matrix_for_query(fname, name, query_length, batch_size):

    NITERS = 20
    query_string = "A" * query_length
    query = gene_seq_to_array(query_string)

    func_name = "search_matrix_for_query"
    func = eqx.filter_jit(search_matrix_for_query)
    contigs = load_genome(f"{DATDIR}/{fname}")

    # Warmup
    time0 = timeit.default_timer()
    result = func(
        contigs, query, batch_size=batch_size, pad_val=4,
        print_every=0,
    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            contigs, query, batch_size=batch_size, pad_val=4,
            print_every=0,
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    write_results(
        func_name, name, NITERS, total_warmup_time, avg_time, std_time,
        batch_size, 
    )

@pytest.mark.skip
@pytest.mark.benchmark
@pytest.mark.parametrize("fname, name, query_length", [
    ['SAMD00000649.contigs.fa.gz', 'run1', 10],
])
@pytest.mark.parametrize("batch_size", [100, 200, 400, 800])
def test_static_search_matrix(fname, name, query_length, batch_size):

    NITERS = 20
    query_string = "A" * query_length
    query = gene_seq_to_array(query_string)

    func_name = "static_search_matrix"
    func = eqx.filter_jit(static_search_matrix)
    contigs = load_genome(f"{DATDIR}/{fname}")

    # Warmup
    time0 = timeit.default_timer()
    result = func(
        contigs, query, 
        array_length=contigs.shape[1], 
        query_length=len(query),

    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            contigs, query, 
            array_length=contigs.shape[1], 
            query_length=len(query)
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    write_results(
        func_name, name, NITERS, total_warmup_time, avg_time, std_time,
        batch_size, 
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("fname, name, query_length", [
    ['SAMD00000649.contigs.fa.gz', 'run1', 10],
])
@pytest.mark.parametrize("batch_size", [100, 200, 400, 800])
def test_static_search_matrix_batched(fname, name, query_length, batch_size):

    NITERS = 20
    query_string = "A" * query_length
    query = gene_seq_to_array(query_string)

    func_name = "static_search_matrix_batched"
    func = eqx.filter_jit(static_search_matrix_batched)
    contigs = load_genome(f"{DATDIR}/{fname}")
    pad_val = 4
    n = contigs.shape[1]
    contigs = pad_matrix_for_batch_size(contigs, batch_size, pad_val=pad_val, axis=1)
    # Warmup
    time0 = timeit.default_timer()
    result = func(
        contigs, query, 
        array_length=contigs.shape[1], 
        query_length=len(query),
        batch_size=batch_size,
    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            contigs, query, 
            array_length=contigs.shape[1], 
            query_length=len(query),
            batch_size=batch_size,
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    write_results(
        func_name, name, NITERS, total_warmup_time, avg_time, std_time,
        batch_size, 
    )
