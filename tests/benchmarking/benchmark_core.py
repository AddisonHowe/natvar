"""Benchmarking core functions

pytest -s --benchmark tests/benchmarking/benchmark_core.py

"""

import pytest
import os
import numpy as np
import timeit
import jax

from ..conftest import DATDIR, TMPDIR

from natvar.io import get_contigs_matrix, process_contig_file, gene_seq_to_array
from natvar.core import search_matrix_for_query


#####################
##  Configuration  ##
#####################

DATDIR = f"{DATDIR}/benchmarking"
OUTDIR = f"{TMPDIR}/benchmarking"

PAD_VAL = 5

os.makedirs(OUTDIR, exist_ok=True)

def load_genome(fpath):
    contigs_list = process_contig_file(fpath)
    contigs = get_contigs_matrix(contigs_list, pad_val=PAD_VAL)
    return contigs

def write_results(func_name, run_name, niters, warmup_time, avg_time, std_time):
    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{run_name}.txt"
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")
        f.write(f"total warmup time: {warmup_time}\n")
        f.write(f"num iterations: {niters}\n")
        f.write(f"avg time: {avg_time}\n")
        f.write(f"std time: {std_time}\n")


##############################################################################
###########################   BEGIN BENCHMARKING   ###########################
##############################################################################


@pytest.mark.benchmark
@pytest.mark.parametrize("fname, name, query_length", [
    ['SAMD00000649.contigs.fa.gz', 'run1', 10],
])
def test_search_matrix_for_query(fname, name, query_length):

    NITERS = 20
    query_string = "A" * query_length
    query = gene_seq_to_array(query_string)

    func_name = "search_matrix_for_query"
    func = search_matrix_for_query
    contigs = load_genome(f"{DATDIR}/{fname}")

    # Warmup
    time0 = timeit.default_timer()
    result = func(
        contigs, query,
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
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    write_results(
        func_name, name, NITERS, total_warmup_time, avg_time, std_time
    )
