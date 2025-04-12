"""Helper functions

"""

import numpy as np

NT_MAP = {c: i for i, c in enumerate(['A', 'C', 'G', 'T', 'N', '_'])}
NT_MAP_INV = {i: c for i, c in enumerate(['A', 'C', 'G', 'T', 'N', '_'])}


def gene_seq_to_array(seq, mapping=NT_MAP):
    """Convert a string sequence to a numy array, using the provided mapping."""
    if mapping is None:
        mapping = NT_MAP
    return np.array([mapping[c] for c in seq], dtype=np.uint8)


def array_to_gene_seq(arr, mapping=NT_MAP_INV, dtype='S1'):
    """Convert a string sequence to a numy array, using the provided mapping."""
    if mapping is None:
        mapping = NT_MAP_INV
    return np.array([mapping[i] for i in arr], dtype=dtype)


def binary_arr_to_int(bin_arr):
    """Convert one or more binary arrays to integer value(s).
    """
    k = bin_arr.shape[-1]
    weights = 1 << np.arange(k)[::-1]
    return bin_arr @ weights


def int_to_binary_arr(int_array, n=None):
    """Convert one or more integers to binary arrays of length n.
    """
    if not isinstance(int_array, np.ndarray):
        int_array = np.array(int_array)
    if n is None:
        n = np.max([int_array.max(), 1]).item().bit_length()
    return (
        (int_array[...,None] >> np.arange(n - 1, -1, -1)) & 1
    ).astype(np.uint8)
