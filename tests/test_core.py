"""Tests for core functions.

"""

import pytest
import numpy as np

from natvar.helpers import binary_arr_to_int, int_to_binary_arr, pad_matrix_for_batch_size
from natvar.core import compare_sequences, count_mutations
from natvar.core import search_matrix_for_query

###################
##  Begin Tests  ##
###################

@pytest.mark.parametrize("bin_arr, expected", [
    [[0, 1, 0], 2],
    [[0, 1, 1], 3],
    [[1, 0, 1], 5],
    [[[0, 1, 0],[0, 1, 1],[1, 0, 1]], [2,3,5]],
    [[0], 0],
    [[1], 1],
])
def test_binary_arr_to_int(bin_arr, expected):
    val = binary_arr_to_int(np.array(bin_arr))
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.parametrize("int_array, n, expected", [
    [0, None, [0]],
    [1, None, [1]],
    [2, None, [1, 0]],
    [0, 1, [0]],
    [1, 1, [1]],
    [2, 2, [1, 0]],
    [0, 3, [0, 0, 0]],
    [1, 3, [0, 0, 1]],
    [2, 3, [0, 1, 0]],
])
def test_int_to_binary_arr(int_array, n, expected):
    val = int_to_binary_arr(int_array, n)
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.parametrize("seq1, seq2, expected", [
    [[0,3,0,0], [0,3,0,1], [0,0,0,1]],
    [[0,0,2,0], [1,0,2,0], [1,0,0,0]],
    [[2,2,2,1], [0,0,0,1], [1,1,1,0]],
    [[[1,1,2,1],[0,0,2,1]], [0,0,0,1], [[1,1,1,0],[0,0,1,0]]],
    [[0,0,0,1], [[1,1,2,1],[0,0,2,1]], [[1,1,1,0],[0,0,1,0]]],
])
def test_compare_sequences(seq1, seq2, expected):
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    val = compare_sequences(seq1, seq2)
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.parametrize("seqs, wt_seq, expected", [
    [[0,3,0,0], [0,3,0,1], 1],
    [[0,0,2,0], [1,0,2,0], 1],
    [[2,2,2,1], [0,0,0,1], 3],
    [[[1,1,2,1],[0,0,2,1]], [0,0,0,1], [3,1]],
])
def test_count_mutations(seqs, wt_seq, expected):
    seqs = np.array(seqs)
    wt_seq = np.array(wt_seq)
    val = count_mutations(seqs, wt_seq)
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.parametrize(
    "matrix, query, min_locs_exp, min_dists_exp", [
    [[[0,1,2,3,3,3,3,3],[3,0,1,2,3,3,3,3],[3,3,3,3,0,1,2,2]], 
     [0,1,2,3], 
     [0, 1, 4],
     [0, 0, 1],
    ],
    [[[0,1,2,3,0,1,2,3],[3,0,1,2,3,3,3,3],[3,3,3,3,0,1,2,2]], 
     [0,1,2,3],
     [0, 1, 4],  # Keeps first occurrence
     [0, 0, 1],
    ],
])
def test_search_matrix_for_query(matrix, query, min_locs_exp, min_dists_exp):
    matrix = np.array(matrix, dtype=np.uint8)
    query = np.array(query, dtype=np.uint8)
    min_locs, min_dists = search_matrix_for_query(matrix, query)
    errors = []
    if not np.allclose(min_locs, min_locs_exp):
        msg = f"Wrong locations."
        msg += f"Expected:\n{min_locs_exp}\nGot:\n{min_locs}"
        errors.append(msg)
    if not np.allclose(min_dists, min_dists_exp):
        msg = f"Wrong distances."
        msg += f"Expected:\n{min_dists_exp}\nGot:\n{min_dists}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    

@pytest.mark.parametrize("m, axis, batch_size, exp_shape", [
    [np.ones([3, 5]), 0, 2, (4, 5)],
    [np.ones([3, 5]), 1, 2, (3, 6)],
])
@pytest.mark.parametrize("pad_val", [4])
def test_pad_matrix_for_batch_size(m, axis, batch_size, exp_shape, pad_val):
    m = pad_matrix_for_batch_size(
        m, batch_size, pad_val, axis,
    )
    assert m.shape == exp_shape, f"Bad shape. Got {m.shape}. Expected {exp_shape}."
    