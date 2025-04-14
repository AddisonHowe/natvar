"""Tests for core functions.

"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from natvar.jax.helpers import binary_arr_to_int, int_to_binary_arr
from natvar.jax.core import search_matrix_for_query

NA = np.nan

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
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("pad_val", [4, 5])
def test_search_matrix_for_query(
    matrix, query, min_locs_exp, min_dists_exp,
    batch_size, pad_val
):
    matrix = jnp.array(matrix, dtype=jnp.uint8)
    query = jnp.array(query, dtype=jnp.uint8)
    min_locs, min_dists = search_matrix_for_query(
        matrix, query, 
        batch_size=batch_size,
        pad_val=pad_val,
    )
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
    