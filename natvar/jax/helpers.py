"""General helper functions with JAX acceleration

"""

import jax.numpy as jnp
import numpy as np


def binary_arr_to_int(bin_arr):
    """Convert one or more binary arrays to integer value(s).
    """
    k = bin_arr.shape[-1]
    weights = 1 << jnp.arange(k)[::-1]
    return bin_arr @ weights.astype(float)


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
