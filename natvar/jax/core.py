"""Core functions with JAX acceleration

"""

import jax
import jax.numpy as jnp
import equinox as eqx

from ..core import count_mutations


def search_matrix_for_query(
        matrix, query, 
        batch_size, 
        pad_val,
        print_every=0,
):
    nseqs, nbases = matrix.shape
    k = len(query)
    nscans = nbases - k + 1
    
    nbatches = int(nscans // batch_size) + (nscans % batch_size != 0)

    def count_mutations_helper(i, matrix, query, k):
        return count_mutations(
            jax.lax.dynamic_slice_in_dim(matrix, i, k, 1), query
        )
    
    @eqx.filter_jit
    def batched_mutation_counter(idxs, matrix, query, k):
        return jax.vmap(count_mutations_helper, (0, None, None, None))(
            idxs, matrix, query, k
        )
    
    if nscans % batch_size != 0:
        matrix = jnp.concat([
            matrix, 
            pad_val * jnp.ones(
                [matrix.shape[0], batch_size - (nscans % batch_size)], 
                dtype=matrix.dtype
            )
        ], axis=1)
    
    min_distances = (k+1) * jnp.ones(matrix.shape[0], dtype=int)
    min_locations = -1 * jnp.ones(matrix.shape[0], dtype=int)

    for batch_idx in range(nbatches):
        if print_every and (batch_idx % print_every) == 0:
            print(f"  Processing batch {1+batch_idx}/{nbatches}", flush=True)
        idxs = jnp.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        distances_arr = batched_mutation_counter(idxs, matrix, query, k)
        distances = jnp.min(distances_arr, axis=0)
        dist_idxs = jnp.argmin(distances_arr, axis=0)
        improved_screen = distances < min_distances
        min_locations = jnp.where(improved_screen, idxs[dist_idxs], min_locations)
        min_distances = jnp.minimum(min_distances, distances)
    
    return min_locations, min_distances


def static_search_array(
        array, query, array_length, query_length,
        scan_range=None,
):
    """Perform a k-query on a length n array, searching over a specified range.

    Args:
        array (np.ndarray[np.uint8]): Array sequence to search. Shape (n,).
        query (np.ndarray[np.uint8]): Query sequences. Shape (k,).
        array_length (int): Length of array to search.
        query_length (int): Length of the query.
        scan_range (np.ndarray[int]): Indices at which to position the query 
            sequence along the array. Shape (r,). If None, indices are set to 
            range(r) with r=n-k+1, i.e. the entire array is searched.

    Returns:
        min_idx (int): Index where the first nearest match occurs.
        min_val (int): Minimal distance found over the specified range.
    
    """
    n = array_length
    k = query_length
    assert len(query) == k, f"Bad query length. Expected {len(query)}. Got {k}."
    assert len(array) == n, f"Bad array length. Expected {len(array)}. Got {n}."
    if scan_range is None:
        scan_range = jnp.arange(n - k + 1)
    
    #TODO: store other stuff in the carry, just don't change it.
    def min_search(carry, i):
        min_idx, min_val = carry
        arr = jax.lax.dynamic_slice(array, (i,), (k,))
        nmuts = count_mutations(arr, query)
        improved = (nmuts < min_val).astype(int)
        new_min_idx = i * improved + min_idx * (1 - improved)
        new_min_val = nmuts * improved + min_val * (1 - improved)
        return (new_min_idx, new_min_val), None
    
    init_state = (0, 1000000)
    final_state, _ = jax.lax.scan(min_search, init_state, scan_range)
    final_min_idx, final_min_val = final_state
    return final_min_idx, final_min_val


def static_search_matrix(
        matrix, query, array_length, query_length,
        scan_range=None,
):
    """Perform a k-query on a matrix of sequences, over a specified range.

    Args:
        matrix (np.ndarray[np.uint8]): Array sequences to search. Shape (m,n).
        query (np.ndarray[np.uint8]): Query sequences. Shape (k,).
        array_length (int): Length of array to search.
        query_length (int): Length of the query.
        scan_range (np.ndarray[int]): Indices at which to position the query 
            sequence along the matrix, operating along column subsets. 
            If None, indices are set to range(n-k+1).

    Returns:
        idxs (jnp.ndarray[int]): Indices where a near match occurs. Shape (m,).
        vals (jnp.ndarray[int]): Minimal distances found. Shape (m,).
    
    """
    final_min_idxs, final_min_vals = jax.vmap(
        static_search_array, (0, None, None, None, None)
    )(matrix, query, array_length, query_length, scan_range)
    return final_min_idxs, final_min_vals


def static_search_matrix_batched(
        matrix, query, array_length, query_length, batch_size,
):
    """Perform a k-query on a matrix of sequences, searching in batches.

    Each batch consists of comparing the query to a set of matrix slices, where
    each slice is a column subset of the matrix. The batch size must be a 
    divisor of `array_length`, i.e. the number of columns in the matrix.

    Args:
        matrix (np.ndarray[np.uint8]): Array sequences to search. Shape (m,n).
        query (np.ndarray[np.uint8]): Query sequences. Shape (k,).
        array_length (int): Length of array to search.
        query_length (int): Length of the query.
        batch_size (int): Number of slices to include in each batch.
        scan_range (np.ndarray[int]): Indices at which to position the query 
            sequence along the matrix, operating along column subsets. 
            If None, indices are set to range(n-k+1).

    Returns:
        idxs (jnp.ndarray[int]): Indices where a near match occurs. Shape (m,).
        vals (jnp.ndarray[int]): Minimal distances found. Shape (m,).
    
    """
    n = array_length
    k = query_length
    nqueries = n - k + 1
    if nqueries % batch_size != 0:
        msg = "Batch size B must divide number of queries Nq=n-k+1."
        msg += f" Got Nq={nqueries}, B={batch_size}."
        raise RuntimeError(msg)
    nbatches = int(nqueries // batch_size)
    
    q_range = jnp.arange(nqueries)
    def search_helper(i):
        return static_search_matrix(
            matrix, query, array_length, query_length, 
            scan_range=jax.lax.dynamic_slice(
                q_range, (i*batch_size,), (batch_size,)
            )
        )
    
    min_idxs, min_vals = jax.lax.map(
        search_helper, jnp.arange(nbatches), batch_size=1
    )    
    idxs = jnp.argmin(min_vals, axis=0)
    final_min_idxs = min_idxs[idxs,jnp.arange(len(idxs))]
    final_min_vals = min_vals[idxs,jnp.arange(len(idxs))]
    return final_min_idxs, final_min_vals


def static_multisearch_matrix_batched(
        matrix, queries, array_length, query_length, batch_size,
):
    search_fn = jax.vmap(static_search_matrix_batched, (None,0,None,None,None))
    return search_fn(matrix, queries, array_length, query_length, batch_size)
