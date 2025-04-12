"""Core functions with JAX acceleration

"""

import jax
import jax.numpy as jnp
import equinox as eqx

from ..core import count_mutations


def search_matrix_for_query(matrix, query, batch_size, pad_val):
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

    PRINT_EVERY = 10
    for batch_idx in range(nbatches):
        if batch_idx % PRINT_EVERY == 0:
            print(f"  Processing batch {1+batch_idx}/{nbatches}", flush=True)
        idxs = jnp.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        distances_arr = batched_mutation_counter(idxs, matrix, query, k)
        distances = jnp.min(distances_arr, axis=0)
        dist_idxs = jnp.argmin(distances_arr, axis=0)
        improved_screen = distances < min_distances
        min_locations = jnp.where(improved_screen, idxs[dist_idxs], min_locations)
        min_distances = jnp.minimum(min_distances, distances)
    
    return min_locations, min_distances



# def search_matrix_for_query(matrix, query):
#     nseqs, nbases = matrix.shape
#     k = len(query)
#     nscans = nbases - k + 1
#     min_distances = (k+1) * np.ones(nseqs, dtype=int)
#     min_locations = -1 * np.ones(nseqs, dtype=int)

#     def compare_slice_i(i, matrix, query):
#         return count_mutations(matrix[:,i:i+len(query)], query)

#     jax.vmap(compare_slice_i, (0, None, None))(
#         jnp.arange(nscans), matrix, query
#     )

#     for i in range(nscans):
#         distances = count_mutations(
#             matrix[:,i:i+k], query
#         )
#         improved_screen = distances < min_distances
#         min_distances[improved_screen] = distances[improved_screen]
#         min_locations[improved_screen] = i
#     return min_locations, min_distances
