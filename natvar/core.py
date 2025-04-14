"""Core functions

"""

import numpy as np


def compare_sequences(seq1, seq2):
    """Screen for all differing positions between seq1 and seq2."""
    return seq1 != seq2


def count_mutations(seqs, wt_seq):
    """Count the number of mutations in a sequence relative to another."""
    mut_screen = compare_sequences(seqs, wt_seq)
    nmuts = mut_screen.sum(axis=-1)
    return nmuts


def search_matrix_for_query(matrix, query):
    nseqs, nbases = matrix.shape
    k = len(query)
    nscans = nbases - k + 1
    min_distances = (k+1) * np.ones(nseqs, dtype=int)
    min_locations = -1 * np.ones(nseqs, dtype=int)
    for i in range(nscans):
        distances = count_mutations(matrix[:,i:i+k], query)
        improved_screen = distances < min_distances
        min_distances[improved_screen] = distances[improved_screen]
        min_locations[improved_screen] = i
    return min_locations, min_distances
