"""Basic file processing and I/O functions.

"""

import numpy as np
import gzip
import re as regex
import warnings
import tqdm as tqdm

from .helpers import gene_seq_to_array


def smart_open(filename, mode='rt', **kwargs):
    """Open a file that may be gzip-compressed or plain text.

    Args:
        filename (str): Path to the file
        mode (str): Mode to open the file.
        **kwargs: Additional arguments passed to open or gzip.open

    Returns:
        file object: Either from open() or gzip.open()
    """
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, **kwargs)
    else:
        return open(filename, mode, **kwargs)


def process_contig_file(fpath):
    """Return list of np.uint8 sequence arrays."""

    def add_contig_to_list(contig_list, contig, reported_length):
        contig_list.append(gene_seq_to_array(contig))
        if reported_length:
            if len(contig) != reported_length:
                msg = "Expected length {} contig. Got length {}.".format(
                    reported_length, len(contig)
                )
                warnings.warn(msg)
        else:
            msg = "No reported length for contig."
            warnings.warn(msg)
    
    with smart_open(fpath, mode='rt') as f:
        contig_list = []
        current_contig=""
        for i, line in enumerate(f):
            if line.startswith('>'):
                # Add previously found contig
                if i > 0:
                    add_contig_to_list(
                        contig_list, current_contig, contig_length_reported
                    )
                current_contig = ""
                current_contig_length = 0
                match = regex.search(r'\blen=(\d+)', line)
                if match:
                    contig_length_reported = int(match.group(1))
                else:
                    contig_length_reported = None
            else:
                line = line.strip()
                current_contig += line
                current_contig_length += len(line)
        # Add last contig
        add_contig_to_list(contig_list, current_contig, contig_length_reported)
    return contig_list


def get_contigs_matrix(contigs_list, pad_val=5):
    """Convert a list of np.uint8 contig sequences to a padded 2d-array."""
    lengths = [len(contig) for contig in contigs_list]
    contigs = pad_val * np.ones(
        [len(contigs_list), np.max(lengths)], dtype=np.uint8
    )
    for i in range(len(contigs_list)):
        contigs[i, 0:lengths[i]] = contigs_list[i]
    return contigs


def load_genome_set(genome_filelist, pad_val):
    """Process a number of genome files into a matrix format."""
    contig_matrices = []
    genome_sizes = []
    nrows = 0
    ncols = 0
    for fpath in genome_filelist:
        m = get_contigs_matrix(fpath)
        r, c = m.shape
        contig_matrices.append(m)
        genome_sizes.append(r)
        nrows += r
        ncols = max(ncols, c)
    genome_sizes = np.array(genome_sizes)
    genome_array = np.zeros([nrows, ncols], dtype=np.uint8)


def read_genome_sizes(genome_filelist, pbar=False):
    """For each genome, determine the number of contigs and the length of each.
    
    Args:
        genome_filelist (list[str]): paths to each genome file to process.
    
    Returns:
        num_contigs (np.ndarray[int]): number of contigs making up each genome.
        contig_lengths (list[np.ndarray[int]]): list of arrays giving contig 
            lengths for each genome.
    """
    num_contigs = []
    contig_lengths = []
    for fpath in tqdm.tqdm(genome_filelist, disable=not pbar):
        contigs = process_contig_file(fpath)
        n = len(contigs)  # number of contigs
        lengths = np.array([len(c) for c in contigs])  # length of each contig
        num_contigs.append(n)
        contig_lengths.append(lengths)
    num_contigs = np.array(num_contigs, dtype=int)
    return num_contigs, contig_lengths
