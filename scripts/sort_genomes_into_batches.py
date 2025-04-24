"""Sort genome files according to their dimensions.

Usage:
    
    python scripts/sort_genomes_into_batches.py \
        -i <genome_file_list> -gl <genome_lengths> -cl <contig_lengths> \
        -b <batch_size> -o <outdir>

"""

import argparse
import os
import time
import gzip
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--genome_file_list', type=str, required=True)
parser.add_argument('-gl', '--genome_lengths', type=str, required=True)
parser.add_argument('-cl', '--contig_lengths', type=str, required=True)
parser.add_argument('-b', '--batch_size', type=int, required=True)
parser.add_argument('-o', '--outdir', type=str, required=True)
args = parser.parse_args()

genome_file_list = args.genome_file_list
genome_lengths_fpath = args.genome_lengths
contig_lengths_fpath = args.contig_lengths
batch_size = args.batch_size
outdir = args.outdir


def read_genome_filepaths(input_fpath):
    """Read into a list the paths contained in the input file."""
    with open(input_fpath, 'r') as f:
        files = [line.strip() for line in f.readlines()]
    return files


def load_contig_lengths(input_fpath: str):
    if input_fpath.endswith(".npy.gz"):
        with gzip.open(contig_lengths_fpath, 'rb') as f:
            contig_lengths = np.load(f, allow_pickle=True)
    elif input_fpath.endswith(".npy"):
        contig_lengths = np.load(input_fpath)
    else:
        raise RuntimeError(f"Cannot process contig lengths file: {input_fpath}")
    # Convert to list, then make type efficient.
    contig_lengths = [x.astype(int) for x in list(contig_lengths)]
    return contig_lengths


########################
##  Begin processing  ##
########################

time0 = time.time()

genome_paths = read_genome_filepaths(genome_file_list)
genome_lengths = np.load(genome_lengths_fpath)
contig_lengths = load_contig_lengths(contig_lengths_fpath)

print("Shape of genome_lengths:", genome_lengths.shape)
print("Length of contig_lengths:", len(contig_lengths))

max_contig_lengths = np.array([np.max(lengths) for lengths in contig_lengths])
print("Shape of max_contig_lengths:", max_contig_lengths.shape)

# Order genomes in increasing order of their maximum contig length.
sorted_contig_idxs = np.argsort(max_contig_lengths)

# Group the genomes into batches of size `batch_size`
groups = []
for i in range(len(max_contig_lengths)):
    if i % batch_size == 0:
        group = []
        groups.append(group)
    genome_idx = sorted_contig_idxs[i]
    group.append(genome_idx)

# Reverse the order of each group, so that first item is the longest.
groups = [np.flip(np.array(g)) for g in groups]

print(f"Constructed {len(groups)} groups.")
print(f"Sizes of last 3 groups:", [len(g) for g in groups[-3:]])

# Determine the maximum genome length (i.e. number of contigs) in each group.
max_genome_lengths = np.zeros(len(groups), dtype=int)
for group_idx, group in enumerate(groups):
    max_genome_lengths[group_idx] = np.max(genome_lengths[group])


# Store results
os.makedirs(outdir, exist_ok=True)
np.save(f"{outdir}/ncontigs_max_by_group.npy", max_genome_lengths)
for group_idx, group in enumerate(groups):
    genome_paths_for_group = [genome_paths[idx] for idx in group]
    np.savetxt(
        f"{outdir}/group_{group_idx}.txt", 
        genome_paths_for_group, fmt='%s'
    )

time1 = time.time()
time_elapsed = time1 - time0
print(f"Finished in {time_elapsed:.3g} seconds")
