"""Read genome sizes.

Inputs:
    --genome_filelist : input text file consisting of paths to genome files.
    --outdir : output directory.
    --verbosity : verbosity level.

Outputs:
    Writes results to an output file.
"""

import argparse
import sys
import os
import time
import numpy as np
import warnings

from .io import read_genome_sizes


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--genome_filelist', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    return parser.parse_args(args)


def main(args):
    genome_filelist = args.genome_filelist
    outdir = args.outdir
    verbosity = args.verbosity

    def printv(s, importance=1, **kwargs):
        if verbosity >= importance:
            print(s, **kwargs)

    printv(f"Processing list of genomes in file: {genome_filelist}")

    
    # Load contigs into a matrix from the contig file.
    printv("Reading input file...")
    time0 = time.time()
    with open(genome_filelist, 'r') as f:
        filelist = f.readlines()
        filelist = [s.strip() for s in filelist]
    genome_lengths, contig_lengths = read_genome_sizes(filelist)
    time1 = time.time()
    printv(f"Finished in {time1-time0:.4f} seconds.")

    os.makedirs(outdir, exist_ok=True)
    np.save(f"{outdir}/genome_lengths.npy", genome_lengths)
    np.save(f"{outdir}/contig_lengths.npy", contig_lengths, allow_pickle=True)
    

#######################
##  Main Entrypoint  ##
#######################

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
