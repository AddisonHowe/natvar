#!/usr/bin/env bash
#=============================================================================
#
# FILE: read_genome_sizes.sh
#
# USAGE: read_genome_sizes.sh filelist outdir
#
# DESCRIPTION: Determine the number of contigs and their lengths for a number 
#  of genome files.
#
# EXAMPLE: sh read_genome_sizes.sh genome_filelist.txt <outdir>
#=============================================================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filelist> <outdir>"
    exit 1
fi

filelist=$1
outdir=$2

read-genome-sizes -i $filelist -o $outdir -v 1 --pbar


