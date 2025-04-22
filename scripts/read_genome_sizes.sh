#!/bin/bash
#=============================================================================
#
# FILE: read_genome_sizes.sh
#
# USAGE: read_genome_sizes.sh filelist
#
# DESCRIPTION: Determine the number of contigs and their lengths for a number 
#  of genome files.
#
# EXAMPLE: sh read_genome_sizes.sh genome_filelist.txt
#=============================================================================

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filelist>"
    exit 1
fi

filelist=$1

read-genome-sizes -i $filelist -o out/genome_sizes -v 1


