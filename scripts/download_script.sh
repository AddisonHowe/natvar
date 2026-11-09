#!/usr/bin/env bash

DATDIR=data

read -p "Downloading data will overwrite existing data.
Do you want to continue? (y/n): " answer
case "$answer" in
    [Yy]* ) echo "Continuing...";;
    [Nn]* ) echo "Aborting."; exit 1;;
    * ) echo "Please answer [y]es or [n]o."; exit 1;;
esac


# Sample IDs to assembly path
wget -O ${DATDIR}/assemblyid_ftps.tsv \
    https://ftp.ebi.ac.uk/pub/databases/ENA2018-bacteria-661k/sampleid_assembly_paths.txt

# Metadata
wget -O ${DATDIR}/metadata_all.tsv \
    https://figshare.com/ndownloader/files/30449922

# # Subset of data corresponding to E. Coli
# cat ${DATDIR}/metadata_all.tsv | grep "Escherichia coli" \
#     > ${DATDIR}/metadata_ecoli_subset.tsv

# # E. Coli sample ids
# awk -F'\t' '{print $1}' ${DATDIR}/metadata_ecoli_subset.tsv \
#     > ${DATDIR}/sampleids_ecoli.txt

# Reference genome
eval "$($CONDA_EXE shell.bash hook)"
conda activate ncbi-datasets-env
datasets download genome accession GCF_000005845.2 \
    --include gff3,rna,cds,protein,genome,seq-report \
    --filename ${DATDIR}/ref_genome.zip
unzip ${DATDIR}/ref_genome.zip -d ${DATDIR}/ref_genome
