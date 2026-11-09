#!/bin/bash

# map_fpath=$1  # text file of sample ids to download with url

datdir=/projects/p31512/aeh581/natvar/data
outdir=/scratch/aeh581/ecoli_exploration/assemblies

assemblies_fpath=${datdir}/sampleids_entero.txt

map_fpath=${datdir}/sampleid_assembly_paths.tsv

mkdir -p $outdir

do_download() {
    local sampid=$1
    local sampurl=$2
    sampurl=$(echo $sampurl | sed 's|^/ebi/ftp|https://ftp.ebi.ac.uk|')
    wget -P ${outdir} ${sampurl}
}

n=10000000000
count=0
awk -F'\t' 'NR==FNR {map[$1]=$2; next} {print $0, map[$1]}' \
    $map_fpath $assemblies_fpath | \
    while read -r sampid sampurl; do
        do_download $sampid $sampurl
        ((count++))
        echo Downloading item $count
        if [[ $count -ge $n ]]; then
            break
        fi
    done
