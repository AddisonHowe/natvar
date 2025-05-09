#!/usr/bin/env bash
#=============================================================================
#
# FILE: compile_ecoli_data.sh
#
# USAGE: compile_ecoli_data.sh
#
# DESCRIPTION: Compile data corresponding to species E. Coli.
#
# EXAMPLE: sh scripts/compile_ecoli_data.sh
#=============================================================================

outdir=data/dataset_ecoli
assemblies_dir=data/assemblies

mkdir -p $outdir

# --- Contruct metadata table ---
mdata_fpath=${outdir}/metadata_subset.tsv
head data/metadata_all.tsv -n 1 > ${mdata_fpath}
awk -F'\t' '$10 == "Escherichia coli"' data/metadata_all.tsv >> ${mdata_fpath}

# --- Contruct list of assembly ids ---
assemblyids_fpath=${outdir}/assemblyids.tsv
awk -F'\t' 'NR > 1 { print $1 }' ${mdata_fpath} > ${assemblyids_fpath}

# --- Construct map from assembly id to ftp filepath ---
id_to_ftp_fpath=${outdir}/assemblyid_to_ftp.tsv
awk -F'\t' 'NR==FNR {map[$1]=$2; next} $1 in map {print $1 "\t" map[$1]}' \
    data/assemblyid_ftps.tsv ${assemblyids_fpath} > $id_to_ftp_fpath

# --- Construct map from assembly id to local filepath ---
assemblymap_fpath=${outdir}/assemblyid_to_fpath.tsv
repstr=/ebi/ftp/pub/databases/ENA2018-bacteria-661k/Assemblies
cat $id_to_ftp_fpath | sed "s|${repstr}|${assemblies_dir}|g" > $assemblymap_fpath

# --- Construct list of local filepaths ---
genome_list_fpath=${outdir}/genome_file_paths.txt
awk -F'\t' '{ print $2 }' ${assemblymap_fpath} > $genome_list_fpath

# --- Check that all files have been downloaded and exist ---
# all_exist=true
# while read -r fpath; do
#     if [ ! -e "$fpath" ]; then
#         echo Missing $fpath
#         all_exist=false
#     fi
# done < $genome_list_fpath

# if $all_exist; then
#     echo "All files exist. Continuing."
# else
#     echo "Some files are missing! Halting"
#     exit 1
# fi

# --- Read genome sizes ---
genome_sizes_dir=${outdir}/genome_sizes
# read-genome-sizes -i $genome_list_fpath -o $genome_sizes_dir -v 1 --pbar

# --- Batch genomes ---
genome_batches_dir=${outdir}/genome_batches
batch_size=1000
python scripts/sort_genomes_into_batches.py \
    -i $genome_list_fpath \
    -gl ${genome_sizes_dir}/genome_lengths.npy \
    -cl ${genome_sizes_dir}/contig_lengths.npy.gz \
    -b $batch_size \
    -o $genome_batches_dir
