#!/bin/bash



results_fpath=$1  # results/multiquery_results_0_30/results_18.tsv.gz

zcat $results_fpath | awk -F'\t' '
    BEGIN {OFS="\t"} {print $1, $2, $3, $4, $5}
' | sort | uniq -cd

