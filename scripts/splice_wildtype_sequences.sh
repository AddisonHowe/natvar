#!/bin/bash

infile="data/wtsequences.csv"

function make_query_list () {
    idx0=$1
    length=$2
    idx1=$((idx0+length))
    awkstart=$((idx0+1))
    awkstop=$((idx0+length))
    awk -v i0=$awkstart -v i1=$awkstop -F',' 'NR > 1 {print substr($5, i0, i1)}' \
        $infile > data/query_list_${idx0}_${idx1}.txt
}

make_query_list 0 30
make_query_list 130 30
