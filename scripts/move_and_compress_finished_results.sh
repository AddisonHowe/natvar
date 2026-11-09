#!/usr/bin/env bash


d=multiquery_results_0_30_rc
# d=multiquery_results_130_160_rc

logdir=logs/multiquery_entero
outdirold=out/multiquery_entero/${d/multiquery_/}
outdirnew=results/dataset_entero/${d}



for f in ${logdir}/${d/results_/}.*.o; do 
    if [[ $(cat $f | grep "Done") ]]; then 
        i=$(echo $f | sed -n 's/.*_\([^\.]*\)\..*/\1/p'); 
        mv ${outdirold}/results_${i}.tsv ${outdirnew}/;
        gzip ${outdirnew}/results_${i}.tsv;
        mv $f ${logdir}/completed/
        mv ${f/.o/.e} ${logdir}/completed/ && \
            echo "Moved and zipped data ${outdirold}/results_${i}.tsv"
    fi; 
done


echo "The following halted due to exceeding time limit:"
for f in ${logdir}/${d/results_/}.*.e; do 
    if [[ $(cat $f | grep "DUE TO TIME LIMIT") ]]; then 
        i=$(echo $f | sed -n 's/.*_\([^\.]*\)\..*/\1/p'); 
        echo $i $f
    fi; 
done

echo Done!
