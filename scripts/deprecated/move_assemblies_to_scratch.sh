#!/bin/bash

# Get current assemblies
ls data/assemblies/ > tmpdelme.txt

# Then wait (might still be downloading)
sleep 10s

# Then move
cat tmpdelme.txt | while read -r s; do 
    echo moving $s
    mv data/assemblies/$s /scratch/aeh581/ecoli_exploration/assemblies/$s; 
done
rm tmpdelme.txt

echo Done!
