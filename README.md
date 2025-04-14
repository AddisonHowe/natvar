# natvar

## Installation and setup

Clone the repository with 

```bash
git clone https://github.com/AddisonHowe/natvar.git
cd natvar
```
and create a conda environment by running
```bash
conda env create -p ./env -f environment.yml
conda activate env
python -m pip install -e ".[dev,jax,jupyter]"
```

Check that tests pass:
```bash
pytest tests
```

## Query a single genome file for a sequence string

We can use the command `query-genome` to query a single genome file for a particular sequence:

```bash
conda activate env
query-genome -q <query> -i <genome_file> \
    [-o <outdir>] [-f <outfname>] [-pl <left_pad>] [-pr <right_pad>]
```

Here, `<query>` is the sequence we wish to search for, and `<genome_file>` is the path to a text file (either in raw text or compressed .gz format) that specifies one genome, as a set of fragments, or "contigs."
The values `<outdir>` and `<outfname>` specify the name of the output directory and the output filename, respectively. 
The `<left_pad>` and `<right_pad>` integer values specify how many bases to include to the left and right of the nearest sequence that is found in the search.

The output file located at `<outdir>/<outfname>` is a tab separated file of the following form:
```
genome_fpath	<genome_fpath>
query_string	<query>
min_distance	<min_distance>
nearest_idxs	<nearest_idxs>
location_on_contigs   <location_on_contigs>
contig_segments	<contig_segments>
time_elapsed    <time_elapsed>
```

where `<min_distance>` is an integer, the distance between the query and the closest matching sequence found within the genome.
`<nearest_idxs>` is a list of length $N_{matches}$ specifying the index of each contig fragment within the genome that contains a sequence with the minimal distance to the query.
`<location_on_contigs>` is a list of length $N_{matches}$, and specifies the position on the corresponding contig where the (nearest) match occurs.
`<contig_segments>` is a list of length $N_{matches}$ containing the (nearest) matching sequences on each of the $N_{matches}$ contigs, with additional bases included based on the values of `<left_pad>` and `<right_pad>`.

## Query a set of genome files for a sequence string

A parallelized algorithm can be run across a number of genome files using the command `query_genome_batch` as follows:

```bash
conda activate env
query-genome-batch -q <query> -i <input_fpath> \
    [-o <outdir>] [-f <outfname>] [-b <batch_size>] [-pl <left_pad>] [-pr <right_pad>]
```

where the arguments are analogous to those above, but with `<input_fpath>` the path to a text file where each row specifies the path to an individual genome file.
For example, an input file might look like
```
data/assemblies/genome_file1.contigs.fa
data/assemblies/genome_file2.contigs.fa
data/assemblies/genome_file3.contigs.fa.gz
```
and can contain both raw and compressed files.
The additional `<batch_size>` argument is an integer (default 1000) that specifies the number of slices to compare at once. 
This value is limited by memory constraints, and should be set as large as possible.

The output file is a tab-separated file where each row corresponds to the results of searching one of the genomes, and whose columns correspond to the output values described above for the `query-genome` command.
