
import numpy as np

datdir = "out/dataset_entero/genome_sizes"
genomes_outfpath = "out/genomes_out"
contigs_outfpath = "out/contigs_out"

N = 100


#########################
##  Join genome files  ##
# #########################

# genome_list = []
# for i in range(N):
#     d = f"{datdir}/process{i}"
#     genome_file = f"{d}/genome_lengths.npy"
#     genome = np.load(genome_file)
#     genome_list.append(genome)

# genomes = np.concatenate(genome_list, dtype=int)
# np.save(genomes_outfpath, genomes)

#########################
##  Join contig files  ##
#########################

contig_list = []
for i in range(N):
    d = f"{datdir}/process{i}"
    contig_file = f"{d}/contig_lengths.npy"
    contig = np.load(contig_file, allow_pickle=True)
    contig_list.append(contig)

contigs = np.concatenate(contig_list, axis=0, dtype=object)
np.save(contigs_outfpath, contigs)
