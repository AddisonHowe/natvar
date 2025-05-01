import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import entropy



#determine probability of a sequence with n differences from a reference within a query length L.
#we wish for the output of this to be less than a specified cutoff. 
def E(n, L):
    N = 4*10**6 #genome length
    comb = math.comb(L, n)
    return N*(comb*3**n/4**L)

def calculate_entropy(seqs):
    seqs = np.array(seqs)
    num_seqs, seq_length = seqs.shape
    entropies = np.zeros(seq_length)
    for i in range(seq_length):
        values, counts = np.unique(seqs[:, i], return_counts=True)
        entropies[i] = entropy(counts)
    return entropies


#Given a gene, all natural variants of that gene from the data. 
def produce_seqs(GENE): 
    seqs = []
    
    #fixed parameters
    tau = 0.1 #threshold
    L = 30 #query length
    m = 85 #number of genome files
    seq_len = 175 #sequence length
    
    NT_MAP = {c: i for i, c in enumerate(['A', 'C', 'G', 'T'])}
    search = pd.read_csv('../data/search.csv')
    index = search[search['gene'] == GENE].index
    query = search['start'].loc[index].tolist()[0]
    
    for j in range(m):
        FPATH = f'../results/multiquery_results_0_30/results_{j}.tsv.gz'
        results = pd.read_csv(FPATH, sep='\t')
        indices = results[results['query_string'] == query].index.tolist()
        for i in indices: 
            if E(int(results['min_distance'][i]), L) < tau: 
                sequence = results['contig_segments'][i]
                numeric_sequence = [NT_MAP[c.upper()] for c in sequence if c.upper() in ['A', 'C', 'T', 'G']]
                if len(numeric_sequence) == seq_len:  
                    seqs.append(numeric_sequence[5:165])
    
    return seqs

#produce entropy plots
def plot_entropy(seqs, GENE):
    
    entropies = calculate_entropy(seqs)
        
    x = np.linspace(-115, 45, 160)    
    plt.scatter(x, entropies, s = 7.5, color = 'blue', label = 'entropy')

    #binding_sites = [(-70, -60), (-45, -20)]
    #for start, end in binding_sites:
    #    plt.axvspan(start, end, color='yellow', alpha=0.3)

    #plt.ylim(0,0.1)
    plt.xlabel('position')
    plt.ylabel('entropy')
    plt.title(f'{GENE} entropy plot')
    plt.savefig(f'../out/entropy_plots/{GENE}_full.png')
    plt.show()
    plt.scatter(x, entropies, s = 7.5, color = 'blue', label = 'entropy')

    #binding_sites = [(-70, -60), (-45, -20)]
    #for start, end in binding_sites:
    #    plt.axvspan(start, end, color='yellow', alpha=0.3)

    plt.ylim(0,0.1)
    plt.xlabel('position')
    plt.ylabel('entropy')
    plt.title(f'{GENE} entropy plot')
    plt.savefig(f'../out/entropy_plots/{GENE}_capped.png')
    

#produce a table of frequencies for each nucleotide at each position
def frequency_table(seqs):
    m = len(seqs)
    n = len(seqs[0])
    table = np.zeros((4, n))
    for i in range(m):
        for j in range(n):
            table[seqs[i][j]][j] += 1/m 
    return table 

#produce a plot of the table
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_table(table, GENE):
    x = np.linspace(-115, 45, 160)  
    
    plt.scatter(x, table[0], color='darkviolet', label='A', s=3)
    plt.scatter(x, table[1], color='firebrick', label='C', s=3)
    plt.scatter(x, table[2], color='orange', label='G', s=3)
    plt.scatter(x, table[3], color='springgreen', label='T', s=3)
    plt.legend()
    plt.xlabel('Position on Promoter')
    plt.ylabel('Nucleotide Frequency')
    plt.savefig(f'../out/entropy_plots/{GENE}_frequncies_full.png')
    plt.show() 
    
    
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.scatter(x, table[0], color='darkviolet', label='A', s=3)
    ax1.scatter(x, table[1], color='firebrick', label='C', s=3)
    ax1.scatter(x, table[2], color='orange', label='G', s=3)
    ax1.scatter(x, table[3], color='springgreen', label='T', s=3)
    ax1.legend()
    ax1.set_xlabel('Position on Promoter')
    ax1.set_ylabel('Nucleotide Frequency')
    ax1.set_ylim(0, 0.05)
    
    ax2.scatter(x, table[0], color='darkviolet', label='A', s=3)
    ax2.scatter(x, table[1], color='firebrick', label='C', s=3)
    ax2.scatter(x, table[2], color='orange', label='G', s=3)
    ax2.scatter(x, table[3], color='springgreen', label='T', s=3)
    ax2.set_ylabel('Nucleotide Frequency')
    ax2.set_ylim(0.95, 1)
    plt.savefig(f'../out/entropy_plots/{GENE}_frequncies_capped.png')
    
    plt.tight_layout()
    plt.show()



    
    