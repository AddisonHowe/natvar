"""Test for entrypoint query-genome-batch

"""

import pytest
import numpy as np

from .conftest import DATDIR, TMPDIR, remove_dir


#####################
##  Configuration  ##
#####################


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
    "infile, outdir, query, pad_left, pad_right, " \
    "exp_dists, exp_idxs, exp_locs, exp_seqs", [
    # Assembly 1
    [
        "test_assembly1_genomes.txt", "out1", 
        "AAAA", 0, 0, 
        [0, 0], [[0], [1]], [[0], [0]], 
        [['AAAA'], ['AAAA']]
    ],
    [
        "test_assembly1_genomes.txt", "out1", 
        "CCCC", 0, 0, 
        [0, 0], [[0], [1]], [[4], [4]], [['CCCC'], ['CCCC']]
    ],
    [
        "test_assembly1_genomes.txt", "out1", 
        "AATT", 0, 0, 
        [2, 2], [[0,1,2], [0,1,2]], [[0,2,0], [0,0,0]], 
        [['AAAA','GGTT','ACGT'], ['AA__','AAAA','ACGT']]
    ],
    [
        "test_assembly1_genomes.txt", "out1", 
        "AATT", 0, 2, 
        [2, 2], [[0,1,2], [0,1,2]], [[0,2,0], [0,0,0]], 
        [['AAAAcc','GGTTtt','ACGTac'], ['AA____','AAAAcc','ACGT__']]
    ],
    [
        "test_assembly1_genomes.txt", "out1", 
        "AATT", 0, 6, 
        [2, 2], [[0,1,2], [0,1,2]], [[0,2,0], [0,0,0]], 
        [['AAAAcccc__','GGTTtt____','ACGTacgt__'], 
         ['AA________','AAAAcccc__','ACGT______']]
    ],
    [
        "test_assembly1_genomes.txt", "out1", 
        "AATT", 2, 6, 
        [2, 2], [[0,1,2], [0,1,2]], [[0,2,0], [0,0,0]], 
        [['__AAAAcccc__','ggGGTTtt____','__ACGTacgt__'], 
         ['__AA________','__AAAAcccc__','__ACGT______']]
    ],

])
@pytest.mark.parametrize("batch_size", [1])
def test_query_genome_batch(
        infile, outdir, query, pad_left, pad_right, 
        exp_dists, exp_idxs, exp_locs, exp_seqs, batch_size,
):
    from natvar.query_genome_batch import parse_args, main
    
    outfname = "q_batch_results.tsv"
    argstring = f"-q {query} -i {DATDIR}/{infile} -o {outdir} " \
                + f"-f {outfname} -pl {pad_left} -pr {pad_right} " \
                + f"--batch_size {batch_size} -v 0"
    
    arglist = argstring.split(" ")
    args = parse_args(arglist)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    
    main(args)

    def read_output(fpath):
        with open(fpath, 'r') as f:
            lines = f.readlines()
            keys = lines[0].strip().split('\t')
            d = {k: [] for k in keys}
            for line in lines[1:]:
                vals = line.strip().split('\t')
                for k, v in zip(keys, vals):
                    d[k].append(v)
        return d, len(lines) - 1

    res, nrows = read_output(f"{args.outdir}/{outfname}")
    print(res)

    errors = []
    for i in range(nrows):
        if not np.allclose(eval(res['min_dist'][i]), exp_dists[i]):
            msg = f"Wrong expected min_dist (row {i+1}). "
            msg += f"Expected {exp_dists[i]}. Got {res['min_dist'][i]}"
            errors.append(msg)
        if not np.allclose(eval(res['nearest_idxs'][i]), exp_idxs[i]):
            msg = f"Wrong expected nearest_idxs (row {i+1}). "
            msg += f"Expected {exp_idxs[i]}. Got {res['nearest_idxs'][i]}"
            errors.append(msg)
        if not np.allclose(eval(res['location_on_contigs'][i]), exp_locs[i]):
            msg = f"Wrong expected location_on_contigs (row {i+1}). "
            msg += f"Expected {exp_locs[i]}. Got {res['location_on_contigs'][i]}"
            errors.append(msg)
        if not np.all(eval(res['contig_segments'][i]) == exp_seqs[i]):
            msg = f"Wrong expected contig_segments (row {i+1}). "
            msg += f"Expected {exp_seqs[i]}. Got {res['contig_segments'][i]}"
            errors.append(msg)

    remove_dir(args.outdir)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

