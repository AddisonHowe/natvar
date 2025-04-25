"""Test for entrypoint multiquery-genome-batch

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
    "infile, queries_fpath, outdir, pad_left, pad_right, " \
    "exp_dists, exp_idxs, exp_locs, exp_seqs, exp_compiles_map", [
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_AAAA_1.txt",
        "out1", 0, 0, 
        [0, 0], [[0], [1]], [[0], [0]], [['AAAA'], ['AAAA']],
        {},
    ],
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_AAAA_2.txt",
        "out1", 0, 0, 
        2*[0, 0], [[0],[0],[1],[1]], 2*[[0], [0]], 2*[['AAAA'], ['AAAA']],
        {},
    ],
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_CCCC_2.txt",
        "out1", 0, 0, 
        2*[0, 0], [[0],[0],[1],[1]], 2*[[4], [4]], 2*[['CCCC'], ['CCCC']],
        {},
    ],
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_AATT_2.txt",
        "out1", 0, 0, 
        2*[2, 2], 2*[[0,1,2], [0,1,2]], [[0,2,0], [0,2,0], [0,0,0],[0,0,0]], 
        [['AAAA','GGTT','ACGT'], ['AAAA','GGTT','ACGT'], 
         ['AA__','AAAA','ACGT'], ['AA__','AAAA','ACGT']],
        {},
    ],
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_AATT_2.txt",
        "out1", 0, 2, 
        2*[2, 2], 2*[[0,1,2], [0,1,2]], [[0,2,0], [0,2,0], [0,0,0], [0,0,0]], 
        [['AAAAcc','GGTTtt','ACGTac'], ['AAAAcc','GGTTtt','ACGTac'], 
         ['AA____','AAAAcc','ACGT__'], ['AA____','AAAAcc','ACGT__']],
        {},
    ],
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_AATT_2.txt",
        "out1", 0, 6, 
        2*[2, 2], 2*[[0,1,2], [0,1,2]], [[0,2,0], [0,2,0], [0,0,0], [0,0,0]], 
        [['AAAAcccc__','GGTTtt____','ACGTacgt__'],
         ['AAAAcccc__','GGTTtt____','ACGTacgt__'],
         ['AA________','AAAAcccc__','ACGT______'],
         ['AA________','AAAAcccc__','ACGT______']],
        {},
    ],
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_AATT_2.txt",
        "out1", 2, 6, 
        2*[2, 2], 2*[[0,1,2], [0,1,2]], [[0,2,0], [0,2,0], [0,0,0], [0,0,0]], 
        [['__AAAAcccc__','ggGGTTtt____','__ACGTacgt__'],
         ['__AAAAcccc__','ggGGTTtt____','__ACGTacgt__'], 
         ['__AA________','__AAAAcccc__','__ACGT______'],
         ['__AA________','__AAAAcccc__','__ACGT______']],
        {},
    ],
    [
        "test_assembly3_genomes.txt", 
        "test_multiqueries/queries_G8_A8.txt",
        "out1", 1, 1, 
        [7, 4, 4, 2, 8, 0, 0, 8, 0, 8],
        [[2], [1], [2], [0], [0], [0], [0], [0,1], [0,1], [0,1]],
        [[0], [0], [0], [0], [0], [0], [0], [0,0], [0,0], [0,0]], 
        [['_ACGT_____',],
         ['_AAAA_____',],
         ['_GGGG_____',],
         ['_AAAAAA___',],
         ['_AAAAAAAA_',],
         ['_AAAAAAAA_',],
         ['_GGGGGGGGg',],
         ['_GGGGGGGGg','_GGGG_____',],
         ['_GGGGGGGGg','_GGGGGGGG_',],
         ['_GGGGGGGGg','_GGGGGGGG_',]],
        {0:2, 1:2, 4:2, 32:2},
    ],
    [
        "test_assembly4_genomes.txt", 
        "test_multiqueries/queries_G8_A8.txt",
        "out1", 1, 1, 
        [8, 6, 7, 4, 4, 2, 8, 0, 0, 8, 0, 8],
        [[0], [0], [2], [1], [2], [0], [0], [0], [0], [0,1], [0,1], [0,1]],
        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0,0], [0,0], [0,0]], 
        [['_AA_______',],
         ['_AA_______',],
         ['_ACGT_____',],
         ['_AAAA_____',],
         ['_GGGG_____',],
         ['_AAAAAA___',],
         ['_AAAAAAAA_',],
         ['_AAAAAAAA_',],
         ['_GGGGGGGGg',],
         ['_GGGGGGGGg','_GGGG_____',],
         ['_GGGGGGGGg','_GGGGGGGG_',],
         ['_GGGGGGGGg','_GGGGGGGG_',]],
         {0:3, 1:3, 4:2, 32:2},
    ],
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_G8_A8.txt",
        "out1", 1, 1, 
        [4, 4, 7, 4], [[1],[0],[2],[1],], [[0],[0],[0],[0]],
        [['_GGGGTTTT_',],
         ['_AAAACCCC_',], 
         ['_ACGT_____',],
         ['_AAAACCCC_',]],
        {},
    ],
    [
        "test_assembly1_genomes.txt", 
        "test_multiqueries/queries_G10_A10.txt",
        "out1", 1, 1, 
        [6, 6, 9, 6], [[1],[0],[2],[1],], [[0],[0],[0],[0]],
        [['_GGGGTTTT___',],
         ['_AAAACCCC___',], 
         ['_ACGT_______',],
         ['_AAAACCCC___',]],
        {},
    ],
])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 32])
@pytest.mark.parametrize("nrows0", [0, 1, 4, 32])
@pytest.mark.filterwarnings("ignore:.*Repadding.*:UserWarning")
def test_multiquery_genome_batch(
        infile, queries_fpath, outdir, pad_left, pad_right, 
        exp_dists, exp_idxs, exp_locs, exp_seqs, exp_compiles_map, 
        batch_size, nrows0
):
    from natvar.multiquery_genome_batch import parse_args, main
    
    outfname = "q_batch_results.tsv"
    argstring = f"-q {DATDIR}/{queries_fpath} -i {DATDIR}/{infile} " \
                + f"-o {outdir} -f {outfname} -pl {pad_left} -pr {pad_right} " \
                + f"--batch_size {batch_size} -v 3 --nrows0 {nrows0}"
    
    errors = []
    try:
        exp_compiles = exp_compiles_map.get(nrows0, None)
        if exp_compiles:
            max_trace_line = f" --jax_debug_max_traces {exp_compiles}"
        else:
            max_trace_line = ""
        arglist = (argstring + max_trace_line).split(" ")
        args = parse_args(arglist)
        args.outdir = f"{TMPDIR}/{args.outdir}"
        main(args)
    except RuntimeError() as e:
        errors.append(e)
        arglist = (argstring + max_trace_line).split(" ")
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

    KEY_MIN_DISTANCE = "min_distance"
    KEY_NEAREST_IDXS = "nearest_idxs"
    KEY_LOC_ON_CONTIGS = "location_on_contigs"
    KEY_CONTIG_SEGMENTS = "contig_segments"
    for i in range(nrows):
        if not np.allclose(eval(res[KEY_MIN_DISTANCE][i]), exp_dists[i]):
            msg = f"Wrong expected {KEY_MIN_DISTANCE} (row {i+1}). "
            msg += f"Expected {exp_dists[i]}. Got {res[KEY_MIN_DISTANCE][i]}"
            errors.append(msg)
        if not np.allclose(eval(res[KEY_NEAREST_IDXS][i]), exp_idxs[i]):
            msg = f"Wrong expected {KEY_NEAREST_IDXS} (row {i+1}). "
            msg += f"Expected {exp_idxs[i]}. Got {res[KEY_NEAREST_IDXS][i]}"
            errors.append(msg)
        if not np.allclose(eval(res[KEY_LOC_ON_CONTIGS][i]), exp_locs[i]):
            msg = f"Wrong expected {KEY_LOC_ON_CONTIGS} (row {i+1}). "
            msg += f"Expected {exp_locs[i]}. Got {res[KEY_LOC_ON_CONTIGS][i]}"
            errors.append(msg)
        if not np.all(eval(res[KEY_CONTIG_SEGMENTS][i]) == exp_seqs[i]):
            msg = f"Wrong expected {KEY_CONTIG_SEGMENTS} (row {i+1}). "
            msg += f"Expected {exp_seqs[i]}. Got {res[KEY_CONTIG_SEGMENTS][i]}"
            errors.append(msg)

    remove_dir(args.outdir)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

