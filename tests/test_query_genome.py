"""Test for entrypoint query-genome 

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
    # Assembly 1 Genome 1
    [
        "assembly_set1/genome1.contigs.fa", "out1", 
        "AAAA", 0, 0, 
        0, [0], [0], ['AAAA']
    ],[
        "assembly_set1/genome1.contigs.fa", "out1", 
        "CCCC", 0, 0, 
        0, [0], [4], ['CCCC']
    ],[
        "assembly_set1/genome1.contigs.fa", "out1", 
        "GGGG", 0, 0, 
        0, [1], [0], ['GGGG']
    ],[
        "assembly_set1/genome1.contigs.fa", "out1", 
        "TTTT", 0, 0, 
        0, [1], [4], ['TTTT']
    ],[
        "assembly_set1/genome1.contigs.fa", "out1", 
        "ACGT", 0, 0, 
        0, [2], [0], ['ACGT']
    ],[
        "assembly_set1/genome1.contigs.fa", "out1", 
        "AAAT", 0, 0, 
        1, [0], [0], ['AAAA']
    ],
    # Assembly 1 Genome 2
    [
        "assembly_set1/genome2.contigs.fa.gz", "out2", 
        "AAAA", 0, 0, 
        0, [1], [0], ['AAAA']
    ],[
        "assembly_set1/genome2.contigs.fa.gz", "out2", 
        "AACC", 0, 0, 
        0, [1], [2], ['AACC']
    ],[
        "assembly_set1/genome2.contigs.fa.gz", "out2", 
        "AACC", 2, 2, 
        0, [1], [2], ['aaAACCcc']
    ],[
        "assembly_set1/genome2.contigs.fa.gz", "out2", 
        "AACC", 3, 3, 
        0, [1], [2], ['_aaAACCcc_']
    ]
])
@pytest.mark.parametrize("use_jax", [False, True])
def test_query_genome(
        infile, outdir, query, pad_left, pad_right, 
        exp_dists, exp_idxs, exp_locs, exp_seqs, use_jax
):
    from natvar.query_genome import parse_args, main
    
    outfname = "q_results.tsv"
    argstring = f"-q {query} -i {DATDIR}/{infile} -o {outdir} " \
                + f"-f {outfname} -pl {pad_left} -pr {pad_right} -v 0"
    if use_jax:
        argstring += " --jax"
    
    arglist = argstring.split(" ")
    args = parse_args(arglist)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    
    main(args)

    def read_output(fpath):
        d = {}
        with open(fpath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                k, v = line.strip().split('\t')
                d[k] = v
        return d

    res = read_output(f"{args.outdir}/{outfname}")

    errors = []
    KEY_MIN_DISTANCE = "min_distance"
    KEY_NEAREST_IDXS = "nearest_idxs"
    KEY_LOC_ON_CONTIGS = "location_on_contigs"
    KEY_CONTIG_SEGMENTS = "contig_segments"
    if not np.allclose(eval(res[KEY_MIN_DISTANCE]), exp_dists):
        msg = f"Wrong expected {KEY_MIN_DISTANCE}. "
        msg += f"Expected {exp_dists}. Got {res[KEY_MIN_DISTANCE]}"
        errors.append(msg)
    if not np.allclose(eval(res[KEY_NEAREST_IDXS]), exp_idxs):
        msg = f"Wrong expected {KEY_NEAREST_IDXS}. "
        msg += f"Expected {exp_idxs}. Got {res[KEY_NEAREST_IDXS]}"
        errors.append(msg)
    if not np.allclose(eval(res[KEY_LOC_ON_CONTIGS]), exp_locs):
        msg = f"Wrong expected {KEY_LOC_ON_CONTIGS}. "
        msg += f"Expected {exp_locs}. Got {res[KEY_LOC_ON_CONTIGS]}"
        errors.append(msg)
    if not np.all(eval(res[KEY_CONTIG_SEGMENTS]) == exp_seqs):
        msg = f"Wrong expected {KEY_CONTIG_SEGMENTS}. "
        msg += f"Expected {exp_seqs}. Got {res[KEY_CONTIG_SEGMENTS]}"
        errors.append(msg)

    remove_dir(args.outdir)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

