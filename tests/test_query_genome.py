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
def test_query_genome(
        infile, outdir, query, pad_left, pad_right, 
        exp_dists, exp_idxs, exp_locs, exp_seqs,
):
    from natvar.query_genome import parse_args, main
    
    outfname = "q_results.tsv"
    argstring = f"-q {query} -i {DATDIR}/{infile} -o {outdir} " \
                + f"-f {outfname} -pl {pad_left} -pr {pad_right} -v 0"
    
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
    if not np.allclose(eval(res['min_distance']), exp_dists):
        msg = "Wrong expected min_distance. "
        msg += f"Expected {exp_dists}. Got {res['min_distance']}"
        errors.append(msg)
    if not np.allclose(eval(res['nearest_idxs']), exp_idxs):
        msg = "Wrong expected nearest_idxs. "
        msg += f"Expected {exp_idxs}. Got {res['nearest_idxs']}"
        errors.append(msg)
    if not np.allclose(eval(res['location_on_contigs']), exp_locs):
        msg = "Wrong expected location_on_contigs. "
        msg += f"Expected {exp_locs}. Got {res['location_on_contigs']}"
        errors.append(msg)
    if not np.all(eval(res['contig_segments']) == exp_seqs):
        msg = "Wrong expected contig_segments. "
        msg += f"Expected {exp_seqs}. Got {res['contig_segments']}"
        errors.append(msg)

    remove_dir(args.outdir)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

