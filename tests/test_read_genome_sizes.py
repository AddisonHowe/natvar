"""Test for entrypoint read-genome-sizes

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
        "infile, outdir, genome_lengths_exp, contig_lengths_exp", [
    ["test_assembly1_genomes.txt", "out1", 
     [3, 3], [[8, 8, 8], [2, 8, 4]]],
    ["test_assembly2_genomes.txt", "out2", 
     [3, 3, 4], [[8, 8, 8], [2, 8, 4], [8, 8, 8, 4]]],
])
def test_query_genome(
        infile, outdir, genome_lengths_exp, contig_lengths_exp
):
    from natvar.read_genome_sizes import parse_args, main
    
    argstring = f"-i {DATDIR}/{infile} -o {outdir} -v 1"
    arglist = argstring.split(" ")
    args = parse_args(arglist)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    
    main(args)

    def read_output(outdir):
        genome_lengths = np.load(f"{outdir}/genome_lengths.npy")
        contig_lengths = np.load(f"{outdir}/contig_lengths.npy", 
                                 allow_pickle=True)
        return genome_lengths, contig_lengths

    genome_lengths, contig_lengths = read_output(args.outdir)

    errors = []
    for i in range(len(genome_lengths)):
        if not np.allclose(genome_lengths[i], genome_lengths_exp[i]):
            msg = f"Wrong genome lengths (item {i}). "
            msg += f"Expected {genome_lengths_exp[i]}. Got {genome_lengths[i]}"
            errors.append(msg)
        if not np.allclose(contig_lengths[i], contig_lengths_exp[i]):
            msg = f"Wrong contig lengths (item {i}). "
            msg += f"Expected {contig_lengths_exp[i]}. Got {contig_lengths[i]}"
            errors.append(msg)

    remove_dir(args.outdir)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
