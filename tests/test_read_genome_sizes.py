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
    # Assembly 1 Genome 1
    ["test_assembly1_genomes.txt", "out1", [3, 3], [[8, 8, 8], [2, 8, 4]]],
    ["test_assembly2_genomes.txt", "out2", [3, 3], [[8, 8, 8], [2, 8, 4]]],
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
    if not np.allclose(genome_lengths, genome_lengths_exp):
        msg = f"Wrong genome lengths. "
        msg += f"Expected {genome_lengths_exp}. Got {genome_lengths}"
        errors.append(msg)
    if not np.allclose(contig_lengths, contig_lengths_exp):
        msg = f"Wrong contig lengths. "
        msg += f"Expected {contig_lengths_exp}. Got {contig_lengths}"
        errors.append(msg)

    remove_dir(args.outdir)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
