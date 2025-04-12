"""Tests for IO and file processing

"""

import pytest
import numpy as np

from .conftest import DATDIR
from natvar.io import process_contig_file 
from natvar.helpers import gene_seq_to_array

@pytest.mark.parametrize("fpath, seqs_exp", [
    [f"{DATDIR}/assembly_set1/genome1.contigs.fa", 
     ["AAAACCCC", "GGGGTTTT", "ACGTACGT"]],
    [f"{DATDIR}/assembly_set1/genome2.contigs.fa.gz", 
     ["AA", "AAAACCCC", "ACGT"]],
])
class TestProcessContigFile:

    def test_types(self, fpath, seqs_exp):
        seqs = process_contig_file(fpath)
        errors = []
        if not isinstance(seqs, list):
            msg = f"Expected list. Got {type(seqs)}"
            errors.append(msg)
        if not isinstance(seqs[0], np.ndarray):
            msg = f"Expected first element np.ndarray. Got {type(seqs[0])}"
            errors.append(msg)
        if not isinstance(seqs[0][0], np.uint8):
            msg = f"Expected array type np.uint8. Got {type(seqs[0][0])}"
            errors.append(msg)
        
    def test_values(self, fpath, seqs_exp):
        seqs = process_contig_file(fpath)
        seqs_exp = [gene_seq_to_array(s) for s in seqs_exp]
        errors = []
        if len(seqs) != len(seqs_exp):
            msg = "Read wrong number of sequences. Expected {}. Got {}.".format(
                len(seqs_exp), len(seqs)
            )
            errors.append(msg)
            return    
        for s, se in zip(seqs, seqs_exp):
            if not np.allclose(s, se):
                msg = f"Expected:\n{se}.\nGot:\n{s}"
                errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


