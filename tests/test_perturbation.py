import random 
from unsupervised_dna import MimicSequence

def test_mimicsequence():
    # test for default config of mimic sequences
    ms = MimicSequence()
    seq = [random.choice("ACGT") for _ in range(100_000)]
    mimic_seq = ms(seq)
    assert mimic_seq != seq, "sequences has not change"