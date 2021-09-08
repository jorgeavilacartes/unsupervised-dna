import random; random.seed(42)
from unsupervised_dna import GenerateFCGR

def test_fcgr():
    gen = GenerateFCGR(destination_folder="data/test", kmer=7)
    seq = "".join(random.choice("ACG") for _ in range(300_000))
    gen.from_seq(seq,"data/test/test.jpg")