from unsupervised_dna import (
    GenerateFCGR,
    GeneratePerturbatedFCGR,
)

KMER = 8
fcgr = GenerateFCGR(destination_folder="data/fcgr",kmer=KMER)
fcgr.from_fasta(path="data/not-aligned/sequences_fasta_2021_07_16/sequences.fasta")
