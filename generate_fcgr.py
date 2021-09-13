from unsupervised_dna import (
    GenerateFCGR,
    GeneratePerturbatedFCGR,
)

KMER = 7
fcgr = GenerateFCGR(destination_folder=f"data/fcgr-{KMER}-mer",kmer=KMER)
fcgr.from_fasta(path="data/not-aligned/sequences_fasta_2021_07_16/sequences.fasta", max_seq=20000)
