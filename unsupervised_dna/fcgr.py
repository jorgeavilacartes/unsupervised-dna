from pathlib import Path
from Bio import SeqIO
from complexcgr import FCGR

from . import (
    MonitorValues, 
    MimicSequence,
)

#TODO: allow transversions and transitions
class GenerateFCGR: 

    def __init__(self, destination_folder: Path, kmer: int): 
        self.destination_folder = Path(destination_folder)
        self.kmer = kmer
        self.fcgr = FCGR(kmer)
        self.counter = 0 # count number of time a sequence is converted to fcgr
        
        # Monitor Values
        self.mv = MonitorValues(["id_seq","path_save","fasta","len_seq",
                                "count_A","count_C","count_G","count_T"])

        # Create destination folder if needed
        self.destination_folder.mkdir(exist_ok=True)


    def from_fasta(self, path: Path):
        "FCGR for all sequences of a fasta file"
        # load fasta file
        fasta = self.load_fasta(path)
        
        # for each sequence save the FCGR
        for j,record in enumerate(fasta):
            if j < 10:
                # get basic information
                seq     = record.seq
                id_seq  = record.id.replace("/","_")
                len_seq = len(seq)
                count_A = seq.count("A")
                count_C = seq.count("C")
                count_G = seq.count("G")
                count_T = seq.count("T")
                
                # Generate and save FCGR for the current sequence
                path_save = self.destination_folder.joinpath("{}-{}".format(str(self.counter).zfill(5)), id_seq)
                self.from_seq(record.seq, path_save)
                self.mv()

        # save metadata
        self.mv.to_csv(self.destination_folder.joinpath("fcgr-metadata.csv"))

    def from_seq(self, seq: str, path_save):
        "Get FCGR from a sequence"
        seq = self.preprocessing(seq)
        chaos = self.fcgr(seq)
        self.fcgr.save(chaos, path_save)
        self.counter +=1

    @staticmethod
    def preprocessing(seq):
        seq = seq.upper()
        for letter in "BDEFHIJKLMOPQRSUVWXYZ":
            seq = seq.replace(letter,"N")
        return seq

    @staticmethod
    def load_fasta(path: Path):
        return SeqIO.parse(path, "fasta")

class GeneratePerturbationFCGR(GenerateFCGR):

    def __init__(self, destination_folder: Path, kmer: int):
        super().__init__(destination_folder, kmer)

    