import logging
logging.basicConfig(filename='data/fcgr.log',  level=logging.INFO, filemode="w")

from tqdm import tqdm
import random ; random.seed(42)
from pathlib import Path
from Bio import SeqIO
from complexcgr import FCGR

from . import (
    MonitorValues, 
    MimicSequence,
)

#TODO: tqdm for FCGR generation
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


    def from_fasta(self, path: Path, max_seq: int = 1000):
        "FCGR for all sequences of a fasta file"
        self.reset_counter()
        # load fasta file
        fasta = self.load_fasta(path)
        
        max_id = len(str(max_seq))
        # for each sequence save the FCGR
        for record in tqdm(fasta, desc="Generating FCGR", total=max_seq):
            if self.counter < max_seq:
                # get basic information
                seq     = record.seq
                id_seq  = record.id.replace("/","_")
                len_seq = len(seq)
                count_A = seq.count("A")
                count_C = seq.count("C")
                count_G = seq.count("G")
                count_T = seq.count("T")
                
                # Generate and save FCGR for the current sequence
                path_save = self.destination_folder.joinpath("{}-{}.jpg".format(str(self.counter).zfill(max_id), id_seq))
                self.from_seq(record.seq, path_save)
                self.mv()
                
            else:
                break

        # save metadata
        self.mv.to_csv(self.destination_folder.joinpath("fcgr-metadata.csv"))
        
    def from_seq(self, seq: str, path_save):
        "Get FCGR from a sequence"
        if not Path(path_save).is_file():
            seq = self.preprocessing(seq)
            chaos = self.fcgr(seq)
            self.fcgr.save(chaos, path_save)
        self.counter +=1

    def reset_counter(self,):
        self.counter=0
        
    @staticmethod
    def preprocessing(seq):
        seq = seq.upper()
        for letter in "BDEFHIJKLMOPQRSUVWXYZ":
            seq = seq.replace(letter,"N")
        return seq

    @staticmethod
    def load_fasta(path: Path):
        return SeqIO.parse(path, "fasta")

class GeneratePerturbatedFCGR(GenerateFCGR):

    def __init__(self, destination_folder: Path, kmer: int, 
                    p_transition: float = 1-10**-4, p_transversion: float = 1-0.5*10**-4,
                    n_transition: int = 3, n_transversion: int = 3, n_both = 3):
        super().__init__(destination_folder, kmer)
        self.n_transition = n_transition
        self.n_transversion = n_transversion
        self.n_both = n_both
        self.ms = MimicSequence(p_transition, p_transversion)
        self.mv = MonitorValues(["id_seq","path_save","fasta","perturbation"])

    def get_perturbations(self, seq, id_seq): 

        for _ in range(self.n_transition):
            perturbation = "transition"
            m_seq = self.ms(seq, transition=True, transversion=False)
            path_save = self.destination_folder.joinpath("{}-{}-{}".format(perturbation, str(self.counter).zfill(5)), id_seq)
            self.from_seq(seq, path_save)
            self.mv()
        
        for _ in range(self.n_transversion):
            perturbation = "transversion"
            m_seq = self.ms(seq, transition=False, transversion=True)
            path_save = self.destination_folder.joinpath("{}-{}-{}".format(perturbation, str(self.counter).zfill(5)), id_seq)
            self.from_seq(seq, path_save)
            self.mv()

        for _ in range(self.n_both):
            perturbation = "transition-transversion"
            m_seq = self.ms(seq, transition=True, transversion=True)
            path_save = self.destination_folder.joinpath("{}-{}-{}".format(perturbation, str(self.counter).zfill(5)), id_seq)
            self.from_seq(seq, path_save)
            self.mv()

    def from_fasta(self, path: Path):
        "FCGR for all sequences of a fasta file"
        # load fasta file
        fasta = self.load_fasta(path)
        
        # for each sequence save the FCGR
        for j,record in enumerate(fasta):

            # get basic information
            seq     = record.seq
            id_seq  = record.id.replace("/","_")
            len_seq = len(seq)
            count_A = seq.count("A")
            count_C = seq.count("C")
            count_G = seq.count("G")
            count_T = seq.count("T")
            
            # Generate and save FCGR for all perturbations
            self.get_perturbations(seq, id_seq)
                
        # save metadata
        self.mv.to_csv(self.destination_folder.joinpath("fcgr-mimicsequences-metadata.csv"))