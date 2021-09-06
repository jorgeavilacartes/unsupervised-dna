from collections import namedtuple, Counter
from pathlib import Path
import random
random.seed(42)

from Bio import SeqIO
from PIL import Image
import numpy as np 
import pandas as pd

# Chaos Game Representation
from complexcgr import FCGR

# Save FCGR generated
BASE = Path().joinpath("data") 
BASE.mkdir(exist_ok=True)

# Define k-mer -> (2**k, 2**k) array
k = 8
fcgr = FCGR(k)

# Load fasta file
path_fasta = "data/not-aligned/sequences_fasta_2021_07_16/sequences.fasta"
fasta = SeqIO.parse(path_fasta , "fasta")

# Generate all FCGR
BASE_CHARS = {"A","C","G","T"}
metadata = namedtuple("metadata",["id","count_by_char","len_seq","not_ACGT","total_not_ACGT","path"])
i = 1
summary = []
while i < 402: 
    print(i)
    record = next(fasta) 
    
    seq = record.seq.upper() 
    
    ## Metadata
    count_chars = dict(Counter(seq))
    not_ACGT = [c for c in count_chars if c not in BASE_CHARS]
    total_not_ACGT = sum([count_chars.get(c,0) for c in not_ACGT])
    
    ## fcgr
    # clean sequence
    for letter in "BDEFHIJKLMOPQRSUVWXYZ":
        seq = seq.replace(letter,"N")
    
    # compute FCGR
    chaos = fcgr(seq)
    
    # save image
    path_npy = BASE.joinpath("{}.npy".format(record.id.replace("/","_")))
    path_jpg = BASE.joinpath("{}.jpg".format(record.id.replace("/","_")))
    #np.save(path_npy, chaos)
    fcgr.save(chaos, path_jpg)
    
    # consolidate metadata with path 
    meta = metadata(record.id, count_chars, len(record.seq), ";".join(not_ACGT), total_not_ACGT, str(path_jpg))
    summary.append(meta)
    i+=1

df_summary = pd.DataFrame(summary)
df_summary["% not_ACGT"] = df_summary.apply(lambda row: np.round((row["total_not_ACGT"]/row["len_seq"]*100),2), axis=1)
df_summary.sort_values(by="% not_ACGT", ascending=False).to_csv("metadata-fcgr.csv")