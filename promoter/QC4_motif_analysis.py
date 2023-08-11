import numpy as np
import pandas as pd
import pickle
import os
import pdb
from tqdm import tqdm
import argparse

import scipy.stats as stats
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from joblib import Parallel, delayed

from gimmemotifs.motif import read_motifs, Motif
from gimmemotifs.scanner import Scanner, scan_to_file
from gimmemotifs.fasta import Fasta
from gimmemotifs.motif.denovo import gimme_motifs

os.environ["XDG_CACHE_HOME"] = "/clusterfs/nilah/aniketh/gimmemotifs"

np.random.seed(97)

ap = argparse.ArgumentParser()
ap.add_argument("fasta_path", help="path to fasta file containing sequences")
ap.add_argument("denovo_output_dir", help="dir to output denovo motif analysis")
ap.add_argument("known_analysis_output_path", help="path to output known motif analysis")
ap.add_argument("pfm_file_path", help="path to PFM file containing known motifs")

args = ap.parse_args()

print(args)

# known motif analysis
scan_to_file(args.fasta_path, args.pfm_file_path, filepath_or_buffer=args.known_analysis_output_path, nreport=1000, fpr=0.01, cutoff=None, bed=False, scan_rc=False, table=False, score_table=False, bgfile=None, genome="hg38", ncpus=32, zscore=True, gcnorm=True, random_state=np.random.RandomState(97), progress=True)

# denovo motif analysis
params = {"genome": "hg38"}
motifs = gimme_motifs(args.fasta_path, args.denovo_output_dir, params=params)