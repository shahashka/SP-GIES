import pandas as pd
from sp_gies.sp_gies import sp_gies, run_pc
from sp_gies.utils import get_scores
import numpy as np
from castle.algorithms import Notears, NotearsNonlinear
import avici
# Test algorithms on subgraph data
# Linear methods: GIES, PC, SP-GIES
# Nonlinear methods:  AVICI, NOTEARS

def run_linear_methods():
    outdir = "./data/regulondb/local_graph/"
    data = pd.read_csv(outdir+"data.csv", header=0)
    ground_truth = pd.read_csv(outdir+"ground_truth.csv",header=None).to_numpy()
    pc_adj = run_pc(data.to_numpy(), outdir)

    data['target'] = np.zeros(data.shape[0])
    gies_adj = sp_gies(data, outdir, skel=None, pc=False)
    sp_gies_adj = sp_gies(data, outdir, skel=None, pc=True)

    get_scores(["PC", "GIES", "SP-GIES"], [pc_adj, gies_adj, sp_gies_adj], ground_truth, get_sid=True)

def run_nonlinear_methods():
    outdir = "./data/regulondb/local_graph/"
    data = pd.read_csv(outdir+"data.csv", header=0).to_numpy()
    ground_truth = pd.read_csv(outdir+"ground_truth.csv",header=None).to_numpy()

    # NOTEARS
    nt = Notears()
    nt.learn(data)
    G_notears = nt.causal_matrix

    # NOTEARS NONLINEAR
    nt = NotearsNonlinear()
    nt.learn(data)
    G_notears_nl = nt.causal_matrix

    #AVICI
    model = avici.load_pretrained(download="scm-v0")
    G_avici = model(data)

    get_scores(["NOTEARS", "NOTEARS-MLP", "AVICI"], [G_notears, G_notears_nl, G_avici], ground_truth, get_sid=True)
run_linear_methods()
run_nonlinear_methods()
