# %%
import pandas as pd
from sp_gies.sp_gies import sp_gies, run_pc
from sp_gies.utils import get_scores, edge_to_adj
from causaldag import unknown_target_igsp,igsp, gsp
import conditional_independence
import numpy as np
from castle.algorithms import Notears, NotearsNonlinear
import avici
import os
import networkx as nx
import matplotlib.pyplot as plt
import itertools
os.environ['CASTLE_BACKEND'] = 'pytorch'
os.environ['CUDA_VISIBLE_DEVICES']="7"
# Test algorithms on subgraph data
# Linear methods: GIES, PC, SP-GIES
# Nonlinear methods:  AVICI, NOTEARS
# Nonparametric methods: GSP

# %%
def vis_network(adj_mat, pos,map,name):
    graph = nx.DiGraph(adj_mat)
    graph = nx.relabel_nodes(graph, map)
    nx.draw(graph, pos=pos, with_labels=True)
    plt.savefig("{}.png".format(name))

def run_linear_methods(ground_truth):
    outdir = "./data/regulondb/local_graph/"
    data = pd.read_csv(outdir+"data.csv", header=0)
    pc_adj = run_pc(data.to_numpy(), outdir)

    data['target'] = np.zeros(data.shape[0])
    gies_adj = sp_gies(data, outdir, skel=None, pc=False)
    skel = pd.read_csv(outdir+"clr_skel.csv", header=None).to_numpy()
    skel = skel+skel.T
    sp_gies_adj = sp_gies(data, outdir, skel=skel, pc=True)

    get_scores(["PC", "CLR", "GIES", "SP-GIES"], [pc_adj,skel, gies_adj, sp_gies_adj], ground_truth, get_sid=True)
    return  pc_adj,skel, gies_adj, sp_gies_adj

def run_nonparametric_methods(ground_truth):
    outdir = "./data/regulondb/local_graph/"
    data = pd.read_csv(outdir+"data.csv", header=0)
    skel = pd.read_csv(outdir+"clr_skel.csv", header=None).to_numpy()
    skel = skel+skel.T
    fixed_gaps = set()
    fixed_adjacencies = set()
    for i,j in itertools.product(np.arange(skel.shape[0]), np.arange(skel.shape[1])):
        if skel[i,j] == 0:
            fixed_gaps.add(frozenset({i,j}))
        else:
            fixed_adjacencies.add(frozenset({i,j}))
    obs_suffstat = conditional_independence.partial_correlation_suffstat(data)
    alpha = 1e-3
    ci_tester = conditional_independence.MemoizedCI_Tester(conditional_independence.partial_correlation_test, 
                                                           obs_suffstat, alpha=alpha)

    nodes = np.arange(len(data.columns))
    gdag = gsp(nodes,ci_tester,initial_permutations=[np.random.choice(nodes, size=len(nodes), replace=False)],fixed_gaps=fixed_gaps)
    gsp_adj =  edge_to_adj(gdag.arcs, list(gdag.nodes))

    get_scores(["GSP"], [gsp_adj], ground_truth, get_sid=True)
    return gsp_adj

def run_nonlinear_methods(ground_truth):
    outdir = "./data/regulondb/local_graph/"
    data = pd.read_csv(outdir+"data.csv", header=0).to_numpy()

    # NOTEARS NONLINEAR
    nt = NotearsNonlinear()
    nt.learn(data)
    G_notears_nl = nt.causal_matrix

    # NOTEARS                                                                                                          
    nt = Notears()
    nt.learn(data)
    G_notears = nt.causal_matrix


    #AVICI
    model = avici.load_pretrained(download="scm-v0")
    G_avici = model(data)

    get_scores(["NOTEARS", "NOTEARS-MLP", "AVICI"], [G_notears, G_notears_nl, G_avici], ground_truth, get_sid=True)
    return G_notears, G_notears_nl, G_avici
# %%
genes = pd.read_csv("./data/regulondb/local_graph/local_nodes.csv", header=None).to_numpy()
genes = [g[0] for g in genes]
map = dict(zip(np.arange(len(genes)),genes))
global_gt = pd.read_csv("./data/regulondb/ground_truth.csv",header=None).to_numpy().T
gt = nx.DiGraph(global_gt)

genes_global = pd.read_csv("./data/regulondb/genes.txt", header=None).to_numpy()
genes_global = [g[0] for g in genes_global]
genes_inds = np.array([genes_global.index(g) for g in genes])
local_gt = global_gt[genes_inds][:,genes_inds]

gt = nx.DiGraph(local_gt)
gt = nx.relabel_nodes(gt, map)
print(gt)
pos = nx.circular_layout(gt)
vis_network(local_gt, pos, map, 'ground_truth')
graphs = []
graphs += run_linear_methods(local_gt)
graphs += run_nonlinear_methods(local_gt)
graphs.append(run_nonparametric_methods(local_gt))
names = ["pc", "clr", "gies", "sp_gies", "notears", "notears_mlp", "avici", "gsp"]
for g,n in zip(graphs,names):
    vis_network(g, pos,map, n)


#run_nonlinear_methods()

# %%
