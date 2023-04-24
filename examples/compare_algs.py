import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import warnings
import time

from causal_learning.utils import get_scores, edge_to_dag, edge_to_adj, adj_to_dag
warnings.filterwarnings("ignore")
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
import matplotlib
matplotlib.rc('font', **font)

# Script to compare PC, GES-O (observational only), GIES-IO (observational + interventional joint), and SP-GIES-IO algorhtms
# on random, dream4 and regulondb datasets


# Evaluate the performance of algorithms on the regulondb dataset
def test_regulondb():
    print("REGULONDB")
    true_adj = pd.read_csv("../data/regulondb/ground_truth.csv", header=None).values
    genes = [i[0] for i in pd.read_csv("../data/regulondb/genes.txt", header=None).values]
    true_graph = adj_to_dag(true_adj, all_nodes=genes)

    aracne_network = pd.read_csv("../data/regulondb/network.txt", sep='\t', header=0)
    clr_network = pd.read_csv("../data/regulondb/adj_mat.csv", header=None).to_numpy()
    sp_gies_network = pd.read_csv("../data/regulondb/clr_skel_sp-gies-adj_mat.csv", header=0).to_numpy()
    gies_network =  pd.read_csv("../data/regulondb/md_10_gies-adj_mat.csv", header=0).to_numpy()
    gies_o_network =  pd.read_csv("../data/regulondb/obs_gies-adj_mat.csv", header=0).to_numpy()
    pc_network = pd.read_csv("../data/regulondb/cupc-adj_mat.csv", header=0).to_numpy()

    #  zero out gene->gene and gene-> tf interactions
    with open('../data/regulondb/tfs.txt') as f:
        lines = f.readlines()
    tfs = lines[0].split("\t")
    edges_pos = [(row['Regulator'][1:-1], row['Target'][1:-1])
                 for i,row in aracne_network.iterrows() if row['Regulator'][1:-1] in
                 genes and row['Target'][1:-1] in genes]
    aracne_graph = edge_to_dag(edges_pos)
    aracne_graph.add_nodes_from(genes)

    threshold = 6.917 # This achieves 60% precision
    clr_network[clr_network < threshold] = 0
    clr_graph = adj_to_dag(clr_network,genes)

    pc_graph = adj_to_dag(pc_network, genes)
    sp_gies_graph = adj_to_dag(sp_gies_network, genes)
    gies_graph = adj_to_dag(gies_network, genes)
    gies_o_graph = adj_to_dag(gies_o_network, genes)

    get_scores(["ARACNE-AP", "CLR",  "SP-GIES-OI", "GIES-OI", "GES-O", "NULL"],
               [aracne_graph, clr_graph, sp_gies_graph, gies_graph, gies_o_graph, 
                np.zeros((len(genes), len(genes)))], true_graph)

# Evaluate the performance of algorithms on the Dream4 size 10 network 3 dataset
def test_dream4():
    print("DREAM4")
    d=3
    print("NETWORK {}".format(d))
    edges = pd.read_csv("../data/insilico_size10_{}/insilico_size10_{}_goldstandard.csv".format(d, d), header=0)
    df = pd.read_csv("../data/insilico_size10_{}/insilico_size10_{}_obs.csv".format(d, d), header=0)

    edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
    true_graph = edge_to_dag(edges_pos)
    nodes = list(df.columns)
    nodes.remove('target')
    true_graph.add_nodes_from(nodes)

    sp_gies_network = pd.read_csv("../data/insilico_size10_{}/sp-gies-adj_mat.csv".format(d), header=0).to_numpy()
    gies_network =  pd.read_csv("../data/insilico_size10_{}/gies-adj_mat.csv".format(d), header=0).to_numpy()
    gies_o_network =  pd.read_csv("../data/insilico_size10_{}/obs_gies-adj_mat.csv".format(d), header=0).to_numpy()
    pc_network = pd.read_csv("../data/insilico_size10_{}/pc_skel_cupc-adj_mat.csv".format(d), header=0).to_numpy()
    igsp_network = pd.read_csv("../data/insilico_size10_{}/igsp_adj.csv".format(d), header=0).to_numpy()

    pc_graph = adj_to_dag(pc_network, nodes)
    sp_gies_graph = adj_to_dag(sp_gies_network, nodes)
    gies_graph = adj_to_dag(gies_network, nodes)
    gies_o_graph = adj_to_dag(gies_o_network, nodes)
    igsp_graph = adj_to_dag(igsp_network, nodes)
    
    get_scores(["PC-O", "GES-O", "GIES-OI","SP-GIES-OI", "IGSP-OI", "NULL"],
               [pc_graph, gies_o_graph, gies_graph, sp_gies_graph, igsp_graph, np.zeros((len(nodes), len(nodes)))], true_graph)

# Evaluate the performance of algorithms on the random datasets of size 10 nodes(Erdos Renyi, scale-free, and small-world)
# Average over 30 generated graphs for each network type
# Results will differ from paper since SP-GIES uses PC skeleton here but in paper we use CLR/ARACNE
def test_random():
    num_nodes = 10
    random = [ "ER","scale","small"]
    num_graphs = 30
    for r in random:
        print("RANDOM {}".format(r))
        sp_gies = []
        pc = []
        gies = []
        gies_o = []
        igsp = []
        ground_truth = []
        for n in range(num_graphs):
            edges = pd.read_csv( "../data/random_test_set_{}_{}/bn_network_{}.csv".format(num_nodes,r, n), header=0)
            df = pd.read_csv("../data/random_test_set_{}_{}/data_{}.csv".format(num_nodes,r,n), header=0)

            edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
            true_graph = edge_to_dag(edges_pos)
            nodes = list(df.columns)
            nodes.remove('target')
            true_graph.add_nodes_from(nodes)
            df = df.drop('target', axis=1)


            sp_gies_network = pd.read_csv("../data/random_test_set_{}_{}/{}_sp-gies-adj_mat.csv".format(num_nodes,r,n), header=0).to_numpy()
            gies_network =  pd.read_csv("../data/random_test_set_{}_{}/{}_gies-adj_mat.csv".format(num_nodes,r,n), header=0).to_numpy()
            gies_o_network = pd.read_csv("../data/random_test_set_{}_{}/obs_{}_gies-adj_mat.csv".format(num_nodes,r, n), header=0).to_numpy()
            pc_network = pd.read_csv("../data/random_test_set_{}_{}/{}_cupc-adj_mat.csv".format(num_nodes,r, n), header=0).to_numpy()
            igsp_network = pd.read_csv("../data/random_test_set_{}_{}/igsp_{}_adj.csv".format(num_nodes,r, n), header=None).to_numpy()

            sp_gies_graph = adj_to_dag(sp_gies_network, nodes)
            gies_graph = adj_to_dag(gies_network, nodes)
            pc_graph = adj_to_dag(pc_network, nodes)
            gies_o_graph = adj_to_dag(gies_o_network, nodes)
            igsp_graph = adj_to_dag(igsp_network, nodes)
            
            pc.append(pc_graph)
            gies.append(gies_graph)
            sp_gies.append(sp_gies_graph)
            gies_o.append(gies_o_graph)
            igsp.append(igsp_graph)
            ground_truth.append(true_graph)

        get_scores(["PC-O", "GES-O", "GIES-OI", "SP-GIES-OI","IGSP-OI", "NULL"],
                   [pc, gies_o, gies, sp_gies, igsp, np.zeros((num_nodes, num_nodes))], ground_truth)

def test_random_large():
    n = 0
    num_nodes=1000
    folder = "../data/random_test_set_scaling_1000_small"
    edges = pd.read_csv( "{}/bn_network_{}.csv".format(folder, n), header=0)
    df = pd.read_csv("{}/data_{}.csv".format(folder,n), header=0)

    edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
    true_graph = edge_to_dag(edges_pos)
    nodes = list(df.columns)
    nodes.remove('target')
    true_graph.add_nodes_from(nodes)
    df = df.drop('target', axis=1)
    n=0
    sp_gies_network = pd.read_csv("{}/{}_sp-gies-adj_mat.csv".format(folder,n), header=0).to_numpy()
    gies_network =  pd.read_csv("{}/{}_gies-adj_mat.csv".format(folder,n), header=0).to_numpy()
    gies_o_network = pd.read_csv("{}/obs_{}_gies-adj_mat.csv".format(folder, n), header=0).to_numpy()
    pc_network = pd.read_csv("{}/{}_cupc-adj_mat.csv".format(folder, n), header=0).to_numpy()

    sp_gies_graph = adj_to_dag(sp_gies_network, nodes)
    gies_graph = adj_to_dag(gies_network, nodes)
    pc_graph = adj_to_dag(pc_network, nodes)
    gies_o_graph = adj_to_dag(gies_o_network, nodes)

    get_scores(["PC-O", "GES-O", "GIES-OI", "SP-GIES-OI", "NULL"],
               [pc_graph, gies_o_graph, gies_graph, sp_gies_graph,  np.zeros((num_nodes, num_nodes))], true_graph)
    
test_regulondb()
test_random()
test_dream4()
test_random_large()
