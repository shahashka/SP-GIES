import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import auc
import itertools
import warnings
import time
import cdt
from cdt.causality.graph import GIES
warnings.filterwarnings("ignore")
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
import matplotlib
matplotlib.rc('font', **font)

# Script to compare PC, GIES-O (observational only), GIES-IO (observational + interventional joint), and SP-GIES-IO algorhtms
# on random, dream4 and regulondb datasets


# Helper function to convert an adjacency matrix into a Networkx Digraph
def adj_to_dag(adj, all_nodes,fixed_edges=None):
    dag = nx.DiGraph()
    dag.add_nodes_from(all_nodes)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if np.abs(adj[i,j]) > 0:
                dag.add_edge(all_nodes[i], all_nodes[j], weight=np.abs(adj[i,j]))
    if fixed_edges:
        dag.add_edges_from(fixed_edges)
    return dag

# Helper function to convert a list of edges into an adjacency matrix
def edge_to_adj(edges, all_nodes):
    adj_mat = np.zeros((len(all_nodes), len(all_nodes)))
    for e in edges:
        start = all_nodes.index(e[0])
        end = all_nodes.index(e[1])
        adj_mat[start,end] = 1
    return adj_mat

# Helper function to convert a list of edges into a Networkx Digraph
def edge_to_dag(edges):
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    return dag

# Helper function to print the SHD, SID, AUC for a set of algorithms and networks
# Also handles averaging over several sets of networks (e.g the random comparison averages over 30 different generated graphs)
def get_scores(alg_names, networks, ground_truth):
    for name, net in zip(alg_names, networks):
        if type(net) == list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            for n,g in zip(net, ground_truth):
                shd += cdt.metrics.SHD(g, n, False)
                sid +=cdt.metrics.SID(g, n)
                auc +=  cdt.metrics.precision_recall(g, n)[0]
            print("{} {} {} {}".format(name, shd/len(net), sid/len(net), auc/len(net)))
        elif type(net) != list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            for g in ground_truth:
                shd += cdt.metrics.SHD(g, net, False)
                sid +=cdt.metrics.SID(g, net)
                auc +=  cdt.metrics.precision_recall(g, net)[0]
            print("{} {} {} {}".format(name, shd/len(ground_truth), sid/len(ground_truth), auc/len(ground_truth)))
        else:
            shd = cdt.metrics.SHD(ground_truth, net, False)
            sid = cdt.metrics.SID(ground_truth, net)
            auc, pr = cdt.metrics.precision_recall(ground_truth, net)
            print("{} {} {} {}".format(name, shd, sid, auc))


# Evaluate the performance of algorithms on the regulondb dataset
def test_regulondb():
    print("REGULONDB")
    true_adj = pd.read_csv("./regulondb/ground_truth.csv", header=None).values
    genes = [i[0] for i in pd.read_csv("./regulondb/genes.txt", header=None).values]
    true_graph = adj_to_dag(true_adj, all_nodes=genes)

    aracne_network = pd.read_csv("./regulondb/network.txt", sep='\t',header=0)
    clr_network = pd.read_csv("./regulondb/adj_mat.csv", header=None).to_numpy()
    sp_gies_network = pd.read_csv("./regulondb/sp-gies-adj_mat.csv", header=0).to_numpy()
    #gies_network =  pd.read_csv("./regulondb/gies-adj_mat.csv", header=0).to_numpy()
    # gies_o_network =  pd.read_csv("./regulondb/gies-o-adj_mat.csv", header=0).to_numpy()
    #pc_network = pd.read_csv("./regulondb/cupc_adj_mat.csv", header=0).to_numpy()

    inds = pd.read_csv("./regulondb/inds.csv", header=None, sep=",").iloc[0].values

    #  zero out gene->gene and gene-> tf interactions
    with open('./regulondb/tfs.txt') as f:
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

    #pc_network = pc_network[inds][:,inds]
    #pc_graph = adj_to_dag(pc_network, genes)
    sp_gies_graph = adj_to_dag(sp_gies_network, genes)
    #gies_graph = adj_to_dag(gies_network, genes)
    # gies_o_graph = adj_to_dag(gies_o_network, genes)

    get_scores(["ARACNE-AP", "CLR",  "SP-GIES-OI", "EMPTY"],
               [aracne_graph, clr_graph, sp_gies_graph,
                np.zeros((len(genes), len(genes)))], true_graph)

# Evaluate the performance of algorithms on the Dream4 size 10 network 3 dataset
def test_dream4():
    print("DREAM4")
    d=3
    print("NETWORK {}".format(d))
    edges = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_goldstandard.csv".format(d, d), header=0)
    df = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_obs.csv".format(d, d), header=0)

    edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
    true_graph = edge_to_dag(edges_pos)
    nodes = list(df.columns)
    nodes.remove('target')
    true_graph.add_nodes_from(nodes)

    sp_gies_network = pd.read_csv("./insilico_size10_{}/sp-gies-adj_mat.csv".format(d), header=0).to_numpy()
    gies_network =  pd.read_csv("./insilico_size10_{}/gies-adj_mat.csv".format(d), header=0).to_numpy()
    gies_o_network =  pd.read_csv("./insilico_size10_{}/obs_gies-adj_mat.csv".format(d), header=0).to_numpy()
    pc_network = pd.read_csv("./insilico_size10_{}/obs_cupc-adj_mat.csv".format(d), header=0).to_numpy()

    pc_graph = adj_to_dag(pc_network, nodes)
    sp_gies_graph = adj_to_dag(sp_gies_network, nodes)
    gies_graph = adj_to_dag(gies_network, nodes)
    gies_o_graph = adj_to_dag(gies_o_network, nodes)

    get_scores(["PC-O", "GIES-O", "GIES-OI","SP-GIES-OI", "EMPTY"],
               [pc_graph, gies_o_graph, gies_graph, sp_gies_graph, np.zeros((len(nodes), len(nodes)))], true_graph)

# Evaluate the performance of algorithms on the random datasets of size 10 nodes(Erdos Renyi, scale-free, and small-world)
# Average over 30 generated graphs for each network type
def test_random():
    num_nodes = 10
    random = [ "ER", "scale", "small"]
    num_graphs = 30
    for r in random:
        print("RANDOM {}".format(r))
        sp_gies = []
        pc = []
        gies = []
        gies_o = []
        ground_truth = []
        for n in range(num_graphs):
            edges = pd.read_csv( "./random_test_set_{}_{}/bn_network_{}.csv".format(num_nodes, r, n), header=0)
            df = pd.read_csv("./random_test_set_{}_{}/data_{}.csv".format(num_nodes, r,n), header=0)

            edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
            true_graph = edge_to_dag(edges_pos)
            nodes = list(df.columns)
            nodes.remove('target')
            true_graph.add_nodes_from(nodes)

            sp_gies_network = pd.read_csv("./random_test_set_{}_{}/sp-gies-adj_mat.csv".format(num_nodes,r,n), header=0).to_numpy()
            gies_network =  pd.read_csv("./random_test_set_{}_{}/gies-adj_mat.csv".format(num_nodes,r,n), header=0).to_numpy()
            gies_o_network =  pd.read_csv("./random_test_set_{}_{}/obs_gies-adj_mat.csv".format(num_nodes, r, n), header=0).to_numpy()
            pc_network = pd.read_csv("./random_test_set_{}_{}/obs_cupc-adj_mat.csv".format(num_nodes, r, n), header=0).to_numpy()
            sp_gies.append(sp_gies_network)
            pc.append(pc_network)
            gies.append(gies_network)
            gies_o.append(gies_o_network)
            ground_truth.append(true_graph)

        get_scores(["PC-O", "GIES-O", "GIES-OI", "SP-GIES-OI", "EMPTY"],
                   [pc, gies_o, gies, sp_gies, np.zeros((num_nodes, num_nodes))], ground_truth)


#test_regulondb()
#test_random()
test_dream4()
