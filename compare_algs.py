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

def edge_to_adj(edges, all_nodes):
    adj_mat = np.zeros((len(all_nodes), len(all_nodes)))
    for e in edges:
        start = all_nodes.index(e[0])
        end = all_nodes.index(e[1])
        adj_mat[start,end] = 1
    return adj_mat

def edge_to_dag(edges):
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    return dag

def pinna_alg(X_obs, X_int, threshold):
    W_D = np.abs(X_int - X_obs)
    W_ND = W_D/X_obs
    W_ZD = (W_D - np.average(W_D, axis=1))/(np.std(W_D, axis=1))
    W_ZR = np.abs(X_int - np.average(X_int, axis=1))/(np.std(X_int, axis=1))

    #Choose the deviation type
    W = W_ZD
    edges = np.argwhere(W > threshold)

    N = nx.DiGraph()
    for e in edges:
        if e[0] != e[1]: # no self loops
            N.add_edge("G{}".format(e[0]+1),"G{}".format(e[1]+1), weight= W[e[0],e[1]])
    N_prime = nx.condensation(N)
    for start,end, in N_prime.edges():
        all_paths = nx.all_simple_edge_paths(N_prime, start, end)
        exists_longer = False
        for path in all_paths:
            if len(path) >=2 :
                exists_longer=True
        if exists_longer:
            members = nx.get_node_attributes(N_prime, 'members')
            for e in itertools.product(members[start], members[end]):
                try:
                    N.remove_edge(e[0], e[1])
                except:
                    continue
        else:
            N_prime[start][end]['weight'] = np.max(W)
    return N

def get_scores(alg_names, networks, ground_truth):
    for name, net in zip(alg_names, networks):
        shd = cdt.metrics.SHD(ground_truth, net, False)
        sid = cdt.metrics.SID(ground_truth, net)
        auc, pr = cdt.metrics.precision_recall(ground_truth, net)
        print("{} {} {} {}".format(name, shd, sid, auc))


def test_regulondb():
    true_adj = pd.read_csv("./regulondb2/ground_truth.csv", header=None).values
    genes = [i[0] for i in pd.read_csv("./regulondb2/genes.txt", header=None).values]
    true_graph = adj_to_dag(true_adj, all_nodes=genes)

    aracne_network = pd.read_csv("./regulondb2/network.txt", sep='\t',header=0)
    clr_network = pd.read_csv("./regulondb2/adj_mat.csv", header=None).to_numpy()

    # TODO write a script to generate all of these from R
    sp_gies_network = pd.read_csv("./regulondb2/sp-gies-adj_mat.csv", header=0).to_numpy()
    #gies_network =  pd.read_csv("./regulondb2/gies-adj_mat.csv", header=0).to_numpy()
    #gies_o_network =  pd.read_csv("./regulondb2/gies-o-adj_mat.csv", header=0).to_numpy()
    #pc_network = pd.read_csv("./regulondb2/cupc_adj_mat.csv", header=0).to_numpy()

    inds = pd.read_csv("./regulondb2/inds.csv", header=None, sep=",").iloc[0].values

    #  zero out gene->gene and gene-> tf interactions
    with open('./regulondb2/tfs_names.txt') as f:
        lines = f.readlines()
    tfs = lines[0].split("\t")
    edges_pos = [(row['Regulator'][1:-1], row['Target'][1:-1]) for i,row in aracne_network.iterrows() if row['Regulator'][1:-1] in genes and row['Target'][1:-1] in genes]
    #weights = [row['MI'] if (row['Regulator'][1:-1] in tfs) else 0 for i,row in aracne_network.iterrows() if row['Regulator'][1:-1] in genes and row['Target'][1:-1] in genes]
    aracne_graph = edge_to_dag(edges_pos)
    aracne_graph.add_nodes_from(genes)

    threshold = 6.917 # This achieves 60% precision
    clr_network[clr_network < threshold] = 0
    clr_graph = adj_to_dag(clr_network,genes)

    #pc_network = pc_network[inds][:,inds]
    #pc_graph = adj_to_dag(pc_network, genes)
    sp_gies_graph = adj_to_dag(sp_gies_network, genes)
    #gies_graph = adj_to_dag(gies_network, genes)
    #gies_o_graph = adj_to_dag(gies_o_network, genes)

    get_scores(["ARACNE-AP", "CLR",  "SP-GIES-OI", "EMPTY"],
               [aracne_graph, clr_graph, sp_gies_graph,
                np.zeros((len(genes), len(genes)))], true_graph)

def test_random():
    random = [ "small_norm"]
    num_graphs = 1
    for r in random:
        stats_clr = np.zeros(3)
        stats_aracne = np.zeros(3)
        stats_empty = np.zeros(3)
        for n in range(num_graphs):
            edges = pd.read_csv( "./random_test_set_1000_{}/bn_network_{}.csv".format(r, n), header=0)
            aracne_network = pd.read_csv("./random_test_set_1000_{}/output/network.txt".format(r,n), sep='\t',
                                         header=0)
            clr_network = pd.read_csv("./random_test_set_1000_{}/adj_mat_{}.csv".format(r,n), header=None)
            df = pd.read_csv("./random_test_set_1000_{}/data_{}.csv".format(r,n), header=0)

            edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
            true_graph = edge_to_dag(edges_pos, type=DAG)
            nodes = list(df.columns)
            nodes.remove('target')
            true_graph.add_nodes_from(nodes)

            edges_pos = [(row['Regulator'], row['Target']) for i, row in aracne_network.iterrows()]
            weights = [row['MI'] for i, row in aracne_network.iterrows()]
            aracne_graph = edge_to_dag(edges_pos, type=DAG, weights=weights)
            aracne_graph.add_nodes_from(nodes)

            A = clr_network.to_numpy()
            threshold = 1.5
            A[np.abs(A) < threshold] = 0
            clr_graph = adj_to_dag(A, nodes)

            shd = cdt.metrics.SHD(true_graph, aracne_graph, False)
            sid = cdt.metrics.SID(true_graph, aracne_graph)
            auc, pr = cdt.metrics.precision_recall(true_graph, aracne_graph)
            stats_aracne += np.array([shd, sid, auc])

            shd = cdt.metrics.SHD(true_graph, clr_graph, False)
            sid = cdt.metrics.SID(true_graph, clr_graph)
            auc, pr = cdt.metrics.precision_recall(true_graph, clr_graph)
            stats_clr += np.array([shd, sid, auc])

            shd = cdt.metrics.SHD(true_graph, np.zeros((len(nodes), len(nodes))), False)
            sid = cdt.metrics.SID(true_graph, np.zeros((len(nodes), len(nodes))))
            auc, pr = cdt.metrics.precision_recall(true_graph, np.zeros((len(nodes), len(nodes))))
            stats_empty += np.array([shd, sid, auc])

        stats_aracne/=num_graphs
        stats_clr/=num_graphs
        stats_empty/=num_graphs

        print(stats_clr)
        print(stats_aracne)
        print(stats_empty)

def test_dream4():
    for d in range(1,6):
        print("NETWORK {}".format(d))
        edges = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_goldstandard.csv".format(d, d), header=0)
        df = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_obs.csv".format(d, d), header=0)
        df_int = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_inter.csv".format(d, d), header=0)
        # observational data only
        aracne_network = pd.read_csv("./insilico_size10_{}/output/network.txt".format(d), sep='\t',header=0)
        clr_network = pd.read_csv("./insilico_size10_{}/adj_mat.csv".format(d), header=None)
        df_int_ko = pd.read_csv("./insilico_size10_{}_inter_ko.csv".format(d, d), header=0)
        df_wt = pd.read_csv("./insilico_size10_{}_wt.csv".format(d, d), header=0)


test_regulondb()
#test_random()
#test_dream4()
