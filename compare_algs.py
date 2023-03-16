import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import auc
import itertools
import warnings
import time
import cdt
from sklearn.metrics import precision_score
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
def get_scores(alg_names, networks, ground_truth, sid=True):
    for name, net in zip(alg_names, networks):
        if type(net) == list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            for n,g in zip(net, ground_truth):
                shd += cdt.metrics.SHD(g, n, False)
                sid += cdt.metrics.SID(g, n) if sid else 0
                auc +=  cdt.metrics.precision_recall(g, n)[0]
            print("{} {} {} {}".format(name, shd/len(net), sid/len(net), auc/len(net)))
        elif type(net) != list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            for g in ground_truth:
                shd += cdt.metrics.SHD(g, net, False)
                sid += cdt.metrics.SID(g, net) if sid else 0
                auc +=  cdt.metrics.precision_recall(g, net)[0]
            print("{} {} {} {}".format(name, shd/len(ground_truth), sid/len(ground_truth), auc/len(ground_truth)))
        else:
            shd = cdt.metrics.SHD(ground_truth, net, False)
            sid = cdt.metrics.SID(ground_truth, net) if sid else 0
            auc, pr = cdt.metrics.precision_recall(ground_truth, net)
            print("{} {} {} {}".format(name, shd, sid, auc))

def test_regulondb_subnetwork():
    print("REGULONDB_SUB")
    true_adj = pd.read_csv("./regulondb/ground_skel_subgraph_local.csv", header=None).values
    true_adj = np.abs(true_adj)
    true_adj[true_adj>1]=1
    np.fill_diagonal(true_adj, 0)
    
    clr_network = pd.read_csv("./regulondb/clr_skel_subgraph_local.csv", header=None).to_numpy()
    #sp_gies_network_realdata = pd.read_csv("./regulondb/subgraph_sp-gies-adj_mat.csv", header=0).to_numpy()
    #sp_gies_network_realdata[np.abs(sp_gies_network_realdata) > 0] = 1
    
    sp_gies_network = pd.read_csv("./regulondb/clr_skel_subgraph_local_sp-gies-adj_mat.csv", header=0).to_numpy()
    #sp_gies_network[np.abs(sp_gies_network) > 0] = 1

    sp_gies_network_int = pd.read_csv("./regulondb/clr_skel_subgraph_local_wint_sp-gies-adj_mat.csv", header=0).to_numpy()
    #sp_gies_network_pc[np.abs(sp_gies_network_pc) > 0] = 1

    gies_network = pd.read_csv("./regulondb/subgraph_local_gies-adj_mat.csv", header=0).to_numpy()
    genes = [i[0] for i in pd.read_csv("./regulondb/genes.txt", header=None).values]
    clr_graph = adj_to_dag(clr_network,genes)
    sp_gies_graph = adj_to_dag(sp_gies_network,genes)
    sp_gies_graph_int = adj_to_dag(sp_gies_network_int,genes)
    gies_graph  = adj_to_dag(gies_network, genes)
    #sp_gies_graph_real = adj_to_dag(sp_gies_network_realdata, genes)
    
    true_graph = adj_to_dag(true_adj, genes)
    get_scores(["CLR", "SP-GIES-CLR-SKEL","SP-GIES-CLR-SKEL-INT", "EMPTY"],
               [clr_graph,sp_gies_graph,sp_gies_graph_int,
                np.zeros((len(genes), len(genes)))], true_graph, sid=False)


    # Evaluate the performance of algorithms on the regulondb dataset
def test_regulondb():
    print("REGULONDB")
    true_adj = pd.read_csv("./regulondb/ground_truth.csv", header=None).values
    true_adj = np.abs(true_adj)
    true_adj[true_adj>1]=1
    print(np.max(true_adj), np.min(true_adj))
    genes = [i[0] for i in pd.read_csv("./regulondb/genes.txt", header=None).values]
    true_graph = adj_to_dag(true_adj, all_nodes=genes)

    aracne_network = pd.read_csv("./regulondb/network.txt", sep='\t',header=0)
    clr_network = pd.read_csv("./regulondb/adj_mat.csv", header=None).to_numpy()
    threshold = 6.917 # This achieves 60% precision
    clr_network[np.abs(clr_network) < threshold] = 0
    clr_network[np.abs(clr_network) >= threshold] = 1
    print(np.allclose(clr_network, clr_network.T, rtol=1e-2, atol=1e-2))
    print(np.sum(clr_network))
    true_adj[np.abs(true_adj) >0] = 1
    print(precision_score(true_adj.flatten(), clr_network.flatten()))
    
    sp_gies_network_pc = pd.read_csv("./regulondb/pc_skel_sp-gies-adj_mat.csv", header=0).to_numpy()
    sp_gies_network_pc[np.abs(sp_gies_network_pc) > 0] = 1
    print(np.sum(sp_gies_network_pc))
    
    sp_gies_network_clr = pd.read_csv("./regulondb/clr_skel_sp-gies-adj_mat.csv", header=0).to_numpy()
    sp_gies_network_clr[np.abs(sp_gies_network_clr) > 0] = 1
    print(np.sum(sp_gies_network_clr))
          
    sp_gies_network_aracne = pd.read_csv("./regulondb/aracne_skel_sp-gies-adj_mat.csv", header=0).to_numpy()
    sp_gies_network_aracne[np.abs(sp_gies_network_aracne) > 0] = 1
    print(np.sum(sp_gies_network_aracne))
    
    #igsp_network = pd.read_csv("./regulondb/igsp_adj.csv", header=0).to_numpy()
    #igsp_o_network = pd.read_csv("./regulondb/obs_igsp_adj.csv", header=0).to_numpy()
    #gies_network =  pd.read_csv("./regulondb/gies-adj_mat.csv", header=0).to_numpy()
    # gies_o_network =  pd.read_csv("./regulondb/gies-o-adj_mat.csv", header=0).to_numpy()
    pc_network = pd.read_csv("./regulondb/cupc-adj_mat.csv", header=0).to_numpy()

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

    clr_graph = adj_to_dag(clr_network,genes)

    #pc_network = pc_network[inds][:,inds]
    pc_graph = adj_to_dag(pc_network, genes)
    sp_gies_graph_pc = adj_to_dag(sp_gies_network_pc, genes)
    sp_gies_graph_aracne = adj_to_dag(sp_gies_network_aracne, genes)
    sp_gies_graph_clr = adj_to_dag(sp_gies_network_clr, genes)

    #igsp_graph = adj_to_dag(igsp_network, genes)
    #igsp_o_graph = adj_to_dag(igsp_o_network, genes)
    #gies_graph = adj_to_dag(gies_network, genes)
    # gies_o_graph = adj_to_dag(gies_o_network, genes)

    get_scores([ "PC", "ARACNE", "SP-GIES-PC","SP-GIES-CLR", "SP-GIES-ARACNE", "EMPTY"],
               [pc_graph, aracne_graph, sp_gies_graph_pc, sp_gies_graph_clr, sp_gies_graph_aracne,
                np.zeros((len(genes), len(genes)))], true_graph)

# Evaluate the performance of algorithms on the Dream4 size 10 network 3 dataset
def test_dream4():
    print("DREAM4")
    for d in range(1,6):
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
        igsp_network = pd.read_csv("./insilico_size10_{}/igsp_adj.csv".format(d), header=0).to_numpy()
        igsp_o_network = pd.read_csv("./insilico_size10_{}/obs_igsp_adj.csv".format(d), header=0).to_numpy()

        pc_graph = adj_to_dag(pc_network, nodes)
        sp_gies_graph = adj_to_dag(sp_gies_network, nodes)
        gies_graph = adj_to_dag(gies_network, nodes)
        gies_o_graph = adj_to_dag(gies_o_network, nodes)
        igsp_graph = adj_to_dag(igsp_network, nodes)
        igsp_o_graph = adj_to_dag(igsp_o_network, nodes)
        get_scores([ "PC", "GES", "GIES-OI", "SP-GIES-OI", "GSP", "IGSP-OI", "EMPTY"],
                   [ pc_graph, gies_o_graph, gies_graph, sp_gies_graph, igsp_o_graph,
                    igsp_graph,  np.zeros((len(nodes), len(nodes)))], true_graph)

# Evaluate the performance of algorithms on the random datasets of size 10 nodes(Erdos Renyi, scale-free, and small-world)
# Average over 30 generated graphs for each network type
def test_random():
    num_nodes = 10
    random = [ "ER","scale","small"]
    num_graphs = 1
    for r in random:
        print("RANDOM {}".format(r))
        sp_gies = []
        pc = []
        gies = []
        gies_o = []
        igsp = []
        igsp_o = []
        ground_truth = []
        for n in range(num_graphs):
            edges = pd.read_csv( "./random_test_set_{}_{}/bn_network_{}.csv".format(num_nodes,r, n), header=0)
            df = pd.read_csv("./random_test_set_{}_{}/data_{}.csv".format(num_nodes,r,n), header=0)

            edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
            true_graph = edge_to_dag(edges_pos)
            nodes = list(df.columns)
            nodes.remove('target')
            true_graph.add_nodes_from(nodes)
            df = df.drop('target', axis=1)


            sp_gies_network = pd.read_csv("./random_test_set_{}_{}/{}_sp-gies-adj_mat.csv".format(num_nodes,r,n), header=0).to_numpy()
            gies_network =  pd.read_csv("./random_test_set_{}_{}/{}_gies-adj_mat.csv".format(num_nodes,r,n), header=0).to_numpy()
            gies_o_network = pd.read_csv("./random_test_set_{}_{}/obs_{}_gies-adj_mat.csv".format(num_nodes,r, n), header=0).to_numpy()
            pc_network = pd.read_csv("./random_test_set_{}_{}/obs_{}_cupc-adj_mat.csv".format(num_nodes,r, n), header=0).to_numpy()
            igsp_network = pd.read_csv("./random_test_set_{}_{}/igsp_{}_adj.csv".format(num_nodes,r, n), header=None).to_numpy()
            #igsp_o_network = pd.read_csv("./random_test_set_{}_{}/obs_igsp_{}_adj.csv".format(num_nodes,r, n), header=None).to_numpy()

            sp_gies_graph = adj_to_dag(sp_gies_network, nodes)
            gies_graph = adj_to_dag(gies_network, nodes)
            pc_graph = adj_to_dag(pc_network, nodes)
            gies_o_graph = adj_to_dag(gies_o_network, nodes)
            igsp_graph = adj_to_dag(igsp_network, nodes)
            #igsp_o_graph = adj_to_dag(igsp_o_network, nodes)

            pc.append(pc_graph)
            gies.append(gies_graph)
            sp_gies.append(sp_gies_graph)
            gies_o.append(gies_o_graph)
            igsp.append(igsp_graph)
            #igsp_o.append(igsp_o_graph)
            ground_truth.append(true_graph)
            if n==0:
                pr = cdt.metrics.precision_recall(true_graph, gies_graph)[1]
                plt.plot([p[0] for p in pr], [r[1] for r in pr], label=r, marker='o')
                plt.legend()
                plt.savefig("./random_test_set_{}_{}/gies_oi_pr.png".format(num_nodes,r))

        get_scores(["PC-O", "GES", "GIES-IO", "SP-GIES-IO", "IGSP-IO", "EMPTY"],
                   [pc, gies_o, gies, sp_gies,igsp, np.zeros((num_nodes, num_nodes))], ground_truth)

def test_random_large():
    n = 0
    num_nodes=1000
    folder = "./random_test_set_fewer_1000_500n_scale"
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
    #igsp_network = pd.read_csv("./random_test_set_{}_{}/igsp_{}_adj.csv".format(num_nodes,random, n), header=None).to_numpy()
    #igsp_o_network = pd.read_csv("./random_test_set_{}_{}/obs_igsp_{}_adj.csv".format(num_nodes,random, n), header=None).to_numpy()

    sp_gies_graph = adj_to_dag(sp_gies_network, nodes)
    gies_graph = adj_to_dag(gies_network, nodes)
    pc_graph = adj_to_dag(pc_network, nodes)
    gies_o_graph = adj_to_dag(gies_o_network, nodes)
    #igsp_graph = adj_to_dag(igsp_network, nodes)
    #igsp_o_graph = adj_to_dag(igsp_o_network, nodes)

    get_scores(["PC-O", "GES", "GIES-IO", "SP-GIES-IO", "EMPTY"],
               [pc_graph, gies_o_graph, gies_graph, sp_gies_graph,  np.zeros((num_nodes, num_nodes))], true_graph)

#test_regulondb()
#test_random()
#test_dream4()
test_regulondb_subnetwork()
#test_random_large()
