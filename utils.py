import networkx as nx
import numpy as np
import itertools
import cdt
from sklearn.metrics import auc

def adj_to_edge(adj, nodes):
    edges = []
    for (row,col) in itertools.product(np.arange(adj.shape[0]), np.arange(adj.shape[1])):
        if adj[row,col] != 0:
            edges.append((nodes[row], nodes[col], {'weight' :adj[row,col]}))
    return edges

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
# Default turn off sid, since it is computationally expensive
def get_scores(alg_names, networks, ground_truth, get_sid=False):
    for name, net in zip(alg_names, networks):
        if type(net) == list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            for n,g in zip(net, ground_truth):
                shd += cdt.metrics.SHD(g, n, False)
                sid += cdt.metrics.SID(g, n) if get_sid else 0
                auc +=  cdt.metrics.precision_recall(g, n)[0]
            print("{} {} {} {}".format(name, shd/len(net), sid/len(net), auc/len(net)))
        elif type(net) != list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            for g in ground_truth:
                shd += cdt.metrics.SHD(g, net, False)
                sid +=cdt.metrics.SID(g, net) if get_sid else 0
                auc +=  cdt.metrics.precision_recall(g, net)[0]
            print("{} {} {} {}".format(name, shd/len(ground_truth), sid/len(ground_truth), auc/len(ground_truth)))
        else:
            shd = cdt.metrics.SHD(ground_truth, net, False)
            sid = cdt.metrics.SID(ground_truth, net) if get_sid else 0
            auc, pr = cdt.metrics.precision_recall(ground_truth, net)
            print("{} {} {} {}".format(name, shd, sid, auc))