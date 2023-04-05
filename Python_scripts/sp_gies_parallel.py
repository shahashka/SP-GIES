import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
import networkx as nx
import functools
from concurrent.futures import ProcessPoolExecutor
import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from utils import edge_to_dag, adj_to_edge, adj_to_dag, get_scores
from sp_gies import sp_gies
rpy2.robjects.numpy2ri.activate()
pcalg = importr('pcalg')
base = importr('base')
# PARALLEL STRUCTURE LEARNING

def markov_blankets(skeleton, data, outdir):
    mrf = nx.from_numpy_array(skeleton)
    MB_X = []
    data_local = []
    maps = []
    for n in mrf.nodes():
        subgraph = nx.ego_graph(mrf, n, radius=2)
        MB_X.append(subgraph)

        # Grab the subset of the data that corresponds to the MB
        columns = list(subgraph.nodes)
        # filter rows for interventional samples that are in subgraph and are observational
        data_sub = data.loc[data['target'].isin(list(np.array(columns) + 1) + [0])]

        target_map = pd.DataFrame({"GENE_ID": list(np.array(columns) + 1),
                                   "COLUMN_INDEX": np.arange(len(columns))})
        maps.append(target_map)

        columns.append(-1)  # include the target column
        data_sub = data_sub.iloc[:, columns]
        data_local.append(data_sub)
    local_structs = zip(MB_X, data_local, maps)
    return local_structs

def local_structure_learn(it, outdir):
    # Run a structure learning algorithm on a subset of the data
    # algorithms in include GIES, IGSP, etc..
    print(it)
    data = pd.read_csv("{}/part_{}/data.csv".format(outdir, it), header=0)
    skel = pd.read_csv("{}/part_{}/skel.csv".format(outdir, it), header=None).to_numpy()
    target_map = pd.read_csv("{}/part_{}/map.csv".format(outdir, it), header=0)
    target_map = dict(zip(target_map.iloc[:, 0], target_map.iloc[:, 1]))
    sp_gies(data, target_map, skel, "{}/part_{}/".format(outdir, it))
    return 0

def partition(skeleton, data, outdir):
    # Partition the skeleton according to some notion of locality
    # Save the sub-skeleton and sub-data in different folders
    local_structs = markov_blankets(skeleton, data, outdir)
    for i,(s, d, m) in enumerate(local_structs):
        if not os.path.exists("{}/part_{}/".format(outdir,i)):
            os.makedirs("{}/part_{}/".format(outdir,i))
        adj_mat = nx.adjacency_matrix(s)
        np.savetxt("{}/part_{}/skel.csv".format(outdir,i), adj_mat.toarray(), delimiter=",")
        d.to_csv("{}/part_{}/data.csv".format(outdir,i), index=False)
        m.to_csv("{}/part_{}/map.csv".format(outdir, i), index=False)
    return 0


def skeleton(data, outdir):
    # Generate a skeleton of the causal graph based on data
    # Methods include PC, GENIE, etc...
    # Save adjacency matrix is outdir

    # Read cuPC generated skeleton
    return pd.read_csv("../random_test_set_100_small/0_cupc-adj_mat.csv", header=0).to_numpy()

def resolve_global(outdir, num_partitions):
    # Read all dags in directory and create a global dag
    # Report number of conflicts
    edge_list = []
    for i in range(num_partitions):
        adj_mat = pd.read_csv("{}/part_{}/sp-gies-adj_mat.csv".format(outdir, i), header=0).to_numpy()
        map = pd.read_csv("{}/part_{}/map.csv".format(outdir, i), header=0)
        nodes = map.iloc[:,0]
        nodes = ["G{}".format(i) for i in nodes]
        edge_list += adj_to_edge(adj_mat, nodes)
    G = nx.DiGraph(edge_list)
    print(0.5 * len( [ 1 for (u,v) in G.edges() if u in G[v] ] ))
    return G

def global_structure_learn(data, outdir):
    partitioned = True
    locally_resolved = True

    if not partitioned:
        global_skel = skeleton(data, outdir)
        partition(global_skel, data, outdir)

    num_partitions = len(os.listdir(outdir))
    if not locally_resolved:
        nthreads = 2
        func_partial = functools.partial(
            local_structure_learn,
            outdir=outdir
        )
        results = []
        chunksize = max(1, num_partitions // nthreads)
        print("Launching processes")
        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            for result in executor.map(func_partial, np.arange(num_partitions), chunksize=chunksize):
                results.append(result)
    G = resolve_global(outdir, num_partitions)
    return G



if __name__ == '__main__':
    data = pd.read_csv("../random_test_set_100_small/data_joint_0.csv", header=0)
    sp_gies_network = pd.read_csv("../random_test_set_100_small/0_sp-gies-adj_mat.csv", header=0).to_numpy()
    nodes = list(data.columns)
    nodes.remove('target')
    sp_gies_graph = adj_to_dag(sp_gies_network, nodes)

    edges = pd.read_csv("../random_test_set_100_small/bn_network_0.csv", header=0)
    edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
    G_true = edge_to_dag(edges_pos)

    outdir =  "./random_test_set_100_small/parallel_test"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    G_est = global_structure_learn(data, outdir)
    print(G_true, G_est)
    print(get_scores(["PARALLEL-SP-GIES", "SP-GIES"], [G_est, sp_gies_graph], G_true, get_sid=True))

