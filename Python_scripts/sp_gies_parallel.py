import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import functools
from concurrent.futures import ProcessPoolExecutor
import os
from utils import edge_to_dag, adj_to_edge, adj_to_dag, get_scores, get_random_graph_data
from sp_gies import sp_gies

# PARALLEL STRUCTURE LEARNING
# (1) Estimate skeleton
# (2) Partition
# (3) Local Learn
# (4) Global resolution

def markov_blankets(skeleton, data):
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

def community_detection(skeleton, data):
    mrf = nx.from_numpy_array(skeleton)
    #communities = nx.community.girvan_newman(mrf)
    #communities = nx.community.louvain_communities(mrf)
    communities = nx.community.label_propagation_communities(mrf)
    #communities = tuple(sorted(c) for c in next(communities))
    comm_subgraphs = []
    data_local = []
    maps = []
    for c in communities:
        columns = list(c)
        comm_subgraphs.append(mrf.subgraph(columns))
        # filter rows for interventional samples that are in subgraph and are observational
        data_sub = data.loc[data['target'].isin(list(np.array(columns) + 1) + [0])]

        target_map = pd.DataFrame({"GENE_ID": list(np.array(columns) + 1),
                                   "COLUMN_INDEX": np.arange(len(columns))})
        maps.append(target_map)

        columns.append(-1)  # include the target column
        data_sub = data_sub.iloc[:, columns]
        data_local.append(data_sub)
    local_structs = zip(comm_subgraphs, data_local, maps)
    return local_structs


def local_structure_learn(it, outdir):
    # Run a structure learning algorithm on a subset of the data
    # algorithms in include GIES, IGSP, etc..
    data = pd.read_csv("{}/part_{}/data.csv".format(outdir, it), header=0)
    skel = pd.read_csv("{}/part_{}/skel.csv".format(outdir, it), header=None).to_numpy()
    target_map = pd.read_csv("{}/part_{}/map.csv".format(outdir, it), header=0)
    target_map = dict(zip(target_map.iloc[:, 0], target_map.iloc[:, 1]))
    sp_gies(data, "{}/part_{}/".format(outdir, it), skel=skel, cupc=True, target_map=target_map)
    return 0

def partition(skeleton, data, outdir):
    # Partition the skeleton according to some notion of locality
    # Save the sub-skeleton and sub-data in different folders
    local_structs = community_detection(skeleton, data)
    for i,(s, d, m) in enumerate(local_structs):
        if not os.path.exists("{}/part_{}/".format(outdir,i)):
            os.makedirs("{}/part_{}/".format(outdir,i))
        adj_mat = nx.adjacency_matrix(s)
        np.savetxt("{}/part_{}/skel.csv".format(outdir,i), adj_mat.toarray(), delimiter=",")
        d.to_csv("{}/part_{}/data.csv".format(outdir,i), index=False)
        m.to_csv("{}/part_{}/map.csv".format(outdir, i), index=False)
        print("Size of MB is {}".format(d.shape[1]))
    return i+1

def resolve_global(outdir, num_partitions):
    # Read all dags in directory and create a global dag
    # Report number of conflicts
    edge_list = []
    node_list = set()
    for i in range(num_partitions):
        adj_mat = pd.read_csv("{}/part_{}/sp-gies-adj_mat.csv".format(outdir, i), header=0).to_numpy()
        map = pd.read_csv("{}/part_{}/map.csv".format(outdir, i), header=0)
        nodes = map.iloc[:,0]
        nodes = ["G{}".format(i) for i in nodes]
        node_list = node_list.union(nodes)
        edge_list += adj_to_edge(adj_mat, nodes)
    G = nx.DiGraph(edge_list)
    G.add_nodes_from(node_list)
    print(0.5 * len( [ 1 for (u,v) in G.edges() if u in G[v] ] ))
    return G

def global_structure_learn(data, global_skel, outdir):
    num_partitions = partition(global_skel, data, outdir)
    
    nthreads = 1
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
    # Generate data using create_random_data
    # Use 100 nodes, change p, k
    # Run SP-GIES and save CUPC skeleton
    # Then run global run, pass in parameters for skeleton
    for k in [2,4,8,16]:
        print("K={}".format(k))
        for p in [0.5]:
            outdir = "./parallel_test_set_100_scale_k={}".format(k)
            arcs, data = get_random_graph_data("scale_free", n=100, nsamples=10000, iv_samples=10,
                                      p=p, k=k, save=True, outdir=outdir )
            sp_gies(data, outdir, skel=None, cupc=True)
            skel = pd.read_csv("{}/cupc-adj_mat.csv".format(outdir), header=0).to_numpy()
            G_true = edge_to_dag(arcs)
            G_sp_gies = pd.read_csv("{}/sp-gies-adj_mat.csv".format(outdir), header=0).to_numpy()
            nodes = list(data.columns)
            nodes.remove('target')
            G_sp_gies = adj_to_dag(G_sp_gies, nodes)
            G_est = global_structure_learn(data, skel, outdir)
            print(G_true, G_est, G_sp_gies)
            print(get_scores(["PARALLEL-SP-GIES", "SP-GIES"], [G_est, G_sp_gies], G_true, get_sid=True))

