import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import functools
from concurrent.futures import ProcessPoolExecutor
import os
from utils import edge_to_dag, adj_to_edge, adj_to_dag, get_scores
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
    communities = nx.community.girvan_newman(mrf)
    communities = tuple(sorted(c) for c in next(communities))
    comm_subgraphs = []
    data_local = []
    maps = []
    for c in communities:
        columns = c
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
    print(it)
    data = pd.read_csv("{}/part_{}/data.csv".format(outdir, it), header=0)
    skel = pd.read_csv("{}/part_{}/skel.csv".format(outdir, it), header=None).to_numpy()
    target_map = pd.read_csv("{}/part_{}/map.csv".format(outdir, it), header=0)
    target_map = dict(zip(target_map.iloc[:, 0], target_map.iloc[:, 1]))
    sp_gies(data, "{}/part_{}/".format(outdir, it), skel, target_map)
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

def global_structure_learn(data, outdir):
    partitioned = False
    locally_resolved = False

    if not partitioned:
        global_skel = skeleton(data, outdir)
        print("Number of edges in skeleton is {}".format(sum(sum(global_skel))))
        partition(global_skel, data, outdir)

    num_partitions = len(os.listdir(outdir))
    if not locally_resolved:
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
    data = pd.read_csv("../random_test_set_100_small/data_joint_0.csv", header=0)
    sp_gies_network = pd.read_csv("../random_test_set_100_small/0_sp-gies-adj_mat.csv", header=0).to_numpy()
    nodes = list(data.columns)
    nodes.remove('target')
    sp_gies_graph = adj_to_dag(sp_gies_network, nodes)

    edges = pd.read_csv("../random_test_set_100_small/bn_network_0.csv", header=0)
    edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
    G_true = edge_to_dag(edges_pos)

    #Visualize for debugging

    pos=pos=nx.spring_layout(G_true)
    nx.draw(G_true,pos=pos )
    plt.show()
    skel = pd.read_csv("../random_test_set_100_small/0_cupc-adj_mat.csv", header=0).to_numpy()
    mrf = nx.from_numpy_array(skel)
    mrf = nx.relabel_nodes(mrf, dict(zip(mrf.nodes, ["G{}".format(i+1) for i in mrf.nodes])))
    nx.draw(mrf, pos =pos )
    plt.show()

    outdir =  "./random_test_set_100_small_gv/parallel_test"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    G_est = global_structure_learn(data, outdir)
    print(G_true, G_est)
    print(get_scores(["PARALLEL-SP-GIES", "SP-GIES"], [G_est, sp_gies_graph], G_true, get_sid=True))

