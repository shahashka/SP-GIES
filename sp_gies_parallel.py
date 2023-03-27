import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
import networkx as nx
import functools
from concurrent.futures import ProcessPoolExecutor

# PARALLEL STRUCTURE LEARNING
# PSEUDO CODE
# Use cuPC to estimate the skeleton of the graph
# Find MB(.)(Markov blanket for each vertex int the graph) with distance=2 for all nodes
# Launch parallel threads/processes in python. each thread is assigned to a chunk of MBs
# In each thread run GIES on a subset of the data corresponding to the markov blanket, use the skeleton as fixedGap (also try no skeleton  to compare)
# Collect all the graphs (on order of number of vertices). report any conflicts

def func(mb,  data, skeleton=A):
    # Use MB to partition data and skeleotn
    # Run GIES with input skeleton (or not)
    return dag


def parallel_structure_learn(data):
    # Use cuPC to estimate skeleton from data

    # Get MB(.)
    mrf = nx.from_numpy_array(A)
    MB_X = []
    nthreads = 8
    nodes = list(data.columns)
    nodes.remove('target')
    for n in mrf.nodes():
        subgraph = nx.ego_graph(mrf, n, radius=2)
        MB_X.append(subgraph)
    func_partial = functools.partial(
        func,
        data = data,
        skeleton = A
    )
    results = []
    chunksize = max(1, len(MB_X) // nthreads)
    print("Launching processes")
    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(func_partial, MB_X, chunksize=chunksize):
            results.append(result)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for dag in results:
        G.add_edges_from(dag.edges())
    # Print number of bidirectional edges


#edges = pd.read_csv("./random_test_set_1000_small_norm/bn_network_0.csv")
#
# # convert edges to the true graph
# edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
# nodes = ["G{}".format(i) for i in range(1,10001)]
# A = edge_to_und_adj(edges_pos, nodes)
