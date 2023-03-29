import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
import cdt
from graphical_models import rand, GaussDAG
import causaldag as cd

ground = pd.read_csv("ground_truth.csv", header=None)
graph = nx.from_numpy_array(ground.to_numpy(), create_using=nx.DiGraph)
degree_dist = np.array(list(graph.degree()))
print("Max degree in ground truth network is {}".format(np.max(np.array(degree_dist[:,-1], dtype=int))))
cycle = nx.simple_cycles(graph)
print("Number of cycles in ground truth netowrk is {}".format(len(sorted(cycle))))

# Ground turth skel is the undirected version of the groudn truth network
skel = pd.read_csv("ground_truth_skel.csv", header=None)
skel_graph = nx.from_numpy_array(ground.to_numpy(), create_using=nx.Graph)
genes = pd.read_csv("genes.txt", header=None)
tfs = pd.read_csv("tfs.txt", header=None, sep="\t")
tfs_in_data = list(set(genes.values.flatten()).intersection(set(tfs.values.flatten())))

# Find all the connected components of the ground truth undirected graph
S = [skel_graph.subgraph(c).copy() for c in nx.connected_components(skel_graph)]

clr_network = pd.read_csv("adj_mat.csv", header=None).values
threshold = 7.5 #6.917 # This achieves 60% precision
clr_network[np.abs(clr_network) < threshold] = 0
clr_network[np.abs(clr_network) >= threshold] = 1
ground[np.abs(ground) >=1] = 1

# limit the direction of edges so that no incoming edges to genes for the CLR network
genes_array = genes.values.flatten()
for i in range(clr_network.shape[0]):
    for j in range(i+1):
        if genes_array[j] not in tfs_in_data:
            clr_network[i][j]=0
            if clr_network[i][j] == 1:
                print("break")
print("Precision of CLR network is {}".format(precision_score(ground.values.flatten(), clr_network.flatten())))

clr_graph = nx.from_numpy_array(clr_network, create_using=nx.Graph)
clr_graph = nx.relabel_nodes(clr_graph, dict(zip(clr_graph.nodes, genes.to_numpy().flatten())))
graph = nx.relabel_nodes(graph, dict(zip(graph.nodes, genes.to_numpy().flatten())))


# CLR nodes are all the nodes in the clr network that are part of a connected component
# This subgraph has 188 nodes
clr_nodes = []
for i, c in enumerate(nx.connected_components(clr_graph)):
    if len(list(c)) > 10:
        print(f"Island {i+1}: {c}")
    if len(list(c)) >= 2:
        clr_nodes = clr_nodes + list(c)
S = [clr_graph.subgraph(c).copy() for c in nx.connected_components(clr_graph)]

# Local nodes are the nodes in one connected component that we care about investigating that has hubs at fliA, flhD, flhC genes
# The local graph as 46 nodes
local_nodes = S[32].nodes

#Extract the local subset of the clr graph, ground truth graph and of the data
data = pd.read_csv("data_smaller.csv", header=None)
clr_nodes_index =[ np.where(genes.to_numpy().flatten() == c)[0][0] for c in local_nodes]
data_clr = data.iloc[:,clr_nodes_index]
data_clr.to_csv("data_smaller_clr_subgraph_local.csv", header=None, index=False)
clr_skel = clr_network[:,clr_nodes_index][clr_nodes_index]
ground_skel_sub = ground.iloc[:,clr_nodes_index].iloc[clr_nodes_index]
np.savetxt("local_graph/clr_skel_subgraph_local.csv", clr_skel,  delimiter=",")
np.savetxt("local_graph/ground_skel_subgraph_local.csv", ground_skel_sub,  delimiter=",")

# Genearte a synethic data set from the local graph topology, use this for sp-gies input
generate_syth = False
if generate_syth:
    G = graph.subgraph(local_nodes).copy()
    G = nx.relabel_nodes(G, dict(zip(local_nodes, np.arange(len(local_nodes)))))
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_edges_from(nx.simple_cycles(G))

    bias = np.random.normal(0, 1, size=len(local_nodes))
    var = np.abs(np.random.normal(0, 1, size=len(local_nodes)))
    bn = GaussDAG(nodes=np.arange(len(local_nodes)), arcs=G.edges, biases=bias,variances=var)

    data_synthetic = bn.sample(10000)
    i_data = []
    for i in np.arange(len(local_nodes)):
        samples = bn.sample_interventional(cd.Intervention({i: cd.ConstantIntervention(val=0)}), 10)
        samples = pd.DataFrame(samples, columns=np.arange(len(local_nodes)))
        i_data.append(samples)
    df_int = pd.concat(i_data)
    df_obs = pd.DataFrame(data=data_synthetic)
    df_obs.columns = local_nodes
    df_int.columns=local_nodes
    df_obs.to_csv("local_graph/data_smaller_clr_subgraph_local_synthetic_n=10000_obs.csv", header=True, index=False)
    df = pd.concat([df_obs, df_int])
    df.columns = local_nodes
    df.to_csv("local_graph/data_smaller_clr_subgraph_local_synthetic_n=10000_obs_int.csv", header=True, index=False)

    targets = np.arange(1,11)
    target_index = list(np.ones(10000))
    for i in np.arange(2,12):
        target_index += list(i* np.ones(10))
    np.savetxt("local_graph/targets_local.csv", targets, delimiter=",")
    np.savetxt("local_graph/targets_inds_local.csv", target_index, delimiter=",")

# Read sp-gies learned network using the clr skeleton prior (global network)
sp_gies_network = pd.read_csv("clr_skel_sp-gies-adj_mat.csv", header=0).values
sp_gies_network[np.abs(sp_gies_network) >0] = 1
sp_gies_graph = nx.from_numpy_array(sp_gies_network, create_using=nx.Graph)

# Read sp-gies learned subnetwork (as defined by clr network ie 188 nodes)
sp_gies_subnetwork = pd.read_csv("subgraph_sp-gies-adj_mat.csv", header=0).to_numpy()
sp_gies_subgraph = nx.from_numpy_array(sp_gies_subnetwork, create_using=nx.DiGraph)
sp_gies_subgraph = nx.relabel_nodes(sp_gies_subgraph, dict(zip(sp_gies_subgraph.nodes, clr_nodes)))

# Get the directed version of hte clr network (1146 nodes)
clr_digraph = nx.from_numpy_array(clr_network, create_using=nx.DiGraph)
clr_digraph = nx.relabel_nodes(clr_digraph, dict(zip(clr_digraph.nodes, genes.to_numpy().flatten())))

# Get the local graph nodes (46) with the ordering that corresponds to the sp-gies network (this may change between runs)
local_nodes = pd.read_csv("local_graph/data_smaller_clr_subgraph_local_synthetic_n=10000_obs_int.csv", header=0).columns

sp_gies_subnetwork_synth_local = pd.read_csv("local_graph/clr_skel_subgraph_local_sp-gies-adj_mat.csv", header=0).to_numpy()
sp_gies_subgraph_synth_local = nx.from_numpy_array(sp_gies_subnetwork_synth_local, create_using=nx.DiGraph)
mapping =  dict(zip(np.arange(len(sp_gies_subgraph_synth_local.nodes)), local_nodes))
sp_gies_subgraph_synth_local = nx.relabel_nodes(sp_gies_subgraph_synth_local, mapping)

sp_gies_subnetwork_synth_local_int = pd.read_csv("local_graph/clr_skel_subgraph_local_wint_sp-gies-adj_mat.csv", header=0).to_numpy()
sp_gies_subgraph_synth_local_int = nx.from_numpy_array(sp_gies_subnetwork_synth_local_int, create_using=nx.DiGraph)
mapping =  dict(zip(np.arange(len(sp_gies_subgraph_synth_local_int.nodes)), local_nodes))
sp_gies_subgraph_synth_local_int = nx.relabel_nodes(sp_gies_subgraph_synth_local_int, mapping)

# Get the positions of the local nodes to use for future plotting
pos = nx.spring_layout(sp_gies_subgraph.subgraph(local_nodes), k=100/np.sqrt(len(local_nodes)), seed=42)
node_size=[len(local_nodes[i])**2 * 60 for i in range(len(pos))]

#  Local graph learned sp-gies network where input was the 188 node network
nodes = nx.draw(sp_gies_subgraph.subgraph(local_nodes), pos=pos,node_color='#ADD8E6', node_size=node_size,with_labels=True)
plt.show()

# Local graph taken from learning over only 46 nodes with synethic data
nx.draw(sp_gies_subgraph_synth_local.subgraph(local_nodes),pos=pos,node_color='#ADD8E6', node_size=node_size,with_labels=True)
plt.show()

# Local graph taken from learning over only 46 nodes with synethic data including internvetional samples
nx.draw(sp_gies_subgraph_synth_local_int.subgraph(local_nodes), pos=pos,node_color='#ADD8E6', node_size=node_size,with_labels=True)
plt.show()

# Local graph taken clr learned network (1146 -> 46 noddes)
nx.draw(clr_digraph.subgraph(local_nodes).to_undirected(), pos=pos,node_color='#ADD8E6', node_size=node_size,with_labels=True)
plt.show()

# local graph taken from ground truth network (1146 -> 46 nodes)
nx.draw(graph.subgraph(local_nodes), pos=pos,node_color='#ADD8E6', node_size=node_size,with_labels=True)
plt.show()

