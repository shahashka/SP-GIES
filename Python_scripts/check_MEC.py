import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from utils import edge_to_dag, adj_to_edge, adj_to_dag, get_scores, edge_to_adj
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
rpy2.robjects.numpy2ri.activate()
pcalg = importr('pcalg')
base = importr('base')
from pgmpy.base import DAG
from graphical_models import rand, GaussDAG
import networkx as nx

def check_mec_w_ges(data, true_graph):
    obs_data = data.drop(columns=['target']).to_numpy()
    score = ro.r.new('GaussL0penObsScore', obs_data)
    ro.r.assign("score", score)
    result = pcalg.gies(ro.r['score'])
    ro.r.assign("result", result)

    rcode = 'result$repr$weight.mat()'
    adj_mat = ro.r(rcode)
    edges = adj_to_edge(adj_mat, ["G{}".format(i+1) for i in range(data.shape[1]-1)])
    G_est = DAG()
    G_est.add_nodes_from(["G{}".format(i+1) for i in range(data.shape[1]-1)])
    G_est.add_edges_from(edges)
    print(G_true)
    print(G_est)

    print(G_true.is_iequivalent(G_est))

# Create a random gaussian DAG and correposning observational dataset. Assume no prior information
def get_random_graph_data(graph_type, nnodes, nsamples):
    if graph_type == 'erdos_renyi':
        random_graph_model = lambda nnodes: nx.erdos_renyi_graph(nnodes, p=0.5, seed=42)
    elif graph_type == 'scale_free':
        random_graph_model = lambda nnodes: nx.barabasi_albert_graph(n, m=2, seed=42)
    elif graph_type == 'small_world':
        random_graph_model = lambda nnodes: nx.watts_strogatz_graph(n, k=2, p=0.5, seed=42)
    else:
        print("Unsupported random graph")
        return
    dag = rand.directed_random_graph(n, random_graph_model)
    nodes_inds = list(dag.nodes)
    bias = np.random.normal(0,1,size=len(nodes_inds))
    var = np.abs(np.random.normal(0,1,size=len(nodes_inds)))

    bn = GaussDAG(nodes= nodes_inds, arcs=dag.arcs, biases=bias,variances=var)
    data = bn.sample(nsamples)
    # if toNormalize:
    #     print("normalizing data")
    #     data = normalize(data, axis=1)

    nodes = ["G{}".format(d+1) for d in nodes_inds]
    arcs = [("G{}".format(i+1),"G{}".format(j+1))  for i,j in dag.arcs]

    df = pd.DataFrame(data=data, columns=nodes)
    df['target'] = np.zeros(data.shape[0])
    return arcs, df

for n in [1000, 5000, 10000]:
    arcs, data = get_random_graph_data("scale_free", nnodes=100, nsamples=n)
    G_true = DAG()
    G_true.add_nodes_from(["G{}".format(i+1) for i in range(data.shape[1]-1)])
    G_true.add_edges_from(arcs)
    check_mec_w_ges(data, G_true)