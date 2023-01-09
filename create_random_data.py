from graphical_models import rand, GaussDAG
import networkx as nx
import numpy as np
import argparse
import pandas as pd
import causaldag as cd
import os
from sklearn.preprocessing import normalize

# Script to generate random data from a random network either Erdos Renyi, scale-free or small world
# Specify the number of samples, number of nodes and number of graphs needed
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_graph', type=str)
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--nsamples', type=int)
    parser.add_argument('--nnodes', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--working_dir', type=str)
    parser.add_argument('--num_graphs',type=int)
    parser.add_argument('--normalize', action='store_true',default=False )
    return parser.parse_args()

# Create a random gaussian DAG and correposning observational dataset. Assume no prior information
def get_random_graph_data(working_dir, graph_type, nsamples, n, parameters, toNormalize=False, seed=None, num_graphs=1):
    if graph_type == 'erdos_renyi':
        random_graph_model = lambda nnodes: nx.erdos_renyi_graph(nnodes, p=parameters["p"], seed=seed)
    elif graph_type == 'scale_free':
        random_graph_model = lambda nnodes: nx.barabasi_albert_graph(n, m=parameters['k'], seed=seed)
    elif graph_type == 'small_world':
        random_graph_model = lambda nnodes: nx.watts_strogatz_graph(n, k=parameters['k'], p=parameters['p'], seed=seed)
    else:
        print("Unsupported random graph")
        return
    for g in range(num_graphs):
        dag = rand.directed_random_graph(n, random_graph_model)
        nodes_inds = list(dag.nodes)
        bias = np.random.normal(0,1,size=len(nodes_inds))
        var = np.abs(np.random.normal(0,1,size=len(nodes_inds)))

        bn = GaussDAG(nodes= nodes_inds, arcs=dag.arcs, biases=bias,variances=var)
        data = bn.sample(nsamples)
        if toNormalize:
            print("normalizing data")
            data = normalize(data, axis=1)

        nodes = ["G{}".format(d+1) for d in nodes_inds]
        arcs = [("G{}".format(i+1),"G{}".format(j+1))  for i,j in dag.arcs]

        df = pd.DataFrame(data=data, columns=nodes)
        df['target'] = np.zeros(data.shape[0])

        if not os.path.isdir(working_dir):
            os.makedirs(working_dir)
        df.to_csv("{}/data_{}.csv".format(working_dir,g), index=False)
        df.iloc[[0]].to_csv("{}/data_wt_{}.csv".format(working_dir, g), index=False)
        start = [e[0]  for e in arcs] # make sure nodes start from 1 to avoid issues with R code
        end = [e[1] for e in arcs] # make sure nodes start from 1 to avoid issues with R code
        df_edges = pd.DataFrame(data=np.column_stack([start, end, np.ones(len(start), dtype=int)]), columns=["start", "end", "edge"])
        df_edges.to_csv('{}/bn_network_{}.csv'.format(working_dir,g), index=False)

        df_params = pd.DataFrame(data=np.column_stack([nodes,bias,var]), columns=['node','bias','variance'])
        df_params.to_csv('{}/bn_params_{}.csv'.format(working_dir,g), index=False)

        i_data = []
        for i in nodes_inds:
            samples = bn.sample_interventional(cd.Intervention({i: cd.ConstantIntervention(val=0)}), 1)
            samples = pd.DataFrame(samples, columns=nodes)
            samples['target'] = i+1
            i_data.append(samples)
        df_int = pd.concat(i_data)
        df_int.to_csv("{}/interventional_data_{}.csv".format(working_dir, g), index=False)
        df_joint = pd.concat([df, df_int])
        df_joint.to_csv("{}/data_joint_{}.csv".format(working_dir, g), index=False)


args = get_args()
get_random_graph_data(args.working_dir, args.random_graph, nsamples=args.nsamples, n=args.nnodes,
                      parameters={"p":args.p, "k":args.k}, toNormalize=args.normalize, seed=args.seed, num_graphs=args.num_graphs)
