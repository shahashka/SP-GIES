import networkx as nx
import numpy as np
import argparse
import pandas as pd
import os
from sp_gies.utils import get_random_graph_data
# Script to generate random data from a random network either Erdos Renyi, scale-free or small world
# Specify the number of samples, number of nodes and number of graphs needed
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_graph', type=str, help='type of random graph: erdos_renyi, small_world or scale_free')
    parser.add_argument('--p', type=float, default=0.5, help='for erdos renyi and small_world networks, p is the probability of edge existence between 2 nodes')
    parser.add_argument('--k', type=int, default=2, help='for small_world and scale free k is the number of initial nodes, roughly corresponds to the number of hubs in the final graph')
    parser.add_argument('--nsamples', type=int, help='number of observational samples')
    parser.add_argument('--nnodes', type=int, help='number of nodes in the graph')
    parser.add_argument('--seed', type=int, help='optional random seed')
    parser.add_argument('--working_dir', type=str, help='working directory to save output files')
    parser.add_argument('--num_graphs',type=int, help='number of random graphs to generate')
    parser.add_argument('--ivnodes', type=int, default=1, help='number of interventional samples per node')
    return parser.parse_args()

# Create a random gaussian DAG and correposning observational dataset. Assume no prior information
def create_data(working_dir, graph_type, nsamples, nnodes, parameters, seed=42, num_graphs=1,ivnodes=1):
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    for g in range(num_graphs):
        (arcs, nodes, bias, var), df_joint = get_random_graph_data(graph_type, nnodes, nsamples, ivnodes, parameters['p'],
                                               parameters['k'], seed=seed, save=False, outdir=working_dir)
        # Save all the needed files (observational only, interventional only, joint data, and network parameters)
        df_joint.to_csv("{}/data_joint_{}.csv".format(working_dir, g), index=False)
        df_obs = df_joint.loc[df_joint['target'] == 0]
        df_obs.to_csv("{}/data_{}.csv".format(working_dir,g), index=False)

        start = [e[0]  for e in arcs] # make sure nodes start from 1 to avoid issues with R code
        end = [e[1] for e in arcs] # make sure nodes start from 1 to avoid issues with R code
        df_edges = pd.DataFrame(data=np.column_stack([start, end, np.ones(len(start), dtype=int)]), columns=["start", "end", "edge"])
        df_edges.to_csv('{}/bn_network_{}.csv'.format(working_dir,g), index=False)
        df_params = pd.DataFrame(data=np.column_stack([nodes,bias,var]), columns=['node','bias','variance'])
        df_params.to_csv('{}/bn_params_{}.csv'.format(working_dir,g), index=False)

        df_int = df_joint.loc[df_joint['target'] != 0]
        df_int.to_csv("{}/interventional_data_{}.csv".format(working_dir, g), index=False)


args = get_args()
create_data(args.working_dir, args.random_graph, nsamples=args.nsamples, nnodes=args.nnodes,
            parameters={"p":args.p, "k":args.k}, seed=args.seed, num_graphs=args.num_graphs,
            ivnodes=args.ivnodes)
