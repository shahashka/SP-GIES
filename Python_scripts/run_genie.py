from GENIE3 import *
import pandas as pd
import numpy as np

# Helper function to convert a list of edges into an adjacency matrix
def edge_to_adj(edges, all_nodes):
    adj_mat = np.zeros((len(all_nodes), len(all_nodes)))
    for e in edges:
        start = all_nodes.index(e[0])
        end = all_nodes.index(e[1])
        adj_mat[start,end] = 1
    return adj_mat

def run_regulon_db():
    data = pd.read_csv("../regulondb/data_smaller.csv", header=None)
    adj = GENIE3(expr_data=data.to_numpy(), tree_method='XG')
    np.savetxt("./regulondb/genie_adj.csv",adj, delimiter=",")


def run_dream4():
    for d in range(3,4):
        data = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_combine.csv".format(d,d))
        obs_data = data.loc[data['target']==0]
        obs_data = obs_data.loc[:, obs_data.columns != 'target'].to_numpy()
        adj = GENIE3(expr_data=obs_data, tree_method='XG')
        np.savetxt("./insilico_size10_{}/genie_adj.csv".format(d),adj, delimiter=",")

def run_random():
    for network in ["small"]:
        for i in range(1):
            data = pd.read_csv("./random_test_set_10_{}/data_joint_{}.csv".format(network, i))
            obs_data = data.loc[data['target'] == 0]
            obs_data = obs_data.loc[:, obs_data.columns != 'target'].to_numpy()
            adj = GENIE3(expr_data=obs_data, tree_method='XG')
            np.savetxt("./random_test_set_10_{}/genie_adj_{}.csv".format(network,i), adj, delimiter=",")



def run_random_large():
    network = "small_norm"
    data = pd.read_csv("./random_test_set_1000_{}/data_joint_{}.csv".format(network, 0))
    obs_data = data.loc[data['target'] == 0]
    obs_data = obs_data.loc[:, obs_data.columns != 'target'].to_numpy()
    adj = GENIE3(expr_data=obs_data, tree_method='XG')
    np.savetxt("./random_test_set_1000_{}/gies_{}_adj.csv".format(network,0), adj, delimiter=",")

run_regulon_db()
run_dream4()
run_random()
run_random_large()