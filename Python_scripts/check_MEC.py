import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from utils import edge_to_dag, adj_to_edge, adj_to_dag, get_scores, edge_to_adj, get_random_graph_data
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
rpy2.robjects.numpy2ri.activate()
pcalg = importr('pcalg')
base = importr('base')
from pgmpy.base import DAG
import networkx as nx
import collections
from causaldag import igsp, gsp
import conditional_independence
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

    print("GES found MEC? {}".format(G_true.is_iequivalent(G_est)))

def check_mec_w_gsp(data, true_graph):
    obs_data = data.loc[data['target'] == 0]
    obs_data_no_targets = obs_data.loc[:, obs_data.columns != 'target'].to_numpy()

    nodes = obs_data.columns

    obs_suffstat = conditional_independence.partial_correlation_suffstat(obs_data_no_targets)
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = conditional_independence.MemoizedCI_Tester(conditional_independence.partial_correlation_test,
                                                           obs_suffstat, alpha=alpha)

    est_dag_obs = gsp(nodes, ci_tester)
    G_est = DAG()
    G_est.add_nodes_from(["G{}".format(i+1) for i in range(data.shape[1]-1)])
    edges = [("G{}".format(i+1), "G{}".format(j+1)) for (i,j) in est_dag_obs.arcs]
    G_est.add_edges_from(edges)
    print(G_true)
    print(G_est)
    print("GSP found MEC? {}".format(G_true.is_iequivalent(G_est)))

def check_consistency_w_gies(data, true_graph):
    target_index = data.loc[:, 'target'].to_numpy()
    targets = np.unique(target_index)[1:]  # Remove 0 the observational target
    target_index_R = target_index + 1  # R indexes from 1
    data = data.drop(columns=['target']).to_numpy()

    nr, nc = data.shape
    D = ro.r.matrix(data, nrow=nr, ncol=nc)
    ro.r.assign("data", D)

    rcode = ','.join(str(int(i)) for i in targets)
    rcode = 'append(list(integer(0)), list({}))'.format(rcode)
    T = ro.r(rcode)
    ro.r.assign("targets", T)

    TI = ro.IntVector(target_index_R)
    ro.r.assign("target_index", TI)

    score = ro.r.new('GaussL0penIntScore', ro.r['data'], ro.r['targets'], ro.r['target_index'])
    ro.r.assign("score", score)
    result = pcalg.gies(ro.r['score'], targets=ro.r['targets'])
    ro.r.assign("result", result)

    rcode = 'result$repr$weight.mat()'
    adj_mat = ro.r(rcode)
    edges = adj_to_edge(adj_mat, ["G{}".format(i+1) for i in range(data.shape[1])])
    G_est = DAG()
    G_est.add_nodes_from(["G{}".format(i+1) for i in range(data.shape[1])])
    G_est.add_edges_from(edges)
    print(G_true)
    print(G_est)

    print("GIES found correct DAG? {}".format(collections.Counter(G_true.edges) == collections.Counter(G_est.edges)))

def check_consistency_w_igsp(data, true_graph):
    obs_data = data.loc[data['target'] == 0]
    int_data = data.loc[data['target'] != 0]
    obs_data_no_targets = obs_data.loc[:, obs_data.columns != 'target'].to_numpy()
    int_data_no_targets = int_data.loc[:, int_data.columns != 'target']
    iv_samples_list = [np.expand_dims(row.to_numpy(), axis=0) for _, row in int_data_no_targets.iterrows()]

    targets_list = set(list(int_data['target'].to_numpy()))
    nodes = targets_list

    setting_list = [dict(interventions=[t]) for t in targets_list]

    obs_suffstat = conditional_independence.partial_correlation_suffstat(obs_data_no_targets)
    invariance_suffstat = conditional_independence.gauss_invariance_suffstat(obs_data_no_targets,
                                                                             iv_samples_list)
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = conditional_independence.MemoizedCI_Tester(conditional_independence.partial_correlation_test,
                                                           obs_suffstat, alpha=alpha)
    invariance_tester = conditional_independence.MemoizedInvarianceTester(
        conditional_independence.gauss_invariance_test,
        invariance_suffstat, alpha=alpha_inv)

    est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester)

    G_est = DAG()
    G_est.add_nodes_from(["G{}".format(i+1) for i in range(data.shape[1]-1)])
    edges = [("G{}".format(i+1), "G{}".format(j+1)) for (i,j) in est_dag.arcs]
    G_est.add_edges_from(edges)
    print(G_true)
    print(G_est)
    print("IGSP found correct DAG? {}".format(collections.Counter(G_true.edges) == collections.Counter(G_est.edges)))


# GES always finds the correct MEC
# Weird that GSP does noe
for n in [1000, 10000, 100000]:
    arcs, data = get_random_graph_data("scale_free", n=10, nsamples=n, iv_samples=0, p=0.5, k=2)
    print(n)
    G_true = DAG()
    G_true.add_nodes_from(["G{}".format(i+1) for i in range(data.shape[1]-1)])
    G_true.add_edges_from(arcs)
    check_mec_w_ges(data, G_true)
    check_mec_w_gsp(data, G_true)

# GIES does not necessarily find the right graph, depends on the graph!
for i in [1, 10, 100, 1000]:
    arcs, data = get_random_graph_data("scale_free", n=10, nsamples=10000, iv_samples=i, p=0.5, k=2)
    print(i, data.shape)
    G_true = DAG()
    G_true.add_nodes_from(["G{}".format(i+1) for i in range(data.shape[1]-1)])
    G_true.add_edges_from(arcs)
    check_consistency_w_gies(data, G_true)
    check_consistency_w_igsp(data, G_true)