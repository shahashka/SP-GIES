import causaldag as cd
import numpy as np
import networkx as nx
import pandas as pd

def cov2dag(cov_mat, dag):
    # See formula https://arxiv.org/pdf/1303.3216.pdf pg. 17
    node_idx = np.arange(len(dag.nodes))
    nodes = dict(zip(dag.nodes, node_idx))
    p = len(nodes)
    amat = np.zeros((p, p))
    variances = np.zeros(p)
    for node in nodes:
        if type(dag) == nx.DiGraph:
            node_parents = [nodes[p] for p in dag.predecessors(node)]
        elif type(dag) == cd.GaussDAG:
            node_parents = [nodes[p] for p in dag.parents_of(node)]
        else:
            node_parents = [nodes[p] for p in dag.get_parents(node)]
        if len(node_parents) == 0:
            variances[nodes[node]] = cov_mat[nodes[node], nodes[node]]
        else:
            S_k_k = cov_mat[nodes[node], nodes[node]]
            S_k_pa = cov_mat[nodes[node], node_parents]
            S_pa_pa = cov_mat[np.ix_(node_parents, node_parents)]
            if len(node_parents) > 1:
                inv_S_pa_pa = np.linalg.inv(S_pa_pa)
            else:
                inv_S_pa_pa = np.array(1 / S_pa_pa)
            node_mle_coefficents = S_k_pa.dot(inv_S_pa_pa)
            error_mle_variance = S_k_k - S_k_pa.dot(inv_S_pa_pa.dot(S_k_pa.T))
            variances[nodes[node]] = error_mle_variance
            amat[node_parents, nodes[node]] = node_mle_coefficents
    graph = cd.GaussDAG.from_amat(amat, variances=variances)
    return graph

class Intervention:
    def __init__(self, name, data, data_inds):
        self.name = name
        self.data = data
        self.data_inds = data_inds
        self.intervention = cd.ConstantIntervention(val=0)

    def get_data(self):
        return self.data
    def get_dict_inds(self):
        my_dict = dict()
        for i in range(self.data_inds, self.data_inds+self.data.shape[0]):
            my_dict[i] = (int)(self.name[1:])
        return my_dict

    def gauss_dag_logpdf(self, dag, data_norm, node):
        cov = np.cov(pd.concat([data_norm, self.data]), rowvar=False)
        gauss_dag = cov2dag(cov, dag)
        nodes = list(self.data.columns)
        phenotype = nodes.index(node) if node == 'P1' else None
        logpdf = gauss_dag.logpdf(self.data.values, special_node=phenotype,
                                  interventions={(int)(self.name[1:]) - 1: self.intervention},
                                  exclude_intervention_prob=False)
        return logpdf


class KO_KD_Intervention(Intervention):
    def __init__(self, name, data, data_inds):
        super().__init__(name, data, data_inds)
        self.intervention = cd.BinaryIntervention(cd.ConstantIntervention(val=data.iloc[0][name]),
                                                  cd.ConstantIntervention(val=data.iloc[1][name]), p=0.5)
