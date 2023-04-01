import conditional_independence
from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.base import DAG
from graphical_model_learning import igsp, unknown_target_igsp
from sp_gies import sp_gies
import numpy as np
import networkx as nx
import pandas as pd
import re

class Learner(object):
    def __init__(self, all_nodes, num_graphs, subset_size, replace):
        self.all_nodes = all_nodes
        self.num_graphs = num_graphs
        self.subset_size=subset_size
        self.replace = replace
        self.node_2ix = dict(zip(all_nodes,
                                 [(int)(re.split('[a-z]+', s, flags=re.IGNORECASE)[-1]) for s in all_nodes]))
    def sample_posterior(self, data):
        print("Super class")

    def adj_to_dag(self, adj):
        dag = DAG()
        dag.add_nodes_from(self.all_nodes)
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == 1:
                    dag.add_edge(self.all_nodes[i], self.all_nodes[j])
        return dag
    def get_index(self, node):
        return self.node_2ix[node]

class HillClimbLearner(Learner):
    def __init__(self,all_nodes, num_graphs, subset_size, replace,
                  fixed_edges, whitelist):
        super().__init__(all_nodes, num_graphs, subset_size, replace)
        self.fixed_edges = fixed_edges
        self.white_list = whitelist
    #            d_sub_wo_targets = d_sub.loc[:, d_sub.columns != 'target']
    # if type(self.white_list) == list and len(self.white_list) == 0:
    #     whitelist = None
    def sample_posterior(self, data):
        posterior = []
        data = data.loc[:, data.columns != 'target']

        for _ in range(self.num_graphs):
            d_sub_inds = np.random.choice(np.arange(data.shape[0]), self.subset_size, replace=self.replace)
            d_sub = data.iloc[d_sub_inds]
            est = HillClimbSearch(d_sub)
            if self.fixed_edges:
                best_model = est.estimate(scoring_method=K2Score(data), fixed_edges=self.fixed_edges,
                                          white_list=self.white_list)
            else:
                best_model = est.estimate(scoring_method=K2Score(data),
                                          white_list=self.white_list)
            posterior.append(best_model)
        return {'dags': posterior}


class GIESLearner(Learner):
    def __init__(self, all_nodes, num_graphs, subset_size, replace, no_interventions, skeleton_file=None):
        super().__init__(all_nodes, num_graphs, subset_size, replace)
        self.no_interventions = no_interventions
        if skeleton_file is not None:
            if skeleton_file.endswith(".txt"):
                self.skeleton = pd.read_csv(skeleton_file, sep='\t', header=0)
            else:
                A = pd.read_csv(skeleton_file, header=0)
                A = A.to_numpy()
                #threshold = 1.5
                #A[np.abs(A) < threshold] = 0
                #A = np.tril(A) + np.triu(A.T, 1)
                self.skeleton = pd.DataFrame(data=(A == 0))
        else:
            self.skeleton = pd.DataFrame(data=np.ones((len(all_nodes), len(all_nodes))))

    def sample_posterior(self, data):
        posterior = []
        edge_interventions = []
        for _ in range(self.num_graphs):
            d_sub_inds = np.random.choice(np.arange(data.shape[0]), self.subset_size, replace=self.replace)
            d_sub = data.iloc[d_sub_inds]
            adj_mat, best_intervention = sp_gies(d_sub, self.skeleton.to_numpy())
            best_model = nx.relabel_nodes(nx.DiGraph(adj_mat),
                                {idx: i for idx, i in enumerate(self.all_nodes)})
            posterior.append(best_model)

            if best_intervention.shape[0] == 0:
                edge_interventions.append('')
                print("no optimal intervention found")
            else:
                edge_interventions.append("G{}".format(int(best_intervention[0])))
        return {'dags':posterior, 'optimal_interventions':edge_interventions}

class IGSPLearner(Learner):
    def __init__(self, all_nodes, num_graphs, subset_size, replace, no_interventions):
        super().__init__(all_nodes, num_graphs, subset_size, replace)
        self.no_interventions = no_interventions

    def sample_posterior(self, data):
        posterior = []
        edge_interventions = []
        for _ in range(self.num_graphs):
            d_sub_inds = np.random.choice(np.arange(data.shape[0]), self.subset_size, replace=self.replace)
            d_sub = data.iloc[d_sub_inds]
            d_sub_wo_targets = d_sub.loc[:, d_sub.columns != 'target']
            setting_list = [{'interventions':[row['target']]} for i,row in d_sub.iterrows()]
            ci_tester = conditional_independence.PlainCI_Tester(conditional_independence.hsic_test,
                                                                d_sub_wo_targets.to_numpy())
            invariance_tester = conditional_independence.PlainInvarianceTester(conditional_independence.hsic_invariance_test,
                                                                               d_sub_wo_targets.to_numpy())
            best_model = unknown_target_igsp(setting_list=setting_list, nodes=set(np.arange(10)), ci_tester=ci_tester,
                              invariance_tester=invariance_tester, verbose=True )
        return {'dags':posterior, 'optimal_interventions':edge_interventions}

