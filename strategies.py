import networkx as nx
import numpy as np
import scipy
import pandas as pd
import math
from intervention import Intervention, KO_KD_Intervention, cov2dag
# from pgmpy.estimators import MaximumLikelihoodEstimator
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.models import BayesianNetwork
import cdt
from collections import Counter
import causaldag as cd
from sklearn.cluster import SpectralClustering
import re
class Strategy(object):
    def __init__(self, learner, observational_data, interventional_data, possible_interventions, genes, priors):
        self.possible_interventions = possible_interventions
        self.genes = genes
        self.obs_data = observational_data
        self.inter_data = interventional_data
        self.data = self.obs_data

        self.priors = priors
        self.posterior = []
        self.learner = learner

        self.num_rounds = None
        self.shd = None
        self.sid = None
        self.auc = None
        self.shd_G1 = None
        self.selected = None
        self.utility = None

        self.gene_2ix = {}

    def log(self):
        print("Log scores to global metric holder")
    def get_name(self):
        return ''

    def init(self, num_rounds, bn_generator, test_learned = False):
        # initialize metric containers
        self.num_rounds = num_rounds
        self.shd = np.zeros(num_rounds+1)
        self.shd_G1 = np.zeros(num_rounds+1)
        self.sid = np.zeros(num_rounds+1)
        self.auc = np.zeros(num_rounds+1)

        self.selected = [] # holds intervention objects
        self.utility = np.zeros((num_rounds, len(self.possible_interventions)))

        # update init score
        self.posterior = self.learner.sample_posterior(self.data)
        self.shd[0] = shd(self.posterior['dags'], bn_generator['adj_mat'])
        self.shd_G1[0] = shd_gene_1(self.posterior['dags'], bn_generator)
        self.sid[0] = sid(self.posterior['dags'], bn_generator['adj_mat'])
        self.auc[0] = auc(self.posterior['dags'], bn_generator['adj_mat'])

        if test_learned:
            if self.shd[0] == 0:
                print("Graph learned before interventions!")

    def run_rounds(self, bn_generator, percent_accept):
        assert(self.shd is not None)
        for n in range(self.num_rounds):
            print("ROUND {}".format(n))
            selected, utility = self.run_strategy()
            print("SELECTED {}".format(selected))
            self.run_exps(selected, bn_generator)
            self.posterior = self.learner.sample_posterior(self.data)

            self.utility[n] = utility
            self.shd[n+1] = shd(self.posterior['dags'], bn_generator['adj_mat'])
            self.shd_G1[n+1] = shd_gene_1(self.posterior['dags'], bn_generator)
            self.sid[n+1] = sid(self.posterior['dags'], bn_generator['adj_mat'])
            self.auc[n+1] = auc(self.posterior['dags'], bn_generator['adj_mat'])
            self.update_priors(percent_accept)
            print("SCORE {}".format(self.shd[n+1]))
        for s in range(len(self.selected)):
            #self.selected[s] =  (int)(re.split('[a-z]+', self.selected[s].name, flags = re.IGNORECASE)[-1])
            self.selected[s] = self.learner.get_index(self.selected[s].name)

    def run_strategy(self):
        print("super class for strategy exectution")
        return 0,0


    # Run an "experiment" given a set of interventions
    def run_exps(self, selected, bn_generator):
        # Sample fake system for new data
        # Select all the data points where selected == 1, randomly choose subset
        # Set the selected column to 0
        # Apply to BN to get experiment results
        if self.inter_data is not None:
            new_i_data,intervention = sample_from_file(selected, self.inter_data, self.obs_data.shape[0], self.learner)
        else:
            new_i_data, intervention = sample_gauss_dag(self.genes,selected, bn_generator['model'], self.obs_data.shape[0], self.learner)

            # new_i_data, intervention = sample_by_modification(self.data, selected,
            #                                                   bn_generator['model'],
            #                                                   self.genes)
        self.data = pd.concat([self.data, new_i_data])
        self.selected.append(intervention)

    # After a set of interventions, update the prior graph based
    # on the most recently sampled dags
    def update_priors(self, percent_acceptance):
        counts = Counter(edge for dag in self.posterior['dags'] for edge in dag.edges)
        for edge, count in counts.items():
            if count / len(self.posterior['dags']) >= percent_acceptance:
                self.priors.append(edge)

class RandomStrategy(Strategy):
    def __init__(self, learner, observational_data, interventional_data, possible_interventions, genes, priors, name):
        super().__init__( learner, observational_data, interventional_data, possible_interventions, genes, priors)
        self.name=name

    def run_strategy(self):
        rand_selected = np.random.choice(list(self.possible_interventions), 1)[0]
        return rand_selected, 0
    def get_name(self):
        return self.name

class DiversitySamplingStrategy(Strategy):
    def __init__(self, learner, observational_data, interventional_data, possible_interventions, genes, priors, name):
        super().__init__( learner, observational_data, interventional_data, possible_interventions, genes, priors)
        self.name=name

    def get_name(self):
        return self.name

    def run_strategy(self):
        # Cluster self.data with spectral cluster
        nclusters = 3
        data_for_clustering = self.data.drop(columns=['target'])
        clustering = SpectralClustering(n_clusters=nclusters,assign_labels = 'discretize', random_state = 0).fit_predict(data_for_clustering.T)

        # Choose from a cluster that hasn't been chosen yet
        # self.selected are intervention objects, need
        chosen_already = [clustering[self.genes.index(i.name)] for i in self.selected]
        next_cluster = [i for i in range(0,nclusters) if i not in chosen_already]
        if len(next_cluster) > 0:
            chosen = np.random.choice(next_cluster, size=1)[0]
            possible_nodes = [i for i in range(len(clustering)) if clustering[i] == chosen]
            if len(possible_nodes) > 0:
                diverse_select = np.random.choice(possible_nodes, 1)[0]
                print(possible_nodes, self.genes)
                return self.genes[diverse_select], 0 # Assume here that all genes are possibel interventions

        # If all clusters have been chosen, randomly select
        rand_selected = np.random.choice(list(self.possible_interventions), 1)[0]
        print("Making random selection")
        return rand_selected, 0




class FixedStrategy(Strategy):
    def __init__(self,  learner, observational_data, interventional_data, possible_interventions, genes, priors, ordering):
        super().__init__( learner, observational_data, interventional_data, possible_interventions, genes, priors)
        self.ordering = ordering

    def run_strategy(self):
        return self.ordering.pop(0), 0

    def get_name(self):
        return 'fixed'


class IGStrategy(Strategy):
    def __init__(self,  learner, observational_data, interventional_data, possible_interventions, genes, priors, name=None):
        super().__init__( learner, observational_data, interventional_data, possible_interventions, genes, priors)
        self.name=name

    # Calculate the probability of the data given a DAG P(D|G) assuming that the
    # DAG is a linear Gaussian model
    def gauss_dag_logpdf(self, dag, data, cov, node):
        gauss_dag = cov2dag(cov, dag)
        nodes = list(data.columns)
        phenotype = nodes.index(node) if node == 'P1' else None
        logpdf = gauss_dag.logpdf(data.values, special_node=phenotype)
        return logpdf

    def set_MLE(self, D, dag):
        bn = cd.GaussDAG(nodes=dag.nodes, arcs=dag.edges)
        bn = cd.GaussDAG.fit_mle(bn,samples=D)
        return bn

    '''
    DEPRECATED PGMPY BNs FOR DISCRETE TOY NETWORKS
    def set_CPTs_theta_MLE(self, D, dag):
        bn = BayesianNetwork()
        bn.add_nodes_from(dag.nodes)
        bn.add_edges_from(dag.edges)
        mle = MaximumLikelihoodEstimator(model=bn, data=D)
        states = 2

        # Incorporate known prior parameters for the phenotype node
        for n in bn.nodes():
            if n.startswith("P"):  # OR gates
                flow = np.array([0, 1])
                cpd = np.array([flow for i in range(states ** len(bn.get_parents(n)))])
                cpd[0] = flow[::-1]
                cpd = np.transpose(cpd)
                tab = TabularCPD(n, states, cpd, evidence=bn.get_parents(n),
                                 evidence_card=[len(p) for p in bn.get_parents(n)])
                bn.add_cpds(tab)

        # Assign MLE parameters to all other nodes
        for p in mle.get_parameters():
            if not p.variable.startswith("P"):
                bn.add_cpds(p)
        return bn
    '''

    def run_strategy(self):
        # Maximize information gain between P1 and P2 where P1 is probability of a graph G given data D
        # and P2 is the probability of a graph G given data D and interventional data D_I where
        # D_I is generated from do(X_i=Val(intervention)

        # For each intervention, calculate the sum of P1*log(P1/P2) over all DAGs
        IG = np.zeros(len(self.possible_interventions))
        data = self.obs_data.loc[:, self.obs_data.columns != 'target']
        D_norm = normalize_datasets([data])[0]

        cov = np.cov(D_norm, rowvar=False)  # numpy uses rows as features
        # dags_w_thetas = [set_CPTs(dag, mutation_rate=mutation_rate) for dag in dags]

        special_node = "P1" if "P1" in self.data.columns else None
        logP = np.array([self.gauss_dag_logpdf(dag, D_norm, cov, node=special_node).sum(axis=0) for dag in self.posterior['dags']])

        dags_w_thetas =  [self.set_MLE(data, dag) for dag in self.posterior['dags']]
        # Loop through all selected interventions and sum the interventional log liklihood
        for int_set in self.selected:
            for d, dag in enumerate(dags_w_thetas):
                logP[d] += int_set.gauss_dag_logpdf(dag, D_norm, special_node).sum(axis=0)
            D_norm = pd.concat([D_norm, int_set.get_data()])

        logP_norm = logP - scipy.special.logsumexp(logP)  # logP(G|D) (logP1 in paper)
        P = np.exp(logP_norm)
        assert (np.sum(P) <= 1.1 and np.sum(P) >= 0.9)

        # Set arbitrary the number of y samples
        M = 1
        # For each possible next intervention, calculate the log liklihood
        for i, int_set in enumerate(self.possible_interventions):
            logQ_norm = np.zeros(len(self.posterior['dags']))
            P2 = np.zeros(len(self.posterior['dags']))
            for d, dag in enumerate(dags_w_thetas):
                for m in range(M):
                    _, int_type = sample_gauss_dag(self.genes,int_set, dag, self.obs_data.shape[0], self.learner)
                    # logQ  = logP(G|D) + logP(y|G)
                    logQ_iter = int_type.gauss_dag_logpdf(dag, D_norm, special_node) + logP[d]
                    # logQ_norm = logP2 in paper
                    logQ_norm[d] += np.sum(logQ_iter - scipy.special.logsumexp(logQ_iter), axis=0)
                    P2[d] = np.exp(logQ_norm[d] - scipy.special.logsumexp(logQ_norm[d]))

            # IG = P1*logP1 - P2*logP2
            IG[i] += np.sum(P * logP_norm - P2 * logQ_norm)
            assert (not math.isinf(IG[i]) and not math.isnan(IG[i]))
        # Choose the intervention with the largest information gain
        selected = self.possible_interventions[np.random.choice(np.where(IG == IG.max())[0])]
        return selected, IG


    def get_name(self):
        if self.name is None:
            return 'information_gain'
        else:
            return self.name


class EdgeOrientationStrategy(Strategy):
    def __init__(self,  learner, observational_data, interventional_data, possible_interventions, genes, priors, name=None):
        super().__init__( learner, observational_data, interventional_data, possible_interventions, genes, priors)
        self.name = name
    def run_strategy(self):
        next_intervention = self.posterior['optimal_interventions']
        for i, intervention in enumerate(next_intervention):
            if intervention == '':
                next_intervention[i] = np.random.choice(self.possible_interventions, 1)[0]
        values, counts = np.unique(next_intervention, return_counts=True)
        m = values[counts.argmax()]
        return m, 0

    def get_name(self):
        if self.name is None:
            return 'edge_orientation'
        else:
            return self.name


# Compute average SHD over all DAGs
def shd(dags, true_dag):
    if true_dag is None:
        return 0
    error = 0
    for d in dags:
        error += cdt.metrics.SHD(true_dag, d, False)
    return error / len(dags)

def shd_gene_1(dags, true_dag):
    error = 0
    if true_dag is None:
        return 0
    true_dag_nx = nx.DiGraph()
    true_dag_nx.add_nodes_from(true_dag['model'].nodes)
    true_dag_nx.add_edges_from(true_dag['model'].arcs)
    for d in dags:
        true_parents = set(true_dag_nx.predecessors("G1"))
        dag_parents = set(d.predecessors("G1"))
        error += len( true_parents- dag_parents)
    return error/len(dags)


def auc(dags, true_dag):
    if true_dag is None:
        return 0
    score = 0
    for d in dags:
        score += cdt.metrics.precision_recall(true_dag, d)[0]
    return score / len(dags)

def sid(dags, true_dag):
    error = 0
    for d in dags:
        error += cdt.metrics.SID(true_dag, d)
    return error / len(dags)

def normalize_datasets(dfs):
    inds = [0] + [df.shape[0] for df in dfs]
    df = pd.concat(dfs)
    mean = df.mean(axis=0)
    stddev = df.std(axis=0)
    df_norm = (df - mean) / stddev
    dfs = [df_norm.iloc[inds[i-1]:inds[i-1]+inds[i]] for i in range(1,len(inds))]
    return dfs

def sample_from_file(intervention, data, data_obs_ind, learner):
    gene_ind = learner.get_index(intervention)
    df = data[data['target'] == gene_ind]
    df_int = df.loc[:, df.columns != 'target']
    if df_int.shape[0] == 2:
        intervention = KO_KD_Intervention(intervention, df_int, data_obs_ind)
    else:
        intervention = Intervention(intervention, df_int, data_obs_ind)
    return df, intervention

def sample_gauss_dag(columns, intervention, bn, data_obs_ind, learner, nsamples=10):
    i_data = bn.sample_interventional(cd.Intervention( {intervention:cd.ConstantIntervention(val=0)}),
                                      nsamples=nsamples)
    i_data = pd.DataFrame(i_data, columns=columns)
    i_data['target'] = learner.get_index(intervention)
    intervention = Intervention(intervention, i_data, data_obs_ind)
    return i_data, intervention

'''
DEPRECATED PGMPY BNs FOR DISCRETE TOY NETWORKS
def sample_by_modification(data, intervention, bn, inputs):
    i_samples = data.loc[(data[intervention] == 1)]
    idx = np.random.choice(np.arange(i_samples.shape[0]), size=np.min([100, i_samples.shape[0]]), replace=False)
    i_samples = i_samples[inputs] # TODO possible interventions here is wrong
    i_samples[intervention] = 0
    i_samples = i_samples.iloc[idx]
    new_i_data = bn.predict(i_samples)
    i_samples = i_samples.reset_index(drop=True)

    new_i_data = pd.concat([i_samples, new_i_data], axis=1)
    new_i_data_norm = normalize_datasets([data, new_i_data])[1]

    new_i_data['target'] = (int(intervention[1:])) * np.ones(new_i_data.shape[0])

    intervention = Intervention(intervention, new_i_data_norm, data.shape[0])
    return new_i_data, intervention

'''
