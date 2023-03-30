import itertools

from pgmpy.models import BayesianNetwork

from learner import HillClimbLearner,  GIESLearner
from strategies import RandomStrategy, IGStrategy, FixedStrategy, EdgeOrientationStrategy, DiversitySamplingStrategy
import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor
import metrics
from ast import literal_eval
import time
import functools
from causaldag import GaussDAG
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str)
    parser.add_argument('--percent_accept', type=float, default=1.0)
    parser.add_argument('--num_repeats', type=int)
    parser.add_argument('--num_graphs', type=int)
    parser.add_argument('--base', action='store_true', default=False)
    parser.add_argument('--larger_graph', action='store_true', default=False)
    parser.add_argument('--nthreads', type=int)
    parser.add_argument('--num_rounds', type=int)
    parser.add_argument('--prior_path', type=str)
    parser.add_argument('--ground_truth_net', type=str)
    parser.add_argument('--obs_data', type=str)
    parser.add_argument('--intervene_data', type=str, default=None)
    parser.add_argument('--seed',type=int, default=None)
    parser.add_argument('--structure_learner', type=str)
    parser.add_argument('--no_interventions', action='store_true', default=False)
    parser.add_argument('--net_params', type=str)
    parser.add_argument('--skeleton_file', type=str, default=None)
    return parser.parse_args()


def edge_to_adj(edges, all_nodes):
    adj_mat = np.zeros((len(all_nodes), len(all_nodes)))
    for e in edges:
        start = all_nodes.index(e[0])
        end = all_nodes.index(e[1])
        adj_mat[start,end] = 1
    return adj_mat

# Load a Bayesian network from a pickle file
# Obscure a path that needs to be discovered using structure learning
# Returns
# bn_generator: The true BN representing the underlying system
# prior_graph: The BN after obscuring paths, represents the prior information
# priors: The edges in the prior_graph
# path_to_obscure: Edges obscured in the graph, hard coded for now
# df: The observational dataset
def load_data(working_dir,bn_file, data_file, interv_file, prior_file, param_file, base):
    with open(prior_file, "r") as f:
        lines = f.read().splitlines()

    priors = []
    for line in lines[1:]:
        priors.append(literal_eval(line))

    edges = pd.read_csv(working_dir+'/'+bn_file, header=0)
    edges = set([(str(r['start']), str(r['end'])) for i,r in edges.iterrows() if r['edge'] == 1])
    df = pd.read_csv(working_dir+"/"+data_file, header=0)

    paths_obscured = [e for e in list(edges) if e not in priors]
    all_nodes = list(df.columns)
    all_nodes.remove('target')
    # If base flag is set, then force the prior to be the full graph
    if base:
        priors = list(edges)
    if interv_file is not None:
        df_interv = pd.read_csv(working_dir+'/'+interv_file, header=0)
        model=edges
        obscured_nodes = list(set([n for edge in paths_obscured for n in edge if n.startswith("G")]))
        genes = [i for i in all_nodes if i.startswith("G")]
        modes = [i for i in all_nodes if i.startswith("R")]
    else:
        df_interv = None
        params = pd.read_csv(working_dir+'/'+param_file, header=0)
        bias = params['bias']
        var = params['variance']
        model = GaussDAG(nodes=all_nodes, arcs=edges, biases=bias,
                      variances=var)
        obscured_nodes = list(set([n for edge in paths_obscured for n in edge]))
        genes = all_nodes
        modes = []
    adj_mat = edge_to_adj(list(edges), all_nodes)

    return {"model":model,"adj_mat":adj_mat }, priors, df, df_interv, all_nodes, obscured_nodes, genes, modes


def gene_from_kmer(top_kmers, function_mapping, non_function_mapping, mapping_kmer):
    # For each row in the gene matrix create determine if the gene is on or off based on the kmer there
    top_genes = set()
    for kmer in top_kmers:
        for mapping in (function_mapping, non_function_mapping):
            for gene, kmer_set in mapping.items():
                if kmer in kmer_set:
                    top_genes.add(gene)
    return top_genes


def create_learner(all_nodes, data, num_graphs, structure_learner,
                  fixed_edges, whitelist, no_interventions, skeleton_file):
    subset_size = min([1000, data.shape[0]])
    replace = data.shape[0] > 1000
    if structure_learner == "hillclimbsearch":
        if type(whitelist) == list and len(whitelist) == 0:
            whitelist = None
        learner = HillClimbLearner(all_nodes, num_graphs, subset_size, replace, fixed_edges, whitelist)
    elif structure_learner == "gies":
        learner = GIESLearner(all_nodes, num_graphs, subset_size, replace, no_interventions, skeleton_file)
    else:
        print("Unsupported structure learner")
        return
    return learner


def func(it,  args):
    bn_generator, priors, data_gene, data_intervene, all_nodes, obscured_nodes, genes, modes = \
        load_data(args.working_dir, args.ground_truth_net, args.obs_data, args.intervene_data, args.prior_path,
                  args.net_params,args.base)
    whitelist = list(itertools.product(obscured_nodes, modes))

    learner_bayesian = create_learner(all_nodes, data_gene, args.num_graphs, args.structure_learner, priors, whitelist, args.no_interventions, None)
    learner_hybrid = create_learner(all_nodes, data_gene, args.num_graphs, args.structure_learner, priors, whitelist, args.no_interventions, args.skeleton_file)

    strategies = [RandomStrategy(learner_bayesian, data_gene, data_intervene, obscured_nodes, genes, priors, name='GIES x random'),
                  IGStrategy(learner_bayesian, data_gene, data_intervene, obscured_nodes, genes, priors, name='GIES x info. gain'),
                   EdgeOrientationStrategy(learner_bayesian, data_gene, data_intervene, obscured_nodes,genes,priors, name="GIES x edge orient."),
                   RandomStrategy(learner_hybrid, data_gene, data_intervene, obscured_nodes, genes, priors,
                                  name='SP-GIES x random'),
                   IGStrategy(learner_hybrid, data_gene, data_intervene, obscured_nodes, genes, priors, name='SP-GIES x info. gain'),
                   EdgeOrientationStrategy(learner_hybrid, data_gene, data_intervene, obscured_nodes,genes,priors, name='SP-GIES x edge orient.')]
    #strategies = [ DiversitySamplingStrategy(learner_hybrid, data_gene, data_intervene, obscured_nodes, genes, priors,
#                                 name='SP-GIES x diversity')]
    for s in strategies:
        s.init(args.num_rounds, bn_generator)
        if it == 0:
            metrics.visualize_posterior(s.posterior['dags'], bn_generator['model'], all_nodes, args.working_dir,
                                         filename="initial_posterior.png")
        s.run_rounds(bn_generator, args.percent_accept)
        print(s.shd)
    return strategies


def run_interventions_full_graph(args):
    func_partial = functools.partial(
        func,
        args = args
    )
    results = []
    chunksize = max(1, args.num_repeats // args.nthreads)
    print("Launching processes")
    with ProcessPoolExecutor(max_workers=args.nthreads) as executor:
        for result in executor.map(func_partial, np.arange(args.num_repeats), chunksize=chunksize):
            results.append(result)
    return results


def run_interventions_full_graph_fixed_order(ordering,args):
    bn_generator, priors, data_gene, data_intervene, all_nodes, obscured_nodes, genes, modes = \
        load_data(args.working_dir, args.ground_truth_net, args.obs_data, args.intervene_data, args.prior_path, args.net_params, args.base)
    scores = np.zeros((args.num_repeats, args.num_rounds+1))
    whitelist = list(itertools.product(obscured_nodes, modes))

    for n in range(args.num_repeats):
        learner = create_learner(all_nodes, data_gene, args.num_graphs, args.structure_learner, priors, whitelist,
                                 args.no_interventions)
        strategy = FixedStrategy( learner, data_gene, data_intervene, obscured_nodes, genes, priors, list(ordering))
        strategy.init(args.num_rounds, bn_generator, test_learned=(n==0))
        strategy.run_rounds(bn_generator, args.percent_accept)
        scores[n] = strategy.shd
    return scores


def run_brute_force(args):
    bn_generator, priors, data_gene, data_intervene, all_nodes, obscured_nodes, genes, modes = \
        load_data(args.working_dir, args.ground_truth_net, args.obs_data, args.intervene_data, args.prior_path,
                  args.net_params, args.base)
    nodes = obscured_nodes
    possible_orderings = list(itertools.permutations(nodes, r=args.num_rounds))
    rand_sub = np.random.choice(np.arange(len(possible_orderings)),size=1,replace=False)
    possible_orderings = [possible_orderings[r] for r in rand_sub]

    fixed_order_fn = functools.partial(
        run_interventions_full_graph_fixed_order,
        args=args
    )
    all_scores = []
    chunksize = max(1, len(possible_orderings) // args.nthreads)
    with ProcessPoolExecutor(max_workers=args.nthreads) as executor:
        for score_ord in executor.map(fixed_order_fn, possible_orderings, chunksize=chunksize):
            all_scores.append(score_ord)
    return np.array(all_scores), possible_orderings, args.num_rounds


def main():
    args = get_args()
    if args.seed:
        print("Setting seed")
        rs = RandomState(MT19937(SeedSequence(123456789)))
        #np.random.seed(args.seed)
    if not args.base and not args.larger_graph:
        print("Running all possible orderings")
        score_ord, possible_orderings, num_rounds = run_brute_force(args)
        metrics.plot_score(score_ord, possible_orderings, num_rounds, args.working_dir, filename="shd_all_orders.png", score_type="SHD")

    strategies = run_interventions_full_graph(args)
    metrics.plot(strategies, args.num_repeats, args.num_rounds, args.working_dir)


if __name__ == '__main__':
    init = time.time()
    main()
    print(time.time() - init)
