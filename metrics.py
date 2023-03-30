import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pgmpy.models
import causaldag as cd
import re
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
class Metrics(object):
    def __init__(self):
        self.metric_dict = {}

    def log_strategy(self, strategy, num_repeats):
        if strategy.get_name() in self.metric_dict.keys():
            # add score, selected, utility to existing
            next_metrics = self.metric_dict[strategy.get_name()]
            repeat = next_metrics['repeat']
            next_metrics['shd'][repeat] = strategy.shd
            next_metrics['sid'][repeat] = strategy.sid
            next_metrics['auc'][repeat] = strategy.auc
            next_metrics['shd_g1'][repeat] = strategy.shd_G1

            next_metrics['selected'][repeat] = strategy.selected
            next_metrics['utility'][repeat] = strategy.utility
            next_metrics['repeat'] = repeat + 1
            self.metric_dict[strategy.get_name()] = next_metrics

        else:
            selected_all = np.zeros((num_repeats, strategy.num_rounds))
            selected_all[0] = strategy.selected

            shd_all = np.zeros((num_repeats, strategy.num_rounds + 1))
            shd_all[0] = strategy.shd

            shd_g1_all = np.zeros((num_repeats, strategy.num_rounds + 1))
            shd_g1_all[0] = strategy.shd_G1

            sid_all = np.zeros((num_repeats, strategy.num_rounds + 1))
            sid_all[0] = strategy.sid

            auc_all = np.zeros((num_repeats, strategy.num_rounds + 1))
            auc_all[0] = strategy.auc

            utility_all = np.zeros((num_repeats, strategy.num_rounds, len(strategy.possible_interventions)))
            utility_all[0] = strategy.utility
            self.metric_dict[strategy.get_name()] ={"shd": shd_all,
                                                    "shd_g1":shd_g1_all,
                                                    "sid": sid_all,
                                                    "auc": auc_all,
                                                    "selected": selected_all ,
                                                    "utility": utility_all,
                                                    "repeat": 1}
    def get_metrics(self):
        strategy_names = list(self.metric_dict.keys())
        shd = []
        shd_g1 = []
        sid = []
        auc = []
        selected = []
        utility = []
        for s in strategy_names:
            shd.append(self.metric_dict[s]['shd'])
            sid.append(self.metric_dict[s]['sid'])
            auc.append(self.metric_dict[s]['auc'])
            shd_g1.append(self.metric_dict[s]['shd_g1'])
            selected.append(self.metric_dict[s]['selected'])
            utility.append(self.metric_dict[s]['utility'])
        return strategy_names, shd, shd_g1, sid, auc, selected, utility

def plot(strategies, num_repeats, num_rounds, working_dir):
    logger = Metrics()
    for r in strategies:
        for s in r:
            logger.log_strategy(s, num_repeats)
    strategy_names, shd, shd_g1, sid, auc, selected, utility = logger.get_metrics()
    fig, [ax_shd, ax_sid, ax_auc] = plt.subplots(1,3, figsize=(25,10))
    interventions = strategies[0][0].possible_interventions

    plot_score(shd, strategy_names, num_rounds, score_type='SHD', axs=ax_shd)
    #plot_score(shd_g1, strategy_names, num_rounds,score_type='SHD for gene 1',a)
    plot_score(sid, strategy_names, num_rounds,score_type='SID', axs=ax_sid)
    plot_score(auc, strategy_names, num_rounds,score_type='AUC', axs=ax_auc)
    lgd = plt.legend(loc='upper right', bbox_to_anchor=(0, -0.1),
          ncol=2, fancybox=True, shadow=True)
    fig.subplots_adjust(bottom=0.3)
    plt.savefig(working_dir+"/all_scores.png", bbox_inches='tight')

    plt.show()

    plot_selected_strategy(strategies[0][0].learner.node_2ix, selected, strategy_names, interventions, num_rounds, working_dir,
                                   filename="selected.png")
    plot_utility(utility, shd, strategy_names, interventions, num_rounds, working_dir, filename1="utility_graph.png",
                    filename2="utility_shd.png")

def plot_score(scores, names, num_rounds, score_type, axs):
    print("plotting shd")
    number_of_plots = len(names)
    colormap = plt.cm.turbo
    axs.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, number_of_plots)])
    xaxis = ["{}".format(i) for i in range(num_rounds)]
    xaxis = ['Init'] + xaxis
    print(type(scores))
    for l, s in zip(names, scores):
    # Plot the SHD over number of interventions for each subgraph
        #conf_int = stats.norm.interval(alpha=0.95, loc=np.mean(s, axis=0), scale=stats.sem(s, axis=0))
        axs.errorbar(xaxis, np.mean(s, axis=0), yerr=np.std(s,axis=0),
                      marker='o', label=l, capsize=5, markersize=10)
    #axs.legend(bbox_to_anchor=(1, 1))
    axs.set_ylabel(score_type)
    axs.set_title("{} over rounds of intervention".format(score_type))
    # plt.savefig(working_dir + "/" + filename)
    # plt.clf()


def plot_selected_strategy(mapping, selected_strategy, names, interventions, num_rounds, working_dir, filename):
    print("plot selected strategy")
    for l, s in zip(names, selected_strategy):
        fig, axs = plt.subplots()
        data = np.zeros(num_rounds)
        labels = ["{}".format(i) for i in range(num_rounds)]
        for i in range(len(interventions)):
            g = list(interventions)[i]
            g = mapping[g]
            count = [(s[:,r] == g).sum() for r in
                     range(s.shape[1])]
            axs.bar(labels, count, width=0.35, bottom=data, label=g)
            data += count
            axs.set_title("Chosen interventions")
            axs.legend()
        plt.savefig(working_dir + "/" + l+ "_" +filename)
        plt.ylabel("Number of repeated runs")
        plt.clf()

def plot_utility(utility, scores, names, interventions, num_rounds, working_dir, filename1, filename2):
    print("plot utility")
    labels = ["{}".format(i) for i in range(num_rounds)]

    for l, s, u in zip(names, scores, utility ):
        fig, axs = plt.subplots()
        for n in range(len(interventions)):
            axs.errorbar(labels, np.mean(u[:,:,n], axis=0),
                           yerr=np.std(u[:,:,n], axis=0), marker='o', capsize=5, label=interventions[n])
        axs.legend()
        axs.set_ylabel("IG")
        axs.set_title("IG over rounds of intervention")
        plt.savefig(working_dir + "/" + l + "_" + filename1)
        plt.clf()

        fig, axs = plt.subplots()
        axs.plot(np.mean(s, axis=0)[1:], np.sum(np.mean(u, axis=0), axis=1), marker='o')
        axs.set_title("Structural Hamming Distance vs Utility")
        axs.set_ylabel("IG")
        axs.set_xlabel("SHD")

        plt.savefig(working_dir + "/" + l + "_" +  filename2)
        plt.clf()

def plot_graph_dist(graph_dist, scores, working_dir, filename):
    m = ["o", "s", "D", "*"]
    fig, axs = plt.subplots()

    test = graph_dist[:,0]
    test_scores = scores[:,0]
    for i in range(test.shape[0]):
        plt.plot(np.arange(test.shape[1]), test[i], marker=m[i%4], label="round {}, SHD is {}".format(i, test_scores[i]))
    axs.set_xlabel("Graph index")
    axs.set_ylabel("P(G|D)")
    axs.legend()
    plt.savefig(working_dir + "/" + filename)
    plt.clf()

def visualize_posterior(dags, true_dag, all_nodes, working_dir, filename):
    # Split according to how true dag is encoded (either as BN or as edge list)
    if type(true_dag) == pgmpy.models.BayesianNetwork:
        edges = list(true_dag.edges())
        ave_edges = dict(zip(edges, np.zeros(len(edges), dtype=int)))
        graph = true_dag
    elif type(true_dag) == cd.GaussDAG:
        edges = list(true_dag.arcs)
        ave_edges = dict(zip(edges, np.zeros(len(edges), dtype=int)))
        graph = nx.DiGraph()
        graph.add_nodes_from(all_nodes)
        graph.add_edges_from(edges)
    else:
        ave_edges = dict(zip(true_dag, np.zeros(len(true_dag), dtype=int)))
        graph = nx.DiGraph()
        graph.add_nodes_from(all_nodes)
        graph.add_edges_from(true_dag)
    for edge in ave_edges.keys():
        for dag in dags:
            if edge in dag.edges():
                ave_edges[edge] += 1
    fig, ax = plt.subplots()

    pos = {'G1': np.array([-0.5, 0.6]),
           'R1': np.array([0, 0.5]),
           'G2': np.array([-0.5, 0.5]),
           'G3': np.array([-0.5, 0.4]),
           'G4': np.array([-0.5, 0.1]),
           'R2': np.array([0, 0]),
           'G5': np.array([-0.5, 0]),
           'G6': np.array([-0.5, -0.1]),
           'R3': np.array([0, -0.5]),
           'G7': np.array([-0.5, -0.6]),
           'G8': np.array([-0.5, -0.5]),
           'G9': np.array([-0.5, -0.4]),
           'P1': np.array([0.5, 0.5])} \
        if "P1" in all_nodes else nx.spring_layout(graph)

    edges = list(graph.edges)
    ave_edges_weights = []
    for e in edges:
        ave_edges_weights.append(ave_edges[e]/len(dags))
    nx.draw(graph, pos=pos, node_color='white', edgelist=edges, ax=ax,
            edge_color=ave_edges_weights, width=5.0, with_labels=True, edge_cmap=plt.cm.Blues)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                               norm=plt.Normalize(vmin=np.min(ave_edges_weights),
                                                  vmax=np.max(ave_edges_weights)))
    sm._A = []
    plt.colorbar(sm, ax=ax)
    ax.set_title("Estimated Posterior from observation data only")
    plt.savefig(working_dir+"/" + filename)


