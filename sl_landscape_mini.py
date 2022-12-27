import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import auc
import itertools
import warnings
import time
import cdt
from cdt.causality.graph import GIES
warnings.filterwarnings("ignore")
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
import matplotlib
matplotlib.rc('font', **font)

def adj_to_dag(adj, all_nodes,fixed_edges=None):
    dag = nx.DiGraph()
    dag.add_nodes_from(all_nodes)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if np.abs(adj[i,j]) > 0:
                dag.add_edge(all_nodes[i], all_nodes[j], weight=np.abs(adj[i,j]))
    if fixed_edges:
        dag.add_edges_from(fixed_edges)
    return dag

def edge_to_adj(edges, all_nodes):
    adj_mat = np.zeros((len(all_nodes), len(all_nodes)))
    for e in edges:
        start = all_nodes.index(e[0])
        end = all_nodes.index(e[1])
        adj_mat[start,end] = 1
    return adj_mat

def edge_to_dag(edges, weights=None):
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    return dag
# Report the SHD, nx graph for a gies run                                                                              
def gies_run(df, true_graph, targets, target_index, fixedGaps=None):
    if targets is not None:
        targets = pd.DataFrame(data=targets)
        target_index = pd.DataFrame(data=target_index)
    # GIES W INTERVENTIONS                                                                                             
    obj = GIES(score="int", verbose=True)
    output = obj.create_graph_from_data(df, targets, target_index, fixedGaps=fixedGaps)[0]
    shd = cdt.metrics.SHD(true_graph, output, False)
    sid = cdt.metrics.SID(true_graph, output)
    auc,pr = cdt.metrics.precision_recall(true_graph, output)
    return shd, sid, auc, output

def test_regulondb():
    true_adj = pd.read_csv("./regulondb2/ground_truth.csv", header=None).values
    df = pd.read_csv("./regulondb2/data.csv", header=None)
    genes = [i[0] for i in pd.read_csv("./regulondb2/genes.txt", header=None).values]
    true_graph = adj_to_dag(true_adj,all_nodes=genes)
    print(len(true_graph.nodes()), len(true_graph.edges()))
    aracne_network = pd.read_csv("./regulondb2/output/network2.txt", sep='\t',header=0)
    clr_network = pd.read_csv("./regulondb2/adj_mat.csv", header=None).to_numpy()
    #pc_network = pd.read_csv("./regulondb2/cupc_adj_mat.csv", header=0).to_numpy()
    sp_gies_network = pd.read_csv("./regulondb2/sp-gies-adj_mat.csv", header=0).to_numpy()
    #gies_network =  pd.read_csv("./regulondb2/gies-adj_mat.csv", header=0).to_numpy()
    #print(pc_network.shape)
    inds = pd.read_csv("./regulondb2/inds.csv", header=None, sep=",").iloc[0].values
    #print(inds, inds.shape)
    #  zero out gene->gene and gene-> tf interactions
    with open('./regulondb2/tfs_names.txt') as f:
        lines = f.readlines()
    tfs = lines[0].split("\t")
    edges_pos = [(row['Regulator'][1:-1], row['Target'][1:-1]) for i,row in aracne_network.iterrows() if row['Regulator'][1:-1] in genes and row['Target'][1:-1] in genes]
    weights = [row['MI'] if (row['Regulator'][1:-1] in tfs) else 0 for i,row in aracne_network.iterrows() if row['Regulator'][1:-1] in genes and row['Target'][1:-1] in genes]
    aracne_graph = edge_to_dag(edges_pos, weights=weights)
    aracne_graph.add_nodes_from(genes)

    threshold = 6.917 # This achieves 60% precision
    clr_network[clr_network < threshold] = 0
    clr_graph = adj_to_dag(clr_network,genes)

    shd = cdt.metrics.SHD(true_graph, aracne_graph, False)
    sid = cdt.metrics.SID(true_graph, aracne_graph)

    auc,pr = cdt.metrics.precision_recall(true_graph,aracne_graph)
    print("ARACNE {} {} {}".format( shd, sid, auc))
    #print(pr)

    shd = cdt.metrics.SHD(true_graph, clr_graph, False)
    sid = 0 #cdt.metrics.SID(true_graph, clr_graph)
    auc,pr = cdt.metrics.precision_recall(true_graph,clr_graph)
    print("CLR {} {} {}".format( shd, sid, auc))
    #print(pr)

    df_new = df.loc[:, df.columns != 'target']
    #PC                                                                                                        
    #pc_network = pc_network[inds][:,inds]
    #print(pc_network.shape)
    #pc_graph = adj_to_dag(pc_network, genes)
    #shd = cdt.metrics.SHD(true_graph, pc_graph, False)
    #sid = cdt.metrics.SID(true_graph, pc_graph)                                                                    
    #auc,pr = cdt.metrics.precision_recall(true_graph,pc_graph)
    #print("PC {} {} {}".format(shd,sid,auc))

    sp_gies_graph = adj_to_dag(sp_gies_network, genes)
    shd = cdt.metrics.SHD(true_graph, sp_gies_graph, False)
    sid = 0#cdt.metrics.SID(true_graph, sp_gies_graph)
    auc, pr = cdt.metrics.precision_recall(true_graph, sp_gies_graph)
    print("SP-GIES {} {} {}".format(shd, sid, auc))

    #gies_graph = adj_to_dag(gies_network, genes)                                                          
    #shd = cdt.metrics.SHD(true_graph, gies_graph, False)                                                         
    #sid = cdt.metrics.SID(true_graph, gies_graph)                                                                
    #auc, pr = cdt.metrics.precision_recall(true_graph, gies_graph)                                               
    #print("GIES-IO {} {} {}".format(shd, sid, auc)) 
    
    # GIES ONLY OBS DATA
    df_new = df_new.iloc[inds].transpose()
    #df_new.to_csv("./regulondb2/data_smaller.csv", header=False, index=False)
    #print('wrote file')
    score, sid, auc, output = gies_run(df_new, true_graph, None, None)
    print("GIES {} {} {}".format(score, sid, auc))
    
    shd = cdt.metrics.SHD(true_graph, np.zeros((len(genes), len(genes))), False)
    sid = 0 #cdt.metrics.SID(true_graph, np.zeros((len(genes), len(genes))))
    auc,pr = cdt.metrics.precision_recall(true_graph,np.zeros((len(genes), len(genes))))
    print("EMPTY {} {} {}".format( shd, sid, auc))
    #print(pr)

def test_ARACNE_CLR_random():
    random = [ "small_norm"]
    num_graphs = 1
    for r in random:
        stats_clr = np.zeros(3)
        stats_aracne = np.zeros(3)
        stats_empty = np.zeros(3)
        for n in range(num_graphs):
            edges = pd.read_csv( "./random_test_set_1000_{}/bn_network_{}.csv".format(r, n), header=0)
            aracne_network = pd.read_csv("./random_test_set_1000_{}/output/network.txt".format(r,n), sep='\t',
                                         header=0)
            clr_network = pd.read_csv("./random_test_set_1000_{}/adj_mat_{}.csv".format(r,n), header=None)
            df = pd.read_csv("./random_test_set_1000_{}/data_{}.csv".format(r,n), header=0)

            edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
            true_graph = edge_to_dag(edges_pos, type=DAG)
            nodes = list(df.columns)
            nodes.remove('target')
            true_graph.add_nodes_from(nodes)

            edges_pos = [(row['Regulator'], row['Target']) for i, row in aracne_network.iterrows()]
            weights = [row['MI'] for i, row in aracne_network.iterrows()]
            aracne_graph = edge_to_dag(edges_pos, type=DAG, weights=weights)
            aracne_graph.add_nodes_from(nodes)

            A = clr_network.to_numpy()
            threshold = 1.5
            A[np.abs(A) < threshold] = 0
            clr_graph = adj_to_dag(A, nodes)

            shd = cdt.metrics.SHD(true_graph, aracne_graph, False)
            sid = cdt.metrics.SID(true_graph, aracne_graph)
            auc, pr = cdt.metrics.precision_recall(true_graph, aracne_graph)
            stats_aracne += np.array([shd, sid, auc])

            shd = cdt.metrics.SHD(true_graph, clr_graph, False)
            sid = cdt.metrics.SID(true_graph, clr_graph)
            auc, pr = cdt.metrics.precision_recall(true_graph, clr_graph)
            stats_clr += np.array([shd, sid, auc])

            shd = cdt.metrics.SHD(true_graph, np.zeros((len(nodes), len(nodes))), False)
            sid = cdt.metrics.SID(true_graph, np.zeros((len(nodes), len(nodes))))
            auc, pr = cdt.metrics.precision_recall(true_graph, np.zeros((len(nodes), len(nodes))))
            stats_empty += np.array([shd, sid, auc])

        stats_aracne/=num_graphs
        stats_clr/=num_graphs
        stats_empty/=num_graphs

        print(stats_clr)
        print(stats_aracne)
        print(stats_empty)

def test_hybrid_dream4():
    for d in range(1,6):
        print("NETWORK {}".format(d))
        edges = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_goldstandard.csv".format(d, d), header=0)
        df = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_obs.csv".format(d, d), header=0)
        df_int = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_inter.csv".format(d, d), header=0)
        # observational data only
        aracne_network = pd.read_csv("./non_bayesian_results/insilico_size10_{}/output/network.txt".format(d), sep='\t',header=0)
        clr_network = pd.read_csv("./non_bayesian_results/insilico_size10_{}/adj_mat.csv".format(d), header=None)
        df_int_ko = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_inter_ko.csv".format(d, d), header=0)
        df_wt = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_wt.csv".format(d, d), header=0)

        # convert edges to the true graph
        edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
        true_graph = edge_to_dag(edges_pos, type=DAG)
        nodes = list(df.columns)
        nodes.remove('target')
        true_graph.add_nodes_from(nodes)

        # convert aracne network to adj matrix
        edges_pos = [(row['Regulator'], row['Target']) for i, row in aracne_network.iterrows()]
        weights = [row['MI'] for i, row in aracne_network.iterrows()]
        nodes = list(df.columns)
        nodes.remove('target')
        A_aracne = edge_to_adj(edges_pos, nodes)
        A_aracne = np.tril(A_aracne) + np.triu(A_aracne.T, 1)

        # clr is already in correct form
        A = clr_network.to_numpy()
        threshold = 1.5
        A[np.abs(A) < threshold] = 0
        A = np.tril(A) + np.triu(A.T, 1)


        X_obs = df_wt.loc[:, df.columns != 'target'].to_numpy()
        X_int = df_int_ko.loc[:, df_int_ko.columns != 'target'].to_numpy()
        pinna = pinna_alg(X_obs, X_int, threshold=3)
        pinna.add_nodes_from(nodes)
        pinna_A = nx.adjacency_matrix(pinna).todense()

        # GIES OBS + INT DATA
        targets_matrix = np.eye(10, 10)
        data = pd.concat([df, df_int])
        target_index = list(data['target'].values + 1)
        data_new = data.loc[:, data.columns != 'target']

        fixedGaps_clr = pd.DataFrame(data=(A == 0))
        fixedGaps_aracne = pd.DataFrame(data=(A_aracne == 0))
        fixedGaps_pinna = pd.DataFrame(data=(pinna_A ==0))
        targets = pd.DataFrame(data=targets_matrix)
        target_index = pd.DataFrame(data=target_index)

        # CLR FIXEDGAP
        data_obs = df.drop('target',axis=1)
        shd, sid, auc, output = gies_run(data_obs, true_graph, None,
                                           None, fixedGaps=fixedGaps_clr)
        print(shd, sid, auc)

        shd, sid, auc, output = gies_run(data_new, true_graph, targets,
                                           target_index, fixedGaps=fixedGaps_clr)
        print(shd, sid, auc)

        # ARACNE FIXEDGAP
        shd, sid, auc, output = gies_run(data_obs, true_graph, None,
                                           None, fixedGaps=fixedGaps_aracne)
        print(shd, sid, auc)

        shd, sid, auc, output = gies_run(data_new, true_graph, targets,
                                           target_index, fixedGaps=fixedGaps_aracne)
        print(shd, sid, auc)



        # # pinna FIXEDGAP
        # shd, sid, auc, output = gies_run(data_new, true_graph, targets,
        #                                    target_index, fixedGaps=fixedGaps_pinna)
        # print(shd, sid, auc)

        shd = cdt.metrics.SHD(true_graph, np.zeros((len(nodes), len(nodes))), False)
        sid = cdt.metrics.SID(true_graph, np.zeros((len(nodes), len(nodes))))
        auc,pr = cdt.metrics.precision_recall(true_graph, np.zeros((len(nodes), len(nodes))))
        #print(shd, sid,auc)


def test_hybrid_random():
    random = ["small"]
    num_graphs = 1
    stats_pc = np.zeros(3)
    stats_clr = np.zeros(3)
    size=1000
    for r in random:
        print(r)
        for n in range(num_graphs):
            edges = pd.read_csv("./random_test_set_{}_{}/bn_network_{}.csv".format(size,r, n), header=0)
            df = pd.read_csv("./random_test_set_{}_{}/data_{}.csv".format(size,r, n), header=0)
            df_int = pd.read_csv("./random_test_set_{}_{}/interventional_data_{}.csv".format(size,r, n), header=0)

            # observational data only
            #aracne_network = pd.read_csv("./random_test_set_{}_{}/output/network.txt".format(size,r,n,n), sep='\t',header=0)
            #clr_network = pd.read_csv("./random_test_set_{}_{}/adj_mat_{}.csv".format(size,r,n), header=None)
            pc_network = pd.read_csv("./random_test_set_{}_{}/cupc_adj_mat.csv".format(size,r,n), header=0)
            # convert edges to the true graph
            edges_pos = [(r['start'], r['end']) for i, r in edges.iterrows() if r['edge'] == 1]
            true_graph = edge_to_dag(edges_pos, type=DAG)
            nodes = list(df.columns)
            nodes.remove('target')
            true_graph.add_nodes_from(nodes)

            # convert aracne network to adj matrix
            #edges_pos = [(row['Regulator'], row['Target']) for i, row in aracne_network.iterrows()]
            #weights = [row['MI'] for i, row in aracne_network.iterrows()]
            nodes = list(df.columns)
            nodes.remove('target')
            #A_aracne = edge_to_adj(edges_pos, nodes)
            #A_aracne = np.tril(A_aracne) + np.triu(A_aracne.T, 1)

            # clr is already in correct form
            #A = clr_network.to_numpy()
            #threshold = 1.5
            #A[np.abs(A) < threshold] = 0
            #A = np.tril(A) + np.triu(A.T, 1)


            A_pc = pc_network.to_numpy()
            # GIES OBS + INT DATA
            targets_matrix = np.eye(len(nodes), len(nodes))
            data = pd.concat([df, df_int])
            target_index = list(data['target'].values + 1)
            data_new = data.loc[:, data.columns != 'target']

            #fixedGaps_clr = pd.DataFrame(data=(A == 0))
            #fixedGaps_aracne = pd.DataFrame(data=(A_aracne == 0))
            fixedGaps_pc = pd.DataFrame(data=(A_pc == 0))
            targets = pd.DataFrame(data=targets_matrix)
            target_index = pd.DataFrame(data=target_index)

            # CLR FIXEDGAP
            #shd, sid, auc, output = gies_run(data_new, true_graph, targets,
            #                                   target_index, fixedGaps=fixedGaps_clr)
            #stats_clr += np.array([shd, sid, auc])

            # ARACNE FIXEDGAP
            #shd, sid, auc, output = gies_run(data_new, true_graph, targets,
            #                                   target_index, fixedGaps=fixedGaps_aracne)
            #stats_aracne += np.array([shd, sid, auc])

                        # ARACNE FIXEDGAP

            print(data_new.shape, fixedGaps_pc.shape)
            shd, sid, auc, output = gies_run(data_new, true_graph, targets,
                                               target_index, fixedGaps=fixedGaps_pc)
            stats_pc += np.array([shd, sid, auc])
            
        stats_pc/=num_graphs
        stats_clr/=num_graphs
        print(stats_clr, stats_pc)


test_regulondb()
