from causaldag import unknown_target_igsp,igsp, gsp
import conditional_independence
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
    data = pd.read_csv("./regulondb/data_smaller.csv", header=None)
    targets = pd.read_csv("./regulondb/targets.csv", header=None)
    targets_index = pd.read_csv("./regulondb/target_index.csv", header=None)
    obs_inds = targets_index.index[targets_index.iloc[:,0]==1]
    obs_data_no_targets = data.iloc[obs_inds].to_numpy()
    
    iv_samples_list = [data.iloc[targets_index.index[targets_index.iloc[:,0]==ind]].to_numpy() for ind in np.arange(1, targets.shape[0]+1)]
    for s in iv_samples_list:
        print(s.shape)

    targets_list = [targets.iloc[i].dropna().tolist() for i in np.arange(targets.shape[0])]

    nodes=np.arange(1,data.shape[1]+1)

    setting_list = [dict(interventions=[t]) if type(t) !=list else dict(interventions=t) for t in targets_list]
    print(obs_data_no_targets.shape, len(targets_list), len(nodes), len(setting_list), len(iv_samples_list))
    obs_suffstat = conditional_independence.partial_correlation_suffstat(obs_data_no_targets)
    invariance_suffstat = conditional_independence.gauss_invariance_suffstat(obs_data_no_targets, iv_samples_list)
    #print(obs_suffstat)
    print(len(invariance_suffstat['contexts']))
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = conditional_independence.MemoizedCI_Tester(conditional_independence.partial_correlation_test,
                                                           obs_suffstat, alpha=alpha)
    invariance_tester = conditional_independence.MemoizedInvarianceTester(conditional_independence.gauss_invariance_suffstat,
                                                                          invariance_suffstat, alpha=alpha_inv)

    est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester)
    est_dag_obs = gsp(nodes,ci_tester)

    np.savetxt("./regulondb/igsp_adj.csv", edge_to_adj(est_dag.arcs, list(est_dag.nodes)), delimiter=",")
    np.savetxt("./regulondb/obs_igsp_adj.csv", edge_to_adj(est_dag_obs.arcs, list(est_dag.nodes)), delimiter=",")


def run_dream4():
    for d in range(3,4):
        data = pd.read_csv("./insilico_size10_{}/insilico_size10_{}_combine.csv".format(d,d))
        obs_data = data.loc[data['target']==0]
        int_data = data.loc[data['target'] != 0]
        targets_list = set(list(int_data['target'].to_numpy()))

        obs_data = obs_data.loc[:, obs_data.columns != 'target'].to_numpy()

        iv_samples_list = [int_data.loc[int_data['target'] == t] for t in targets_list]
        iv_samples_list = [inter.loc[:, inter.columns != 'target'] for inter in iv_samples_list]

        setting_list = [dict(interventions=[t]) for t in targets_list]

        obs_suffstat = conditional_independence.partial_correlation_suffstat(obs_data)
        invariance_suffstat = conditional_independence.gauss_invariance_suffstat(obs_data, iv_samples_list)
        alpha = 1e-3
        alpha_inv = 1e-3
        ci_tester = conditional_independence.MemoizedCI_Tester(conditional_independence.partial_correlation_test,
                                                               obs_suffstat, alpha=alpha)
        invariance_tester = conditional_independence.MemoizedInvarianceTester(conditional_independence.gauss_invariance_test,
                                                                              invariance_suffstat, alpha=alpha_inv)

        est_dag = igsp(setting_list, targets_list, ci_tester, invariance_tester)
        est_dag_obs = gsp(nodes, ci_tester)

        np.savetxt("./insilico_size10_{}/igsp_adj.csv".format(d), edge_to_adj(est_dag.arcs, list(est_dag.nodes)), delimiter=",")
        np.savetxt("./insilico_size10_{}/obs_igsp_adj.csv".format(d), edge_to_adj(est_dag_obs.arcs, list(est_dag.nodes)), delimiter=",")

def run_random():
    for network in ["ER", "scale", "small"]:
        for i in range(30):
            data = pd.read_csv("./random_test_set_10_{}/data_joint_{}.csv".format(network, i))
            obs_data = data.iloc[data['target'] == 0]
            int_data = data.iloc[data['target'] != 0]
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
                conditional_independence.gauss_invariance_suffstat,
                invariance_suffstat, alpha=alpha_inv)

            est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester)
            est_dag_obs = gsp(nodes, ci_tester)

            np.savetxt("./random_test_set_10_{}/igsp_{}_adj.csv".format(network,i), edge_to_adj(est_dag.arcs, list(est_dag.nodes)), delimiter=",")
            np.savetxt("./random_test_set_10_{}/obs_igsp_{}_adj.csv".format(network,i), edge_to_adj(est_dag_obs.arcs, list(est_dag.nodes)), delimiter=",")



def run_random_large():
        data = pd.read_csv("./random_test_set_1000_{}/data_joint_{}.csv".format("small_world", 0))
        obs_data = data.iloc[data['target'] == 0]
        int_data = data.iloc[data['target'] != 0]
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
            conditional_independence.gauss_invariance_suffstat,
            invariance_suffstat, alpha=alpha_inv)

        est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester)
        est_dag_obs = gsp(nodes, ci_tester)

        np.savetxt("./random_test_set_1000_{}/igsp_{}_adj.csv".format("small_norm",0), edge_to_adj(est_dag.arcs, list(est_dag.nodes)), delimiter=",")
        np.savetxt("./random_test_set_1000_{}/obs_igsp_{}_adj.csv".format("small_norm",0), edge_to_adj(est_dag_obs.arcs, list(est_dag.nodes)), delimiter=",")

run_regulon_db()
# run_dream4()
# run_random()
run_random_large()
