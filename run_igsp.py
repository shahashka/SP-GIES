from causaldag import unknown_target_igsp
import conditional_independence
import pandas as pd
import numpy as np

def run_regulon_db():
    data = pd.read_csv("./regulondb/data_smaller.csv")
    targets = pd.read_csv("./regulondb/targets.csv")
    targets_index = pd.read_csv("./regulondb/targets.csv")

    obs_data_no_targets = data.iloc[targets_index==1].to_numpy()
    iv_samples_list = [data.iloc[targets_index==ind] for ind in np.arange(2, targets.shape[0]+1)]

    targets_list = []
    for i in np.arange(targets.shape[0]):
        targets_list.append(targets.iloc[i])
    nodes=np.arange(1,data.shape[1]+1)

    setting_list = [dict(known_interventions=[t]) if type(t) !=list else dict(known_interventions=t) for t in targets_list]

    obs_suffstat = conditional_independence.partial_correlation_suffstat(obs_data_no_targets)
    invariance_suffstat = conditional_independence.gauss_invariance_suffstat(obs_data_no_targets, iv_samples_list)
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = conditional_independence.MemoizedCI_Tester(conditional_independence.partial_correlation_test,
                                                           obs_suffstat, alpha=alpha)
    invariance_tester = conditional_independence.MemoizedInvarianceTester(conditional_independence.gauss_invariance_test,
                                                                          invariance_suffstat, alpha=alpha_inv)

    est_dag, _ = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
    est_adj = edge_to_adj(est_dag.arcs, list(est_dag.nodes))
    np.savetxt("./regulondb/igsp_adj.csv", est_adj, delimeter=",")

def run_dream4():
    data = pd.read_csv("./insilico_size10_3/ insilico_size10_3_combine.csv")
    obs_data = data.iloc[data['target']==0]
    int_data = data.iloc[data['target']!=0]
    obs_data_no_targets = obs_data.loc[:, obs_data.columns != 'target'].to_numpy()
    int_data_no_targets = int_data.loc[:, int_data.columns != 'target']
    iv_samples_list = [np.expand_dims(row.to_numpy(),axis=0) for _,row in int_data_no_targets.iterrows()]

    targets_list = set(list(int_data['target'].to_numpy()))
    nodes=targets_list

    setting_list = [dict(known_interventions=[t]) for t in targets_list]

    obs_suffstat = conditional_independence.partial_correlation_suffstat(obs_data_no_targets)
    invariance_suffstat = conditional_independence.gauss_invariance_suffstat(obs_data_no_targets, iv_samples_list)
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = conditional_independence.MemoizedCI_Tester(conditional_independence.partial_correlation_test,
                                                           obs_suffstat, alpha=alpha)
    invariance_tester = conditional_independence.MemoizedInvarianceTester(conditional_independence.kci_invariance_test,
                                                                          invariance_suffstat, alpha=alpha_inv)

    est_dag, _ = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
    est_adj = edge_to_adj(est_dag.arcs, list(est_dag.nodes))
    np.savetxt("./insilico_size10_3/igsp_adj.csv", est_adj, delimeter=",")

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

            setting_list = [dict(known_interventions=[t]) for t in targets_list]

            obs_suffstat = conditional_independence.partial_correlation_suffstat(obs_data_no_targets)
            invariance_suffstat = conditional_independence.gauss_invariance_suffstat(obs_data_no_targets,
                                                                                     iv_samples_list)
            alpha = 1e-3
            alpha_inv = 1e-3
            ci_tester = conditional_independence.MemoizedCI_Tester(conditional_independence.partial_correlation_test,
                                                                   obs_suffstat, alpha=alpha)
            invariance_tester = conditional_independence.MemoizedInvarianceTester(
                conditional_independence.kci_invariance_test,
                invariance_suffstat, alpha=alpha_inv)

            est_dag, _ = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
            est_adj = edge_to_adj(est_dag.arcs, list(est_dag.nodes))
            np.savetxt("./insilico_size10_3/igsp_adj.csv", est_adj, delimeter=",")

run_regulon_db()
run_dream4()
run_random()