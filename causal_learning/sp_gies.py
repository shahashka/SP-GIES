import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from causal_learning.utils import edge_to_dag, adj_to_edge, adj_to_dag, get_scores
rpy2.robjects.numpy2ri.activate()
pcalg = importr('pcalg')
base = importr('base')

def cu_pc(data, outdir):
    '''
      Python wrapper for cuPC. CUDA implementation of the PC algorithm

               Parameters:
                       data (numpy ndarray): Observational data with dimensions n x p
                       outdir (str): Another decimal integer

               Returns:
                       np.ndarray representing the adjancency matrix for skeleton (CPDAG) with dimensions p x p
       '''
    with open("../cupc/cuPC.R") as file:
        string = ''.join(file.readlines())
    cupc = SignatureTranslatedAnonymousPackage(string, "cupc")
    ro.r.assign("data", data)
    rcode = 'cor(data)'
    corMat = ro.r(rcode)
    ro.r.assign("corrolationMatrix", corMat)

    p = data.shape[1]
    ro.r.assign("p",p)

    rcode = 'list(C = corrolationMatrix, n = nrow(data))'
    suffStat = ro.r(rcode)
    ro.r.assign("suffStat", suffStat)

    cuPC_fit = cupc.cu_pc(ro.r['suffStat'],p=ro.r['p'],alpha=0.05)
    ro.r.assign("cuPC_fit", cuPC_fit)

    rcode = 'as(cuPC_fit@graph, "matrix")'
    skel = ro.r(rcode)
    ro.r.assign("skel", skel)

    rcode = "write.csv(skel,row.names = FALSE, file = paste('{}', 'cupc-adj_mat.csv',sep = ''))".format(outdir)
    ro.r(rcode)
    return skel


def sp_gies(data, outdir, skel=None, cupc=False, target_map=None):
    '''
      Python wrapper for SP-GIES. Uses skeleton estimation to restrict edge set to GIES learner

               Parameters:
                       data (pandas DataFrame): DataFrame containing observational and interventional samples.
                                                Must contain a column named 'target' which specifies the index of the node
                                                that was intervened on to obtain the sample (assumes single interventions only).
                                                This indexes from 1 for R convenience.
                                                For observational samples the corresponding target should be 0
                       outdir (str): The directory to save the final adjacency matrix to
                       skel (numpy ndarray): an optional initial skeleton with dimensions p x p
                       cupc (bool): a flag to indicate if skeleton estimation should be done with cupc. If False
                                    and no skel is specified, then assumed no skeleton i.e. reverts to GIES algorithm
                       target_map (dict): An optional dictionary to map the 'target' column of the input dataset to the indices
                                         in the dataframe. This is only needed for the parallel implmentation of SP-GIES where the
                                        full graph is paritioned and indices need to be tracked

               Returns:
                       np.ndarray representing the adjancency matrix for the final learned graph
       '''
    if skel is None:
        if cupc:
            obs_data = data.loc[data['target']==0]
            obs_data = obs_data.drop(columns=['target'])
            obs_data = obs_data.to_numpy()
            skel = cu_pc(obs_data, outdir)
        else:
            skel = np.ones((data.shape[1], data.shape[1]))
    fixed_gaps = np.array((skel == 0), dtype=int)
    target_index = data.loc[:, 'target'].to_numpy()
    if target_map is not None:
        target_index = np.array([0 if i == 0 else target_map[i] + 1 for i in target_index])
    targets = np.unique(target_index)[1:]  # Remove 0 the observational target
    target_index += 1  # R indexes from 1
    data = data.drop(columns=['target']).to_numpy()

    nr, nc = data.shape
    D = ro.r.matrix(data, nrow=nr, ncol=nc)
    ro.r.assign("data", D)

    rcode = ','.join(str(int(i)) for i in targets)
    rcode = 'append(list(integer(0)), list({}))'.format(rcode)
    T = ro.r(rcode)
    ro.r.assign("targets", T)

    TI = ro.IntVector(target_index)
    ro.r.assign("target_index", TI)

    nr, nc = fixed_gaps.shape
    FG = ro.r.matrix(fixed_gaps, nrow=nr, ncol=nc)
    ro.r.assign("fixed_gaps", FG)
    if data.shape[1] > 1:
        score = ro.r.new('GaussL0penIntScore', ro.r['data'], ro.r['targets'], ro.r['target_index'])
        ro.r.assign("score", score)
        result = pcalg.gies(ro.r['score'], fixedGaps=ro.r['fixed_gaps'], targets=ro.r['targets'], maxDegree=10)
        ro.r.assign("result", result)

        rcode = 'result$repr$weight.mat()'
        adj_mat = ro.r(rcode)
    else:
        adj_mat = np.zeros((1,1))
    ro.r.assign("adj_mat", adj_mat)
    rcode = 'write.csv(adj_mat, row.names = FALSE,' \
            ' file = paste("{}", "sp-gies-adj_mat.csv", sep=""))'.format(outdir)
    ro.r(rcode)
    return adj_mat