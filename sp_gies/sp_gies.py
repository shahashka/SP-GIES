import pandas as pd
import numpy as np
import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
rpy2.robjects.numpy2ri.activate()
pcalg = importr('pcalg')
base = importr('base')

def run_pc(data, outdir, alpha):
    '''
      Python wrapper for PC.

               Parameters:
                       data (numpy ndarray): Observational data with dimensions n x p
                       outdir (str): Path to save the adjacency matrix
                       alpha (float): threshold for test statistic

               Returns:
                       np.ndarray representing the adjancency matrix for skeleton (CPDAG) with dimensions p x p
       '''
    ro.r.assign("data", data)
    rcode = 'cor(data)'
    corMat = ro.r(rcode)
    ro.r.assign("corrolationMatrix", corMat)

    p = data.shape[1]
    ro.r.assign("p",p)

    rcode = 'list(C = corrolationMatrix, n = nrow(data))'
    suffStat = ro.r(rcode)
    ro.r.assign("suffStat", suffStat)
    ro.r.assign('alpha', alpha)

    rcode = 'pc(suffStat,p=p,indepTest=gaussCItest,skel.method="stable.fast",alpha=alpha)'
    pc_fit = ro.r(rcode)
    ro.r.assign("pc_fit", pc_fit)

    rcode = 'as(pc_fit@graph, "matrix")'
    skel = ro.r(rcode)
    ro.r.assign("skel", skel)

    rcode = "write.csv(skel,row.names = FALSE, file = paste('{}/', 'pc-adj_mat.csv',sep = ''))".format(outdir)
    ro.r(rcode)
    return skel

def cu_pc(data, outdir, alpha):
    '''
      Python wrapper for cuPC. CUDA implementation of the PC algorithm

               Parameters:
                       data (numpy ndarray): Observational data with dimensions n x p
                       outdir (str): Path to save the adjacency matrix
                       alpha (float): threshold for test statistic

               Returns:
                       np.ndarray representing the adjancency matrix for skeleton (CPDAG) with dimensions p x p
       '''
    with open("./cupc/cuPC.R") as file:
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

    cuPC_fit = cupc.cu_pc(ro.r['suffStat'],p=ro.r['p'],alpha=alpha)
    ro.r.assign("cuPC_fit", cuPC_fit)

    rcode = 'as(cuPC_fit@graph, "matrix")'
    skel = ro.r(rcode)
    ro.r.assign("skel", skel)

    rcode = "write.csv(skel,row.names = FALSE, file = paste('{}/', 'cupc-adj_mat.csv',sep = ''))".format(outdir)
    ro.r(rcode)
    return skel


def sp_gies(data, outdir, skel=None, pc=False, alpha=1e-3):
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
                       pc (bool): a flag to indicate if skeleton estimation should be done with the PC. If False
                                    and no skel is specified, then assumed no skeleton i.e. reverts to GIES algorithm.
                                    Will use the GPU accelerated version of the PC if avaiable, otherwise reverts to pcalg
                                    implementation of PC
               Returns:
                       np.ndarray representing the adjancency matrix for the final learned graph
       '''
    if skel is None:
        if pc:
            obs_data = data.loc[data['target']==0]
            obs_data = obs_data.drop(columns=['target'])
            obs_data = obs_data.to_numpy()
            if os.path.exists("../R_scripts/Skeleton.so"):
                skel = cu_pc(obs_data, outdir, alpha)
            else:
                skel = run_pc(obs_data, outdir,alpha)
        else:
            skel = np.ones((data.shape[1], data.shape[1]))
    fixed_gaps = np.array((skel == 0), dtype=int)
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
    ro.r.assign("lambda", 0)

    nr, nc = fixed_gaps.shape
    FG = ro.r.matrix(fixed_gaps, nrow=nr, ncol=nc)
    ro.r.assign("fixed_gaps", FG)
    if data.shape[1] > 1:
        #rcode = 'new("GaussL0penIntScore", data = data, targets=targets, target.index=target_index, lambda=0)'
        #score = ro.r(rcode)
        score = ro.r.new('GaussL0penIntScore', ro.r['data'], ro.r['targets'], ro.r['target_index'])
        ro.r.assign("score", score)
        result = pcalg.gies(ro.r['score'], fixedGaps=ro.r['fixed_gaps'], targets=ro.r['targets'])#, adaptive='triples')
        ro.r.assign("result", result)

        rcode = 'result$repr$weight.mat()'
        adj_mat = ro.r(rcode)
    else:
        adj_mat = np.zeros((1,1))
    ro.r.assign("adj_mat", adj_mat)
    rcode = 'write.csv(adj_mat, row.names = FALSE,' \
            ' file = paste("{}/", "sp-gies-adj_mat.csv", sep=""))'.format(outdir)
    ro.r(rcode)
    return adj_mat
