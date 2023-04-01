import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from utils import edge_to_dag, adj_to_edge, adj_to_dag, get_scores
rpy2.robjects.numpy2ri.activate()
pcalg = importr('pcalg')
base = importr('base')
def sp_gies(data, skel, outdir=None, target_map=None):
    fixed_gaps = np.array((skel == 0), dtype=int)
    target_index = data.loc[:, 'target'].to_numpy()
    if target_map is not None:
        target_index = np.array([0 if i == 0 else target_map[i] + 1 for i in target_index])

    # Assume that targets are each column
    targets = np.unique(target_index)[1:] # Remove 0 the observational target
    if len(targets) > 0:
        # R indexes from 1 (+1) and observational target gets added later (+1)
        target_index = np.array([np.where(targets==i)[0][0]+2 if i > 0 else 1 for i in target_index])
    else:
        target_index += 1
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

    score = ro.r.new('GaussL0penIntScore', ro.r['data'], ro.r['targets'], ro.r['target_index'])
    ro.r.assign("score", score)
    result = pcalg.gies(ro.r['score'], fixedGaps=ro.r['fixed_gaps'], targets=ro.r['targets'], maxDegree=10)
    ro.r.assign("result", result)

    rcode = 'result$repr$weight.mat()'
    adj_mat = ro.r(rcode)
    ro.r.assign("adj_mat", adj_mat)

    rcode =  'pcalg::opt.target(result$essgraph, max.size = 1, use.node.names = TRUE)'
    opt_intervene = ro.r(rcode)
    if outdir is not None:
        rcode = 'write.csv(adj_mat, row.names = FALSE,' \
            ' file = paste("{}", "sp-gies-adj_mat.csv", sep=""))'.format(outdir)
        ro.r(rcode)
    return adj_mat, opt_intervene

def gies(data, outdir=None, target_map=None):
    target_index = data.loc[:, 'target'].to_numpy()
    if target_map is not None:
        target_index = np.array([0 if i == 0 else target_map[i] + 1 for i in target_index])

    # Assume that targets are each column
    targets = np.unique(target_index)[1:] # Remove 0 the observational target
    if len(targets) > 0:
        # R indexes from 1 (+1) and observational target gets added later (+1)
        target_index = np.array([np.where(targets==i)[0][0]+2 if i > 0 else 1 for i in target_index])
    else:
        target_index += 1
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

    score = ro.r.new('GaussL0penIntScore', ro.r['data'], ro.r['targets'], ro.r['target_index'])
    ro.r.assign("score", score)
    result = pcalg.gies(ro.r['score'], targets=ro.r['targets'], maxDegree=10)
    ro.r.assign("result", result)

    rcode = 'result$repr$weight.mat()'
    adj_mat = ro.r(rcode)
    ro.r.assign("adj_mat", adj_mat)

    rcode =  'pcalg::opt.target(result$essgraph, max.size = 1, use.node.names = TRUE)'
    opt_intervene = ro.r(rcode)
    if outdir is not None:
        rcode = 'write.csv(adj_mat, row.names = FALSE,' \
            ' file = paste("{}", "gies-adj_mat.csv", sep=""))'.format(outdir)
        ro.r(rcode)
    return adj_mat, opt_intervene