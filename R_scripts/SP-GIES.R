library(pcalg)
library(tictoc)
library(igraph)
source("../cupc/cuPC.R")

## This file contains functions for calling the SP-GIES algorithm which is a two-part algorithm that first calls
## cupc to estimate a skeleton using the PC algorithm. Then runs GIES from the pcalg algorithm to estimate the final
## graph given observational and interventional data

# Given file paths for the dataset, targets and target indices, run SP-GIES
# Optionally provide a path to a pre-generated skeleton
run_from_file_sp_gies <- function(dataset_path, target_path, target_index_path, save_path, threshold=0, skeleton_path=FALSE, save_pc=FALSE) {
    dataset <- read.table(dataset_path, sep=",", header=FALSE)
    targets <- read.table(target_path, sep=",", header=FALSE)
    targets.index <- read.table(target_index_path, sep=",", header=FALSE)
    print(dim(dataset))

    targets <- split(targets, 1:nrow(targets))
    targets <- lapply(targets, function(x) x[!is.na(x)])
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    targets.index <- unlist(targets.index)

    if (is.character(skeleton_path)) {
        skeleton <- read.table(skeleton_path, sep=",", header=FALSE)
        skeleton <- as(skeleton,"matrix")
        skeleton[abs(skeleton) < threshold] = 0
	skeleton[abs(skeleton) >= threshold] = 1
	skeleton <- as.data.frame(skeleton)
        skeleton <- skeleton == 0
        class(skeleton) <- "logical"
	sp_gies_from_skeleton(dataset, targets, targets.index, skeleton, save_path, save_pc)
    }
    else {
        sp_gies(dataset, targets, targets.index, save_path, save_pc)
    }
}

# Given dataset (matrix of size # samples x # nodes), targets (list),  targets.index (list of size #samples) run the SP-GIES algorithm
# and save the adjacency matrix in the save_path. Also saves the adjacency matrix of the skeleton (output of the cupc algorithm)
# if save_pc is set to TRUE
# Also prints the time to solution which includes calculating the sufficient statistics for the PC algorithm
sp_gies <- function(dataset, targets, targets.index, save_path, save_pc=FALSE, max_degree=integer(0)) {
    tic("cupc")
    tic("gies")
    corrolationMatrix <- cor(dataset)
    p <- ncol(dataset)
    suffStat <- list(C = corrolationMatrix, n = nrow(dataset))
    cuPC_fit <- cu_pc(suffStat, p=p, alpha=0.01)
    print("The total time consumed by cuPC is:")
    toc()

    # TODO we don't want this included in time to solution
    if (save_pc) {
        write.csv(as(cuPC_fit@graph, "matrix") ,row.names = FALSE, file = paste(save_path, 'cupc-adj_mat.csv',sep = ''))
    }


    fixedGaps <- as(cuPC_fit@graph,"matrix")
    fixedGaps <- as.data.frame(fixedGaps)
    fixedGaps <- fixedGaps == 0
    class(fixedGaps) <- "logical"

    score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
    result <- pcalg::gies(score, fixedGaps=fixedGaps, targets=targets, maxDegree=max_degree)
    print("The total time consumed by SP-GIES is:")
    toc()
    print(max(degree(graph_from_adjacency_matrix(result$repr$weight.mat(), weighted=TRUE))))
    write.csv(result$repr$weight.mat() ,row.names = FALSE, file = paste(save_path, 'sp-gies-adj_mat.csv',sep = ''))
 }

# Same method as above, except skeleton comes from another algorithm. fixedGaps is a logical array that
# is  FALSE for all edges in the skeleton and TRUE otherwise
 sp_gies_from_skeleton <- function(dataset, targets, targets.index, fixedGaps, save_path, save_pc=FALSE) {
    tic()
    score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
    result <- pcalg::gies(score, fixedGaps=fixedGaps, targets=targets)
    print("The total time consumed by SP-GIES is:")
    toc()
    write.csv(result$repr$weight.mat() ,row.names = FALSE, file = paste(save_path, 'sp-gies-adj_mat.csv',sep = ''))
 }

