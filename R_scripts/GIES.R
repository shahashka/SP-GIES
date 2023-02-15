library(pcalg)
library(tictoc)
library(igraph)
## This file contains functions for calling the GIES algorithm in the pcalg library

# Given file paths for the dataset, targets and target indices, run GIES
# The dataset file should contain a matrix of values with each row corresponding to samples and each column corresponding to a random variable in the network.
# The targets file should contain a list of possible interventions to obtain the samples in the dataset.
# The targets index file should contain a list of size # of samples. Each value corresponds to the index of the target list
# See pcalg::gies documentation for more (https://rdrr.io/cran/pcalg/man/gies.html)
run_from_file_gies <- function(dataset_path, target_path, target_index_path, save_path, max_degree=integer(0)) {
    dataset <- read.table(dataset_path, sep=",", header=FALSE)
    print(dim(dataset))

    if (is.null(target_path)){
	 targets <- list(integer(0))
       	 targets.index <- rep(1,dim(dataset)[1])
    }

    else {    
    	 targets <- read.table(target_path, sep=",", header=FALSE)
    	 targets <- split(targets, 1:nrow(targets))
    	 targets <- lapply(targets, function(x) x[!is.na(x)])
    	 targets_init <- list(integer(0))
    	 targets <- append(targets_init, targets)

    	 targets.index <- read.table(target_index_path, sep=",", header=FALSE)
    	 targets.index <- unlist(targets.index)
	 }
    gies(dataset, targets, targets.index, save_path, max_degree)
}

# Given dataset (matrix of size # samples x # nodes), targets (list),  targets.index (list of size #samples)
# run the GIES algorithm and save the adjacency matrix in the save_path
# Also prints the time to solution
gies <- function(dataset, targets, targets.index, save_path, max_degree=integer(0)){
    tic()
    score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
    result <- pcalg::gies(score, fixedGaps=NULL, targets=targets, maxDegree=max_degree)
    print("The total time consumed by GIES is:")
    toc()
    write.csv(result$repr$weight.mat() ,row.names = FALSE, file = paste(save_path, 'gies-adj_mat.csv',sep = ''))
    plot(x = result$essgraph, y = "ANY")
    print("test")
}

# # Example read data
# dataset_path <- file.path("../regulondb/data_smaller.csv", fsep=.Platform$file.sep)
# target_path <- file.path("../regulondb/targets.csv", fsep=.Platform$file.sep)
# target_index_path <- file.path("../regulondb/target_index.csv", fsep=.Platform$file.sep)

#run_from_file_gies(dataset_path, target_path, target_index_path)