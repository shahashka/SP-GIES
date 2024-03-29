library(pcalg)
library(tictoc)

source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")
source("IGSP.R")

## Scaling study used in paper. Runs small-world random network dataset of size 10,100,1000,2000 nodes.
## Compares GIES and SP-GIES. SP-GIES achieves ~4x speedup
## First run gen_scaling_data.sh script to generate all data needed for this study

run_sp_gies <- function(num_nodes, max_degree) {
    # # read data
    dataset_path <- file.path(paste("../data/random_test_set_scaling_",as.character(num_nodes),"_small/data_joint_0.csv", sep=""), fsep=.Platform$file.sep)
    dataset <- read.table(dataset_path, sep=",", header=TRUE)
    # load target, target index files
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    targets.index <- targets.index + 1
    sp_gies(dataset, targets, targets.index, save_path="./", max_degree=max_degree)
}

run_gies <- function(num_nodes, max_degree) {
    # # read data
    dataset_path <- file.path(paste("../data/random_test_set_scaling_",as.character(num_nodes),"_small/data_joint_0.csv", sep=""), fsep=.Platform$file.sep)
    dataset <- read.table(dataset_path, sep=",", header=TRUE)
    # load target, target index files
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    targets.index <- targets.index + 1
    gies(dataset, targets, targets.index, save_path="./", max_degree=max_degree)
}

run_igsp <- function(num_nodes) {
    obs_dataset_path <- file.path(paste("../data/random_test_set_scaling_",as.character(num_nodes),"_small/data_0.csv", sep=""), fsep=.Platform$file.sep)
    int_dataset_path <- file.path(paste("../data/random_test_set_scaling_",as.character(num_nodes),"_small/interventional_data_0.csv", sep=""), fsep=.Platform$file.sep)
    obs_data = read.table(obs_dataset_path, sep=",", header=TRUE)
    obs_data <- obs_data[,1:ncol(obs_data)-1]

    iv_data = read.table(int_dataset_path, sep=",", header=TRUE)
    iv_data <- iv_data[,1:ncol(iv_data)-1]

    igsp(obs_data, iv_data, save_path = "./")
}

graph_nodes <- list(10,100,1000,2000)
maximum_degree <- list(integer(0), 100)
num_repeats = 3

for (n in 1:num_repeats) {
    for (i in graph_nodes) {
        print(paste("Number of nodes is ", i))
        run_igsp(i)
    }
}

for (m in maximum_degree) {
    print(paste("Max degree is ", m))
    for (n in 1:num_repeats) {
        for (i in graph_nodes) {
            print(paste("Number of nodes is ", i))
            run_sp_gies(i, m)
        }
    }
}

for (m in maximum_degree) {
    print(paste("Max degree is ", m))
    for (n in 1:num_repeats) {
        for (i in graph_nodes) {
            print(paste("Number of nodes is ", i))
            run_gies(i, m)
        }
    }
}

