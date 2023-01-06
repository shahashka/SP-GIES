library(pcalg)
library(tictoc)

source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")

## Scaling study used in paper. Runs small-world random network dataset of size 10,100,1000,2000 nodes.
## Compares GIES and SP-GIES. SP-GIES achieves ~4x speedup
## First run gen_scaling_data.sh script to generate all data needed for this study

run_sp_gies <- function(num_nodes) {
    # # read data
    dataset_path <- file.path(paste("../random_test_set_fewer_",as.character(num_nodes),"_small/data_joint_0.csv", sep=""), fsep=.Platform$file.sep)
    dataset <- read.table(dataset_path, sep=",", header=TRUE)
    # load target, target index files
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    targets.index <- targets.index + 1
    sp_gies(dataset, targets, targets.index, save_path="./")
}

run_gies <- function(num_nodes) {
    # # read data
    dataset_path <- file.path(paste("../random_test_set_fewer_",as.character(num_nodes),"_small/data_0.csv", sep=""), fsep=.Platform$file.sep)
    # load target, target index files
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    gies(dataset, targets, targets.index, save_path="./")
}


graph_nodes <- list(10,100,1000,2000)
num_repeats = 3
for (n in 1:num_repeats) {
    for (i in graph_nodes) {
        print(paste("Number of nodes is ", i))
        run_sp_gies(i)
    }
}

for (n in 1:num_repeats) {
    for (i in graph_nodes) {
        print(paste("Number of nodes is ", i))
        run_gies(i)
        print("\n")
    }
}

# let's see if it can run the largest one as a test
#for (n in 1:num_repeats) {
#print("Number of nodes is 10,000")
#run_sp_gies(10000)
#print("\n")
#}

# let's see if it can run the largest one as a test
#for (n in 1:num_repeats) {
#print("Number of nodes is 10,000")
#run_gies(10000)
#print("\n")
#}

