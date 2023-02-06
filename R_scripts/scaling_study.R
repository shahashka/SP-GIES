library(pcalg)
library(tictoc)

source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")
source("IGSP.R")

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
    dataset_path <- file.path(paste("../random_test_set_fewer_",as.character(num_nodes),"_small/data_joint_0.csv", sep=""), fsep=.Platform$file.sep)
    dataset <- read.table(dataset_path, sep=",", header=TRUE)

    # load target, target index files
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    gies(dataset, targets, targets.index, save_path="./")
}

run_igsp <- function(num_nodes) {
    obs_dataset_path <- file.path(paste("../random_test_set_fewer_",as.character(num_nodes),"_small/data_0.csv", sep=""), fsep=.Platform$file.sep)
    int_dataset_path <- file.path(paste("../random_test_set_fewer_",as.character(num_nodes),"_small/interventional_data_0.csv", sep=""), fsep=.Platform$file.sep)
    obs_data = read.table(obs_dataset_path, sep=",", header=TRUE)
    obs_data <- obs_data[,1:ncol(obs_data)-1]

    iv_data = read.table(int_dataset_path, sep=",", header=TRUE)
    iv_data <- iv_data[,1:ncol(iv_data)-1]

    #get data as input
    data.list = list()
    t.list = list()
    data.list[[1]] = obs_data
    i = 2
    # Loop through interventional data rows and add to list
    cols = colnames(iv_data)
    for (row in (1:dim(iv_data)[1]) ) {
    	d = t(iv_data[row])
	colnames(d) <- cols
	rownames(d) <- row
    	data.list[[i]] = d
        t.list[[i]] = row
        i = i + 1
    }
    method <- "hsic.gamma"

    #prepare for sufficient statistics and intervention targets
    #suffstat <- list(data=data.list[[1]], ic.method=method)
    suffstat <- list(C=cor(data.list[[1]]), n=nrow(data.list[[1]]))
    alpha <-1e-3


    print(data.list[[1]])
    print(data.list[[2]])
    
    #include observational dataset as an intervention
    intdata <- lapply(1:length(t.list), function(t) cbind(data.list[[t]], intervention_index=t) )
    inttargets <- t.list[1:length(t.list)]
    grspdag <- sp.restart.alg(suffstat, intdata, inttargets, alpha)
}

graph_nodes <- list(10,100,1000,2000)
num_repeats = 3

for (n in 1:num_repeats) {
    for (i in graph_nodes) {
        print(paste("Number of nodes is ", i))
        run_igsp(i)
    }
}

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

