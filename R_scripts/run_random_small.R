source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")

# Run random networks small world size 10
num_nodes=10
num_graphs=30
for (network in list('ER', 'scale', 'small')) {
    for (x in 0:(num_graphs-1)) {
    	print(x)
        folder <- paste("../random_test_set_",num_nodes, "_", network, "/",sep="")
	    dataset_OI_path <- paste(folder,"data_joint_", x,".csv", sep="")
        dataset <- read.table(dataset_OI_path, sep=",", header=TRUE)
        targets.index <- dataset[,ncol(dataset)]
        targets <- as.matrix(unique(targets.index))
        targets <- targets[targets > 0]
        targets_init <- list(integer(0))
        targets <- append(targets_init, targets)
        dataset <- dataset[,1:ncol(dataset)-1]
        targets.index <- targets.index + 1
        # SP-GIES_OI
        sp_gies(dataset, targets, targets.index, save_path=paste(folder, x, "_", sep=""), save_pc=TRUE)
        # GIES-OI
        gies(dataset, targets, targets.index, save_path=paste(folder, x, "_", sep=""))

        dataset_O_path <- paste(folder, "data_", x, ".csv", sep="")
        dataset <- read.table(dataset_O_path, sep=",", header=TRUE)
        targets.index <- dataset[,ncol(dataset)]
        targets <- as.matrix(unique(targets.index))
        targets <- targets[targets > 0]
        targets_init <- list(integer(0))
        targets <- append(targets_init, targets)
        dataset <- dataset[,1:ncol(dataset)-1]
        targets.index <- targets.index + 1
        # GES-O
        gies(dataset, targets, targets.index, save_path=paste(folder, "/obs_", x, "_",sep=""))

    }
}