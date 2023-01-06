source("../cupc/cuPC.R")
source("../SP-GIES/GIES.R")
source("../SP-GIES/SP-GIES.R")

# Run RegulonDB
#dataset_path <- file.path("../regulondb/data_smaller.csv", fsep=.Platform$file.sep)
#target_path <- file.path("../regulondb/targets.csv", fsep=.Platform$file.sep)
#target_index_path <- file.path("../regulondb/target_index.csv", fsep=.Platform$file.sep)
#run_from_file_sp_gies(dataset_path, target_path, target_index_path, save_path="../regulondb/", save_pc=TRUE)
#run_from_file_gies(dataset_path, target_path, target_index_path, save_path="../regulondb/")


# Run DREAM4 insilico network #3
dataset_OI_path <- file.path(paste("../insilico_size10_3/insilico_size10_3_combine.csv", sep=""), fsep=.Platform$file.sep)
dataset <- read.table(dataset_OI_path, sep=",", header=TRUE)
targets.index <- dataset[,ncol(dataset)]
targets <- as.matrix(unique(targets.index))
targets <- targets[targets > 0]
targets_init <- list(integer(0))
targets <- append(targets_init, targets)
dataset <- dataset[,1:ncol(dataset)-1]
targets.index <- targets.index + 1
print(targets)
print(targets.index)
print(dim(dataset))
sp_gies(dataset, targets, targets.index, save_path="../insilico_size10_3/")
gies(dataset, targets, targets.index, save_path="../insilico_size10_3/")

dataset_O_path <- file.path(paste("../insilico_size10_3/insilico_size10_3_obs.csv", sep=""), fsep=.Platform$file.sep)
dataset <- read.table(dataset_O_path, sep=",", header=TRUE)
targets.index <- dataset[,ncol(dataset)]
targets <- as.matrix(unique(targets.index))
targets <- targets[targets > 0]
targets_init <- list(integer(0))
targets <- append(targets_init, targets)
dataset <- dataset[,1:ncol(dataset)-1]
targets.index <- targets.index + 1
sp_gies(dataset, targets, targets.index, save_path="../insilico_size10_3/obs_", save_pc=TRUE)
gies(dataset, targets, targets.index, save_path="../insilico_size10_3/obs_")

# Run random networks small world size 10
num_nodes=10
num_graphs=30
for (network in list('ER', 'scale', 'small')) {
    for (x in 0:29) {
    	print(x)
        dataset_OI_path <- file.path(paste("../random_test_set_",as.character(num_nodes), "_", network, "/data_joint_", x,".csv", sep=""), fsep=.Platform$file.sep)
        dataset <- read.table(dataset_OI_path, sep=",", header=TRUE)
        targets.index <- dataset[,ncol(dataset)]
        targets <- as.matrix(unique(targets.index))
        targets <- targets[targets > 0]
        targets_init <- list(integer(0))
        targets <- append(targets_init, targets)
        dataset <- dataset[,1:ncol(dataset)-1]
        targets.index <- targets.index + 1
        sp_gies(dataset, targets, targets.index, save_path=paste("../random_test_set_",as.character(num_nodes),"_", network,"/", sep=""))
        gies(dataset, targets, targets.index, save_path=paste("../random_test_set_",as.character(num_nodes),"_", network,"/", sep=""))

        dataset_O_path <- file.path(paste("../random_test_set_",as.character(num_nodes), "_", network, "/data_", x,".csv", sep=""), fsep=.Platform$file.sep)
        dataset <- read.table(dataset_O_path, sep=",", header=TRUE)
        targets.index <- dataset[,ncol(dataset)]
        targets <- as.matrix(unique(targets.index))
        targets <- targets[targets > 0]
        targets_init <- list(integer(0))
        targets <- append(targets_init, targets)
        dataset <- dataset[,1:ncol(dataset)-1]
        targets.index <- targets.index + 1
	sp_gies(dataset, targets, targets.index, save_path=paste("../random_test_set_",as.character(num_nodes),"_", network, "/obs_", sep=""), save_pc=TRUE)
        gies(dataset, targets, targets.index, save_path=paste("../random_test_set_",as.character(num_nodes),"_", network, "/obs_", sep=""))
    }
}