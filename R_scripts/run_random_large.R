source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")

# Run small world random network small world size 1000
folder <- "../random_test_set_1000_small/"
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
sp_gies(dataset, targets, targets.index, save_path=paste(folder, x, "_", sep=""))
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
