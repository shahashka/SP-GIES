source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")


## File to run GIES, SP-GIES, IGSP on all three datasets and save adjacency matrices in each folder.
## Handles unique file formats of each dataset

# Run RegulonDB
dataset_path <- file.path("../regulondb/data_smaller.csv", fsep=.Platform$file.sep)
target_path <- file.path("../regulondb/targets.csv", fsep=.Platform$file.sep)
target_index_path <- file.path("../regulondb/target_index.csv", fsep=.Platform$file.sep)
#
<<<<<<< HEAD
run_from_file_sp_gies(dataset_path, target_path, target_index_path, skeleton_path="../regulondb/adj_mat.csv", threshold=6.917 ,save_path="../regulondb/clr_skel_trip_", save_pc=TRUE)
=======
run_from_file_sp_gies(dataset_path, target_path, target_index_path, threshold=6.917, skeleton_path="../regulondb/adj_mat.csv", save_path="../regulondb/clr_skel_", save_pc=TRUE)
run_from_file_sp_gies(dataset_path, target_path, target_index_path, threshold=1, skeleton_path="../regulondb/pc_cupc-adj_mat.csv", save_path="../regulondb/pc_skel_", save_pc=TRUE)
run_from_file_sp_gies(dataset_path, target_path, target_index_path, threshold=1, skeleton_path="../regulondb/ground_truth_skel.csv", save_path="../regulondb/skel_", save_pc=TRUE)

>>>>>>> 8d336edf4816a8e38ac7b11ff436c1024d1fb784
# run_from_file_gies(dataset_path, target_path, target_index_path, save_path="../regulondb/md_100_", max_degree=100)
# run_from_file_gies(dataset_path, NULL, NULL, save_path="../regulondb/obs_md_100_", max_degree=100)

#
# # Run DREAM4 insilico network #3
<<<<<<< HEAD
for (x in 1:5) {
    directory <- paste("../insilico_size10_", x, "/", sep="")
    dataset_OI_path <- file.path(paste(directory, "/insilico_size10_", x, "_combine.csv", sep=""), fsep=.Platform$file.sep)
    dataset <- read.table(dataset_OI_path, sep=",", header=TRUE)
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    targets.index <- targets.index + 1

    # Use skeleton from CLR as input to sp-gies algorithm
    skeleton <- read.table(paste("../non_bayesian_results/insilico_size10_", x, "/adj_mat.csv", sep=""), sep=",", header=FALSE)
    skeleton <- as(skeleton,"matrix")
    #make symmetric
    threshold=1.5
    #skeleton[lower.tri(skeleton)] = t(skeleton)[lower.tri(skeleton)]
    skeleton[abs(skeleton) < threshold] = 0
    skeleton[abs(skeleton) >= threshold] = 1    
    skeleton <- as.data.frame(skeleton)
    skeleton <- skeleton == 0
    class(skeleton) <- "logical"
    sp_gies_from_skeleton(dataset, targets, targets.index,skeleton,save_path=directory)
    gies(dataset, targets, targets.index, save_path=directory)

    dataset_O_path <- file.path(paste(directory, "/insilico_size10_", x, "_obs.csv", sep=""), fsep=.Platform$file.sep)
    dataset <- read.table(dataset_O_path, sep=",", header=TRUE)
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    targets.index <- targets.index + 1
    sp_gies(dataset, targets, targets.index, save_path= paste(directory, "/obs_", sep=""), save_pc=TRUE)
    gies(dataset, targets, targets.index, save_path= paste(directory, "/obs_", sep=""))
}
=======
# for (x in 1:5) {
#     directory = paste(paste("../insilico_size10_" x, "/", sep=""))
#     dataset_OI_path <- file.path(paste(directory, "/insilico_size10_", x, "_combine.csv", sep=""), fsep=.Platform$file.sep)
#     dataset <- read.table(dataset_OI_path, sep=",", header=TRUE)
#     targets.index <- dataset[,ncol(dataset)]
#     targets <- as.matrix(unique(targets.index))
#     targets <- targets[targets > 0]
#     targets_init <- list(integer(0))
#     targets <- append(targets_init, targets)
#     dataset <- dataset[,1:ncol(dataset)-1]
#     targets.index <- targets.index + 1
#
#     # Use skeleton from CLR as input to sp-gies algorithm
#     skeleton <- read.table(paste(directory, "adj_mat.csv", sep=""), sep=",", header=FALSE)
#     skeleton <- as(skeleton,"matrix")
#     #make symmetric
#     skeleton[lower.tri(skeleton)] = t(skeleton)[lower.tri(skeleton)]
#     skeleton <- as.data.frame(skeleton)
#     skeleton <- skeleton == 0
#     class(skeleton) <- "logical"
#     sp_gies_from_skeleton(dataset, targets, targets.index, skeleton, save_path=directory)
#     gies(dataset, targets, targets.index, save_path=directory)
#
#     dataset_OI_path <- file.path(paste(directory, "/insilico_size10_", x, "_obs.csv", sep=""), fsep=.Platform$file.sep)
#     dataset <- read.table(dataset_O_path, sep=",", header=TRUE)
#     targets.index <- dataset[,ncol(dataset)]
#     targets <- as.matrix(unique(targets.index))
#     targets <- targets[targets > 0]
#     targets_init <- list(integer(0))
#     targets <- append(targets_init, targets)
#     dataset <- dataset[,1:ncol(dataset)-1]
#     targets.index <- targets.index + 1
#     gies(dataset, targets, targets.index, save_path= paste(directory, "/obs_", x,sep=""))
# )
# }
>>>>>>> 8d336edf4816a8e38ac7b11ff436c1024d1fb784

#
#
# # # Run random networks small world size 10
# num_nodes=10
# num_graphs=30
# for (network in list('ER', 'scale', 'small')) {
#     for (x in 0:num_graphs-1) {
#     	print(x)
#         folder <- paste("../random_test_set_",num_nodes, "_", network, "/",sep="")
# 	dataset_OI_path <- paste(folder,"data_joint_", x,".csv", sep="")
#         dataset <- read.table(dataset_OI_path, sep=",", header=TRUE)
#         targets.index <- dataset[,ncol(dataset)]
#         targets <- as.matrix(unique(targets.index))
#         targets <- targets[targets > 0]
#         targets_init <- list(integer(0))
#         targets <- append(targets_init, targets)
#         dataset <- dataset[,1:ncol(dataset)-1]
#         targets.index <- targets.index + 1
#         sp_gies(dataset, targets, targets.index, save_path=paste(folder, x, "_", sep=""))
#         gies(dataset, targets, targets.index, save_path=paste(folder, x, "_", sep=""))
#
#         dataset_O_path <- paste(folder, "data_", x, ".csv", sep="")
#         dataset <- read.table(dataset_O_path, sep=",", header=TRUE)
#         targets.index <- dataset[,ncol(dataset)]
#         targets <- as.matrix(unique(targets.index))
#         targets <- targets[targets > 0]
#         targets_init <- list(integer(0))
#         targets <- append(targets_init, targets)
#         dataset <- dataset[,1:ncol(dataset)-1]
#         targets.index <- targets.index + 1
# 	    sp_gies(dataset, targets, targets.index, save_path=paste(folder, "/obs_", x, "_",sep=""), save_pc=TRUE)
#         gies(dataset, targets, targets.index, save_path=paste(folder, "/obs_", x, "_",sep=""))
#
#     }
# }