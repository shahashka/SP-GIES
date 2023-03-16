source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")


## File to run GIES, SP-GIES, IGSP on all three datasets and save adjacency matrices in each folder.
## Handles unique file formats of each dataset

# Run RegulonDB
dataset_path <- file.path("../regulondb/data_smaller.csv", fsep=.Platform$file.sep)
dataset_path_subgraph_obs <- file.path("../regulondb/data_smaller_clr_subgraph_local_synthetic_n=10000_obs.csv", fsep=.Platform$file.sep)
dataset_path_subgraph <- file.path("../regulondb/data_smaller_clr_subgraph_local_synthetic_n=10000_obs_int.csv", fsep=.Platform$file.sep)
target_path <- file.path("../regulondb/targets_local.csv", fsep=.Platform$file.sep)
target_index_path <- file.path("../regulondb/targets_inds_local.csv", fsep=.Platform$file.sep)

run_from_file_sp_gies(dataset_path_subgraph, target_path, target_index_path, threshold=1, skeleton_path="../regulondb/clr_skel_subgraph_local.csv", save_path="../regulondb/clr_skel_subgraph_local_wint_")

run_from_file_sp_gies(dataset_path_subgraph_obs, target_path, target_path, target_index_path, threshold=1, skeleton_path="../regulondb/clr_skel_subgraph_local.csv", save_path="../regulondb/clr_skel_subgraph_local_",  obs=TRUE)


#run_from_file_gies(dataset_path_subgraph, NULL, NULL, save_path="../regulondb/subgraph_local_")

#
# # Run DREAM4 insilico network #3
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
