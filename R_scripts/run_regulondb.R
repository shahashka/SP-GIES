source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")

# Load RegulonDB data
dataset_path <- file.path("../regulondb/data_smaller_w_header.csv", fsep=.Platform$file.sep)
target_path <- file.path("../regulondb/targets.csv", fsep=.Platform$file.sep)
target_index_path <- file.path("../regulondb/target_index.csv", fsep=.Platform$file.sep)

# Threshold selected from CLR paper to ensure 60% precision in CLR skeleton
# SP-GIES-OI
run_from_file_sp_gies(dataset_path, target_path, target_index_path, threshold=6.917, skeleton_path="../regulondb/adj_mat.csv", save_path="../regulondb/clr_skel_")

run_from_file_sp_gies(dataset_path, target_path, target_index_path, threshold=6.917, skeleton_path="../regulondb/adj_mat.csv", save_path="../regulondb/clr_skel_lamb0_", lambda=0)

run_from_file_sp_gies(dataset_path, target_path, target_index_path, threshold=1.5, skeleton_path="../regulondb/genie_adj.csv", save_path="../regulondb/genie_skel")

run_from_file_sp_gies(dataset_path, target_path, target_index_path, save_path="../regulondb/", save_pc=TRUE)


# GIES-OI
run_from_file_gies(dataset_path, target_path, target_index_path, save_path="../regulondb/md_10_", max_degree=10)
# GES-O
run_from_file_gies(dataset_path, NULL, NULL, save_path="../regulondb/obs_", max_degree=10)
