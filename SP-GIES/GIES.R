library(pcalg)
library(tictoc)
# # Example read data
dataset_path <- file.path("../regulondb2/data_smaller.csv", fsep=.Platform$file.sep)
target_path <- file.path("../regulondb2/targets.csv", fsep=.Platform$file.sep)
target_index_path <- file.path("../regulondb2/target_index.csv", fsep=.Platform$file.sep)


run_from_file_gies <- function(dataset_path, target_path, target_index_path, save_path) {
    dataset <- read.table(dataset_path, sep=",", header=FALSE)
    print(dim(dataset))

    targets <- read.table(target_path, sep=",", header=FALSE)
    targets <- split(targets, 1:nrow(targets))
    targets <- lapply(targets, function(x) x[!is.na(x)])
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)

    targets.index <- read.table(target_index_path, sep=",", header=FALSE)
    targets.index <- unlist(targets.index)

    gies(dataset, targets, targets.index, save_path)
}

gies <- function(dataset, targets, targets.index, save_path){
    tic()
    score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
    result <- pcalg::gies(score, fixedGaps=NULL, targets=targets)
    print("The total time consumed by GIES is:")
    toc()
    write.csv(result$repr$weight.mat() ,row.names = FALSE, file = paste(save_path, 'gies-adj_mat.csv',sep = '');
}

#run_from_file_gies(dataset_path, target_path, target_index_path)