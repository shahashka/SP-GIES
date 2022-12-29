library(pcalg)
library(graph)
library(MASS)
library(tictoc)

source("../cupc/cuPC.R")

# # read data
dataset_path <- file.path("../regulondb2/data_smaller.csv", fsep=.Platform$file.sep)
target_path <- file.path("../regulondb2/targets.csv", fsep=.Platform$file.sep)
target_index_path <- file.path("../regulondb2/target_index.csv", fsep=.Platform$file.sep)

run_from_file_sp_gies <- function(dataset_path, target_path, target_index_path, header) {
    dataset <- read.table(dataset_path, sep=",", header=header)
    targets <- read.table(target_path, sep=",", header=header)
    targets.index <- read.table(target_index_path, sep=",", header=header)
    print(dim(dataset))

    targets <- split(targets, 1:nrow(targets))
    targets <- lapply(targets, function(x) x[!is.na(x)])
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    targets.index <- unlist(targets.index)

    sp_gies(dataset, targets, targets.index)
}

sp_gies <- function(dataset, targets, targets.index) {
    tic()
    corrolationMatrix <- cor(dataset)
    p <- ncol(dataset)
    suffStat <- list(C = corrolationMatrix, n = nrow(dataset))
    cuPC_fit <- cu_pc(suffStat, p=p, alpha=0.05)

    fixedGaps <- as(cuPC_fit@graph,"matrix")
    fixedGaps <- as.data.frame(fixedGaps)
    fixedGaps <- fixedGaps == 0
    class(fixedGaps) <- "logical"

    score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
    result <- pcalg::gies(score, fixedGaps=fixedGaps, targets=targets)
    print("The total time consumed by SP-GIES is:")
    toc()

    write.csv(result$repr$weight.mat() ,row.names = FALSE, file = 'sp-gies-adj_mat.csv');
 }

 run_from_file_sp_gies(dataset_path, target_path, target_index_path)