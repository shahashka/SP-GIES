library(pcalg)
library(tictoc)
source("../cupc/cuPC.R")

# # Example read data
dataset_path <- file.path("../regulondb2/data_smaller.csv", fsep=.Platform$file.sep)
target_path <- file.path("../regulondb2/targets.csv", fsep=.Platform$file.sep)
target_index_path <- file.path("../regulondb2/target_index.csv", fsep=.Platform$file.sep)

run_from_file_sp_gies <- function(dataset_path, target_path, target_index_path, save_path, save_pc=FALSE) {
    dataset <- read.table(dataset_path, sep=",", header=FALSE)
    targets <- read.table(target_path, sep=",", header=FALSE)
    targets.index <- read.table(target_index_path, sep=",", header=FALSE)
    print(dim(dataset))

    targets <- split(targets, 1:nrow(targets))
    targets <- lapply(targets, function(x) x[!is.na(x)])
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    targets.index <- unlist(targets.index)

    sp_gies(dataset, targets, targets.index, save_path, save_pc)
}

sp_gies <- function(dataset, targets, targets.index, save_path, save_pc=FALSE) {
    tic()
    corrolationMatrix <- cor(dataset)
    p <- ncol(dataset)
    suffStat <- list(C = corrolationMatrix, n = nrow(dataset))
    cuPC_fit <- cu_pc(suffStat, p=p, alpha=0.05)
    if (save_pc) {
        write.csv(as(cuPC_fit@graph, "matrix") ,row.names = FALSE, file = paste(save_path, 'cupc-adj_mat.csv',sep = '');
    }

    fixedGaps <- as(cuPC_fit@graph,"matrix")
    fixedGaps <- as.data.frame(fixedGaps)
    fixedGaps <- fixedGaps == 0
    class(fixedGaps) <- "logical"

    score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
    result <- pcalg::gies(score, fixedGaps=fixedGaps, targets=targets)
    print("The total time consumed by SP-GIES is:")
    toc()

    write.csv(result$repr$weight.mat() ,row.names = FALSE, file = paste(save_path, 'sp-gies-adj_mat.csv',sep = '');
 }

