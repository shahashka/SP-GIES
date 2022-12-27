library(pcalg)
library(graph)
library(MASS)
library(tictoc)

# # read data
dataset_path <- file.path("../regulondb2/data_smaller.csv", fsep=.Platform$file.sep)
dataset <- read.table(dataset_path, sep=",", header=FALSE)
print(dim(dataset))

targets <- read.table("../regulondb2/targets.csv", sep=",", header=FALSE)
targets <- split(targets, 1:nrow(targets))
targets <- lapply(targets, function(x) x[!is.na(x)])
targets_init <- list(integer(0))
targets <- append(targets_init, targets)

targets.index <- read.table("../regulondb2/target_index.csv", sep=",", header=FALSE)
targets.index <- unlist(targets.index)

tic()
score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
result <- pcalg::gies(score, fixedGaps=NULL, targets=targets)
print("The total time consumed by GIES is:")
toc()

write.csv(result$repr$weight.mat() ,row.names = FALSE, file = 'gies-adj_mat.csv');
