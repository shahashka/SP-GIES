library(pcalg)
library(graph)
library(MASS)
library(tictoc)

source("../cupc/cuPC.R")
#
# # read data
dataset_path <- file.path("../regulondb2/data_smaller.csv", fsep=.Platform$file.sep)
dataset <- read.table(dataset_path, sep=",", header=FALSE)
print(dim(dataset))

tic()
corrolationMatrix <- cor(dataset)
p <- ncol(dataset)
suffStat <- list(C = corrolationMatrix, n = nrow(dataset))
print("suff stats done")
cuPC_fit <- cu_pc(suffStat, p=p, alpha=0.05)
print("The total time consumed by cuPC is:")
toc()


targets <- read.table("../regulondb2/targets.csv", sep=",", header=FALSE)
targets <- split(targets, 1:nrow(targets))
targets <- lapply(targets, function(x) x[!is.na(x)])
targets_init <- list(integer(0))
targets <- append(targets_init, targets)

fixedGaps <- as(cuPC_fit@graph,"matrix")
fixedGaps <- as.data.frame(fixedGaps)
fixedGaps <- fixedGaps == 0
class(fixedGaps) <- "logical"

targets.index <- read.table("../regulondb2/target_index.csv", sep=",", header=FALSE)
targets.index <- unlist(targets.index)
print(targets.index)

tic()
score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
result <- pcalg::gies(score, fixedGaps=fixedGaps, targets=targets)
print("The total time consumed by GIES portion of SP-GIES is:")
toc()

write.csv(result$repr$weight.mat() ,row.names = FALSE, file = 'sp-gies-adj_mat.csv');

