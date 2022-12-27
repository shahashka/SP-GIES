library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)

source("../cupc/cuPC.R")

run_sp_gies <- function(num_nodes) {
# # read data
dataset_path <- file.path(paste("../test_set_random_fewer",as.character(num_nodes),"_small/data_0.csv", sep=""), fsep=.Platform$file.sep)
dataset <- read.table(dataset_path, sep=",", header=TRUE)

tic()
corrolationMatrix <- cor(dataset)
p <- ncol(dataset)
suffStat <- list(C = corrolationMatrix, n = nrow(dataset))
cuPC_fit <- cu_pc(suffStat, p=p, alpha=0.05)
targets <- read.table("../regulondb2/targets.csv", sep=",", header=FALSE)
targets <- split(targets, 1:nrow(targets))
targets <- lapply(targets, function(x) x[!is.na(x)])
targets_init <- list(integer(0))
targets <- append(targets_init, targets)
fixedGaps <- as(cuPC_fit@graph,"matrix")
fixedGaps <- fixedGaps == 0
class(fixedGaps) <- "logical"
targets.index <- read.table("../regulondb2/target_index.csv", sep=",", header=FALSE)
targets.index <- unlist(targets.index)
score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
result <- pcalg::gies(score, fixedGaps=fixedGaps, targets=targets)
print("The total time consumed by SP-GIES is:")
toc()
}

run_gies <- function(num_nodes) {
# # read data
dataset_path <- file.path(paste("../test_set_random_fewer",as.character(num_nodes),"_small/data_0.csv", sep=""), fsep=.Platform$file.sep)
dataset <- read.table(dataset_path, sep=",", header=TRUE)

tic()
targets <- read.table("../regulondb2/targets.csv", sep=",", header=FALSE)
targets <- split(targets, 1:nrow(targets))
targets <- lapply(targets, function(x) x[!is.na(x)])
targets_init <- list(integer(0))
targets <- append(targets_init, targets)
targets.index <- read.table("../regulondb2/target_index.csv", sep=",", header=FALSE)
targets.index <- unlist(targets.index)
score <- new("GaussL0penIntScore", data = dataset, targets=targets, target.index=targets.index)
result <- pcalg::gies(score, fixedGaps=NULL, targets=targets)
print("The total time consumed by GIES is:")
toc()
}


graph_nodes <- list(10,100,1000)
num_repeats = 3
for (n in 1:num_repeats) {
    for (i in graph_nodes) {
        print("Number of nodes is ", i)
        run_sp_gies(i)
        print("\n")
    }
}

for (n in 1:num_repeats) {
    for (i in graph_nodes) {
        print("Number of nodes is ", i)
        run_gies(i)
        print("\n")
    }
}

# let's see if it can run the largest one as a test
#for (n in 1:num_repeats) {
#print("Number of nodes is 10,000")
#run_sp_gies(10000)
#print("\n")
#}

# let's see if it can run the largest one as a test
#for (n in 1:num_repeats) {
#print("Number of nodes is 10,000")
#run_gies(10000)
#print("\n")
#}

