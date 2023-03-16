source("../cupc/cuPC.R")
source("GIES.R")
source("SP-GIES.R")

# Run DREAM4 insilico networks
for (x in 1:5) {
    directory = paste(paste("../insilico_size10_" x, "/", sep=""))
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
    skeleton <- read.table(paste(directory, "adj_mat.csv", sep=""), sep=",", header=FALSE)
    skeleton <- as(skeleton,"matrix")
    #make symmetric
    skeleton[lower.tri(skeleton)] = t(skeleton)[lower.tri(skeleton)]
    skeleton <- as.data.frame(skeleton)
    skeleton <- skeleton == 0
    class(skeleton) <- "logical"
    # SP-GIES-OI
    sp_gies_from_skeleton(dataset, targets, targets.index, skeleton, save_path=directory)
    # GIES-OI
    gies(dataset, targets, targets.index, save_path=directory)

    dataset_OI_path <- file.path(paste(directory, "/insilico_size10_", x, "_obs.csv", sep=""), fsep=.Platform$file.sep)
    dataset <- read.table(dataset_O_path, sep=",", header=TRUE)
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    targets.index <- targets.index + 1
    #GES-O
    gies(dataset, targets, targets.index, save_path= paste(directory, "/obs_", x,sep=""))
)
}

