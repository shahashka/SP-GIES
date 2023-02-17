source("GIES.R")
source("SP-GIES.R")

get_non_i_essential_count <- function(result) {
    edges <- result$essgraph$.in.edges
    count <- 0
    for (i in names(edges)) {
        p <- edges[[i]]
        for (q in p) {
          index_of_node = which(i==names(edges))
          name_of_q = paste("G",q,sep="")
          if (is.element(index_of_node, edges[[name_of_q]]) & (index_of_node < q)) {
                count <- count + 1
          }
      }
    }
    return(count)
}

num_nodes=10
network="ER"
results_gies = list()
results_sp_gies = list()
for (iset in (0:10)) {
  ave_count_g <- 0
  ave_count_sp <- 0
  for (x in (0:29)) {
    folder <- paste("../random_test_set_", network, "/",sep="")
    dataset_OI_path <- paste(folder,"data_joint_", x,".csv", sep="")
    dataset <- read.table(dataset_OI_path, sep=",", header=TRUE)
    dataset <- dataset[dataset['target'] <= iset,]
    targets.index <- dataset[,ncol(dataset)]
    targets <- as.matrix(unique(targets.index))
    targets <- targets[targets > 0]
    targets_init <- list(integer(0))
    targets <- append(targets_init, targets)
    dataset <- dataset[,1:ncol(dataset)-1]
    targets.index <- targets.index + 1
    skeleton <- read.table(paste("../random_test_set_", network, "/adj_mat_", x, ".csv", sep=""), sep=",", header=FALSE)
    skeleton <- as(skeleton,"matrix")
    #make symmetric
    threshold=0.5
    skeleton[abs(skeleton) < threshold] = 0
    skeleton[abs(skeleton) >= threshold] = 1
    skeleton <- as.data.frame(skeleton)
    skeleton <- skeleton == 0
    class(skeleton) <- "logical"
    
    result_g <- gies(dataset, targets, targets.index, save_path=paste(folder, x, "_", sep=""))
    result_sp <- sp_gies_from_skeleton(dataset, targets, targets.index, skeleton, save_path=paste(folder, x, "_", sep=""))
    
    count <- get_non_i_essential_count(result_g)
    ave_count_g <- ave_count_g + count
    
    count <- get_non_i_essential_count(result_sp)
    ave_count_sp <- ave_count_sp + count
  }
  ave_count_g = ave_count_g/30
  results_gies <- append(results_gies, ave_count_g)
  
  ave_count_sp = ave_count_sp/30
  results_sp_gies <- append(results_sp_gies, ave_count_sp)
}
print(results_gies)
print(results_sp_gies)

