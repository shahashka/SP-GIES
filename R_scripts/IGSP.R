# CODE TAKEN FROM https://github.com/csquires/utigsp/blob/master/R_algs/run_igsp.R
# Related paper Permutation-Based Causal Structure Learning with Unknown Intervention Targets
library(pcalg)
library(kpcalg)
library(graph)
library(sets)
library(tictoc)
set.seed(1)


#input to Greedy SP with random restarts: mat -- precision matrix, order -- initial permutation
sp.restart.alg <- function(suffstat, intdata, inttargets, alpha){
    tic()
	#set up the initial parameters for all functions
	p <- ncol(suffstat$C)
	intervention_col<-which( colnames(intdata[[1]])=="intervention_index" )
	invariance_cache=matrix(data=NA_integer_,p, p)

	#checks f^(I)(x_b) invariant to I (I_(a\b) U nullset) conditioned on a subset of Union( I_{a\b}\a)
	checkInvariance <- function(a, b, a_no_b) {
		#add observational data
		if(!is.na(invariance_cache[[a, b]])) {
			return(invariance_cache[[a, b]])
		}
		k <- c(1, a_no_b)

		#get rows for k
		kdata<- lapply(k, function(t) intdata[[t]])
		#now combine the intervention data vertically for CI testing
		kdata_binded<-do.call("rbind", kdata)

		#union of a without b without a. (s <- set to condition on)
		v<-setdiff(lapply(a_no_b, function(t) inttargets[[t]]), a)
		v_subsets<- 2^as.set(v) #powerset

		for (s in v_subsets) {
			if (kernelCItest(b, intervention_col, s, suffStat = list(data=kdata_binded, ic.method="hsic.gamma")) > alpha) {
				invariance_cache[[a, b]] <<-T
				return(T)
			}
		}
		invariance_cache[[a, b]] <<-F
		return(F)
	}

	#I-contradicting edges
	contCItest <- function(i, j){
		k1 <- which(sapply(inttargets, function(a) i %in% a))
		k2 <- which(sapply(inttargets, function(a) j %in% a))
		i_no_j <- setdiff(k1, k2)
		j_no_i <- setdiff(k2, k1)

		if  ( (length(i_no_j)>0 && checkInvariance(i, j, i_no_j)) || (length(j_no_i)>0 && !checkInvariance(j, i, j_no_i)) ){
			return(T)
		}

		return(F)
	}

	#I-covered edges
	#tests if a covered edge is NOT i-covered
	icovtest <- function(i, j){
		k1 <- which(sapply(inttargets, function(a) i %in% a))
		k2 <- which(sapply(inttargets, function(a) j %in% a))
		i_no_j <- setdiff(k1, k2)
		return (length(i_no_j)>0 && !checkInvariance(i, j, i_no_j))
	}

	#get new dag based on edge flip
	get.newdag <- function(dag, contdag, order, edge, vorders){


		#get the new orders
		a <- which(order == edge[1])
		b <- which(order == edge[2])
		order <- order[c(0:(a-1), b, a:(b-1), (b+1):(p+1))[2:(p+1)]]
		#check if the new order has been visited
		if(list(order) %in% vorders) return(NULL)
		#if it has not been visited, check if this edge is an I-covered edge
		par <- subset(1:p, dag[,edge[1]] == 1)
		if(icovtest(edge[1], edge[2])) return(NULL)
		#then you can continue
		dag[edge[1], edge[2]] <- 0
		dag[edge[2], edge[1]] <- 1
		contdag[edge[1], edge[2]] <- 0
		contdag[edge[2], edge[1]] <- contCItest(edge[2], edge[1])

		#parent set of the flipped components
		if(length(par) != 0){
			dag[par, edge[1]] <- sapply(1:length(par), function(i) gaussCItest(par[i], edge[1], c(par[-i], edge[2]), suffstat) < alpha)
			dag[par, edge[2]] <- sapply(1:length(par), function(i) gaussCItest(par[i], edge[2], par[-i], suffstat) < alpha)
			contdag[par, edge[1]] <- sapply(1:length(par), function(i) if(dag[par[i], edge[1]] != 0) contCItest(par[i], edge[1]) else 0)
			contdag[par, edge[2]] <- sapply(1:length(par), function(i) if(dag[par[i], edge[2]] != 0) contCItest(par[i], edge[2]) else 0)
		}
		#get updates of the number of contradicting edges
		return(list(dag=dag, contdag=contdag, order=order))
	}

	#get the initial dag
	init.dag <- function(order){
		revorder <- sapply(1:p, function(t) which(order==t))
		return(sapply(1:p, function(j) sapply(1:p, function(i) if(revorder[i] < revorder[j]) gaussCItest(i, j, order[c(1:(revorder[j]-1))[-revorder[i]]], suffstat) < alpha else 0)))
	}

	#get the initial dag
	init.contdag <- function(dag, order){
		revorder <- sapply(1:p, function(t) which(order==t))
		return(sapply(1:p, function(j) sapply(1:p, function(i) if(dag[i, j] != 0) contCItest(i, j) else 0)))
	}

	#the stack for visited orders
	sing.restart <- function(order){
		vorders <- list()
		vtrace <- list()
		vdags <- list()
		dag <- init.dag(order)
		contdag <- init.contdag(dag, order)
		mindag <- list(dag=dag, n=sum(contdag != 0))
		while(TRUE){
			#get the list of covered edges
			cov.edge <- which(dag != 0, arr.ind = TRUE)
			cov.edge <- data.frame(subset(cov.edge, apply(cov.edge, 1, function(x) all.equal(c(dag[-x[1], x[1]]), c(dag[-x[1], x[2]])) == TRUE)))
			#get the list of DAGs after I-covered edge reversals
			rdags <- if(nrow(cov.edge) > 0) apply(cov.edge, 1, function(edge) get.newdag(dag, contdag, order, edge, vorders)) else list()
			if(length(rdags) > 0) rdags <- subset(rdags, sapply(rdags, function(t) !is.null(t)))
			select <- which(sapply(rdags, function(rdag) sum(rdag$dag != 0) < sum(dag != 0)) == TRUE)
			#start the searching
			if((length(rdags) > 0 && length(vtrace) != 3) || length(select) != 0){
				if(length(select) != 0){
					vorders <- list()
					vtrace <- list()
					vdags <- list()
					order <- rdags[[select[1]]]$order
					dag <- rdags[[select[1]]]$dag
					mindag <- list(dag=dag, n=sum(rdags[[select[1]]]$contdag != 0))
				}else{
					vorders <- append(vorders, list(order))
					vtrace <- append(vtrace, list(order))
					vdags <- append(vdags, list(dag))
					order <- rdags[[1]]$order
					dag <- rdags[[1]]$dag
					if(sum(rdags[[1]]$contdag != 0) < mindag$n) mindag <- list(dag=dag, n=sum(rdags[[1]]$contdag != 0))
				}
			}else{
				if(length(vtrace) == 0)
					break
				vorders <- append(vorders, list(order))
				order <- tail(vtrace, 1)[[1]]
				vtrace <- head(vtrace, -1)
				dag <- tail(vdags, 1)[[1]]
				vdags <- head(vdags, -1)
			}
		}
		# print(order)
		return(mindag)
	}

	#main part of the algorithm
	start.order <- lapply(1:10, function(x) sample(1:p, p, replace=F))
	dag.list <- lapply(start.order, function(x) sing.restart(x))
	edgenum.list <- sapply(dag.list, function(dag) sum(dag$dag != 0))
	minidx <- which(edgenum.list == min(edgenum.list))
	contedgenum.list <- sapply(dag.list, function(dag) dag$n)
	minidx <- minidx[which.min(contedgenum.list[minidx])]
	print("The total time consumed by IGSP is:")
    toc()
	return(dag.list[[minidx]]$dag)
}

igsp <- function(obs_data, iv_data, save_path, obs_only=FALSE) {
    #get data as input
    data.list = list()
    t.list = list()
    data.list[[1]] = obs_data
    i = 2
    # Loop through interventional data rows and add to list
    cols = colnames(iv_data)
    for (row in (1:dim(iv_data)[1]) ) {
    	d = t(iv_data[row])
	    colnames(d) <- cols
	    rownames(d) <- row
    	data.list[[i]] = d
        t.list[[i]] = row
        i = i + 1
    }
    method <- "hsic.gamma"

    #prepare for sufficient statistics and intervention targets
    #suffstat <- list(data=data.list[[1]], ic.method=method)
    suffstat <- list(C=cor(data.list[[1]]), n=nrow(data.list[[1]]))
    alpha <-1e-3
    if (obs_only) {
        t.list=list(1)
    }
    #include observational dataset as an intervention
    intdata <- lapply(1:length(t.list), function(t) cbind(data.list[[t]], intervention_index=t) )
    inttargets <- t.list[1:length(t.list)]
    grspdag <- sp.restart.alg(suffstat, intdata, inttargets, alpha)
    write.csv(as.matrix(grspdag) ,row.names = FALSE, file = paste(save_path, 'igsp-adj_mat.csv',sep = ''))
}

run_from_file_igsp <- function(dataset_path, target_path, target_index_path, save_path) {
    # Read data, split into observational and interventional
    dataset <- read.table(dataset_path, sep=",", header=FALSE)
    print(dim(dataset))
    targets <- read.table(target_path, sep=",", header=FALSE)
    targets <- split(targets, 1:nrow(targets))
    targets <- lapply(targets, function(x) x[!is.na(x)])

    targets.index <- read.table(target_index_path, sep=",", header=FALSE)
    targets.index <- unlist(targets.index)
    obs_data_inds = targets.index[targets.index==1]
    int_data_inds = targets.index[targets.index>1]

    obs_data = dataset[obs_data_inds]
    iv_data = dataset[int_data_inds]

    data.list = list()
    t.list = list()
    data.list[[1]] = obs_data
    i = 2
    # Loop through interventional data rows and add to list
    cols = colnames(iv_data)
    for (target in (1:length(targets))) {
    	rows = int_data_inds[int_data_inds==targets]
    	rows = iv_data[rows]
    	data.list[[i]] = rows
        t.list[[i]] = target
        i = i + 1
    }
    method <- "hsic.gamma"

    #prepare for sufficient statistics and intervention targets
    #suffstat <- list(data=data.list[[1]], ic.method=method)
    suffstat <- list(C=cor(data.list[[1]]), n=nrow(data.list[[1]]))
    alpha <-1e-3

    #include observational dataset as an intervention
    intdata <- lapply(1:length(t.list), function(t) cbind(data.list[[t]], intervention_index=t) )
    inttargets <- t.list[1:length(t.list)]
    grspdag <- sp.restart.alg(suffstat, intdata, inttargets, alpha)
    write.csv(grspdag.mat() ,row.names = FALSE, file = paste(save_path, 'igsp-adj_mat.csv',sep = ''))
}
