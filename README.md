# SP-GIES
Skeleton-Primed GIES algorithm

Create conda environment using:
```
conda create --name <name> --file environment.yml 
```
To use the cupc submodule, initialize after cloning and compile:
```
git submodule init
git submodule update
nvcc -O3 --shared -Xcompiler -fPIC -o SP-GIES/Skeleton.so cupc/cuPC-S.cu
```
To install relevant R packages use the installation script in R:
```
Rscript install.R
```
