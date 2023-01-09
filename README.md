# SP-GIES
Skeleton-Primed GIES algorithm

### Setup
Create conda environment using:
```
conda create --name <name> --file environment.yml 
```
To use the cupc and CDT submodules, initialize after cloning and compile:
```
git submodule init
git submodule update
nvcc -O3 --shared -Xcompiler -fPIC -o SP-GIES/Skeleton.so cupc/cuPC-S.cu
pip install ./CausalDiscoveryToolbox
```
To install relevant R packages use the installation script in R:
```
Rscript install.R
```

### Data
The repo contains three types of data: (1) Gaussian random data from Erdos Renyi, small-world and scale-free random networks.
(2) DREAM4 insilico network challenge (https://www.synapse.org/#!Synapse:syn3049712/wiki/74628) (3) RegulonDB gene regulatory network dataset (http://regulondb.ccg.unam.mx/)

To create random data, use the script ```create_random_data.py```. For examples of usage, check ```gen_scaling_data.sh```
To create DREAM4 insilico data in the correct format run the following to create the correct csv files:
``` 
python convert_dream4 -d insilico_size10_3 
```

### Run
To generate estimated graphs for each dataset (and for examples on how to run structure learners), from```SP-GIES/``` folder run 
```
Rscript run_all_datasets.R
```
To replicate results from the paper run the following to get scores for GIES and SP-GIES on each dataset
```
python compare_algs.py
``` 