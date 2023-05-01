# SP-GIES
Skeleton-Primed GIES algorithm. A simple two step structure learning algorithm for estimating the causal graph between
random variables in a system. Step (1) uses observational structure learners: PC, ARACNE-AP or 
CLR to estimate the skeleton. Step (2) uses GIES with the skeleton as input to restrict the edge set of the optimization. 
This achieves up to 4x speedup compared to GIES. 

### Setup
Clone the repository:
```
git clone https://github.com/shahashka/SP-GIES.git
```
Create conda environment using:
```
conda env create --name <name> --file=environment.yml 
```

To use the cupc and CDT submodules, initialize after cloning and compile:
```
git submodule init
git submodule update
nvcc -O3 --shared -Xcompiler -fPIC -o R_scripts/Skeleton.so cupc/cuPC-S.cu
ln -s R_scripts/Skeleton.so  examples/Skeleton.so
pip install ./CausalDiscoveryToolbox
```
To install relevant R packages use the installation script in R:
```
Rscript install.R
```

### Installation
To install the package from source, execute the following command from the root of the cloned repository:

```
python -m pip install .
```

### Data
The repo contains three types of data located in the ```/data``` folder : (1) Gaussian random data from Erdos Renyi, small-world and scale-free random networks.
(2) DREAM4 insilico network challenge (https://www.synapse.org/#!Synapse:syn3049712/wiki/74628) (3) RegulonDB gene regulatory network dataset (http://regulondb.ccg.unam.mx/)

To create random data, use the script ```examples/create_random_data.py```. For examples of usage, check ```examples/gen_scaling_data.sh```
To create DREAM4 insilico data in the correct format run the following to create the correct csv files:
``` 
python examples/convert_dream4.py -d data/insilico_size10_3 
```

### Run
To generate estimated graphs for each dataset (and for examples on how to run structure learners), use any of the 
```run_``` scripts in the ```R_scripts/``` e.g. (note that you must run these from the ```R_scripts/``` directory )
```
Rscript run_dream4.R
```
To replicate results from the paper run the following to get scores for GIES and SP-GIES on each dataset
```
python examples/compare_algs.py
``` 

For users intending to use SP-GIES on their own datasets, a Python wrapper for SP-GIES is also available in causal_learning/sp_gies.py. 
For an example on how to use this wrapper, as well as the input file requirements, expected run time etc...see the Jupyter notebook 
[`use_sp_gies.ipynb`](https://github.com/shahashka/SP-GIES/blob/main/examples/use_sp_gies.ipynb)

## License

sp-gies has a MIT license, as seen in the [`LICENSE.md`](https://github.com/shahashka/SP-GIES/blob/main/LICENSE.md) file.

## Citation

If you use this algorithm in your research, please cite this paper:

```bibtex
@article{shah2023causal,
  title={Causal Discovery and Optimal Experimental Design for Genome-Scale Biological Network Recovery},
  author={Shah, Ashka and Ramanathan, Arvind and Hayot-Sasson, Valerie and Stevens, Rick},
  journal={arXiv preprint arXiv:2304.03210},
  year={2023}
}
```
