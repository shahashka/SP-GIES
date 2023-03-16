#!/bin/bash

#for n in 1000;
#do
n=1000
python create_random_data.py --random_graph small_world --p 0.5 --k 2 --nsamples 500 --nnodes $n --seed 42 --working_dir ./random_test_set_fewer_${n}_500n_small --num_graphs 1 --ivnodes 10

python create_random_data.py --random_graph scale_free --p 0.5 --k 2 --nsamples 500 --nnodes $n --seed 42 --working_dir ./random_test_set_fewer_${n}_500n_scale --num_graphs 1 --ivnodes 10
#done

