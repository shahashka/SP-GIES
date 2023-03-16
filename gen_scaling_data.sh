#!/bin/bash

for n in 10 100 1000 2000;
do
python create_random_data.py --random_graph small_world --p 0.5 --nsamples 1000 --nnodes $n --seed 42 --working_dir ./random_test_set_fewer_${n}_small --num_graphs 1   
done

n=1000
python create_random_data.py --random_graph small_world --p 0.5 --k 2 --nsamples 500 --nnodes $n --seed 42 --working_dir ./random_test_set_fewer_${n}_500n_small --num_graphs 1 --ivnodes 10

python create_random_data.py --random_graph scale_free --p 0.5 --k 2 --nsamples 500 --nnodes $n --seed 42 --working_dir ./random_test_set_fewer_${n}_500n_scale --num_graphs 1 --ivnodes 10
