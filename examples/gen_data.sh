#!/bin/bash

# Generate data for scaling study 
for n in 10 100 1000 2000;
do
python create_random_data.py --random_graph small_world --p 0.5 --nsamples 1000 --nnodes $n --seed 42 --working_dir ../data/random_test_set_scaling_${n}_small --num_graphs 1
done

# Generate data for random network evaluation
python create_random_data.py --random_graph erdos_renyi --nsamples 100 --nnodes 10 --seed 42 --working_dir ../data/random_test_set_10_ER --num_graphs 30

python create_random_data.py --random_graph small_world --nsamples 100 --nnodes 10 --seed 42 --working_dir ../data/random_test_set_10_small --num_graphs 30

python create_random_data.py --random_graph scale_free --nsamples 100 --nnodes 10 --seed 42 --working_dir ../data/random_test_set_10_scale --num_graphs 30

