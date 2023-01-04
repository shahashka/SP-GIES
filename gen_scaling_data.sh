#!/bin/bash

for n in 10 100 1000 2000;
do
python create_random_data.py --random_graph small_world --p 0.5 --nsamples 1000 --nnodes $n --seed 42 --working_dir ./random_test_set_fewer_${n}_small --num_graphs 1   
done
