#!/bin/bash

count=33
max_count=43
while [ $count -le $max_count ]
do  
    echo $count
    python algos_tasks/sac_Reach.py --n_count $count 
    # python algos_tasks/sac_Slide.py --n_count $count 
    count=$((count + 1))
done