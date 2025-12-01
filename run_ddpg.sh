#!/bin/bash

count=34 #100
max_count=43
while [ $count -le $max_count ]
do  
    echo $count
    python algos_tasks/ddpg_Reach.py --n_count $count 
    count=$((count + 1))
done