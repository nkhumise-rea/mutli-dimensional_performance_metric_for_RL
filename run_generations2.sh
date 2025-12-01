#!/bin/bash

iter=1
max_iter=5
while [ $iter -le $max_iter ]
do  
    echo $iter
    python generation_td3.py --n_iteration $iter
    iter=$((iter + 1))
done
