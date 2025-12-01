#!/bin/bash

dim=2 #100
max_dim=4
while [ $dim -le $max_dim ]
do  
    echo $dim
    python train_autoencoder.py --encoding_dim $dim
    dim=$((dim + 1))
done