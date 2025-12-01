#!/bin/bash

# count=100
# max_count=105
# while [ $count -le $max_count ]
# do  
#     echo $count
#     python algos_tasks/sac_Reach.py --n_count $count 
#     count=$((count + 1))
# done

# count=100
# max_count=105
# while [ $count -le $max_count ]
# do  
#     echo $count
#     python algos_tasks/ddpg_Reach.py --n_count $count 
#     count=$((count + 1))
# done

# count=100
# max_count=105
# while [ $count -le $max_count ]
# do  
#     echo $count
#     python algos_tasks/td3_Reach.py --n_count $count 
#     count=$((count + 1))
# done

count=1
max_count=5
while [ $count -le $max_count ]
do  
echo $count
python algos_tasks/ddpg_Reach.py --n_iteration $count

# python algos_tasks/sac_Reach.py --n_iteration $count

python algos_tasks/td3_Reach.py --n_iteration $count
count=$((count + 1))
done