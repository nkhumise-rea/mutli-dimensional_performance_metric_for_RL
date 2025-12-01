#!/bin/bash



# X=120 # Number of minutes to wait (2hrs)
# Y=5 # Number of minutes to wait (5min)
# # sleep "${X}m"   # Wait X minutes
# python generation_ddpg.py

# sleep "${Y}m"   # Wait X minutes
# python generation_td3.py

# iter=1
# max_iter=5
# while [ $iter -le $max_iter ]
# do  
#     echo $iter
#     python generation_sac.py --n_iteration $iter
#     iter=$((iter + 1))
# done

iter=2
max_iter=4
while [ $iter -le $max_iter ]
do  
    echo $iter
    # python generation_sac.py --n_iteration $iter
    python generation_td3.py --n_iteration $iter
    # python generation_ddpg.py --n_iteration $iter
    iter=$((iter + 1))
done

python generation_sac.py --n_iteration 3


# iter=1
# max_iter=5
# while [ $iter -le $max_iter ]
# do  
#     echo $iter
#     python generation_td3.py --n_iteration $iter
#     iter=$((iter + 1))
# done
