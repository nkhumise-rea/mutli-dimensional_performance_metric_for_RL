#!/bin/bash

EXAMPLES=rl_reliability/examples
python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py
# python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py --resampling permute
# python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py --resampling bootstrap