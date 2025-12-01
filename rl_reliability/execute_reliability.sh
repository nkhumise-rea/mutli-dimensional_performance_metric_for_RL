#!/bin/bash

EXAMPLES=rl_reliability_metrics/examples
python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py
python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py --resampling permute
python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py --resampling bootstrap

python3 $EXAMPLES/tf_agents_mujoco_subset/permutation_tests.py
python3 $EXAMPLES/tf_agents_mujoco_subset/bootstrap_confidence_intervals.py
python3 $EXAMPLES/tf_agents_mujoco_subset/plots.py