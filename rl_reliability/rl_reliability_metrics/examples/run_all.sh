#!/bin/bash



python tf_agents_mujoco_subset/evaluate_metrics.py
python tf_agents_mujoco_subset/evaluate_metrics.py --resampling permute
python tf_agents_mujoco_subset/evaluate_metrics.py --resampling bootstrap

python tf_agents_mujoco_subset/permutation_tests.py
python tf_agents_mujoco_subset/bootstrap_confidence_intervals.py
python tf_agents_mujoco_subset/plots.py