#!/usr/bin/env bash

# Script to run experiments.

#python main.py --algorithm="DQN" --comet --namestr="DQN Acrobot validation"

ipython --pdb ./rlscripts/main.py -- --normalize_score --comet --algorithm="DQN" --tag="DQN_randomStarts10x10" --namestr="DQN_timeLogs" --max_timesteps=300000 --start_timesteps=10000 --evaluate_every=1000 --save_every=100000 --polyak_target_update
