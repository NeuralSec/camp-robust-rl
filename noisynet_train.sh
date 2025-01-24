#!/bin/bash -l

# Activate virtual environment
source .rlenv/bin/activate

export env="cartpole_simple"
export sigma=0.2

# Train the model
python noisynet_train.py    --exp_name      "train_dqn_noisynet" \
                            --env_id        ${env} \
                            --torch_deterministic \
                            --env_sigma      ${sigma} \
                            --seed 0