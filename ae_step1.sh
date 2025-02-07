#!/bin/bash -l

# Activate virtual environment
source .rlenv/bin/activate

# Train agents using CAMP
export env="cartpole_simple"
export mode="camp"
export config_path="config"
export sigma=0.2
for lamda_loop in 1.0
do
python dqn_train.py     --exp_name        "train_dqn_${mode}" \
                        --config_path     ${config_path} \
                        --env_id          ${env}  \
                        --torch_deterministic    \
                        --train_mode      ${mode} \
                        --num_models      1       \
                        --lamda           ${lamda_loop}\
                        --env_sigma       ${sigma}\
                        --seed            0     
done

# Train agents using Gaussian
export env="cartpole_simple"
export mode="baseline"
export config_path="config"
export sigma=0.2
for lamda_loop in 1.0
do
python dqn_train.py     --exp_name        "train_dqn_${mode}" \
                        --config_path     ${config_path} \
                        --env_id          ${env}  \
                        --torch_deterministic    \
                        --train_mode      ${mode} \
                        --num_models      1       \
                        --lamda           ${lamda_loop}\
                        --env_sigma       ${sigma}\
                        --seed            0     
done

# Train agents using NoisyNet
export env="cartpole_simple"
export sigma=0.2
python noisynet_train.py    --exp_name      "train_dqn_noisynet" \
                            --env_id        ${env} \
                            --torch_deterministic \
                            --env_sigma      ${sigma} \
                            --seed 0

# deactivate venv
deactivate