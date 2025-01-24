#!/bin/bash -l

# Activate virtual environment
source .rlenv/bin/activate

export id="cartpole_simple"
export version="noisynet"

for sigma_loop in 0.0 0.2 0.4 0.6 0.8 1.0
do
export env_sigma=${sigma_loop}
export path="./exp/train_dqn_${version}/${id}/wrapper_sigma=${env_sigma}-seed=0/checkpoints/${id}_restore.pth"

# Train the model
python noisynet_test.py     --exp_name          "test_dqn_${version}"     \
                            --env_id            ${id}                \
                            --torch_deterministic                    \
                            --env_sigma         ${env_sigma}         \
                            --num_evals         10000                  \
                            --checkpoint_path   ${path}              \
                            --seed              20                   \
                            --store_all_rewards  # turn on when test atari games
                            #--no_save                          
done