#!/bin/bash -l

# Activate virtual environment
source .rlenv/bin/activate

export env="cartpole_simple"
export mode="camp"
export config_path="config"
export sigma=0.2

# Set the following paths for CAMP distillation in Atari games. Comment out otherwise.
#export ref_path="exp/train_dqn_baseline/${env}/wrapper_sigma=${sigma}-seed=0/checkpoints/${env}_restore.pth"
#export primary_path="exp/train_dqn_baseline/${env}/wrapper_sigma=${sigma}-seed=0/checkpoints/${env}_restore.pth"

for lamda_loop in 1.0 #0.5 1.0 2.0 4.0 8.0 16.0
do
# Set the following path to resume training from a checkpoint
#export primary_path="exp/train_dqn_camp/${env}/wrapper_sigma=${sigma}-lamda=${lamda}-seed=0/checkpoints/${env}_restore.pth"

# Train the model
python dqn_train.py     --exp_name        "train_dqn_${mode}" \
                        --config_path     ${config_path} \
                        --env_id          ${env}  \
                        --torch_deterministic    \
                        --train_mode      ${mode} \
                        --num_models      1       \
                        --lamda           ${lamda_loop}\
                        --env_sigma       ${sigma}\
                        --seed            0      # \
                        #--distill                 \
                        #--distill_path    ${ref_path} \
                        #--load_checkpoint ${primary_path}       
done