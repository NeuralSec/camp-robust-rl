#!/bin/bash -l

# Activate virtual environment
source .rlenv/bin/activate

export id="cartpole_simple"
export method="camp"
export lamda=1.0

for sigma_loop in 0.0 0.2 0.4 0.6 0.8 1.0
do
export env_sigma=${sigma_loop}

#baseline checkpoint path
if [${method} -eq "baseline"]
then
export ckpt_path="./exp/train_dqn_${method}/${id}/wrapper_sigma=${env_sigma}-seed=0/checkpoints/${id}_restore.pth"
# camp checkpoint path
else
export ckpt_path="./exp/train_dqn_${method}/${id}/wrapper_sigma=${env_sigma}-lamda=${lamda}-seed=0/checkpoints/${id}_restore.pth"
fi

# Train the model
python dqn_test.py  --exp_name          "test_dqn_${method}" \
                    --env_id            ${id}                \
                    --torch_deterministic                    \
                    --num_models        1                    \
                    --env_sigma         ${env_sigma}         \
                    --num_evals         10000                \
                    --checkpoint_path   ${ckpt_path}         \
                    --lamda             ${lamda}             \
                    --seed              20                   
                    #--store_all_rewards  # turn on when test atari games
                    #--no_save
done