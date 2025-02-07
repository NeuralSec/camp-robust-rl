#!/bin/bash -l

# Activate virtual environment
source .rlenv/bin/activate
cd attacks

# Attack CAMP agent with APGD
export id="cartpole_simple"
export env_sigma=0.2
export thresh=0.0
export ckpt_path="../exp/train_dqn_camp/${id}/wrapper_sigma=${env_sigma}-lamda=1.0-seed=0/checkpoints/${id}_restore.pth"
for eps in 0.2 0.4 0.6 0.8 1.0
do
python cartpole_simple_apgd.py  --checkpoint_path  ${ckpt_path}\
                                --attack_eps       ${eps}    \
                                --q_threshold      ${thresh} \
                                --num_evals        100
done

# Attack Gaussian agent with APGD
export id="cartpole_simple"
export env_sigma=0.2
export thresh=0.0
export ckpt_path="../exp/train_dqn_baseline/${id}/wrapper_sigma=${env_sigma}-seed=0/checkpoints/${id}_restore.pth"
for eps in 0.2 0.4 0.6 0.8 1.0
do
python cartpole_simple_apgd.py  --checkpoint_path  ${ckpt_path}\
                                --attack_eps       ${eps}    \
                                --q_threshold      ${thresh} \
                                --num_evals        100
done

# Plot attack result
python cartpole_simple_plot_attacks.py --attack_type "apgd" --num_evals 100

# deactivate venv
deactivate