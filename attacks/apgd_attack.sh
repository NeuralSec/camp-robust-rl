#!/bin/bash -l

# Activate virtual environment
source ../.rlenv/bin/activate

export id="cartpole_simple"
export env_sigma=0.2
export thresh=0.0

# Path to clean model without defense or with Gaussian
#export ckpt_path="../exp/train_dqn_baseline/${id}/wrapper_sigma=${env_sigma}-seed=0/checkpoints/${id}_restore.pth"
# Path to model trained by CAMP
export ckpt_path="../exp/train_dqn_camp/${id}/wrapper_sigma=${env_sigma}-lamda=1.0-seed=0/checkpoints/${id}_restore.pth"

for eps in 0.2 0.4 0.6 0.8 1.0 # Cartpole
#for eps in 0.4 0.8 1.2 1.6 2.0 # Highway
#for eps in 0.05 0.15 0.25 0.35 # Atari games
do
#Run correspoinding script in {cartpole_simple_apgd.py, cartpole_multiframe_apgd.py, pong_1r_apgd.py, freeway_apgd.py, highway_apgd.py, bankheist_apgd.py}
python cartpole_simple_apgd.py  --checkpoint_path  ${ckpt_path}\
                                --attack_eps       ${eps}    \
                                --q_threshold      ${thresh} \
                                --num_evals        1000 #\
                                #--store_all_rewards
done