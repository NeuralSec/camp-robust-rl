#!/bin/bash -l

# Activate virtual environment
source .rlenv/bin/activate

# Test the trained CAMP agent 
export id="cartpole_simple"
export method="camp"
export lamda=1.0
for sigma_loop in 0.2
do
export env_sigma=${sigma_loop}
export ckpt_path="./exp/train_dqn_${method}/${id}/wrapper_sigma=${env_sigma}-lamda=${lamda}-seed=0/checkpoints/${id}_restore.pth"
python dqn_test.py  --exp_name          "test_dqn_${method}" \
                    --env_id            ${id}                \
                    --torch_deterministic                    \
                    --num_models        1                    \
                    --env_sigma         ${env_sigma}         \
                    --num_evals         10000                \
                    --checkpoint_path   ${ckpt_path}         \
                    --lamda             ${lamda}             \
                    --seed              20
done

# Test the trained Gaussian agent 
export id="cartpole_simple"
export method="baseline"
export lamda=1.0
for sigma_loop in 0.2
do
export env_sigma=${sigma_loop}
export ckpt_path="./exp/train_dqn_${method}/${id}/wrapper_sigma=${env_sigma}-seed=0/checkpoints/${id}_restore.pth"
python dqn_test.py  --exp_name          "test_dqn_${method}" \
                    --env_id            ${id}                \
                    --torch_deterministic                    \
                    --num_models        1                    \
                    --env_sigma         ${env_sigma}         \
                    --num_evals         10000                \
                    --checkpoint_path   ${ckpt_path}         \
                    --lamda             ${lamda}             \
                    --seed              20
done

# Test the trained NoisyNet agent
export id="cartpole_simple"
export version="noisynet"
for sigma_loop in 0.2
do
export env_sigma=${sigma_loop}
export path="./exp/train_dqn_${version}/${id}/wrapper_sigma=${env_sigma}-seed=0/checkpoints/${id}_restore.pth"
python noisynet_test.py     --exp_name          "test_dqn_${version}"\
                            --env_id            ${id}                \
                            --torch_deterministic                    \
                            --env_sigma         ${env_sigma}         \
                            --num_evals         10000                \
                            --checkpoint_path   ${path}              \
                            --seed              20                                            
done


# Certify and plot results
export id="cartpole_simple"
export lamda=1.0
export to_plot="comparison"
python cert.py      --exp_name    "cert_dqn" \
                    --env_id      ${id}    \
                    --num_models  1 \
                    --lamda       ${lamda} \
                    --to_plot     ${to_plot}

# deactivate venv
deactivate