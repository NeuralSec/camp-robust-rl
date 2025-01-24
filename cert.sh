#!/bin/bash -l

# Activate virtual environment
source .rlenv/bin/activate

export id="cartpole_simple"
export lamda=1.0
export to_plot="comparison"  # Select from {"comparison", "ablation"}

# Train the model
python cert.py      --exp_name    "cert_dqn" \
                    --env_id      ${id}    \
                    --num_models  1 \
                    --lamda       ${lamda} \
                    --to_plot     ${to_plot}
                            
                            