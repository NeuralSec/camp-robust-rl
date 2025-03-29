## The Code of Our Paper "CAMP in the Odyssey: Provably Robust Reinforcement Learning with Certified Radius Maximization" (USENIX Security Symposium'25)

We introduced CAMP and policy imitation to enhance the certified robustness of deep reinforcement learning agents. Policy imitation enables seamless integration of the CAMP loss into DQN training, significantly boosting the certified expected return of DQN agents under policy smoothing certification.


### 1. Environment Setup

The code requires Python version 3.11 and a Linux environment.

```
cd ${repository_name}
python -m venv .rlenv
source .rlenv/bin/activate
pip install --upgrade pip --no-cache-dir
pip install -r requirements.txt --no-cache-dir
deactivate
```
Our code is developed on CSIRO Virga HPC with SUSE Linux Enterprise Server SLE15. 
```requirements.txt``` includes a comprehensive list of dependencies, rather than just the minimal required ones, to facilitate the recreation of the virtual environment used during our experiments.
We have observed that when installing dependencies on some Ubuntu distributions (tested on WSL2 Ubuntu 24.04.1 LTS), running ```pip install -r requirements.txt --no-cache-dir``` may result in errors like segmentation fault due to conflicting dependency versions. 
If this occurs, rerunning the same command immediately after the error often resolves the issue.


### 2. Train Agents
Run the following script to train DQN using **CAMP** or the **Gaussian baseline**.
```
bash dqn_train.sh
```
Set arguments in the script:\
```env_id```: Set RL environment. Select from {```"cartpole_simple"```, ```"cartpole_multiframe"```, ```"highway"```, ```"pong1r"```, ```"freeway"```, ```"bankheist"```}.\
```train_mode```: Set training method. Select from {```"baseline"```, ```"camp"```}. ```"baseline"``` refers to the Gaussian baseline.\
```config_path```: Select ```"config"``` for Cartpole and Highway. Use ```"atari_config"``` for Atari games.\
```env_sigma```: Set the train-time noise scale. Please refer to the paper for the settings in our experiments.\
```lamda```: The $\lambda$ value for CAMP. Leave it to any value in the ```"baseline"``` mode.

Arguments for **Atari games**:\
```distill```: Turn on policy distillation when training CAMP agents in Atari game environments.\
```distill_path```: Set the path to the source policy to distill from.


Otherwise, run the following script to train agents with **NoisyNet**:
```
bash noisynet_train.sh
```
Arguments to set in the script:\
```env_id```: Set RL environment. Select from {"cartpole_simple", "cartpole_multiframe", "highway"}.\
```env_sigma```: Set the train-time noise scale. Please refer to the paper for the settings in our experiments.

To monitor training with TensorBoard, open a new terminal, navigate to the root directory of the repository with the virtual environment activated, and run ```tensorboard --logdir="exp"```. 
This will allow you to visualize the training progress.

### 3. Test Agents
Test an agent, trained by either CAMP or Gaussian, for multiple runs to obtain and save testing rewards.
```
bash dqn_test.sh
```
In the script, set the following arguments to load the corresponding agent for testing.\
```env_id```: Set RL Environment. Select from {```"cartpole_simple"```, ```"cartpole_multiframe"```, ```"highway"```, ```"pong1r"```, ```"freeway"```, ```"bankheist"```}.\
```env_sigma```: The train-time noise scale.\
```checkpoint_path```: The path to the checkpoint for testing (use the provided ones in the script).\
```lamda```: $\lambda$ values of the CAMP agent to load.\
```store_all_rewards```: Storing reward at each step instead of only saving the episodic return. Turn on when testing in {```"highway"```, ```"freeway"```, ```"bankheist"```} for correct certification results.

To test a NoisyNet agent, run:
```
bash noisynet_test.sh
```
Arguments setting:\
```env_id```: Set RL Environment. Select from {```"cartpole_simple"```, ```"cartpole_multiframe"```, ```"highway"```}.\
```env_sigma```: The train-time noise scale.\
```checkpoint_path```: The path to the checkpoint for testing (use the provided ones in the script).\
```store_all_rewards```: Storing reward at each step instead of only saving the episodic return. Turn on when testing in {```"highway"```} for correct certification results.


### 4. Certify Expected Return and Plot Results
Load the rewards saved in Step 3 and certify the lower bound of the expected return. Return the certified expected return in curve plots.
```
bash cert.sh
```
Arguments setting:\
```env_id```: Set the RL environment the agent is in. Select from {```"cartpole_simple"```, ```"cartpole_multiframe"```, ```"highway"```, ```"pong1r"```, ```"freeway"```, ```"bankheist"```}.\
```lamda```: The $\lambda$ value of the results to be loaded.\
```to_plot```: Which figure to plot. Select from {```"comparison"```, ```"ablation"```}. ```"comparison"``` plots the certified expected returns from different training methods in various environments. ```"ablation"``` plot the ablation study on $\lambda$.\
You may adjust ```sigma_list``` in Lines 139 and 143 for plotting the results from corresponding $\sigma$ configurations.

### 5. Attack Agents and Plot Results
We adopt PGD and APGD attacks to evaluate the empirical robustness of agents. This part requires a little bit of modification of the script: **the to-be-executed ```.py``` file should be selected according to the RL environment.** Please follow the comments in the script. Run the following script for **PGD**:
```
cd attacks
bash attack.sh
``` 
Arguments setting:\
```checkpoint_path```: The path to the checkpoint that will be attacked (use the provided ones in the script by only adjusting the ```env_sigma``` value).\
```attack_eps```: Total perturbation budget.\
```store_all_rewards```: Storing reward at each step instead of only saving the episodic return. Turn on when testing in {```"highway"```, ```"freeway"```, ```"bankheist"```} for correctly plotting the results.

Use the following for **APGD**:
```
cd attacks
bash apgd_attack.sh
```
Arguments setting:\
```checkpoint_path```: The path to the checkpoint that will be attacked (use the provided ones in the script by only adjusting the ```env_sigma``` value).\
```attack_eps```: Total perturbation budget.\
```store_all_rewards```: Storing reward at each step instead of only saving the episodic return. Turn on when testing in {```"highway"```, ```"freeway"```, ```"bankheist"```} for correctly plotting the results.

Use ```attacks/<env_id>_plot_attacks.py``` to plot attack results by setting ```attack_type``` to either ```pgd``` or ```apgd```.

### 6. USENIX Artifact Evaluation
Four scripts are provided for USENIX artifact evaluation.
```
bash ae_step1.sh
bash ae_step2.sh
bash ae_step3.sh
bash ae_step4.sh
```
These scripts enable a fuss-free reproduction of the training-testing-certification pipeline and empirical robustness evaluation tasks in "Cartpole-1" under a specific hyper-parameter setting ($\sigma=0.2$). 
This configuration represents one of the major improvement cases (*i.e.*, "Cartpole" and "Highway"), and other configurations can be evaluated by simply setting the ```env_id``` and ```env_sigma``` arguments in the scripts.
Otherwise, please follow Sections 1-5 for detailed instructions on running experiments in different environments with various hyper-parameter settings. 
