## The Code of Our Paper "CAMP in the Odyssey: Provably Robust Reinforcement Learning with Certified Radius Maximization"

We introduced CAMP and policy imitation to enhance the certified robustness of deep reinforcement learning agents. Policy imitation enables seamless integration of the CAMP loss into DQN training, significantly boosting the certified expected return of DQN agents under policy smoothing certification.


### 1. Environment Setup

The code requires Python version 3.11 and a Linux environment.

```
cd ${repository_name}
python -m venv .rlenv
pip install --upgrade pip --no-cache-dir
pip install -r requirements.txt --no-cache-dir
```

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
```lambda```: The $\lambda$ value for CAMP. Leave it to any value in the ```"baseline"``` mode.

Arguments for **Atari games**:\
```distill```: Turn on policy distillation when training CAMP agents in Atari game environments.\
```distill_path```: Set the path to the source policy to distill from.


Otherwise, run the following script to train agents with **NoisyNet**:
```
bash noisynet_train.sh
```
Arguments to set in the script:\
```env_id```: Set RL environment. Select from {"cartpole_simple", "cartpole_multiframe", "highway"}.\
```env_sigma```: Set the train-time noise scale. Please refer to the paper for the settings in our experiments.\


### 3. Test Agents
Test an agent, trained by either CAMP or Gaussian, for multiple runs to obtain and save testing rewards.
```
bash dqn_test.sh
```
In the script, set the following arguments to load the corresponding agent for testing.\
```env_id```: Set RL Environment. Select from {```"cartpole_simple"```, ```"cartpole_multiframe"```, ```"highway"```, ```"pong1r"```, ```"freeway"```, ```"bankheist"```}.\
```env_sigma```: The train-time noise scale.\
```checkpoint_path```: The path to the checkpoint for testing (use the provided ones in the script).\
```lambda```: $\lambda$ values of the CAMP agent to load.\
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
Load the rewards saved in Step 3 and certify the low bound of the expected return. Return the certified expected return in curve plots.
```
bash cert.sh
```
Arguments setting:\
```env_id```: Set the RL environment the agent is from. Select from {```"cartpole_simple"```, ```"cartpole_multiframe"```, ```"highway"```, ```"pong1r"```, ```"freeway"```, ```"bankheist"```}.\
```lambda```: The $\lambda$ value of the results to be loaded.\
```to_plot```: Which figure to plot. Select from {```"comparison"```, ```"ablation"```}. ```"comparison"``` plots the certified expected returns from different training methods in various environments. ```"ablation"``` plot the ablation study on $\lambda$.


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
