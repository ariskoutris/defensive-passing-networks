import numpy as np
import pandas as pd

WYSCOUT_ID = 5414111
SKILLCORNER_ID = 952209

NETWORKS_PATH = f'../../data/networks/match_{SKILLCORNER_ID}/'
PASSES_DF_PATH = NETWORKS_PATH + 'passes_df.csv'

DATA_PATH= f'../../data/'
XT_PLOT_PATH = DATA_PATH + 'smoothed_xt.csv'
METADATA_PATH = DATA_PATH + f'skillcorner/{SKILLCORNER_ID}_metadata.csv'

passes_df = pd.read_csv(PASSES_DF_PATH)
xt_table = pd.read_csv(XT_PLOT_PATH)

cell_width = 100 / xt_table.shape[1]
cell_height = 100 / xt_table.shape[0]

# prepare with pitch_dict, outside the function
pitch_length = 105
pitch_width = 68
xt_rows, xt_cols = 68, 105
cell_width = pitch_length / xt_cols
cell_height = pitch_width / xt_rows


pitch_dict = {
    'pitch_length': pitch_length,
    'pitch_width': pitch_width,
    'xt_rows': xt_rows,
    'xt_cols': xt_cols,
    'cell_width': cell_width,
    'cell_height': cell_height,
    'xt_table': xt_table,
    'xt_table_np': xt_table.to_numpy(),
}




import gym

from gym.envs.registration import register
from stable_baselines3 import PPO  # Example algorithm: Proximal Policy Optimization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import sys

exp_version = 'maxiter200_rescale1000'
exp_dir = 'logs_{}'.format(exp_version)
# if directory does not exist makedir
import os
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
sys.stdout = open(os.path.join(exp_dir, 'out.txt'), 'w')
sys.stderr = open(os.path.join(exp_dir, 'err.txt'), 'w')

register(
    id='DefenderPosEnv',  # Unique identifier
    entry_point='env:DefenderPosEnv',  # Specify the path to your environment class
    # entry_point='rl.env:DefenderPosEnv',  # Specify the path to your environment class
)

env = gym.make("DefenderPosEnv", pitch_dict=pitch_dict, passes_df=passes_df)



# Initialize the RL model
model = PPO("MlpPolicy", env, verbose=1)  # "MlpPolicy" uses a fully connected network

eval_callback = EvalCallback(
    env, 
    best_model_save_path=exp_dir, 
    log_path=exp_dir, 
    eval_freq=1000, 
    deterministic=True, 
    render=True
)

print('starting...')

# Train the model
timesteps = 100000
model.learn(total_timesteps=timesteps, callback=eval_callback)
print('training completed')

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print('evaluation completed')
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Test the trained model
obs, _ = env.reset()
for step in range(10000):
    action, _states = model.predict(obs, deterministic=True)  # Choose the best action
    obs, reward, done, truncated, info = env.step(action)
    print('eval_reward: ', reward)
    # env.render()  # Render the environment
    if done or truncated:
        obs, _ = env.reset()

env.close()