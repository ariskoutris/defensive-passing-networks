import numpy as np
import gym
from gym import spaces

import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from utils_parallel import get_dxt_parallel, threat_aggregator

from resp import responsibility_parallel

class DefenderPosEnv(gym.Env):
    """
    Custom RL Environment for positioning defenders optimally on the field.


    ASSUMPTIONS:
    - 11 defenders
    - 11 attackers: 10 recipients + 1 passer
    - undefined if total number of players not 22 (in case red cards, injuries etc.)

    Action Space: Continuous changes to player coordinates within a circle of radius max_radius
    Observation Space: Players' positions on the field: 11 defenders, 10 recipients, 1 passer coordinates (x, y)
    - +1 play_direction

    Objective: Relocate defenders to minimize the xt generated by the opponents
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, pitch_dict, passes_df, max_radius=5.0, threat_agg='max'):
        super(DefenderPosEnv, self).__init__()

        self.pitch_dict = pitch_dict
        self.passes_df = passes_df
        self.threat_agg = threat_aggregator(threat_agg)

        self.frame_id_list = passes_df['frame'].unique()
        coeff = 1.5
        self.min_x = -(self.pitch_dict['pitch_length'] / 2) * coeff
        self.max_x = (self.pitch_dict['pitch_length'] / 2) * coeff
        self.min_y = -(self.pitch_dict['pitch_width'] / 2) * coeff
        self.max_y = (self.pitch_dict['pitch_width'] / 2) * coeff
        # maximum radius for the defenders to be placed from the actual position
        self.max_radius = max_radius
        
        # Define action space: Continuous changes to player coordinates => 11 (angle, radius) => 22 dimensions
        low_action = np.tile(np.array([0, 0]), (11, 1))
        high_action = np.tile(np.array([2*np.pi, self.max_radius]), (11, 1))
        self.action_space = spaces.Box(
            low=low_action, 
            high=high_action, 
            shape=(11, 2),
            dtype=np.float32
        )
        # Define observation space: Players' positions on the field => 22 (x,y) => 44 dimensions
        low_action = np.tile(np.array([self.min_x, self.min_y]), (22, 1))
        high_action = np.tile(np.array([self.max_x, self.max_y]), (22, 1))
        self.observation_space = spaces.Box(
            low=low_action,  # Min bounds for all 10 pairs
            high=high_action,  # Max bounds for all 10 pairs
            shape=(22, 2),  # 22 pairs of (x, y) coordinates
            dtype=np.float32
        )

        # Initial state
        self.state = None
        self.prev_state = np.zeros((22,2))
        self.count = 0

        self.frame_id = None # picked frame_id for the passes_df
        self.play_dir = None
        self.state_init = None
        

        self.reward_scale = 1000
        self.max_iters = 200
        self.iters = 0
        
        self.reward_init = 0
        self.reward_final = 0
        self.reward = 0

        

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        # uniformly sample a frame_id from the passes_df
        print('reward_init:', self.reward_init, 'reward_final:', self.reward_final)

        self.frame_id = np.random.choice(self.frame_id_list)
        print('frame_id:', self.frame_id)

        self.state_init = self.get_init(self.frame_id)
        self.state = self.state_init
        assert self.state.shape == (22, 2), AssertionError('state dimension wrong: {}'.format(self.state.shape))
        
        self.iters = 0
        self.reward_init = - self.get_action_reward(reward_scale=self.reward_scale)

        return self.state, {}

    def step(self, action):
        """Execute one step in the environment."""
        # Update player positions with actions
        action_alpha = action[:,0]
        action_radius = action[:,1].reshape(-1,1)
        action_net = np.stack([np.cos(action_alpha), np.sin(action_alpha)], axis=1) * action_radius
        
        assert action_net.shape == (11,2), AssertionError('action dimension invalid: {}'.format(action_net.shape))

        reward_before = - self.get_action_reward(reward_scale=self.reward_scale)
        # update states after the action
        self.state[:11] = self.state_init[:11] + action_net
        

        # opponent plays the optimal pass at that moment if threat_agg=max, there could be different policies, eg. expected threat etc.
        # Reward: Encourage players to move closer to the goal
        reward = - self.get_action_reward(reward_scale=self.reward_scale) # minimize the threat, maximize negated threat value
        self.reward = reward
        done = False
        # done = # TODO => define condition later

        print('reward_before:', reward_before)
        print('reward:', reward)

        # print(self.state)
        # print(action)
        if np.allclose(self.state, self.prev_state, atol=1e-4):
            self.count += 1
            if self.count > 10:
                print('converged, done!')
                done = True
            else:
                self.count = 0
        self.prev_state = self.state
        
        self.iters += 1
        if self.iters > self.max_iters:
            # print('max_iters reached, done!')
            done = True

        self.reward_final = reward
        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        # TODO: Implement rendering => ARIS Plot, for all defender => from original position to new position or previous position to new position
        """Visualize the environment."""
        # print(f"State: {self.state}")
        print(f"Reward: {self.reward}")
        pass
        # plt.figure(figsize=(10, 5))
        # plt.xlim(0, self.field_size[0])
        # plt.ylim(0, self.field_size[1])
        
        # # Draw the field
        # plt.gca().add_patch(plt.Rectangle((0, 0), *self.field_size, fill=False, edgecolor='black'))
        
        # # Draw players
        # for i, pos in enumerate(self.state):
        #     plt.plot(pos[0], pos[1], 'bo', label=f"Player {i+1}" if i == 0 else "")
        
        # # Draw goal position
        # plt.plot(self.goal_position[0], self.goal_position[1], 'rx', label="Goal")
        
        # plt.legend()
        # plt.show()

    def close(self):
        """Close the environment."""
        pass


    def get_init(self, frame_id):
        data = self.passes_df
        data = data[data['frame']==frame_id] # filter pass
        
        self.play_dir = data['play_direction'].unique()
        assert len(self.play_dir) == 1, AssertionError('should only get 1 player_dir for the pass')
        self.play_dir = self.play_dir[0]

        recipients = data[data['tracking.is_teammate'] & ~data['tracking.is_self'] & (data['tracking.object_id'] != -1)].copy()
        defenders = data[~data['tracking.is_teammate'] & ~data['tracking.is_self'] & (data['tracking.object_id'] != -1)].copy()

        defender_coord = defenders[['tracking.x', 'tracking.y']].values # 11,2
        recipient_coord = recipients[['tracking.x', 'tracking.y']].values # 10,2

        passer_id = data['player.id.skillcorner'].values[0]
        passer_coord = data[data['tracking.object_id'] == passer_id][['tracking.x', 'tracking.y']].values[0].reshape(1,2) # 1,2

        state = np.concatenate([defender_coord, recipient_coord, passer_coord]) # 22,2

        return state


    def get_action_reward(self, reward_scale=1.0):
        def_loc = self.state[0:11, :]
        recip_loc = self.state[11:21, :]
        passer_loc = self.state[-1, :]

        # get dxt for each pass option
        x_start = passer_loc[0] # scalar
        y_start = passer_loc[1]
        x_end = recip_loc[:,0] # 10,1
        y_end = recip_loc[:,1]

        dxt_arr = get_dxt_parallel(x_start, y_start, x_end, y_end, self.play_dir, self.pitch_dict) # 10,1

        # calculate responsibility
        resp_matrix = responsibility_parallel(def_loc, recip_loc, passer_loc) # (11,10)

        reward = np.prod(1-resp_matrix, axis=0) * dxt_arr
        reward = self.threat_agg(reward)
        
        return reward*reward_scale