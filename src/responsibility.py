
import pandas as pd
import numpy as np

import sys
sys.path.append('../')
from utils import *

pd.set_option('display.max_columns', None)

from pathlib import Path


# Set path to save the resulting dataframe. Otherwise, set to None.
SAVE_PATH = '../data/'

DATA_PATH = '../data/'
PASSES_DF_PATH = DATA_PATH + 'passes_df.pkl'
WYSCOUT_TO_SKILLCORNER = DATA_PATH + 'wyscout2skillcorner.csv'


# Load data
passes_df = pd.read_pickle(PASSES_DF_PATH)
wyscout2skillcorner = pd.read_csv(WYSCOUT_TO_SKILLCORNER).drop(columns='id')
passes_df.rename(columns={'player.id': 'player.id.wyscout', 'pass.recipient.id': 'pass.recipient.id.wyscout', 'tracking.player_id': 'tracking.player.id.skillcorner'}, inplace=True)
passes_df.drop(columns=['team.id', 'opponentTeam.id'], inplace=True)


# TODO: An id value of 0 doesn't map to any player. Look into this.
passes_df = passes_df[passes_df['pass.recipient.id.wyscout'] != 0]


# Map Wyscout IDs to Skillcorner IDs
passes_df = passes_df.merge(wyscout2skillcorner[['player_id_wy', 'player_id_sk']],
                            left_on='player.id.wyscout', 
                            right_on='player_id_wy', 
                            how='left')
passes_df = passes_df.merge(wyscout2skillcorner[['player_id_wy', 'player_id_sk']],
                            left_on='pass.recipient.id.wyscout', 
                            right_on='player_id_wy', 
                            how='left')
passes_df.rename(columns={'player_id_sk_x': 'player.id.skillcorner', 'player_id_sk_y': 'pass.recipient.id.skillcorner'}, inplace=True)
passes_df.drop(columns=['player_id_wy_x', 'player_id_wy_y'] , inplace=True)


# Add information about the tracked object
passes_df['tracking.is_teammate'] = (passes_df['team.name'] == passes_df['tracking.team_name'])
passes_df['tracking.is_self'] = (passes_df['player.id.skillcorner'] == passes_df['tracking.player.id.skillcorner'])


# Apply responsibility function
passes_df['responsibility'] = passes_df.apply(responsibility, axis=1, pass_length_factor=1)
passes_df['responsibility'] = np.where(passes_df['tracking.is_teammate'], 0, passes_df['responsibility'])


if SAVE_PATH:
    passes_df.to_pickle(SAVE_PATH + 'passes_resp_df.pkl')