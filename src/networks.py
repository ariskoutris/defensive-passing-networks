import pandas as pd
from utils import *


# Set Wyscout and Skillcorner IDs
WYSCOUT_ID = 5414111
SKILLCORNER_ID = 952209


# Set path to save the resulting dataframes. Otherwise, set to None.
SAVE_PATH = f'../data/networks/match_{SKILLCORNER_ID}/'
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH + 'defender_dyads/', exist_ok=True)

DATA_PATH = f'../data/networks/match_{SKILLCORNER_ID}/'
PASSES_DF_PATH = DATA_PATH + 'passes_df.pkl'



# Passes Network
passes_df = pd.read_pickle(PASSES_DF_PATH)

# Keep only players of the opposing team, who are defenders with responsibility greater than 0.
passes_df = passes_df[~passes_df['tracking.is_teammate']]
passes_df = passes_df[passes_df['responsibility'] > 0]



# Passes Network
column_mask = ['player.id.skillcorner', 'pass.recipient.id.skillcorner', 'location.x', 'location.y' , 'pass.endLocation.x', 'pass.endLocation.y', 'dxt', 'responsibility']
group_column_mask = [col for col in column_mask if col != 'responsibility']
passes_network = passes_df[column_mask].groupby(group_column_mask).agg({'responsibility': 'sum'}).reset_index()
passes_network.to_csv(SAVE_PATH + 'passes_network.csv', index=True)



# Defender Responsibility Network
for tracking_object_id in passes_df['tracking.object_id'].unique():
    player_passes_df = passes_df[passes_df['tracking.object_id'] == tracking_object_id]
    defender_responsibility_network = player_passes_df[player_passes_df['responsibility'] > 0]

    column_mask = ['player.id.skillcorner', 'pass.recipient.id.skillcorner', 'location.x', 'location.y' , 'pass.endLocation.x', 'pass.endLocation.y', 'dxt', 'responsibility']
    defender_responsibility_network = defender_responsibility_network[column_mask]

    defender_responsibility_network.to_csv(SAVE_PATH + f'defender_dyads/{tracking_object_id}.csv', index=True)



# Aggregate Statistics for each Defender
defender_stats = passes_df.groupby('tracking.object_id').agg(
    average_responsibility=('responsibility', 'mean'),
    average_dxt=('dxt', 'mean'),
    group_size=('responsibility', 'size')
).reset_index()

defender_stats.to_csv(SAVE_PATH + 'defender_stats.csv', index=False)



# Defender Dyads
passes_df_cp = passes_df.reset_index()
pass_filt_df = passes_df_cp.groupby('frame').filter(lambda x: len(x) >= 2)

relevant_cols = ['frame', 'tracking.object_id_x', 'tracking.object_id_y',  'dxt_x', 'responsibility_x', 'responsibility_y']
joint_df = pass_filt_df.merge(pass_filt_df, on='frame')[relevant_cols]
joint_df = joint_df[joint_df['tracking.object_id_x'] < joint_df['tracking.object_id_y']]
joint_df['joint_resp'] = joint_df['responsibility_x'] + joint_df['responsibility_y']
joint_df.rename(columns={'dxt_x': 'dxt'}, inplace=True)

columns_joint_group = ['frame', 'tracking.player.id.skillcorner_x', 'tracking.player.id.skillcorner_y', 'joint_resp']
defender_dyads_network = joint_df.groupby(['tracking.object_id_x', 'tracking.object_id_y']).agg(
    joint_responsibility_mean=('joint_resp', 'mean'),
    joint_responsibility_sum=('joint_resp', 'sum'),
    group_size=('joint_resp', 'size')
    ).reset_index()

defender_dyads_network.to_csv(SAVE_PATH + 'defender_dyads_network.csv', index=False)

