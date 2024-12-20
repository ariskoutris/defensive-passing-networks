import pandas as pd
from utils import *
from resp import responsibility    
    
# Set Wyscout and Skillcorner IDs
WYSCOUT_ID = 5414111
SKILLCORNER_ID = 952209


# Set path to save the resulting dataframe. Otherwise, set to None.
SAVE_PATH = f'../data/networks/match_{SKILLCORNER_ID}/'
os.makedirs(SAVE_PATH, exist_ok=True)

# Define flag to choose how to save passes_df
USE_PICKLE = False

DATA_PATH = '../data/'
WYSCOUT_PATH = DATA_PATH + 'wyscout/'
SKILLCORNER_PATH = DATA_PATH + 'skillcorner/'
XT_PLOT_PATH = DATA_PATH + 'smoothed_xt.csv'
WYSCOUT_TO_SKILLCORNER = DATA_PATH + 'wyscout2skillcorner.csv'



# Load Wyscout Data
wyscout_data = wyscout_to_df(WYSCOUT_PATH + str(WYSCOUT_ID) + ".json")

# Load SkillCorner Data
metadata = pd.read_csv(SKILLCORNER_PATH + str(SKILLCORNER_ID) + "_metadata.csv")
tracking_df = pd.read_csv(SKILLCORNER_PATH + str(SKILLCORNER_ID) + "_tracking.csv")
lineup_df = pd.read_csv(SKILLCORNER_PATH + str(SKILLCORNER_ID) + "_lineup.csv")
play_direction_df = pd.read_csv(SKILLCORNER_PATH + str(SKILLCORNER_ID) + "_play_direction.csv")



# Filter Pass Events
passes_df = wyscout_data[wyscout_data['type.primary'] == 'pass']

# Filter out inaccurate passes
accurate_only = True
if accurate_only:
    passes_df = passes_df[passes_df['pass.accurate'] == True]

# Filter out passes with invalid recipient id
passes_df = passes_df[passes_df['pass.recipient.id'] != 0]

cols = ['videoTimestamp', 'matchPeriod', 'player.id', 'pass.recipient.id', 'location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y']
passes_df = passes_df.loc[:,cols]
passes_df['matchPeriod'] = passes_df['matchPeriod'].apply(lambda x: int(x.split('H')[0]))
passes_df['videoTimestamp'] = passes_df['videoTimestamp'].astype(float)
passes_df['pass.recipient.id'] = passes_df['pass.recipient.id'].astype(int)
passes_df['pass.endLocation.x'] = passes_df['pass.endLocation.x'].astype(int)
passes_df['pass.endLocation.y'] = passes_df['pass.endLocation.y'].astype(int)


# Map Wyscout IDs to SkillCorner IDs
wyscout2skillcorner = pd.read_csv(WYSCOUT_TO_SKILLCORNER).drop(columns='id')
passes_df.rename(columns={'player.id': 'player.id.wyscout', 'pass.recipient.id': 'pass.recipient.id.wyscout'}, inplace=True)

passes_df = passes_df.merge(wyscout2skillcorner[['player_id_sk', 'player_id_wy', 'team_name_sk']], left_on='player.id.wyscout', right_on='player_id_wy', how='left')
passes_df.drop(columns=['player_id_wy'], inplace=True)
passes_df.rename(columns={'player_id_sk': 'player.id.skillcorner'}, inplace=True)
passes_df.rename(columns={'team_name_sk': 'team.name'}, inplace=True)

passes_df = passes_df.merge(wyscout2skillcorner[['player_id_sk', 'player_id_wy']], left_on='pass.recipient.id.wyscout', right_on='player_id_wy', how='left')
passes_df.drop(columns=['player_id_wy'], inplace=True)
passes_df.rename(columns={'player_id_sk': 'pass.recipient.id.skillcorner'}, inplace=True)

cols = ['videoTimestamp', 'matchPeriod', 'team.name', 'player.id.wyscout', 'player.id.skillcorner', 'pass.recipient.id.wyscout', 'pass.recipient.id.skillcorner', 'location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y']
passes_df = passes_df.loc[:,cols]



# Play Direction
play_direction_dict = play_direction_df.drop(columns='match_id').set_index(['team_name', 'half']).to_dict()['play_direction']
player_team_dict = lineup_df[['player_id', 'team_name']].set_index('player_id').to_dict()['team_name']

def get_play_direction(row):
    team_name = player_team_dict[row['player.id.skillcorner']]
    return play_direction_dict[(team_name, row['matchPeriod'])]

passes_df['play_direction'] = passes_df.apply(get_play_direction, axis=1)



# Compute ΔxT
xt_table = pd.read_csv(XT_PLOT_PATH)

cell_width = 100 / xt_table.shape[1]
cell_height = 100 / xt_table.shape[0]

def get_xt_index(x, y):
    x_index = int(min(x // cell_width, xt_table.shape[1] - 1))
    y_index = int(min(y // cell_height, xt_table.shape[0] - 1))
    return x_index, y_index

def get_xt_value(x, y):
    x_index, y_index = get_xt_index(x, y)
    return xt_table.iat[y_index, x_index]

start_xt = passes_df.apply(lambda row: get_xt_value(row['location.x'], row['location.y']), axis=1)
end_xt = passes_df.apply(lambda row: get_xt_value(row['pass.endLocation.x'], row['pass.endLocation.y']), axis=1)

passes_df['dxt'] = end_xt - start_xt



# Sync skillcorner tracking data with wyscout pass events
framerate = metadata.loc[0,'fps']

def videotime_to_frame(videotime):
    return int(videotime * framerate)

passes_df.insert(1, 'frame', passes_df['videoTimestamp'].apply(videotime_to_frame))
passes_df.drop(columns='videoTimestamp', inplace=True)

tracking_df.rename(columns={'frame_id': 'frame'}, inplace=True)
tracking_df.set_index('frame', inplace=True)
passes_df.set_index('frame', inplace=True)

passes_df = passes_df.join(tracking_df, how='inner', validate='one_to_many')

columns_to_drop = ['match_id', 'half', 'timestamp', 'extrapolated']
existing_columns_to_drop = [col for col in columns_to_drop if col in passes_df.columns]
passes_df.drop(columns=existing_columns_to_drop, inplace=True)

columns_to_prefix = ['object_id', 'x', 'y', 'z']
prefix = 'tracking.'
passes_df.rename(columns={col: prefix + col for col in columns_to_prefix}, inplace=True)



# Normalize Pitch Coordinates
pitch_length = metadata['pitch_length'].values[0]
pitch_width = metadata['pitch_width'].values[0]

start_locations = passes_df.apply(
    lambda row: wyscout_to_pitch(row['location.x'], row['location.y'], pitch_length, pitch_width, row['play_direction']), 
    axis=1)
end_locations = passes_df.apply(
    lambda row: wyscout_to_pitch(row['pass.endLocation.x'], row['pass.endLocation.y'], pitch_length, pitch_width, row['play_direction']), 
    axis=1)

passes_df[['location.x.norm', 'location.y.norm']] = passes_df[['location.x','location.y']]
passes_df[['pass.endLocation.x.norm', 'pass.endLocation.y.norm']] = passes_df[['pass.endLocation.x','pass.endLocation.y']]

passes_df[['location.x', 'location.y']] = start_locations.apply(pd.Series)
passes_df[['pass.endLocation.x', 'pass.endLocation.y']] = end_locations.apply(pd.Series)



# Identify Tracking Objects
def is_self(row):
    return row['player.id.skillcorner'] == row['tracking.object_id']

def is_teammate(row):
    if row['tracking.object_id'] == -1:
        return False
    player_team = player_team_dict[row['player.id.skillcorner']]
    tracking_player_team = player_team_dict[row['tracking.object_id']]
    return player_team == tracking_player_team

def is_opponent(row):
    if row['tracking.object_id'] == -1:
        return False
    return not row['tracking.is_teammate']

passes_df['tracking.is_self'] = passes_df.apply(is_self, axis=1)
passes_df['tracking.is_teammate'] = passes_df.apply(is_teammate, axis=1)
passes_df['tracking.is_opponent'] = passes_df.apply(is_opponent, axis=1)
passes_df['tracking.is_ball'] = passes_df['tracking.object_id'] == -1



# Defender Responsibility
passes_df['responsibility'] = passes_df.apply(responsibility, axis=1)
passes_df['responsibility'] = np.where(passes_df['tracking.is_teammate'], 0, passes_df['responsibility'])



# Expected Threat Gain in case of Interception
passes_df['possible_interception_point'] = passes_df.apply(closest_point, axis = 1)
passes_df['interception_point_x'] = passes_df['possible_interception_point'].apply(lambda point: point[0])
passes_df['interception_point_y'] = passes_df['possible_interception_point'].apply(lambda point: point[1])

passes_df = calculate_interception_xt(passes_df, xt_table)
passes_df['interception_xt'] = np.where(passes_df['tracking.is_teammate'], 0, passes_df['interception_xt'])
passes_df['threat_by_pressing'] = passes_df['responsibility'] * passes_df['interception_xt']

# Organize Columns
col_mask = ['matchPeriod', 'team.name', 'player.id.wyscout',
    'player.id.skillcorner', 'pass.recipient.id.wyscout',
    'pass.recipient.id.skillcorner', 'location.x', 'location.y',
    'pass.endLocation.x', 'pass.endLocation.y', 'location.x.norm', 'location.y.norm', 'pass.endLocation.x.norm',
    'pass.endLocation.y.norm', 'play_direction', 'dxt',
    'tracking.object_id', 'tracking.x', 'tracking.y', 'tracking.z',
    'tracking.is_self', 'tracking.is_teammate',
    'tracking.is_opponent', 'tracking.is_ball', 'responsibility',
    'interception_point_x', 'interception_point_y', 'interception_xt', 'threat_by_pressing'
]
rename_cols = {
    'interception_point_x': 'tracking.interception.x',
    'interception_point_y': 'tracking.interception.y',
    'interception_xt': 'tracking.interception.xt',
}
passes_df[col_mask].rename(columns=rename_cols)



location_mismatch = passes_df[passes_df['tracking.is_self']].apply(lambda row: np.linalg.norm([row['location.x'] - row['tracking.x'], row['location.y'] - row['tracking.y']],), axis=1)
print('Wyscout to Skillcorner MSE Location Mismatch', location_mismatch.mean())

with pd.option_context('display.max_columns', None):
    print(passes_df.sample(5))

if SAVE_PATH:
    if USE_PICKLE:
        passes_df.to_pickle(SAVE_PATH + 'passes_df.pkl')
    else:
        passes_df.to_csv(SAVE_PATH + 'passes_df.csv')
    print("Successfully saved")