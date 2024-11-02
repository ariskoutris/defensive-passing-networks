import pandas as pd
from utils import *
   
    
# Set Wyscout and Skillcorner IDs
WYSCOUT_ID = 5414111
SKILLCORNER_ID = 952209


# Set path to save the resulting dataframe. Otherwise, set to None.
SAVE_PATH = '../data/'

DATA_PATH = '../data/'
WYSCOUT_PATH = DATA_PATH + 'wyscout/'
SKILLCORNER_PATH = DATA_PATH + 'skillcorner/'
XT_PLOT_PATH = DATA_PATH + 'xt_plot_worldcup.csv'
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

cols = ['matchTimestamp', 'matchPeriod', 'player.id', 'pass.recipient.id', 'location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y']
passes_df = passes_df.loc[:,cols]
passes_df['matchPeriod'] = passes_df['matchPeriod'].apply(lambda x: int(x.split('H')[0]))
passes_df['pass.recipient.id'] = passes_df['pass.recipient.id'].astype(int)
passes_df['pass.endLocation.x'] = passes_df['pass.endLocation.x'].astype(int)
passes_df['pass.endLocation.y'] = passes_df['pass.endLocation.y'].astype(int)



# Map Wyscout IDs to SkillCorner IDs
wyscout2skillcorner = pd.read_csv(WYSCOUT_TO_SKILLCORNER).drop(columns='id')
passes_df.rename(columns={'player.id': 'player.id.wyscout', 'pass.recipient.id': 'pass.recipient.id.wyscout'}, inplace=True)

passes_df = passes_df.merge(wyscout2skillcorner[['player_id_sk', 'player_id_wy']], left_on='player.id.wyscout', right_on='player_id_wy', how='left')
passes_df.drop(columns=['player_id_wy'], inplace=True)
passes_df.rename(columns={'player_id_sk': 'player.id.skillcorner'}, inplace=True)

passes_df = passes_df.merge(wyscout2skillcorner[['player_id_sk', 'player_id_wy']], left_on='pass.recipient.id.wyscout', right_on='player_id_wy', how='left')
passes_df.drop(columns=['player_id_wy'], inplace=True)
passes_df.rename(columns={'player_id_sk': 'pass.recipient.id.skillcorner'}, inplace=True)

cols = ['matchTimestamp', 'matchPeriod', 'player.id.wyscout', 'player.id.skillcorner', 'pass.recipient.id.wyscout', 'pass.recipient.id.skillcorner', 'location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y']
passes_df = passes_df.loc[:,cols]



# Play Direction
play_direction_dict = play_direction_df.drop(columns='match_id').set_index(['team_name', 'half']).to_dict()['play_direction']
player_team_dict = lineup_df[['player_id', 'team_name']].set_index('player_id').to_dict()['team_name']

def get_play_direction(row):
    team_name = player_team_dict[row['player.id.skillcorner']]
    return play_direction_dict[(team_name, row['matchPeriod'])]


passes_df['play_direction'] = passes_df.apply(get_play_direction, axis=1)


# Match skillcorner tracking data with wyscout pass event Data
tracking_df.drop(columns=['frame_id', 'extrapolated'], inplace=True)   

tracking_df['timestamp'] = tracking_df.apply(standardize_timestamp, axis=1)
tracking_df.set_index(['timestamp', 'half'], inplace=True)
tracking_df.rename_axis(index={'timestamp': 'timestamp', 'half': 'period'}, inplace=True)

passes_df['quantizedTimestamp'] = passes_df['matchTimestamp'].apply(round_to_tenth_of_second)
passes_df.set_index(['quantizedTimestamp', 'matchPeriod'], inplace=True)
passes_df.rename_axis(index={'quantizedTimestamp': 'timestamp', 'matchPeriod': 'period'}, inplace=True)

passes_df = passes_df.join(tracking_df, how='inner', validate='one_to_many')
passes_df.drop(columns=['match_id', 'matchTimestamp'], inplace=True)
passes_df['object_id'] = passes_df['object_id'].astype(int)

columns_to_prefix = ['object_id', 'x', 'y', 'z']
prefix = 'tracking.'
passes_df.rename(columns={col: prefix + col for col in columns_to_prefix}, inplace=True)

passes_df.reset_index(inplace=True)



# Normalize Pitch Coordinates
pitch_length = metadata['pitch_length'].values[0]
pitch_width = metadata['pitch_width'].values[0]

start_locations = passes_df.apply(
    lambda row: wyscout_to_pitch(row['location.x'], row['location.y'], pitch_length, pitch_width), 
    axis=1)
end_locations = passes_df.apply(
    lambda row: wyscout_to_pitch(row['pass.endLocation.x'], row['pass.endLocation.y'], pitch_length, pitch_width), 
    axis=1)

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


passes_df['is_self'] = passes_df.apply(is_self, axis=1)
passes_df['is_teammate'] = passes_df.apply(is_teammate, axis=1)



# Compute ΔxT
# TODO: Compute ΔxT using denormalized pitch coordinates.
# TODO: Compute ΔxT taking into account team's goal direction.
xt_table = pd.read_csv(XT_PLOT_PATH)

cell_width = 100 / xt_table.shape[1]
cell_height = 100 / xt_table.shape[0]

def get_xt_index(x, y):
    x_index = min(int(x // cell_width), xt_table.shape[1] - 1)
    y_index = min(int(y // cell_height), xt_table.shape[0] - 1)
    return x_index, y_index

start_xts = passes_df.apply(lambda row: xt_table.iat[get_xt_index(row['location.x'], row['location.y'])[1], 
                                                           get_xt_index(row['location.x'], row['location.y'])[0]], axis=1)
end_xts = passes_df.apply(lambda row: xt_table.iat[get_xt_index(row['pass.endLocation.x'], row['pass.endLocation.y'])[1], 
                                                         get_xt_index(row['pass.endLocation.x'], row['pass.endLocation.y'])[0]], axis=1)
passes_df.loc[:,'dxt'] = end_xts - start_xts


if SAVE_PATH:
    passes_df.to_pickle(SAVE_PATH + 'passes_df.pkl')

with pd.option_context('display.max_columns', None):
    print(passes_df.head())
