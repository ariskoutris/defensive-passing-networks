import pandas as pd

from resp import responsibility
from utils import calculate_potential_dxt


# calculate all options for passer in a pass
def attacker_options(passes_df, frame, pitch_dict):
    """
    passes_df: DataFrame with pass information
    frame: frame number, pass to consider
    pitch_dict: dictionary with pitch dimensions: pitch_length, pitch_width, xt_table, cell_width, cell_height
    """
    # TODO: prepare with pitch_dict, outside the function
    # pitch_length = 105
    # pitch_width = 68
    # xt_rows, xt_cols = 68, 105
    # cell_width = pitch_length / xt_cols
    # cell_height = pitch_width / xt_rows
    
    # filter frame/event => get specific pass
    data = passes_df[passes_df['frame'] == frame]

    # apply function to generate potential changes in xt, considering all options
    data['potential_dxt'] = data.apply(calculate_potential_dxt, axis=1, args=(pitch_dict,))

    # Remove unnecessary columns
    passes_table = data[['frame', 'player.id.skillcorner', 'location.x', 'location.y', 'play_direction', 'tracking.object_id', 'tracking.x', 'tracking.y', 'tracking.is_teammate', 'tracking.is_self', 'potential_dxt']]


    # for each pass and pass option, locate every defender - expand the dataframe
    def generate_full_defender_dataset(data):
        full_rows = []
        
        player_positions = data[(data['tracking.is_teammate']) | (data['tracking.is_self'])]
            
        # Identify defenders in the frame
        defender_positions = data[~data['tracking.is_teammate'] & ~data['tracking.is_self']]
        
        # Ensure there are 11 defenders (limit to 11 if more)
        defender_positions = defender_positions.head(11)
        
        # Generate rows: for each player position, associate all 11 defenders
        for _, player in player_positions.iterrows():
            for _, defender in defender_positions.iterrows():
                new_row = player.copy()
                new_row['defender_tracking.x'] = defender['tracking.x']
                new_row['defender_tracking.y'] = defender['tracking.y']
                full_rows.append(new_row)
        
        # Convert the list of rows into a DataFrame
        full_dataset = pd.DataFrame(full_rows)
        
        return full_dataset
    
    # evaluate each defender responsibility for each pass and each attacker option
    passes_table = generate_full_defender_dataset(passes_table)

    # remove the passer from the options
    passes_table = passes_table[~(passes_table['tracking.is_teammate'] & passes_table['tracking.is_self'])]

    # rename columns to match responsibility function input
    passes_table = passes_table.rename(columns={
        'tracking.x': 'pass.endLocation.x', # potantial pass receiver location
        'tracking.y': 'pass.endLocation.y',
        'defender_tracking.x': 'tracking.x', # defender location
        'defender_tracking.y': 'tracking.y'})

    # apply responsibility function
    passes_table['responsibility'] = passes_table.apply(responsibility, axis=1)

    # calculate expected threat value for each potantial pass
    expected_threat = dict()

    # iterate through passes
    for obj in passes_table['tracking.object_id'].unique():
        id = passes_table.iloc[0]['player.id.skillcorner']
        dxt = passes_table[passes_table['tracking.object_id'] == obj].iloc[0]['potential_dxt']
        x_loc = passes_table[passes_table['tracking.object_id'] == obj].iloc[0]['pass.endLocation.x']
        y_loc = passes_table[passes_table['tracking.object_id'] == obj].iloc[0]['pass.endLocation.y']
        # for all defender, responsibility effect on xt: xt := xt * (1-resp_1) * (1-resp_2) ...
        for idx, row in passes_table[passes_table['tracking.object_id'] == obj].iterrows():
            dxt = dxt * (1 - row['responsibility'])
        expected_threat[(int(id), float(x_loc), float(y_loc), int(obj))] = float(dxt)
            

    # create a dataframe for attacker options
    data = [
        {'passer_id': id, 'recipient_player_id': teammate_id, 'recipient_loc_x': loc_x, 'recipient_loc_y': loc_y, 'expected_dxt': max_value}
        for (id, loc_x, loc_y, teammate_id), max_value in expected_threat.items()]

    # Create DataFrame
    attacker_options_dataframe = pd.DataFrame(data)

    
    return attacker_options_dataframe
