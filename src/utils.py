import numpy as np
import os
import zipfile
import json
import pandas as pd
from datetime import datetime, timedelta

def round_to_tenth_of_second(timestamp):
    dt = datetime.strptime(timestamp, '%H:%M:%S.%f')
    microsecond = dt.microsecond
    rounded_microsecond = round(microsecond, -5)
    if rounded_microsecond == 1000000:
        dt += timedelta(seconds=1)
        rounded_microsecond = 0
    dt = dt.replace(microsecond=rounded_microsecond)
    return f"{dt.strftime('%H:%M:%S')}.{rounded_microsecond:06}"

def standardize_timestamp(row):
    ts = row['timestamp']
    if row['half'] == 2:
        ts += 45 * 60 * 1000
    td = timedelta(milliseconds=ts)
    total_seconds = int(td.total_seconds())
    microseconds = int(td.microseconds)
    return f"{total_seconds // 3600:02}:{(total_seconds % 3600) // 60:02}:{total_seconds % 60:02}.{microseconds:06}"

def wyscout_to_df(filepath):
    with open(filepath, encoding='utf8') as f:
        js = json.load(f)
    df = pd.json_normalize(js['events'])
    return df

def unzip_all_files_in_dir(directory, target_directory):
    for file in os.listdir(directory):
        if file.endswith('.zip'):
            with zipfile.ZipFile(directory + file, 'r') as zip_ref:
                zip_ref.extractall(target_directory)
                


def wyscout_to_pitch(x, y, pitch_length, pitch_width, direction):
    """
    Converts Wyscout coordinates (x, y) to pitch coordinates for an upward-playing direction.

    Parameters:
    - x: Wyscout x-coordinate (0 to 100)
    - y: Wyscout y-coordinate (0 to 100)
    - pitch_length: Length of the pitch in meters (default: 105 meters)
    - pitch_width: Width of the pitch in meters (default: 68 meters)

    Returns:
    - (new_x, new_y): Transformed pitch coordinates
    """
    # Transform y-coordinate based on Wyscout ranges
    if y <= 19:
        new_y = ((pitch_width / 2 - 20.16) * y / 19) - (pitch_width / 2)
    elif 19 < y <= 37:
        new_y = 11 * (y - 19) / 18 - 20.16
    elif 37 < y < 63:
        new_y = 9.16 * (y - 50) / 13
    elif 63 <= y < 81:
        new_y = 11 * (y - 63) / 18 + 9.16
    else:  # y >= 81
        new_y = ((pitch_width / 2 - 20.16) * (y - 81) / 19) + 20.16

    # Transform x-coordinate based on Wyscout ranges
    if x <= 6:
        new_x = 5.5 * x / 6
    elif 6 < x <= 10:
        new_x = 5.5 * (x - 6) / 4 + 5.5
    elif 10 < x <= 16:
        new_x = 5.5 * (x - 10) / 6 + 11
    elif 16 < x < 84:
        new_x = ((pitch_length - 33) * (x - 16) / 68) + 16.5
    elif 84 <= x < 90:
        new_x = 5.5 * (x - 90) / 6 + pitch_length - 11
    elif 90 <= x < 94:
        new_x = 5.5 * (x - 94) / 4 + pitch_length - 5.5
    else:  # x >= 94
        new_x = pitch_length - 5.5 * (100 - x) / 6

    if direction == 'TOP_TO_BOTTOM':
        new_x = pitch_length / 2 - new_x
        new_y = new_y
    elif direction == 'BOTTOM_TO_TOP':
        new_x = new_x - pitch_length / 2
        new_y = -new_y
        
    return new_x, new_y

def passes_avg_coord(passes_df, pitch_length, pitch_width, SAVE_PATH=None):
    """
    Takes passes_df with wyscout info
    Returns average coordinates for the players where they pass
    """
    passes_df_coord = passes_df.groupby(['player.id.skillcorner']).agg(
                    location_x_avg=('location.x', 'mean'),
                    location_y_avg=('location.y', 'mean'),
                    team_name=('team.name', 'first')
                    ).reset_index()
    passes_df_coord_avg = \
    passes_df_coord.apply(\
    lambda row: wyscout_to_pitch(row['location_x_avg'], row['location_y_avg'], pitch_length, pitch_width, 'TOP_TO_BOTTOM' \
                                 if row['team.name']=='France' else 'BOTTOM_TO_TOP'), axis=1)

    passes_df_coord['location_x_avg'] = passes_df_coord_avg.apply(lambda x: x[0])
    passes_df_coord['location_y_avg'] = passes_df_coord_avg.apply(lambda x: x[1])
    if SAVE_PATH:
        passes_df_coord.to_csv(SAVE_PATH + 'passes_df_coord.csv', index=False)
    
    return passes_df_coord



# adjust coordinates for correct dxt calculation
def adjust_coordinates(x, y, direction, pitch_dict):
    pitch_length = pitch_dict['pitch_length']
    pitch_width = pitch_dict['pitch_width']

    if direction == 'TOP_TO_BOTTOM':
        # get transpose of coordinates
        x = - x + pitch_length/2
        y = -y + pitch_width/2

    elif direction == 'BOTTOM_TO_TOP':
        x = x + pitch_length/2
        y = y + pitch_width/2
    # consider out of the pitch locations and map them to the edges of the pitch
    x = max(min(x, pitch_length), 0)
    y = max(min(y, pitch_width), 0)

    return x, y

def get_xt_index(x, y, pitch_dict):
    xt_table = pitch_dict['xt_table']
    cell_width = pitch_dict['cell_width']
    cell_height = pitch_dict['cell_height']

    # map locations to xt table
    x_index = int(min(x // cell_width, xt_table.shape[1] - 1))
    y_index = int(min(y // cell_height, xt_table.shape[0] - 1))
    return x_index, y_index

# Get XT value for a given location
def get_xt_value(x, y, direction, pitch_dict):
    xt_table = pitch_dict['xt_table']
    
    adjusted_x, adjusted_y = adjust_coordinates(x, y, direction, pitch_dict)
    x_index, y_index = get_xt_index(adjusted_x, adjusted_y, pitch_dict)
    return xt_table.iat[y_index, x_index]

# Calculate potantial dxt given the player passes to certain teammate
def calculate_potential_dxt(row, pitch_dict):
    if row['tracking.is_teammate'] and not row['tracking.is_self']:
        start_xt = get_xt_value(row['location.x'], row['location.y'], row['play_direction'], pitch_dict)
        end_xt = get_xt_value(row['tracking.x'], row['tracking.y'], row['play_direction'], pitch_dict)
        return end_xt - start_xt
    return 0

# determine the expected threat of a potential interception
def calculate_interception_xt(joined_df, xt_table, pitch_length=105, pitch_width=68, xt_rows=68, xt_cols=105):

    # Calculate cell dimensions for the XT grid
    cell_width = pitch_length / xt_cols
    cell_height = pitch_width / xt_rows

    def adjust_coordinates(x, y, direction):
        """
        Adjust coordinates based on play direction.
        """
        if direction == 'BOTTOM_TO_TOP':  # Consider opponent team's perspective
            x = -x + pitch_length / 2
            y = -y + pitch_width / 2
        elif direction == 'TOP_TO_BOTTOM':
            x = x + pitch_length / 2
            y = y + pitch_width / 2
        x = max(min(x, pitch_length), 0)
        y = max(min(y, pitch_width), 0)
        return x, y

    def get_xt_index(x, y):
        """
        Get the XT table index based on adjusted coordinates.
        """
        x_index = int(min(x // cell_width, xt_table.shape[1] - 1))
        y_index = int(min(y // cell_height, xt_table.shape[0] - 1))
        return x_index, y_index

    def get_xt_value(x, y, direction):
        """
        Get XT value for a given location and play direction.
        """
        adjusted_x, adjusted_y = adjust_coordinates(x, y, direction)
        x_index, y_index = get_xt_index(adjusted_x, adjusted_y)
        return xt_table.iat[y_index, x_index]

    # Apply the XT value calculation to each row in the DataFrame
    joined_df['interception_xt'] = joined_df.apply(
        lambda row: get_xt_value(row['interception_point_x'], row['interception_point_y'], row['play_direction']),
        axis=1
    )

    return joined_df

# find the closest point of a defender to pass trajectory
def closest_point(row, ball_speed=12.0, defender_speed=6.0):
    start_x = row['location.x']
    start_y = row['location.y']
    end_x = row['pass.endLocation.x']
    end_y = row['pass.endLocation.y']
    player_x = row['tracking.x']
    player_y = row['tracking.y']
    
    # Vector from start to end of the pass
    pass_vector = np.array([end_x - start_x, end_y - start_y])
    pass_length = np.linalg.norm(pass_vector)
    
    if pass_length == 0:
        return 0  # No pass, no responsibility
    
    # Unit vector along the pass trajectory
    pass_unit_vector = pass_vector / pass_length

    # Time for the ball to travel the length of the pass
    ball_time = pass_length / ball_speed

    # Maximum distance the defender can travel in the same time
    max_defender_distance = defender_speed * ball_time

    # Width of the triangle at the far end (cone edge)
    max_width = 2 * max_defender_distance

    # Find the projection of the player onto the pass vector
    player_vector = np.array([player_x - start_x, player_y - start_y])
    projection_length = np.dot(player_vector, pass_unit_vector)
    
    # Clamp the projection length to the range [0, pass_length]
    projection_length = max(0, min(projection_length, pass_length))

    # Find the closest point on the pass trajectory
    closest_point = np.array([start_x, start_y]) + projection_length * pass_unit_vector

    

    return closest_point
