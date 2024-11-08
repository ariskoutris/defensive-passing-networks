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
def responsibility(row, max_distance=10, pass_length_factor=1.0, end_location_factor=0.5, start_location_factor=0.5):
    
    start_x = row['location.x']
    start_y = row['location.y']
    end_x = row['pass.endLocation.x']
    end_y = row['pass.endLocation.y']
    
    player_x = row['tracking.x']
    player_y = row['tracking.y']
    
    # Vector from start to end of the pass
    pass_vector = np.array([end_x - start_x, end_y - start_y])
    # Vector from start of the pass to the player's position
    player_vector = np.array([player_x - start_x, player_y - start_y])
    
    pass_length = np.linalg.norm(pass_vector)
    
    
    if pass_length > 0:
        pass_unit_vector = pass_vector / pass_length
        projection_length = np.dot(player_vector, pass_unit_vector)
        
        # Clamp projection_length to the range [0, pass_length] to account for the endpoints of the pass
        projection_length = max(0, min(projection_length, pass_length))
        
        # Find the point on the pass closest to the player
        closest_point = np.array([start_x, start_y]) + projection_length * pass_unit_vector
        
        # Distance from player to the closest point on the pass
        distance_to_pass = np.linalg.norm(np.array([player_x, player_y]) - closest_point)
    else:
        # If the pass length is zero, set distance to the player's distance from the start point
        distance_to_pass = np.linalg.norm(np.array([player_x, player_y]) - np.array([start_x, start_y]))
    
    # Calculate distance from the player to the end location of the pass
    distance_to_end_location = np.linalg.norm(np.array([player_x, player_y]) - np.array([end_x, end_y]))
    distance_to_start_location = np.linalg.norm(np.array([player_x, player_y]) - np.array([start_x, start_y]))

    
    # Calculate responsibility with pass length and distance to pass scaling
    if distance_to_pass < max_distance and row['tracking.object_id'] != -1:
        raw_responsibility = (1 - (distance_to_pass / max_distance)) * np.power(1 + pass_length_factor, pass_length)
        
        # Normalize the responsibility to be between 0 and 1
        max_possible_responsibility = np.power(1 + pass_length_factor, pass_length)
        responsibility_score = raw_responsibility / max_possible_responsibility
        
        start_location_adjustment = max(0, 1 - (start_location_factor * (distance_to_start_location / max_distance)))
        
        normalized_adjustment = 1 + (start_location_adjustment - 1) * start_location_factor
        
        # Apply normalized adjustment
        responsibility_score *= normalized_adjustment

    else:
        # If the player is too far, they get no responsibility
        responsibility_score = 0
        
    
        
    return responsibility_score

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