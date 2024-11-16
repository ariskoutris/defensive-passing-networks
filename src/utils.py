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
def responsibility(row, ball_speed=12.0, defender_speed=6.0):
    start_x = row['location.x']
    start_y = row['location.y']
    end_x = row['pass.endLocation.x']
    end_y = row['pass.endLocation.y']
    player_x = row['tracking.x']
    player_y = row['tracking.y']

    # Pass vector and length
    pass_vector = np.array([end_x - start_x, end_y - start_y])
    pass_length = np.linalg.norm(pass_vector)

    if pass_length == 0:
        return 0  # No pass, no responsibility

    # Ball and defender travel distances
    ball_time = pass_length / ball_speed
    max_defender_distance = defender_speed * ball_time

    # Unit vector along the pass trajectory
    pass_unit_vector = pass_vector / pass_length

    # Perpendicular vector to the pass trajectory
    perp_vector = np.array([-pass_unit_vector[1], pass_unit_vector[0]])

    # Vector from start of pass to defender's position
    player_vector = np.array([player_x - start_x, player_y - start_y])

    # Projection of the defender onto the pass trajectory
    projection_length = np.dot(player_vector, pass_unit_vector)

    # Clamp projection length to [0, pass_length] to ensure it stays within the triangle
    projection_length = max(0, min(projection_length, pass_length))

    # Closest point on the pass trajectory
    closest_point = np.array([start_x, start_y]) + projection_length * pass_unit_vector

    # Perpendicular distance from defender to the pass trajectory
    perpendicular_distance = np.linalg.norm(player_vector - (projection_length * pass_unit_vector))

    # Calculate the triangle width at the projection point
    triangle_width_at_point = 2 * defender_speed * (projection_length / ball_speed)

    # Half-width of the triangle
    half_width = triangle_width_at_point / 2

    # Determine if the defender is inside the triangle
    if perpendicular_distance <= half_width and projection_length <= pass_length and row['tracking.object_id'] != 1:
        # Responsibility is based on the perpendicular distance ratio
        responsibility_score = 1 - (perpendicular_distance / half_width)
    else:
        # Defender is outside the triangle, no responsibility
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