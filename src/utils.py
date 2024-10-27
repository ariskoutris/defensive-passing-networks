# Re-defining the function since the environment was reset

def wyscout_to_pitch(x, y, pitch_length=105, pitch_width=68):
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

    return new_x, new_y



# test_points = {
#     "Top Left Corner": (0, 0),
#     "Bottom Left Corner": (0, 100),
#     "Top Right Corner": (100, 0),
#     "Bottom Right Corner": (100, 100),
#     "Center Circle": (50, 50),
#     "Left Penalty Spot": (10, 50),
#     "Right Penalty Spot": (90, 50),
#     "Left Penalty Area Top": (6, 37),
#     "Left Penalty Area Bottom": (6, 63),
#     "Right Penalty Area Top": (94, 37),
#     "Right Penalty Area Bottom": (94, 63)
# }



# import pandas as pd
# converted_points = {name: wyscout_to_pitch(x, y) for name, (x, y) in test_points.items()}

# converted_points_df = pd.DataFrame(converted_points).T
# import ace_tools as tools; tools.display_dataframe_to_user(name="Converted Wyscout Points", dataframe=converted_points_df)

# converted_points_df
