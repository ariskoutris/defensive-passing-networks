import numpy as np
import pandas as pd

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

    # Unit vector along the pass trajectory
    pass_unit_vector = pass_vector / pass_length

    # Perpendicular vector to the pass trajectory
    # perp_vector = np.array([-pass_unit_vector[1], pass_unit_vector[0]])

    # Vector from start of pass to defender's position
    player_vector = np.array([player_x - start_x, player_y - start_y])

    # Projection of the defender onto the pass trajectory
    projection_length = np.dot(player_vector, pass_unit_vector)

    # Clamp projection length to [0, pass_length] to ensure it stays within the triangle
    projection_length = max(0, min(projection_length, pass_length))

    # Perpendicular distance from defender to the pass trajectory
    perpendicular_distance = np.linalg.norm(player_vector - (projection_length * pass_unit_vector))

    # Calculate the triangle width at the projection point
    triangle_width_at_point = 2 * defender_speed * (projection_length / ball_speed)

    # Half-width of the triangle
    half_width = triangle_width_at_point / 2

    # Determine if the defender is inside the triangle
    if perpendicular_distance <= half_width and projection_length <= pass_length and row['tracking.object_id'] != -1:
        # Responsibility is based on the perpendicular distance ratio
        responsibility_score = 1 - (perpendicular_distance / half_width)
    else:
        # Defender is outside the triangle, no responsibility
        responsibility_score = 0

    return responsibility_score