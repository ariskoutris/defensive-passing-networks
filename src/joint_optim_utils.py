import numpy as np
import matplotlib.pyplot as plt

import optim_utils

def extract_defender_positions(defenders_info):
    init_positions = []
    for v in defenders_info.values():
        x,y = v['location']
        init_positions.append(x)
        init_positions.append(y)
    init_positions = np.array(init_positions)
    return init_positions

def update_defenders_info(defenders_info, new_positions):
    updated_dict = {}
    for i, defender_id in zip(range(0, len(new_positions), 2), defenders_info.keys()):
        x = new_positions[i]
        y = new_positions[i+1]
        updated_dict[defender_id] = {'location': (x, y)}
    return updated_dict

def create_joint_xt_func_vectorized(passes_df, frame_id, threat_map):
    passer_location, recipients_info, defenders_info, play_direction = optim_utils.retrieve_player_positions(passes_df, frame_id)
    
    for id, recipient in recipients_info.items():
        recipient_location = recipient['location']
        threat = threat_map.get_dxt(
            *passer_location,
            *recipient_location,
            play_direction
        )
        recipients_info[id]['threat_of_pass'] = threat

    def xt_func(defender_locations):
        defenders_info_ = update_defenders_info(defenders_info, defender_locations)
        return optim_utils.passers_expected_threat_vectorized(
            passer_location, recipients_info, defenders_info_
        )

    return xt_func

def generate_bounds(init_positions, radius):
    bounds = []
    for pos in init_positions:
        bounds.append((pos - radius, pos + radius))
    return bounds

def joint_optimization_report(opt_func, x_init, x_opt):
    initial_score = opt_func(x_init)
    optimal_score = opt_func(x_opt,)
    improvement = optimal_score - initial_score
    
    distances = []
    for i in range(0,len(x_init)-1,2):
        x1,y1 = x_init[i], x_init[i+1]
        x2,y2 = x_opt[i], x_opt[i+1]
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)    
        distances.append(dist)
    distance = np.mean(distances)    
    
    results = {
        'init_val': initial_score,
        'opt_val': optimal_score,
        'improvement': improvement,
        'improvement_perc': 100 * improvement / abs(initial_score),
        'avg_distance': distance
    }
    
    return results

def visualize_positions_optimization(attackers, ball_owner, defenders_init, defenders_opt):
    fig, ax = plt.subplots(figsize=(12, 8))

    for attacker in attackers.values():
        x, y = attacker['location']
        ax.scatter(x, y, color='blue', marker='o', label='Attacker' if 'Attacker' not in ax.get_legend_handles_labels()[1] else "")

    x, y = ball_owner['location']
    ax.scatter(x, y, color='green', marker='*', s=200, label='Ball Owner')

    for idx, defender in defenders_init.items():
        x, y = defender['location']
        ax.scatter(x, y, color='red', marker='s', label='Defender Initial' if 'Defender Initial' not in ax.get_legend_handles_labels()[1] else "")

    for idx, defender in defenders_opt.items():
        x, y = defender['location']
        ax.scatter(x, y, color='orange', marker='^', label='Defender Optimized' if 'Defender Optimized' not in ax.get_legend_handles_labels()[1] else "")

    for defender_id in defenders_init.keys():
        x_init, y_init = defenders_init[defender_id]['location']
        x_opt, y_opt = defenders_opt[defender_id]['location']
        ax.arrow(x_init, y_init, x_opt - x_init, y_opt - y_init, 
             head_width=0.5, head_length=0.7, fc='k', ec='k', 
             length_includes_head=True)

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Defender Positions Optimization')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

    plt.show()