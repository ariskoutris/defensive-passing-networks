import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plot_utils


class ThreatMap:
    def __init__(self, threat_map, pitch_length, pitch_width) -> None:
        self.threat_map = threat_map
        self.pitch_length = pitch_length
        self.pitch_width = pitch_length
        self.cell_width = threat_map.shape[1] // pitch_length
        self.cell_length = threat_map.shape[0] // pitch_width
        self.direction = 'TOP_TO_BOTTOM'
    
    def adjust_coordinates(self, x, y, direction):
        if direction == 'TOP_TO_BOTTOM':
            x = - x + self.pitch_length/2
            y = -y + self.pitch_width/2
        elif direction == 'BOTTOM_TO_TOP':
            x = x + self.pitch_length/2
            y = y + self.pitch_width/2
        return x, y

    def get_xt_index(self, x, y, direction):
        x_adj, y_adj = self.adjust_coordinates(x, y, direction)
        x_index = int(min(x_adj // self.cell_width, self.threat_map.shape[1] - 1))
        y_index = int(min(y_adj // self.cell_length, self.threat_map.shape[0] - 1))
        return x_index, y_index

    def get_xt_value(self, x, y, direction):
        x_index, y_index = self.get_xt_index(x, y, direction)
        return self.threat_map.iat[y_index, x_index]

    def get_dxt(self, x_start, y_start, x_end, y_end, direction):
        start_xt = self.get_xt_value(x_start, y_start, direction)
        end_xt = self.get_xt_value(x_end, y_end, direction)
        return end_xt - start_xt


def all_negative_dxt_frames(passes_df, threat_map):
    
    def all_neg(scores):
        for v in scores.values():
            if v > 0:
                return False
        return True

    all_negative_frame = []
    pass_frames = passes_df['frame'].unique()
    
    for frame in pass_frames:
        opt_func = create_xt_func(-1, passes_df, frame, threat_map)
        pass_plotter = plot_utils.PassPlotter(passes_df, frame, threat_map.pitch_length, threat_map.pitch_length)
        defender_location = pass_plotter.get_player_location(-1)
        x_origin, y_origin = defender_location    
        true_scores = opt_func(x_origin, y_origin)

        if all_neg(true_scores):
            all_negative_frame.append(frame)
            
    return all_negative_frame

def responsibility(start_x, start_y, end_x, end_y, player_x, player_y, ball_speed=12.0, defender_speed=6.0):

    pass_vector = np.array([end_x - start_x, end_y - start_y])
    pass_length = np.linalg.norm(pass_vector)

    if pass_length == 0:
        return 0
    
    pass_unit_vector = pass_vector / pass_length
    player_vector = np.array([player_x - start_x, player_y - start_y])

    projection_length = np.dot(player_vector, pass_unit_vector)
    projection_length = max(0, min(projection_length, pass_length))

    perpendicular_distance = np.linalg.norm(player_vector - (projection_length * pass_unit_vector))
    triangle_width_at_point = 2 * defender_speed * (projection_length / ball_speed)

    half_width = triangle_width_at_point / 2
    if perpendicular_distance <= half_width and projection_length <= pass_length:
        responsibility_score = 1 - (perpendicular_distance / half_width)
    else:
        responsibility_score = 0

    return responsibility_score

def passers_expected_threat(defender_x, defender_y, defender_id, passer_loc, recipients, defenders):
    
    defenders.loc[defenders['tracking.object_id'] == defender_id, ['tracking.x', 'tracking.y']] = [defender_x, defender_y]

    expected_threat_dict = {}
    for _, recipient in recipients.iterrows():
        threat_of_pass = recipient['threat_of_pass']
        prob_success = 1
        for _, defender in defenders.iterrows(): 
            resp_value = responsibility(*passer_loc, recipient['tracking.x'], recipient['tracking.y'], defender['tracking.x'], defender['tracking.y'])
            prob_success *= (1 - resp_value)
        expected_threat_dict[recipient['tracking.object_id']] = threat_of_pass * prob_success
     
    return expected_threat_dict

def create_xt_func(defender_id, data, frame_id, threat_map):
    data = data[data['frame'] == frame_id]
    
    recipients = data[data['tracking.is_teammate'] & ~data['tracking.is_self']].copy()
    defenders = data[data['tracking.is_opponent']].copy()
    
    passer_id = data['player.id.skillcorner'].values[0]
    passer_location = data[data['tracking.object_id'] == passer_id][['tracking.x', 'tracking.y']].values[0]
    
    play_direction =  data['play_direction'].values[0]
    
    threats = []
    for _, recipient in recipients.iterrows():
        recipient_location = recipient['tracking.x'], recipient['tracking.y']
        threats.append(threat_map.get_dxt(*passer_location, *recipient_location, play_direction))
    recipients['threat_of_pass'] = threats
    
    def xt_func(x,y):
        return passers_expected_threat(x, y, defender_id, passer_location, recipients, defenders)
    
    return xt_func

def threat_aggregator(mode='softmax', k=3, temp=1, only_positive=False):
    if mode == 'max':
        return lambda x: max(x.values())
    elif mode == 'softmax':
        def softmax(x):
            x = np.array(list(x.values()))
            exp_x = np.exp(x/temp)
            softmax_scores = exp_x / np.sum(exp_x) 
            return sum(softmax_scores * x)
        return softmax
    elif mode == 'topK':
        return lambda x: sum(sorted(x.values(), reverse=True)[:k])
    elif mode == 'mean':
        return lambda x: sum(x.values()) / len(x)
    elif mode == 'sum':
        return lambda x: sum(x.values())
    elif mode == 'maxKey':
        return lambda x: max(x, key=x.get)
    else:
        raise ValueError('Invalid mode')
    
def compute_function_grid(func, x0, y0, radius, bound_x, bound_y, grid_res):
    x = np.linspace(max(x0 - radius, -bound_x/2), min(x0 + radius, bound_x/2), grid_res)
    y = np.linspace(max(y0 - radius, -bound_y/2), min(y0 + radius, bound_y/2), grid_res)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(X[i,j],Y[i,j])
    return X,Y,Z
       
def optimize_func(opt_func, x0, y0, radius, bound_x, bound_y, grid_res):
    
    X, Y, Z = compute_function_grid(opt_func, x0, y0, radius, bound_x, bound_y, grid_res)
    
    Z_opt = opt_func(x0, y0)
    x_opt, y_opt = x0, y0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            distance = np.sqrt((X[i, j] - x0)**2 + (Y[i, j] - y0)**2)
            if distance <= radius and Z[i, j] > Z_opt:
                Z_opt = Z[i, j]
                x_opt, y_opt = X[i, j], Y[i, j]
    
    return Z_opt, x_opt, y_opt

def visualize_func(func, x0, y0, radius, bound_x, bound_y, grid_res, contour_levels=20):
    
    X, Y, Z = compute_function_grid(func, x0, y0, radius, bound_x, bound_y, grid_res)
    
    _, ax = plt.subplots(figsize=(10, 6))
    cont = ax.contourf(X, Y, Z, levels=contour_levels)
    cbar = plt.colorbar(cont)
    cbar.set_label('Expected Threat')
    
    return ax

def visualize_best_pass(func, x0, y0, radius, bound_x, bound_y, grid_res=30, return_recipients=False):
    
    X, Y, Z = compute_function_grid(func, x0, y0, radius, bound_x, bound_y, grid_res)
            
    unique_labels = np.unique(Z)
    num_labels = len(unique_labels)
    label_to_class = {label: idx for idx, label in enumerate(unique_labels)}
    class_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    P_class = np.vectorize(label_to_class.get)(Z)
    cmap = plt.cm.get_cmap('viridis', num_labels)
    
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(X, Y, P_class, cmap=cmap, shading='auto')
    cbar = plt.colorbar(mesh, ticks=np.arange(num_labels))
    cbar.ax.set_yticklabels([str(class_to_label[i]) for i in range(num_labels)])
    cbar.set_label('Pass Receiver')
    ax = plt.gca()

    if return_recipients:
        return ax, unique_labels
        
    return ax

def optimization_report(opt_func, x_def, y_def, x_opt, y_opt):
    initial_score = opt_func(x_def, y_def)
    optimal_score = opt_func(x_opt, y_opt)
    improvement = optimal_score - initial_score
    distance = np.sqrt((x_opt - x_def)**2 + (y_opt - y_def)**2)
    
    results = {
        'init_val': initial_score,
        'opt_val': optimal_score,
        'improvement': improvement,
        'improvement_perc': 100 * improvement / abs(initial_score),
        'init_x': x_def,
        'init_y': y_def,
        'opt_x': x_opt,
        'opt_y': y_opt,
        'distance': distance
    }
    
    return results
    
def get_defender_passes(passes_df, defender_id, offset=0):
    defender_passes = passes_df[passes_df['tracking.object_id'] == defender_id]

    mask_lr = defender_passes['tracking.x'] >= (defender_passes['location.x'] - offset)
    mask_lr = mask_lr & (defender_passes['play_direction'] == 'BOTTOM_TO_TOP')
    mask_rl = defender_passes['tracking.x'] <= (defender_passes['location.x'] + offset)
    mask_rl = mask_rl & (defender_passes['play_direction'] == 'TOP_TO_BOTTOM')
    mask = mask_lr | mask_rl
    defender_passes = defender_passes[mask]
    
    frame_ids = defender_passes['frame'].unique()
    return frame_ids

def optimize_defender_pass(passes_df, frame_id, defender_id, threat_map, pitch_length, pitch_width, radius=3, grid_res=20, mode='softmax', temp=0.03):
    threat_func = create_xt_func(defender_id, passes_df, frame_id, threat_map)
    threat_agg = threat_aggregator(mode=mode, temp=temp)
    opt_func = lambda x, y: -threat_agg(threat_func(x, y))
    
    pass_plotter = plot_utils.PassPlotter(passes_df, frame_id, pitch_length, pitch_width)
    x_def, y_def = pass_plotter.get_player_location(defender_id)

    _, x_opt, y_opt = optimize_func(opt_func, x_def, y_def, radius=radius, bound_x=pitch_length, bound_y=pitch_width, grid_res=grid_res)
    results = optimization_report(opt_func, x_def, y_def, x_opt, y_opt)
    return results

def optimize_defender(passes_df, defender_id, threat_map, pitch_length, pitch_width, radius=3, grid_res=5, mode='softmax', temp=0.03):
    defender_passes = get_defender_passes(passes_df, defender_id)
    results = {}
    for frame_id in tqdm(defender_passes, leave=False):
        results[frame_id] = optimize_defender_pass(passes_df, frame_id, defender_id, threat_map, pitch_length, pitch_width, radius=radius, grid_res=grid_res, mode=mode, temp=temp)
    pass_results = pd.DataFrame.from_dict(results, orient='index')
    agg_results = pass_results[['improvement', 'improvement_perc', 'distance']].agg(['mean', 'median', 'std'])
    return pass_results, agg_results
