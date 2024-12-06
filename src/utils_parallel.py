import pandas as pd
import numpy as np


# adjust coordinates for correct dxt calculation
def adjust_coordinates_parallel(x, y, direction, pitch_dict):
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
    x = np.maximum(np.minimum(x, pitch_length), 0)
    y = np.maximum(np.minimum(y, pitch_width), 0)

    return x, y

def get_xt_index_parallel(x, y, pitch_dict):
    xt_table = pitch_dict['xt_table']
    cell_width = pitch_dict['cell_width']
    cell_height = pitch_dict['cell_height']

    # map locations to xt table
    x_index = np.minimum(x // cell_width, xt_table.shape[1] - 1).astype(np.int32)
    y_index = np.minimum(y // cell_height, xt_table.shape[0] - 1).astype(np.int32)
    return x_index, y_index

# Get XT value for a given location
def get_xt_value_parallel(x, y, direction, pitch_dict):
    xt_table = pitch_dict['xt_table_np']
    adjusted_x, adjusted_y = adjust_coordinates_parallel(x, y, direction, pitch_dict)
    x_index, y_index = get_xt_index_parallel(adjusted_x, adjusted_y, pitch_dict)
    return xt_table[y_index, x_index]

def get_dxt_parallel(x_start, y_start, x_end, y_end, direction, pitch_dict):
    start_xt = get_xt_value_parallel(x_start, y_start, direction, pitch_dict)
    end_xt = get_xt_value_parallel(x_end, y_end, direction, pitch_dict)
    return end_xt - start_xt




def threat_aggregator(mode='max', k=3, temp=1):
    if mode == 'max':
        return np.max
    # elif mode == 'softmax':
    #     def softmax(x):
    #         x = np.array(list(x.values()))
    #         exp_x = np.exp(x/temp)
    #         softmax_scores = exp_x / np.sum(exp_x) 
    #         return sum(softmax_scores * x)
    #     return softmax
    #     return lambda x: sum([np.exp(v) for v in x.values()])
    # elif mode == 'topK':
        # return lambda x: sum(sorted(x.values(), reverse=True)[:k])
    elif mode == 'mean':
        return np.mean
    elif mode == 'sum':
        return np.sum
    else:
        raise ValueError('Invalid mode')