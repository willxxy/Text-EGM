import os
import wfdb
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path', type = str, default = None, help='Please specify the path to the folder containing the data.')
    return parser.parse_args()

def z_score_normalization(data):
    mean_val = np.mean(data, axis=(0, 1), keepdims=True)
    std_val = np.std(data, axis=(0, 1), keepdims=True)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

def segment_signal(data, segment_length, step_size = None):
    
    n_time_points, n_electrodes, n_placements = data.shape
    
    if step_size != None:
        n_segments = 1 + (n_time_points - segment_length) // step_size
        segmented_data = np.zeros((n_segments, segment_length, n_electrodes, n_placements))

        for i in range(n_segments):
            start_idx = i * step_size
            end_idx = start_idx + segment_length
            segmented_data[i] = data[start_idx:end_idx, :, :]
            
    elif step_size == None:
        n_segments = n_time_points // segment_length
        truncated_data = data[:n_segments * segment_length]
        segmented_data = truncated_data.reshape(n_segments, segment_length, data.shape[1], data.shape[2])
    
    return segmented_data

def read_all(path):
    all_signals = []
    for i in os.listdir(path):
        if 'qrs' in i:
            file_name = i.split('.')[0]
            record = wfdb.rdrecord(f'./{path}/{file_name}')
            egm_signals = record.p_signal[:, 3:]
            all_signals.append(egm_signals)

    min_shape = min(array.shape[0] for array in all_signals)
    sliced_arrays = [array[:min_shape] for array in all_signals]

    stacked_array = np.stack(sliced_arrays, axis=-1)
    
    return stacked_array


def split_dict_by_catheter_afib(input_dict):

    train_dict, test_dict, val_dict = {}, {}, {}
    for key, value in input_dict.items():
        _, catheter_num, _, _ = key
        if catheter_num < 21:
            train_dict[key] = value
        elif 21 <= catheter_num < 26:
            test_dict[key] = value
        elif 26 <= catheter_num:
            val_dict[key] = value
    return train_dict, test_dict, val_dict

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def main(args):
    egm_signals = read_all(args)
    normalized_egm = z_score_normalization(egm_signals)
    segmented_data = segment_signal(normalized_egm, 1000)

    segmented_data_dict = {}
    n_segments, segment_length, n_electrodes, n_placements = segmented_data.shape
    for i in range(n_electrodes):
        for j in range(n_placements):
            for k in range(n_segments):
                key = (i, j, k, 1)
                segmented_data_dict[key] = segmented_data[k, :, i, j]

    feature_dicts = {
            'segmented_data': segmented_data_dict,
            }
    
    concatenated_features = {}

    for key in feature_dicts['segmented_data'].keys():
        concatenated_feature = np.concatenate([feature_dicts[feature_name][key] for feature_name in feature_dicts])        
        concatenated_features[key] = concatenated_feature

    train, test, val = split_dict_by_catheter_afib(concatenated_features)

    ensure_directory_exists('../data')

    np.save('../data/train_intra.npy', train)
    np.save('../data/val_intra.npy', val)
    np.save('../data/test_intra.npy', test)

if __name__ == '__main__':
    args = get_args()
    main(args)
