import numpy as np
import pandas as pd
from tqdm import tqdm

import tools.utils as utils

def get_df(data, uft_vars=['lowT_av', 'upT_av', 'lwc1V_av'], actos_vars=['sonic1', 'sonic2', 'sonic3']):
    """
    Converts the dictionary to a nice, synced DataFrame.
    """
    time_uft = data['time_av']
    time_actos = data['time']
    
    uft_arrays = [data[var] for var in uft_vars]
    actos_arrays = [data[var] for var in actos_vars]
    
    time_uft, uft_arrays = utils.fast_synchronize(time_uft, *uft_arrays) # Synchronize UFT to ACTOS
    
    # Convert lists to dictionaries for named DF columns
    uft_dict = {uft_vars[i]: uft_arrays[i] for i in range(len(uft_vars))}
    actos_dict = {actos_vars[i]: actos_arrays[i] for i in range(len(actos_arrays))}
    
    # Create the ACTOS dataframe, format it properly
    df_actos = pd.DataFrame(data=actos_dict, index=time_actos)
    df_actos.index.name = 'time'
    df_actos = df_actos.reset_index()
    
    # Resample
    df_actos = df_actos.set_index(pd.TimedeltaIndex(df_actos.time, unit='ms'))
    df_actos = df_actos.resample(rule='10L').mean()

    # Back to nice timestamps/index
    df_actos['time'] = df_actos.index.astype(int) // 1000000 # Convert nanoseconds to milliseconds
    df_actos = df_actos.reset_index(drop=True)
    #df_actos = df.set_index('time')
    
    df_uft = pd.DataFrame(data=uft_dict, index=time_uft)
    df_uft.index.name = 'time'
    df_uft = df_uft.reset_index()
    
    df_full = pd.merge(df_uft, df_actos, on='time')
    
    return df_full

def add_labels(df_data, df_points):
    """
    To the first dataframe, add labels corresponding to the occurence of a jump, based on the second dataframe.
    """
    df_data = df_data.copy()
    
    df_data['low_label'] = np.in1d(df_data.time, df_points[df_points.thermometer == 0].value).astype(int)
    df_data['up_label']  = np.in1d(df_data.time, df_points[df_points.thermometer == 1].value).astype(int)
    
    return df_data

def add_angles(df_data):
    """
    Assuming the dataframe has columns sonic1/2/3, add the "theta" and "phi" columns.
    """
    
    df_data = df_data.copy()
    
    df_data['theta'] = np.arccos(df_data.sonic3 / (df_data.sonic1**2 + df_data.sonic2**2 + df_data.sonic3**2)**(1/2))
    df_data['phi']   = np.arccos(df_data.sonic2 / (df_data.sonic1**2 + df_data.sonic2**2)**(1/2))
    
    return df_data

def generate_positive(df, size=20, feature_names=['lowT_av', 'upT_av', 'sonic1', 'sonic2', 'sonic3', 'lwc1V_av'], label_name='low_label'):
    """
    Generates time and temperature vectors of length $size (preferably divisible by 4), containing a jump.
    """
    time_list    = []
    feature_list = []
    
    labels = df[label_name].values
    time = df.time.values
    features = df[feature_names].values

    for i, val in tqdm(enumerate(labels), total=len(labels)):
        window = labels[i:i+size] # size: how many points around a jump

        left   = i + (1*size//4)
        center = i + (2*size//4)
        right  = i + (3*size//4)

        if right < len(labels) and (labels[left] == 1 or labels[center] == 1 or labels[right] == 1):
            time_list.append(time[i:i+size])
            feature_list.append(features[i:i+size])

    # Remove edge windows, possibly of different length
    if time_list[-1].shape != time_list[0].shape:
        time_list    = list(filter(lambda x: x.shape == time_list[0].shape,    time_list))
        feature_list = list(filter(lambda x: x.shape == feature_list[0].shape, feature_list))

    return np.array(time_list), np.array(feature_list)

def generate_negative(df, size=20, safe=50, feature_names=['lowT_av', 'upT_av', 'sonic1', 'sonic2', 'sonic3', 'lwc1V_av'], label_name='low_label'):
    """
    Generates time and temperature vectors of length 20, hopefully not containing a jump, in a safe distance from actual jumps.
    """
    labels = df[label_name].values.copy()
    time = df.time.values
    features = df[feature_names].values

    time_list    = []
    feature_list = []

    pad = (safe - size) // 2
    for i, val in tqdm(enumerate(labels), total=len(labels)):
        window = labels[i:i+safe]
        if window.max() == 0 and i + safe < len(labels):
            time_list.append(time[i+pad:i+safe-pad])
            feature_list.append(features[i+pad:i+safe-pad])
            labels[i:i+safe] = 1
    
    return np.array(time_list), np.array(feature_list)