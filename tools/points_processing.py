import json
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

#import dataholders as dh
import tools.utils as utils


def prepare_points(path='../data/points.txt'):
    """
    Parses the output of label app to a dataframe with timestamps of all jump points, including
    the nearest points from the original time vector, converted to milliseconds
    """
    
    with open('data/points.txt') as f:
        lines = f.readlines()
    
    lines = [json.loads(line.strip()) for line in lines]
    
    lines = list(filter(lambda x: x['mode'] == 'jump', lines)) # choose only jump points
    
    df = pd.DataFrame(lines)
    
    # Cast everything to numeric datatypes
    df['noiseStep'] = df['noiseStep'].astype(int)
    df['thermometer'] = df['thermometer'].astype(int)
    df['value'] = df['value'].astype(float)
    df['value'] = (1000 * df['value']).astype(int) # Storing timestamps in int's for easier comparing
    
    df.value = df.value.apply(lambda x: x + 1 if x % 10 != 0 else x) # Fix a dumb glitch
    
    return df

def prepare_data(points_path='data/points.txt', data_path='data/data16.pickle'):
    """
    Reads the marked jump points and the time and temperature vectors, with time measured in milliseconds.
    """
    
    df = prepare_points(points_path)
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
        
    time, lowT, upT = all_data['time_av'], all_data['lowT_av'], all_data['upT_av']
    time, [lowT, upT] = utils.fast_synchronize(time, lowT, upT)
    
    low_labels = np.in1d(time, df[df.thermometer == 0].value)
    up_labels  = np.in1d(time, df[df.thermometer == 1].value)

    return time, low_labels, up_labels, lowT, upT

def smear_labels(a: np.ndarray, size: int) -> np.ndarray:
    """
    Gives positive labels to points within $size points of actual positive labels.
    
    Args:
        a: np.array, labels array
        size: int, how far away the labels are smeared
    """
    
    pad = size # Amount of zero's to add on both sides
    a_padded = np.concatenate((np.zeros(pad), a, np.zeros(pad)))
    rolled = utils.rolling_window(a_padded, size*2 + 1)
    
    return rolled.max(1).astype(a.dtype)

def generate_positive(labels, time, temperature, size=20):
    """
    Generates time and temperature vectors of length $size (preferably divisible by 4), containing a jump.
    """
    time_data        = []
    temperature_data = []
    
    for i, val in tqdm(enumerate(labels), total=len(labels)):
        window = labels[i:i+size] # size: how many points around a jump
        
        left   = i + (1*size//4)
        center = i + (2*size//4)
        right  = i + (3*size//4)

        if labels[left:left+1] == 1 or labels[center:center+1] == 1 or labels[right:right+1] == 1:
            time_data.append(time[i:i+size])
            temperature_data.append(temperature[i:i+size])
        
    if time_data[-1].shape != (size,):
        time_data        = list(filter(lambda x: x.shape == (size,),        time_data))
        temperature_data = list(filter(lambda x: x.shape == (size,), temperature_data))

    return np.array(time_data), np.array(temperature_data)

def generate_negative(labels, time, temperature, size=20, safe=50):
    """
    Generates time and temperature vectors of length 20, hopefully not containing a jump, in a safe distance from actual jumps.
    """
    labels = labels.copy()

    time_data        = []
    temperature_data = []

    pad = (safe - size) // 2
    for i, val in tqdm(enumerate(labels), total=len(labels)):
        window = labels[i:i+safe]
        if window.max() == 0 and i + safe < len(labels):
            time_data.append(time[i+pad:i+safe-pad])
            temperature_data.append(temperature[i+pad:i+safe-pad])
            labels[i:i+safe] = 1
    
    return np.array(time_data), np.array(temperature_data)