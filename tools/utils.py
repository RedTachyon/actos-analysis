import numpy as np
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score



def rolling_window(series, window):
    """
    Helper function for performing rolling window computations.
    Adds an extra dimension to the array, which can then be used to do whatever you need to do on windows.

    Args:
        series: np.array, series to be unrolled
        window: int, size of the rolling window

    Returns:
        np.array
    """
    shape = series.shape[:-1] + (series.shape[-1] - window + 1, window)
    strides = series.strides + (series.strides[-1],)
    return np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)


def read_pickle(path):
    """Reads an array from the path, using pickle"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def read_data(path, low=275000, high=300000):
    """Reads data and preformats it."""
    data = read_pickle(path)

    Y1 = data['lowT_av'].squeeze()
    Y2 = data['upT_av'].squeeze()
    # LWC = data['lwc1V_av']
    X = np.arange(Y1.shape[0]) / 100.
    # X = data['time_av'].squeeze()
    if low is not None and high is not None:
        X = X[low:high]
        Y1 = Y1[low:high]
        Y2 = Y2[low:high]

    return X, Y1, Y2


def array_range(a, low, high, ref=None):
    """
    Returns the array limited to values in selected range.
    """
    if ref is None:
        ref = a
    return a[np.logical_and(ref >= low, ref < high)]


def fast_synchronize(time_uft, *arrays, how='linear') -> (np.ndarray, np.ndarray):
    """
    Synchronizes the UFT time and record vector to conform with the ACTOS timestamps, and hopefully does so quickly.
    """
    
    time_uft = rolling_window(time_uft, 2).mean(1, dtype=int)
    narrays = [rolling_window(array, 2).mean(1) for array in arrays]
    
    return time_uft, narrays

def slow_synchronize(time_actos: np.ndarray, time_uft: np.ndarray, *arrays, how='linear') -> (np.ndarray, np.ndarray):
    """
    Synchronizes the UFT time and record vector to conform with the ACTOS timestamps.
    """
    
    narrays = [[] for array in arrays]
    
    for t in tqdm(time_uft):
        try:
            index = np.where(time_uft == t - 5)[0][0]
        except IndexError:
            continue
            
        for i, array in enumerate(arrays):
            narrays[i].append((array[index] + array[index + 1])/2)
        
    narrays = list(map(np.array, narrays))
    
    return time_uft, narrays

def write_report(classifier, train_data, train_labels, test_data, test_labels, keras=False, threshold=.5):
    """
    Evaluates the classifier.
    """
    train_preds = classifier.predict(train_data)
    test_preds = classifier.predict(test_data)
    
    if keras:
        train_preds = train_preds > threshold
        test_preds = test_preds > threshold
    
    print("Training data (less important):")
    print(" Accuracy: %.5f\n Precision: %.5f\n Recall: %.5f\n \033[91m F1 score (class 1): %.5f\n\033[0m F1 score (class 0): %.5f" 
          % (accuracy_score(train_labels, train_preds),
             precision_score(train_labels, train_preds),
             recall_score(train_labels, train_preds),
             f1_score(train_labels, train_preds),
             f1_score(train_labels, train_preds, pos_label=0)
            ))
    
    print("Test data (more important):")
    print(" Accuracy: %.5f\n Precision: %.5f\n Recall: %.5f\n \033[91m F1 score (class 1): %.5f\n\033[0m F1 score (class 0): %.5f" 
          % (accuracy_score(test_labels, test_preds),
             precision_score(test_labels, test_preds),
             recall_score(test_labels, test_preds),
             f1_score(test_labels, test_preds),
             f1_score(test_labels, test_preds, pos_label=0)
            ))