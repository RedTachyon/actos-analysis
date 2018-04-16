import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import curve_fit


import utils
#import heuristics
#import filters


class TempData:
    """
    Holds the data about a single temperature time series.
    """

    def __init__(self, time, temperature):
        """

        Args:
            time: array, holding the time indices
            temperature: array, holding the temperature readings
        """
        self.time = time
        self.temperature = temperature

        self.baseline = None
        self.processed = None
        self.detected = None
        self.filtered = None
        self.mask = None

    def preprocess(self, method=lambda x, *args, **kwargs: x, *args, **kwargs):
        """
        Filters the temperature with a chosen function.

        Args:
            method: function, filter to be applied to the temperature

        Returns:
            filtered_temperature, array
        """
        self.baseline = method(self.temperature, *args, **kwargs)
        self.filtered = self.temperature - self.baseline
        # return self.baseline

    def process(self, method, window):
        """
        Converts the temperature data to values proportional to the likelihood of being jumps.

        Args:
            method: function array => array, algorithm to be used
            window: int, size of the rolling window

        Returns:
            array
        """

        if self.baseline is None:
            raise AttributeError("You must run preprocessing first. "
                                 "If you don't know what to use, just use an identity filter")

        assert window % 2

        pad = window // 2

        strided_data = utils.rolling_window(self.filtered, window)

        self.processed = np.hstack((np.zeros(pad), method(strided_data), np.zeros(pad)))
        # return self.processed

    def detect(self, threshold):
        """
        Sets the values above threshold to True, and the rest to False.

        Args:
            threshold: float

        Returns:
            boolean array
        """
        if self.processed is None:
            raise AttributeError("You must run processing first. "
                                 "If you don't know what to use, just use a simple max - min filter")

        self.mask = self.processed > threshold
        detected = np.where(self.mask)[0] / 100.
        detected += self.time[0]

        self.detected = detected

        # return self.detected

    def plot(self, base=False):
        plt.plot(self.time, self.temperature, linewidth=.5, color='b')
        if base:
            plt.plot(self.time, self.baseline, linewidth=1.2, color='y')
        if self.detected is not None:
            plt.vlines(self.detected, 13.8, 14.6, alpha=.25, linewidth=2.5, colors='r')
        plt.show()

    def plot_base(self):
        plt.plot(self.time, self.temperature, linewidth=1.2, color='r')
        plt.plot(self.time, self.baseline, linewidth=1.2)
        plt.show()


class WindData:
    """Holds the information about wind-related data."""

    def __init__(self, wind_x, wind_y, wind_z, degrees=False):
        self.wind = (wind_x, wind_y, wind_z)
        self.theta, self.phi = None, None

        self._get_angles(degrees=degrees)

    def _get_angles(self, degrees=False):
        self.theta = np.arccos(self.wind[2] / np.sqrt(self.wind[0] ** 2 + self.wind[1] ** 2 + self.wind[2] ** 2))
        self.theta = self.theta.ravel()

        self.phi = np.arccos(self.wind[1] / np.sqrt(self.wind[0] ** 2 + self.wind[1] ** 2))
        self.phi = self.phi.ravel()

        if degrees:
            self.theta *= 180 / np.pi
            self.phi *= 180 / np.pi


class TempWindData:
    """
    Holds the temperature and wind data from a single flight.
    """

    def __init__(self, path_temp, path_wind, voltage=False):
        self.path_temp = path_temp
        self.path_wind = path_wind
        self.voltage = voltage
        
        self.v1, self.v2, self.v3, self.X_temp, self.X_wind, self.T1, self.T2 = None, None, None, None, None, None, None
        self.time = None
        self.lwc = None

        self._load_data()
        self._synchronize()

    def _load_data(self):
        """Loads the temperature and wind data from .mat files."""
        wind_vars = ['sonic1', 'sonic2', 'sonic3']
        if self.voltage:
            wind_vars.append('sonicPRT')
        temp_vars = ['lowT_av', 'upT_av', 'time_av', 'lwc1V_av']
        if self.voltage:
            temp_vars += ['lowV_av', 'upV_av']
        winddata = loadmat(self.path_wind, variable_names=wind_vars)
        tempdata = loadmat(self.path_temp, variable_names=temp_vars)

        self.v1, self.v2, self.v3 = winddata['sonic1'].ravel(), winddata['sonic2'].ravel(), winddata['sonic3'].ravel()
        self.X_wind = np.arange(self.v1.shape[0]) / 100.

        self.X_temp = tempdata['time_av'].ravel()
        self.T1 = tempdata['lowT_av'].ravel()
        self.T2 = tempdata['upT_av'].ravel()
        self.lwc = tempdata['lwc1V_av'].ravel()
        if self.voltage:
            self.V1 = tempdata['lowV_av'].ravel()
            self.V2 = tempdata['upV_av'].ravel()
            self.T_base = winddata['sonicPRT'].ravel()

    def _synchronize(self):
        """Synchronizes the wind and temperature data."""
        low = np.max((self.X_temp[0], self.X_wind[0]))
        self.X_temp -= (self.X_temp[self.X_temp >= low][0] - self.X_wind[self.X_wind >= low][0])

        high = np.min((self.X_temp[-1], self.X_wind[-1])) + 1e-9  # Includes a small constant to keep the same shape

        self.v1 = utils.array_range(self.v1, low, high, self.X_wind)
        self.v2 = utils.array_range(self.v2, low, high, self.X_wind)
        self.v3 = utils.array_range(self.v3, low, high, self.X_wind)
        self.T_base = utils.array_range(self.T_base, low, high, self.X_wind)
        self.X_wind = utils.array_range(self.X_wind, low, high, self.X_wind)

        self.T1 = utils.array_range(self.T1, low, high, self.X_temp)
        self.T2 = utils.array_range(self.T2, low, high, self.X_temp)
        self.lwc = utils.array_range(self.lwc, low, high, self.X_temp)
        if self.voltage:
                self.V1 = utils.array_range(self.V1, low, high, self.X_temp)
                self.V2 = utils.array_range(self.V2, low, high, self.X_temp)
        
        self.X_temp = utils.array_range(self.X_temp, low, high, self.X_temp)
        

        self.time = self.X_wind
        self.X_temp = self.time

    def cut_time(self, low, high):
        """Restricts the data to a specific time."""
        self.v1 = utils.array_range(self.v1, low, high, self.X_wind)
        self.v2 = utils.array_range(self.v2, low, high, self.X_wind)
        self.v3 = utils.array_range(self.v3, low, high, self.X_wind)
        self.T_base = utils.array_range(self.T_base, low, high, self.X_wind)
        self.X_wind = utils.array_range(self.X_wind, low, high, self.X_wind)

        self.T1 = utils.array_range(self.T1, low, high, self.X_temp)
        self.T2 = utils.array_range(self.T2, low, high, self.X_temp)
        self.lwc = utils.array_range(self.lwc, low, high, self.X_temp)
        if self.voltage:
            self.V1 = utils.array_range(self.V1, low, high, self.X_temp)
            self.V2 = utils.array_range(self.V2, low, high, self.X_temp)
        self.X_temp = utils.array_range(self.X_temp, low, high, self.X_temp)
            
        self.time = utils.array_range(self.time, low, high, self.time)
        
    def smooth_temperatures(self, method='mean', window=3):
        if method == 'mean':
            self.T1_smooth = filters.mean_filter(self.T1, window)
            self.T2_smooth = filters.mean_filter(self.T2, window)
        elif method == 'median':
            self.T1_smooth = filters.median_filter(self.T1, window)
            self.T2_smooth = filters.median_filter(self.T2, window)
        else:
            raise ValueError("Unsupported method. Use only 'mean' or 'median'")
        
        self.T1_res = self.T1 - self.T1_smooth
        self.T2_res = self.T2 - self.T2_smooth
        
    def get_temp_from_voltage(self, window=None):
        """
        Callibrates the voltage with the base temperature and puts the new temperature estimates to self.T1, self.T2
        """
        assert self.voltage

        self.T1_old = self.T1
        self.T2_old = self.T2
        
        linear = lambda x, a, b: a*x + b
        # To do: add the possibility to callibrate using a specific period
        if window is None:
            popt1, _ = curve_fit(linear, self.V1, self.T_base)
            popt2, _ = curve_fit(linear, self.V2, self.T_base)
        else:
            popt1, _ = curve_fit(linear, utils.rolling_window(self.V1, window).mean(1)[::window], 
                                 utils.rolling_window(self.T_base, window).mean(1)[::window])
            popt2, _ = curve_fit(linear, utils.rolling_window(self.V2, window).mean(1)[::window], 
                                 utils.rolling_window(self.T_base, window).mean(1)[::window])
            
        self.T1 = linear(self.V1, *popt1)
        self.T2 = linear(self.V2, *popt2)
        
        
    def get_angles(self, degrees=False):
        self.theta = np.arccos(self.v3 / np.sqrt(self.v1 ** 2 + self.v2 ** 2 + self.v3 ** 2))
        self.theta = self.theta.ravel()

        self.phi = np.arccos(self.v2 / np.sqrt(self.v1 ** 2 + self.v2 ** 2))
        self.phi = self.phi.ravel()

        if degrees:
            self.theta *= 180 / np.pi
            self.phi *= 180 / np.pi
        
        
    def get_temp_deviations(self, window):
        baseline1 = utils.rolling_window(self.T1, window).mean(1)
        baseline2 = utils.rolling_window(self.T2, window).mean(1)
        
        self.T1_dev = self.T1[window//2:-window//2+1] - baseline1
        self.T2_dev = self.T2[window//2:-window//2+1] - baseline2
        
        self.DT = self.T1 - self.T2
        
        baselineD = utils.rolling_window(self.DT, window).mean(1)
        self.DT_dev = self.DT[window//2:-window//2+1] - baselineD
    
    
    def get_angle_deviations(self, window):
        assert hasattr(self, 'theta')
        
        baseline_theta = utils.rolling_window(self.theta, window).mean(1)
        baseline_phi = utils.rolling_window(self.phi, window).mean(1)
        
        self.theta_dev = self.theta[window//2:-window//2+1] - baseline_theta
        self.phi_dev = self.phi[window//2:-window//2+1] - baseline_phi

if __name__ == '__main__':
    temp_path = 'data/raw/uft_flight07.mat'
    wind_path = 'data/raw/actos_flight07.mat'

    holder = TempWindData(temp_path, wind_path)

    holder.cut_time(2750, 3000)

    print(holder.v1.shape)
