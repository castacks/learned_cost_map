import argparse
import numpy as np

import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simps

import os
import yaml


class Buffer:
    '''Maintains a scrolling buffer to maintain a window of data in memory

    Args:
        buffer_size: 
            Int, number of data points to keep in buffer
    '''
    def __init__(self, buffer_size, padded=False, pad_val=None):
        self.buffer_size = buffer_size
        if not padded:
            self._data = []
            self.data = np.array(self._data)
        else:
            assert pad_val is not None, "For a padded array, pad_val cannot be None."
            self._data = [pad_val] * buffer_size
            self.data = np.array(self._data)
        
    def insert(self, data_point):
        self._data.append(data_point)
        if len(self._data) > self.buffer_size:
            self._data = self._data[1:]
        self.data = np.array(self._data)

    def insert_many(self, data_list):
        self._data.extend(data_list)
        if len(self._data) > self.buffer_size:
            self._data = self._data[len(self._data)-self.buffer_size:]
        self.data = np.array(self._data)

    def get_data(self):
        return self.data

    def show(self):
        print(self.data)


def psd(x, fs):
    '''Return Poswer
    '''
    # f, Pxx = scipy.signal.periodogram(x, fs=fs)
    f, Pxx = scipy.signal.welch(x, fs)

    return f, Pxx

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Taken from: https://raphaelvallat.com/bandpower.html

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        # nperseg = (2 / low) * sf
        nperseg = None

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def cost_function(data, sensor_freq, cost_name, cost_stats, freq_range=None, num_bins=None):
    '''Average bandpower in bins of 10 Hz in z axis

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - num_bins:
            Number of bins to split data into
    '''
    cost = 1000000

    # import pdb;pdb.set_trace()
    if "bins" in cost_name:
        assert num_bins is not None, "num_bins should not be None"
        freq_width = (sensor_freq//2)/num_bins
        bins_start = [i*freq_width + 1 for i in range(num_bins)]
        bins_end   = [(i+1)*freq_width + 1 for i in range(num_bins)]

        bps = []
        for i in range(num_bins):
            bp_z = bandpower(data, sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=False)
            total_bp = bp_z
            bps.append(total_bp)

        cost = np.mean(bps)

        # Normalize cost:
        cost = (cost-cost_stats["min"])/(cost_stats["max"]-cost_stats["min"])
        cost = max(min(cost, 1), 0)

    elif "band" in cost_name:
        assert freq_range is not None, "range should not be None"
        bp_z = bandpower(data, sensor_freq, freq_range, window_sec=None, relative=False)

        cost = bp_z

        # Normalize cost:
        cost = (cost-cost_stats["min"])/(cost_stats["max"]-cost_stats["min"])
        cost = max(min(cost, 1), 0)

    else:
        raise NotImplementedError("cost_name needs to include bins or band")

    return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--coststats_dir', type=str, required=True, help='Path to cost_statistics.yaml file that contains statistics for IMU cost functions.')
    args = parser.parse_args()

    # Find all trajectory directories
    # trajectories_dir = os.path.join(args.data_dir, "Trajectories")
    # Below: set up for cluster:
    trajectories_dir = args.data_dir
    traj_dirs = list(filter(os.path.isdir, [os.path.join(trajectories_dir,x) for x in sorted(os.listdir(trajectories_dir))]))

    ## Initialize cost parameters
    # cost_stats_dir = "/home/mateo/Data/SARA/TartanCost/cost_statistics.yaml"
    # Below: Set up for cluster

    cost_stats_dir = args.coststats_dir

    # cost_stats_dir = "/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/ros/cost_statistics.yaml"
    with open(cost_stats_dir, 'r') as f:
        all_costs_stats = yaml.safe_load(f)
    cost_name = "freq_band_1_30"
    cost_stats = all_costs_stats[cost_name]
    sensor_name = "imu_z"
    min_freq = 1
    max_freq = 30

    for i, d in enumerate(traj_dirs):
        if "preview" in d:
            continue
        print("=====")
        print(f"Labeling directory {d}")

        ## Load IMU data
        imu_dir = os.path.join(d, "imu")
        imu_fp = os.path.join(imu_dir, "imu.npy")
        imu_data = np.load(imu_fp)

        ## Load IMU timestamps file
        imu_txt = os.path.join(imu_dir, "timestamps.txt")
        imu_times = np.loadtxt(imu_txt)

        ## Load image_left timestamps to use for reference for cost labeling
        image_txt = os.path.join(d, "image_left", "timestamps.txt")
        image_times = np.loadtxt(image_txt)

        ## Initialize buffer
        pad_val = 9.81
        imu_freq = 125.0
        num_seconds = 1
        buffer_size = int(num_seconds*imu_freq)  # num_seconds*imu_freq
        buffer = Buffer(buffer_size, padded=True, pad_val=pad_val)

        ## Initialize cost array
        cost_vals = []
        cost_times = []

        start_imu_idx = 0
        for i, img_time in enumerate(image_times):
            # Get index range for imu_data to look at for a given image
            end_imu_idx = np.searchsorted(imu_times, img_time, side="right")

            # Get imu data and linear acceleration in z for that segment
            imu_segment = imu_data[start_imu_idx:end_imu_idx, :]
            # # Make sure we don't have single elements instead of arrays 
            # if (end_imu_idx - start_imu_idx) == 1:
            #     imu_segment = np.array([imu_segment])

            if len(imu_segment) > 0:
                lin_acc_z = imu_segment[:, -1].tolist()
            else:
                lin_acc_z = []

            # Add this much data into buffer
            buffer.insert_many(lin_acc_z)

            # Calculate cost for buffer.data
            cost = cost_function(buffer.data, imu_freq, cost_name, cost_stats, freq_range=[min_freq, max_freq], num_bins=None)

            # Append cost to cost_vals and img_time to cost_times
            cost_vals.append(cost)
            cost_times.append(img_time)

            # Update start_imu_idx
            start_imu_idx = end_imu_idx

        # import pdb;pdb.set_trace()
        # Write cost_vals and cost_times to own folder in the trajectory
        cost_dir = os.path.join(d, "cost")
        if not os.path.exists(cost_dir):
            os.makedirs(cost_dir)
        
        cost_val_fp = os.path.join(cost_dir, "float.npy")
        cost_times_fp = os.path.join(cost_dir, "timestamps.txt")

        np.save(cost_val_fp, np.array(cost_vals))
        np.savetxt(cost_times_fp, np.array(cost_times))