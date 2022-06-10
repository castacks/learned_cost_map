import numpy as np
import os
import yaml
import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simps
import argparse



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

    def get_data(self):
        return self.data

    def show(self):
        print(self.data)


def psd(x, fs):
    '''Return Power Spectral Density of a signal
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


def cost_function(data, sensor_freq, cost_name, freq_range=None, num_bins=None):
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


    elif "band" in cost_name:
        assert freq_range is not None, "range should not be None"
        bp_z = bandpower(data, sensor_freq, freq_range, window_sec=None, relative=False)

        cost = bp_z


    else:
        raise NotImplementedError("cost_name needs to include bins or band")

    return cost




def main(data_dir1, data_dir2, output_dir):
    # dataset_dir = '/home/mateo/Data/SARA/TartanCost'
    # trajectories_dir = os.path.join(dataset_dir, "Trajectories")
    # annotations_dir  = os.path.join(dataset_dir, "Annotations")
    # dir_names = sorted(os.listdir(trajectories_dir))

    data_dirs = [data_dir1, data_dir2]
    dir_names = [sorted(os.listdir(d)) for d in data_dirs]
    # dir_names =sorted(os.listdir(data_dir))

    all_costs = []

    # import pdb;pdb.set_trace()

    for i, data_dir in enumerate(data_dirs):
        for dir in dir_names[i]:
            if "preview" in dir:
                continue
            print(f"Obtaining costs from directory: {os.path.join(data_dir, dir)}")

            ## Load IMU data
            trajectory_dir = os.path.join(data_dir, dir)
            imu_data = []
            imu_freq = 125.0
            imu_fp = os.path.join(trajectory_dir, "imu", "imu.npy")
            imu_data = np.load(imu_fp)

            # Set cost function parameters
            cost_name = "freq_band_1_30"
            min_freq = 1
            max_freq = 30
            freq_range=[min_freq, max_freq]

            ## Create Buffer for specific trajectory
            window_length = 1  # This length is in seconds
            window_num_points  = int(window_length * imu_freq)  # Number of points that will be stored in the buffer
            pad_val = 9.81 # Value of linear acceleration in z axis when static
            imu_buffer = Buffer(window_num_points, padded=True, pad_val=pad_val)

            print(f"IMU shape: {imu_data.shape}")

            for i in range(imu_data.shape[0]):
                imu_buffer.insert(float(imu_data[i][-1:]))
                all_costs.append(cost_function(imu_buffer.get_data(), imu_freq, cost_name, freq_range=freq_range))

    all_costs = np.array(all_costs)
    min_cost = float(np.min(all_costs))
    max_cost = float(np.percentile(all_costs, 95))
    mean_cost = float(np.mean(all_costs))
    std_cost = float(np.std(all_costs))

    cost_functions = {
        'freq_band_1_30': {
            'max': max_cost,
            'mean': mean_cost,
            'min': min_cost,
            'std': std_cost
        }
    }

    cost_statistics_fp =os.path.join(output_dir, 'wanda_cost_statistics.yaml')
    with open(cost_statistics_fp, 'w') as outfile:
        yaml.safe_dump(cost_functions, outfile, default_flow_style=False)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir1', type=str, required=True, help='Path to the directory that contains the first set of trajectories.')
    parser.add_argument('--data_dir2', type=str, required=True, help='Path to the directory that contains the second set of trajectories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where the output stats file will be saved.')

    args = parser.parse_args()

    data_dir_1 = args.data_dir1
    data_dir_2 = args.data_dir2
    output_dir = args.output_dir
    main(data_dir_1, data_dir_2, output_dir)

