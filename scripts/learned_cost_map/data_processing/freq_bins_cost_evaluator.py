import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simps
from PIL import Image
from sklearn.decomposition import PCA
import csv

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


def cost_function(data, sensor_freq, num_bins):
    '''Average bandpower in bins of 10 Hz in z axis

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - num_bins:
            Number of bins to split data into
    '''

    freq_width = (sensor_freq//2)/num_bins
    print(f"For num_bins: {num_bins}, freq_width is: {freq_width}")
    bins_start = [i*freq_width + 1 for i in range(num_bins)]
    bins_end   = [(i+1)*freq_width + 1 for i in range(num_bins)]

    # print(f"Bins start: {bins_start}, bins_end: {bins_end}")

    bps = []
    for i in range(num_bins):
        bp_z = bandpower(data, sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=False)
        total_bp = bp_z
        bps.append(total_bp)

    cost = np.mean(bps)

    return cost


def main():
    dataset_dir = '/home/mateo/Data/SARA/TartanCost'
    trajectories_dir = os.path.join(dataset_dir, "Trajectories")
    annotations_dir  = os.path.join(dataset_dir, "Annotations")
    dir_names = sorted(os.listdir(trajectories_dir))

    VISUALIZE = True
    
    header = ['trajectory', 'avg_speed', 'avg_score', "1_bin", "2_bins", "3_bins", "5_bins", "10_bins", "15_bins", "30_bins"]
    csv_data = []
    
    for dir in dir_names:
        trajectory_dir = os.path.join(trajectories_dir, dir)
        annotation_dir = os.path.join(annotations_dir, dir+".yaml")

        imu_data = []
        imu_freq = None
        imu_per_img = 0

        # Load IMU data from specific trajectory 
        imu_dir = os.path.join(trajectory_dir, "imu")
        sorted_imu_fps = sorted(os.listdir(imu_dir))
        for imu_seq_fp in sorted_imu_fps:
            fp = os.path.join(imu_dir, imu_seq_fp)
            imu_seq = np.load(fp)
            imu_data.append(imu_seq)
            imu_per_img = imu_seq.shape[0]
        imu_data = np.vstack(imu_data)

        # Load IMU frequency
        with open(annotation_dir, 'r') as f:
            annotation = yaml.safe_load(f)
        imu_freq = annotation["sensors"]["imu"]["freq"]

        img_data = []
        img_freq = None

        img_dir = os.path.join(trajectory_dir, "image_left_color")
        sorted_img_fps = sorted(os.listdir(img_dir))
        for img_fp in sorted_img_fps:
            fp = os.path.join(img_dir, img_fp)
            img = np.asarray(Image.open(fp))
            img_data.append(img)

        img_freq = annotation["sensors"]["image_left_color"]["freq"]
        img_idx = 0
        time_int = 0

        # Iterate data and add sequentially to buffer (and increment time correspondingly) to simulate online data acquisition

        window_length = 1  # This length is in seconds
        window_num_points  = int(window_length * imu_freq)  # Number of points that will be stored in the buffer
        pad_val = np.zeros((3,))  # Will store 3D IMU linear acceleration
        imu_buffer = Buffer(window_num_points, padded=True, pad_val=pad_val)
        # import pdb;pdb.set_trace()
        time = np.arange(0,window_length-1/imu_freq, 1/imu_freq)

        cost_buffer = Buffer(window_num_points, padded=True, pad_val=0.0)

        fig = plt.figure()
        fig.suptitle('Cost visualizer')
        img_viewer = fig.add_subplot(221)
        imu_viewer = fig.add_subplot(222)
        freq_viewer = fig.add_subplot(223)
        cost_viewer = fig.add_subplot(224)

        cost_1_list = []
        cost_2_list = []
        cost_3_list = []
        cost_4_list = []
        cost_5_list = []
        cost_6_list = []
        cost_7_list = []

        avg_speed = annotation["average_speed"]

        avg_score = annotation["human_scores"]["average"]


        # import pdb;pdb.set_trace()
        for i in range(imu_data.shape[0]):
            imu_buffer.insert(imu_data[i][-3:])

            cost_1_list.append(cost_function(imu_buffer.get_data()[:,2], imu_freq, num_bins=1))
            cost_2_list.append(cost_function(imu_buffer.get_data()[:,2], imu_freq, num_bins=2))
            cost_3_list.append(cost_function(imu_buffer.get_data()[:,2], imu_freq, num_bins=3))
            cost_4_list.append(cost_function(imu_buffer.get_data()[:,2], imu_freq, num_bins=5))
            cost_5_list.append(cost_function(imu_buffer.get_data()[:,2], imu_freq, num_bins=10))
            cost_6_list.append(cost_function(imu_buffer.get_data()[:,2], imu_freq, num_bins=15))
            cost_7_list.append(cost_function(imu_buffer.get_data()[:,2], imu_freq, num_bins=30))


        
        cost_1  = np.mean(cost_1_list)
        cost_2  = np.mean(cost_2_list)
        cost_3  = np.mean(cost_3_list)
        cost_4  = np.mean(cost_4_list)
        cost_5  = np.mean(cost_5_list)
        cost_6  = np.mean(cost_6_list)
        cost_7  = np.mean(cost_7_list)


        csv_data.append([dir, avg_speed, avg_score, cost_1, cost_2, cost_3, cost_4, cost_5, cost_6, cost_7])


        # import pdb;pdb.set_trace()
        # Run cost function with each window of data

    # Visualize window of data and cost function
    with open('/home/mateo/Data/SARA/TartanCost/freq_bins_cost_functions.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for elem in csv_data:
            writer.writerow(elem)

    # Save cost function output in annotation for each cost function 

if __name__=="__main__":
    main()



