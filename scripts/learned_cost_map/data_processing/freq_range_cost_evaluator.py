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


def cost_function_1(data, sensor_freq):
    '''Bandpower in 1-3 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [1, 3], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_2(data, sensor_freq):
    '''Bandpower in 1-5 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [1, 5], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_3(data, sensor_freq):
    '''Bandpower in 1-8 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [1, 8], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_4(data, sensor_freq):
    '''Bandpower in 1-12 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [1, 12], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_5(data, sensor_freq):
    '''Bandpower in 1-15 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [1, 15], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_6(data, sensor_freq):
    '''Bandpower in 1-20 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [1, 20], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_7(data, sensor_freq):
    '''Bandpower in 1-30 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [1, 30], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_8(data, sensor_freq):
    '''Bandpower in 3-5 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [3, 5], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_9(data, sensor_freq):
    '''Bandpower in 3-8 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [3, 8], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_10(data, sensor_freq):
    '''Bandpower in 3-12 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [3, 12], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_11(data, sensor_freq):
    '''Bandpower in 3-15 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [3, 15], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_12(data, sensor_freq):
    '''Bandpower in 3-20 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [3, 20], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_13(data, sensor_freq):
    '''Bandpower in 3-30 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [3, 30], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_14(data, sensor_freq):
    '''Bandpower in 5-8 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [5, 8], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_15(data, sensor_freq):
    '''Bandpower in 5-12 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [5, 12], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_16(data, sensor_freq):
    '''Bandpower in 5-15 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [5, 15], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_17(data, sensor_freq):
    '''Bandpower in 5-20 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [5, 20], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_18(data, sensor_freq):
    '''Bandpower in 5-30 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [5, 30], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_19(data, sensor_freq):
    '''Bandpower in 8-12 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [8, 12], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_20(data, sensor_freq):
    '''Bandpower in 8-15 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [8, 15], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_21(data, sensor_freq):
    '''Bandpower in 8-20 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [8, 20], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_22(data, sensor_freq):
    '''Bandpower in 8-30 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [8, 30], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_23(data, sensor_freq):
    '''Bandpower in 12-15 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [12, 15], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_24(data, sensor_freq):
    '''Bandpower in 12-20 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [12, 20], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_25(data, sensor_freq):
    '''Bandpower in 12-30 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [12, 30], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_26(data, sensor_freq):
    '''Bandpower in 15-20 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [15, 20], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_27(data, sensor_freq):
    '''Bandpower in 15-30 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [15, 30], window_sec=None, relative=False)

    cost = bp_z

    return cost

def cost_function_28(data, sensor_freq):
    '''Bandpower in 20-30 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''
    bp_z = bandpower(data, sensor_freq, [20, 30], window_sec=None, relative=False)

    cost = bp_z

    return cost

def main():
    dataset_dir = '/home/mateo/Data/SARA/TartanCost'
    trajectories_dir = os.path.join(dataset_dir, "Trajectories")
    annotations_dir  = os.path.join(dataset_dir, "Annotations")
    dir_names = sorted(os.listdir(trajectories_dir))

    VISUALIZE = True
    
    header = ['trajectory', 'avg_speed', 'avg_score', "cost_1", "cost_2", "cost_3", "cost_4", "cost_5", "cost_6", "cost_7", "cost_8", "cost_9", "cost_10","cost_11","cost_12","cost_13","cost_14","cost_15","cost_16","cost_17","cost_18","cost_19","cost_20","cost_21","cost_22","cost_23","cost_24","cost_25","cost_26","cost_27","cost_28"]
    csv_data = []
    
    # for dir in ["000195"]:
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
        cost_8_list = []
        cost_9_list = []
        cost_10_list = []
        cost_11_list = []
        cost_12_list = []
        cost_13_list = []
        cost_14_list = []
        cost_15_list = []
        cost_16_list = []
        cost_17_list = []
        cost_18_list = []
        cost_19_list = []
        cost_20_list = []
        cost_21_list = []
        cost_22_list = []
        cost_23_list = []
        cost_24_list = []
        cost_25_list = []
        cost_26_list = []
        cost_27_list = []
        cost_28_list = []

        avg_speed = annotation["average_speed"]

        avg_score = annotation["human_scores"]["average"]


        # import pdb;pdb.set_trace()
        for i in range(imu_data.shape[0]):
            imu_buffer.insert(imu_data[i][-3:])

            cost_1_list.append(cost_function_1(imu_buffer.get_data()[:,2], imu_freq))
            cost_2_list.append(cost_function_2(imu_buffer.get_data()[:,2], imu_freq))
            cost_3_list.append(cost_function_3(imu_buffer.get_data()[:,2], imu_freq))
            cost_4_list.append(cost_function_4(imu_buffer.get_data()[:,2], imu_freq))
            cost_5_list.append(cost_function_5(imu_buffer.get_data()[:,2], imu_freq))
            cost_6_list.append(cost_function_6(imu_buffer.get_data()[:,2], imu_freq))
            cost_7_list.append(cost_function_7(imu_buffer.get_data()[:,2], imu_freq))
            cost_8_list.append(cost_function_8(imu_buffer.get_data()[:,2], imu_freq))
            cost_9_list.append(cost_function_9(imu_buffer.get_data()[:,2], imu_freq))
            cost_10_list.append(cost_function_10(imu_buffer.get_data()[:,2], imu_freq))
            cost_11_list.append(cost_function_11(imu_buffer.get_data()[:,2], imu_freq))
            cost_12_list.append(cost_function_12(imu_buffer.get_data()[:,2], imu_freq))
            cost_13_list.append(cost_function_13(imu_buffer.get_data()[:,2], imu_freq))
            cost_14_list.append(cost_function_14(imu_buffer.get_data()[:,2], imu_freq))
            cost_15_list.append(cost_function_15(imu_buffer.get_data()[:,2], imu_freq))
            cost_16_list.append(cost_function_16(imu_buffer.get_data()[:,2], imu_freq))
            cost_17_list.append(cost_function_17(imu_buffer.get_data()[:,2], imu_freq))
            cost_18_list.append(cost_function_18(imu_buffer.get_data()[:,2], imu_freq))
            cost_19_list.append(cost_function_19(imu_buffer.get_data()[:,2], imu_freq))
            cost_20_list.append(cost_function_20(imu_buffer.get_data()[:,2], imu_freq))
            cost_21_list.append(cost_function_21(imu_buffer.get_data()[:,2], imu_freq))
            cost_22_list.append(cost_function_22(imu_buffer.get_data()[:,2], imu_freq))
            cost_23_list.append(cost_function_23(imu_buffer.get_data()[:,2], imu_freq))
            cost_24_list.append(cost_function_24(imu_buffer.get_data()[:,2], imu_freq))
            cost_25_list.append(cost_function_25(imu_buffer.get_data()[:,2], imu_freq))
            cost_26_list.append(cost_function_26(imu_buffer.get_data()[:,2], imu_freq))
            cost_27_list.append(cost_function_27(imu_buffer.get_data()[:,2], imu_freq))
            cost_28_list.append(cost_function_28(imu_buffer.get_data()[:,2], imu_freq))

        
        cost_1  = np.mean(cost_1_list)
        cost_2  = np.mean(cost_2_list)
        cost_3  = np.mean(cost_3_list)
        cost_4  = np.mean(cost_4_list)
        cost_5  = np.mean(cost_5_list)
        cost_6  = np.mean(cost_6_list)
        cost_7  = np.mean(cost_7_list)
        cost_8  = np.mean(cost_8_list)
        cost_9  = np.mean(cost_9_list)
        cost_10 = np.mean(cost_10_list)
        cost_11 = np.mean(cost_11_list)
        cost_12 = np.mean(cost_12_list)
        cost_13 = np.mean(cost_13_list)
        cost_14 = np.mean(cost_14_list)
        cost_15 = np.mean(cost_15_list)
        cost_16 = np.mean(cost_16_list)
        cost_17 = np.mean(cost_17_list)
        cost_18 = np.mean(cost_18_list)
        cost_19 = np.mean(cost_19_list)
        cost_20 = np.mean(cost_20_list)
        cost_21 = np.mean(cost_21_list)
        cost_22 = np.mean(cost_22_list)
        cost_23 = np.mean(cost_23_list)
        cost_24 = np.mean(cost_24_list)
        cost_25 = np.mean(cost_25_list)
        cost_26 = np.mean(cost_26_list)
        cost_27 = np.mean(cost_27_list)
        cost_28 = np.mean(cost_28_list)

        csv_data.append([dir, avg_speed, avg_score, cost_1, cost_2, cost_3, cost_4, cost_5, cost_6, cost_7, cost_8, cost_9, cost_10, cost_11, cost_12, cost_13, cost_14, cost_15, cost_16, cost_17, cost_18, cost_19, cost_20, cost_21, cost_22, cost_23, cost_24, cost_25, cost_26, cost_27, cost_28])


        # import pdb;pdb.set_trace()
        # Run cost function with each window of data

    # Visualize window of data and cost function
    with open('/home/mateo/Data/SARA/TartanCost/freq_cost_functions.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for elem in csv_data:
            writer.writerow(elem)

    # Save cost function output in annotation for each cost function 

if __name__=="__main__":
    main()



# fig = plt.figure()
# img_viewer = fig.add_subplot(111)
# for i in range(imu_data.shape[0]):
#     imu_buffer.insert(imu_data[i][-3:])
#     img_viewer.clear()
#     img_viewer.plot(imu_buffer.get_data()[:, 0], color='red')
#     img_viewer.plot(imu_buffer.get_data()[:, 1], color='green')
#     img_viewer.plot(imu_buffer.get_data()[:, 2], color='blue')
#     img_viewer.set_ylim([-10, 20])
#     plt.pause(0.01)