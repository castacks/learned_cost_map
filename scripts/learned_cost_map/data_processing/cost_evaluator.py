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


def CostFunction(data, sensor_freq, min_freq=0, max_freq=100, sensor_name=None):
    # duration = data.shape[0] / sensor_freq
    # freq_magnitude = rfft(data)
    # freq_bins = rfftfreq(data.shape[0], 1/sensor_freq)
    cost = bandpower(data, sensor_freq, min_freq, max_freq)

    cost = min(1, max(cost, 0))
    return cost

def cost_function_a(data, sensor_freq):
    '''Bandpower of all frequency ranges in all axes

    Args:
        - data:
            Nx3 input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
    '''

    bp_x = bandpower(data[:,0], sensor_freq, [1, sensor_freq//2], window_sec=None, relative=False)
    bp_y = bandpower(data[:,1], sensor_freq, [1, sensor_freq//2], window_sec=None, relative=False)
    bp_z = bandpower(data[:,2], sensor_freq, [1, sensor_freq//2], window_sec=None, relative=False)

    cost = bp_x+bp_y+bp_z

    # print("==========")
    # print("Cost function a: Sum of bandpower of signal in 3 axes")
    # print(f"Bandpower in x: {bp_x}")
    # print(f"Bandpower in y: {bp_y}")
    # print(f"Bandpower in z: {bp_z}")
    # print(f"Sum of bandpowers: {cost}")
    # print("==========")
    return cost


def cost_function_b(data, sensor_freq):
    '''Bandpower of all frequency ranges in z axis

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''
    bp_z = bandpower(data, sensor_freq, [1, sensor_freq//2], window_sec=None, relative=False)

    cost = bp_z

    # print("==========")
    # print("Cost function b: Bandpower of all frequencies in Z axis")
    # print(f"Bandpower in z: {bp_z}")
    # print(f"Sum of bandpowers: {cost}")
    # print("==========")
    return cost


def cost_function_c(data, sensor_freq, num_bins):
    '''Average bandpower in num_bins in all axes

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''
    
    freq_width = (sensor_freq//2)/num_bins
    bins_start = [i*freq_width + 1 for i in range(num_bins)]
    bins_end   = [(i+1)*freq_width + 1 for i in range(num_bins)]

    print(f"Bins start: {bins_start}, bins_end: {bins_end}")

    bps = []
    for i in range(num_bins):
        bp_x = bandpower(data[:,0], sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=False)
        bp_y = bandpower(data[:,1], sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=False)
        bp_z = bandpower(data[:,2], sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=False)
        total_bp = bp_x + bp_y + bp_z
        bps.append(total_bp)

    cost = np.mean(bps)
    # print("==========")
    # print("Cost function c: Average bandpower in num_bins in all axes")
    # print(f"Analyzing {num_bins} bins with frequency band of {freq_width} Hz")
    # print(f"Bandpower bins are: {bps}")
    # print(f"Sum of bandpowers: {cost}")
    # print("==========")
    return cost


def cost_function_d(data, sensor_freq, num_bins):
    '''Average bandpower in bins of 10 Hz in z axis

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''

    freq_width = (sensor_freq//2)/num_bins
    bins_start = [i*freq_width + 1 for i in range(num_bins)]
    bins_end   = [(i+1)*freq_width + 1 for i in range(num_bins)]

    print(f"Bins start: {bins_start}, bins_end: {bins_end}")

    bps = []
    for i in range(num_bins):
        bp_z = bandpower(data, sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=False)
        total_bp = bp_z
        bps.append(total_bp)

    cost = np.mean(bps)
    # print("==========")
    # print("Cost function d: Average bandpower in num_bins in Z axis")
    # print(f"Analyzing {num_bins} bins with frequency band of {freq_width} Hz")
    # print(f"Bandpower bins are: {bps}")
    # print(f"Mean of bandpowers: {cost}")
    # print("==========")
    return cost

def cost_function_e(data, sensor_freq):
    '''First PCA component of PSD for matrix of three axes together

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''

    f_x, Pxx_x = psd(data[:,0], sensor_freq)
    f_y, Pxx_y = psd(data[:,1], sensor_freq)
    f_z, Pxx_z = psd(data[:,2], sensor_freq)

    # import pdb;pdb.set_trace()

    stacked_power = np.vstack([Pxx_x, Pxx_y, Pxx_z])

    pca = PCA(n_components=2)
    pca.fit(stacked_power)
    cost = pca.singular_values_[0]

    # print("==========")
    # print("Cost function e: First PCA component of PSD for matrix of three axes stacked together")
    # print(f"First singular value is: {cost}")
    # print("==========")

    return cost
    


def cost_function_f(data, sensor_freq):
    '''First PCA component of CWT with Ricker wavelet for all three axes

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''
    widths = np.arange(1,sensor_freq//2)
    cwtmatr_x = scipy.signal.cwt(data[:,0], scipy.signal.ricker, widths)
    cwtmatr_y = scipy.signal.cwt(data[:,1], scipy.signal.ricker, widths)
    cwtmatr_z = scipy.signal.cwt(data[:,2], scipy.signal.ricker, widths)
    
    stacked_flat = np.vstack([cwtmatr_x.flatten(), cwtmatr_y.flatten(), cwtmatr_z.flatten()])

    pca = PCA(n_components=3)
    pca.fit(stacked_flat)
    # print("Singular values: ")
    # print(pca.singular_values_)
    cost = pca.singular_values_[0]

    # print("==========")
    # print("Cost function f: First PCA component of CWT with Ricker wavelet for all three axes")
    # print(f"First singular value is: {cost}")
    # print("==========")

    return cost


def cost_function_g(data, sensor_freq, num_bins):
    '''Average of relative powers for n sensor bands in all axes

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''

    freq_width = (sensor_freq//2)/num_bins
    bins_start = [i*freq_width + 1 for i in range(num_bins)]
    bins_end   = [(i+1)*freq_width + 1 for i in range(num_bins)]

    print(f"Bins start: {bins_start}, bins_end: {bins_end}")

    bps = []
    for i in range(num_bins):
        bp_x = bandpower(data[:,0], sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=True)
        bp_y = bandpower(data[:,1], sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=True)
        bp_z = bandpower(data[:,2], sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=True)
        total_bp = bp_x + bp_y + bp_z
        bps.append(total_bp)

    cost = np.mean(bps)
    # print("==========")
    # print("Cost function g: Average bandpower in num_bins in all axes")
    # print(f"Analyzing {num_bins} bins with frequency band of {freq_width} Hz")
    # print(f"Bandpower bins are: {bps}")
    # print(f"Sum of bandpowers: {cost}")
    # print("==========")
    return cost

def cost_function_h(data, sensor_freq, num_bins):
    '''Average of relative powers for n sensor bands in z axis

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''

    freq_width = (sensor_freq//2)/num_bins
    bins_start = [i*freq_width + 1 for i in range(num_bins)]
    bins_end   = [(i+1)*freq_width + 1 for i in range(num_bins)]

    print(f"Bins start: {bins_start}, bins_end: {bins_end}")

    bps = []
    for i in range(num_bins):
        bp_z = bandpower(data, sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=True)
        total_bp = bp_z
        bps.append(total_bp)

    cost = np.mean(bps)
    # print("==========")
    # print("Cost function h: Average bandpower in num_bins in all axes")
    # print(f"Analyzing {num_bins} bins with frequency band of {freq_width} Hz")
    # print(f"Bandpower bins are: {bps}")
    # print(f"Sum of bandpowers: {cost}")
    # print("==========")
    return cost


def cost_function_i(data, sensor_freq):
    '''Ratio of power above 15 hz to power below 15 hz in all axes

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''
    bp_x_low = bandpower(data[:,0], sensor_freq, [3, 15], window_sec=None, relative=False)
    bp_y_low = bandpower(data[:,1], sensor_freq, [3, 15], window_sec=None, relative=False)
    bp_z_low = bandpower(data[:,2], sensor_freq, [3, 15], window_sec=None, relative=False)

    bp_x_high = bandpower(data[:,0], sensor_freq, [15, sensor_freq//2], window_sec=None, relative=False)
    bp_y_high = bandpower(data[:,1], sensor_freq, [15, sensor_freq//2], window_sec=None, relative=False)
    bp_z_high = bandpower(data[:,2], sensor_freq, [15, sensor_freq//2], window_sec=None, relative=False)

    bp_x_ratio = bp_x_high/bp_x_low
    bp_y_ratio = bp_y_high/bp_y_low
    bp_z_ratio = bp_z_high/bp_z_low
    cost = np.mean([bp_x_ratio, bp_y_ratio, bp_z_ratio])

    # print("==========")
    # print("Cost function i: Ratio of power above 15 hz to power below 15 hz in all axes")
    # print(f"Bandpower ratio in x: {bp_x_ratio}")
    # print(f"Bandpower in y: {bp_y_ratio}")
    # print(f"Bandpower in z: {bp_z_ratio}")
    # print(f"Sum of bandpowers: {cost}")
    # print("==========")
    return cost

def cost_function_j(data, sensor_freq):
    '''Ratio of power above 15 hz to power below 15 hz in Z axis

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''
    bp_z_low = bandpower(data, sensor_freq, [3, 15], window_sec=None, relative=False)

    bp_z_high = bandpower(data, sensor_freq, [15, sensor_freq//2], window_sec=None, relative=False)

    bp_z_ratio = bp_z_high/bp_z_low
    cost = bp_z_ratio

    # print("==========")
    # print("Cost function j: Ratio of power above 15 hz to power below 15 hz in Z axis")
    # print(f"Ratio of bandpowers: {cost}")
    # print("==========")
    return cost

def cost_function_k(data, sensor_freq):
    '''Bandpower in 5-8 Hz range in Z

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq: 
            Frequency of the recorded signal
        - min_freq:
            Minimum frequency band to be included
        - max_freq:
            Maximum frequency band to be included
    '''
    bp_z = bandpower(data, sensor_freq, [5, 8], window_sec=None, relative=False)

    cost = bp_z

    # print("==========")
    # print("Cost function k: Bandpower in 3-8 Hz range in Z")
    # print(f"Ratio of bandpowers: {cost}")
    # print("==========")
    return cost

def main():
    dataset_dir = '/home/mateo/Data/SARA/TartanCost'
    trajectories_dir = os.path.join(dataset_dir, "Trajectories")
    annotations_dir  = os.path.join(dataset_dir, "Annotations")
    dir_names = sorted(os.listdir(trajectories_dir))

    VISUALIZE = True
    
    header = ['trajectory', 'avg_speed', 'avg_score', 'cost_a', 'cost_b', 'cost_c', 'cost_d', 'cost_e', 'cost_f', 'cost_g', 'cost_h', 'cost_i', 'cost_j', 'cost_k']
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

        cost_a_list = []
        cost_b_list = []
        cost_c_list = []
        cost_d_list = []
        cost_e_list = []
        cost_f_list = []
        cost_g_list = []
        cost_h_list = []
        cost_i_list = []
        cost_j_list = []
        cost_k_list = []

        avg_speed = annotation["average_speed"]

        avg_score = annotation["human_scores"]["average"]


        # import pdb;pdb.set_trace()
        for i in range(imu_data.shape[0]):
            imu_buffer.insert(imu_data[i][-3:])

            # data_x = imu_buffer.get_data()[:,0]
            # f_x, Pxx_x = psd(data_x, imu_freq)
            # data_y = imu_buffer.get_data()[:,1]
            # f_y, Pxx_y = psd(data_y, imu_freq)
            # data_z = imu_buffer.get_data()[:,2]
            # f_z, Pxx_z = psd(data_z, imu_freq)
            # img_viewer.clear()
            # imu_viewer.clear()
            # freq_viewer.clear()
            # cost_viewer.clear()

            # img_viewer.imshow(img_data[i//imu_per_img])

            # imu_viewer.plot(time, imu_buffer.get_data()[:, 0], color='red', label="X axis")
            # imu_viewer.plot(time, imu_buffer.get_data()[:, 1], color='green', label="Y axis")
            # imu_viewer.plot(time, imu_buffer.get_data()[:, 2], color='blue', label="Z axis")
            # imu_viewer.set_ylim([-10, 20])
            # imu_viewer.grid()
            # imu_viewer.legend(loc="upper left")
            # imu_viewer.set_title("IMU Linear Acceleration")
            # imu_viewer.set_xlabel("Time step [n]")
            # imu_viewer.set_ylabel("Linear acceleration [m/s^2]")
            # freq_viewer.semilogy(f_x, Pxx_x, c='r', label="X axis")
            # freq_viewer.semilogy(f_y, Pxx_y, c='g', label="Y axis")
            # freq_viewer.semilogy(f_z, Pxx_z, c='b', label="Z axis")
            # freq_viewer.set_ylim([1e-7, 1e2])
            # freq_viewer.grid()
            # freq_viewer.legend()
            # freq_viewer.set_xlabel('Frequency [Hz]')
            # freq_viewer.set_ylabel('PSD [V^2/Hz]')

            # widths = np.arange(1,31)
            # cwtmatr_x = scipy.signal.cwt(data_x, scipy.signal.ricker, widths)
            # cwtmatr_y = scipy.signal.cwt(data_y, scipy.signal.ricker, widths)
            # cwtmatr_z = scipy.signal.cwt(data_z, scipy.signal.ricker, widths)
            # print(cwtmatr_x.shape)
            # print(cwtmatr_x)
            # print(abs(cwtmatr_x).max())
            # max_val = 30
            # freq_viewer.imshow(cwtmatr_x, cmap='coolwarm', aspect='auto', vmax=max_val, vmin=-max_val)
            # freq_viewer.set_title("Continuous Wavelet Transform, Ricker wavelet")
            # freq_viewer.set_xlabel("Time step [n]")
            # freq_viewer.set_ylabel("Frequency [Hz]")

            cost_a_list.append(cost_function_a(imu_buffer.get_data()[:,0:3], imu_freq))
            cost_b_list.append(cost_function_b(imu_buffer.get_data()[:,2], imu_freq))
            cost_c_list.append(cost_function_c(imu_buffer.get_data()[:,0:3], imu_freq, num_bins=5))
            cost_d_list.append(cost_function_d(imu_buffer.get_data()[:,2], imu_freq, num_bins=5))
            cost_e_list.append(cost_function_e(imu_buffer.get_data()[:,0:3], imu_freq))
            cost_f_list.append(cost_function_f(imu_buffer.get_data()[:,0:3], imu_freq))
            cost_g_list.append(cost_function_g(imu_buffer.get_data()[:,0:3], imu_freq, num_bins=5))
            cost_h_list.append(cost_function_h(imu_buffer.get_data()[:,2], imu_freq, num_bins=5))
            cost_i_list.append(cost_function_i(imu_buffer.get_data()[:,0:3], imu_freq))
            cost_j_list.append(cost_function_j(imu_buffer.get_data()[:,2], imu_freq))
            cost_k_list.append(cost_function_k(imu_buffer.get_data()[:,2], imu_freq))

            # cost_buffer.insert(cost)
            # cost_viewer.plot(time, cost_buffer.get_data(), c="b")
            # # cost_viewer.set_ylim([0, 20])
            # cost_viewer.grid()

            # plt.pause(0.01)
        
        cost_a = np.mean(cost_a_list)
        cost_b = np.mean(cost_b_list)
        cost_c = np.mean(cost_c_list)
        cost_d = np.mean(cost_d_list)
        cost_e = np.mean(cost_e_list)
        cost_f = np.mean(cost_f_list)
        cost_g = np.mean(cost_g_list)
        cost_h = np.mean(cost_h_list)
        cost_i = np.mean(cost_i_list)
        cost_j = np.mean(cost_j_list)
        cost_k = np.mean(cost_k_list)

        csv_data.append([dir, avg_speed, avg_score, cost_a, cost_b, cost_c, cost_d, cost_e, cost_f, cost_g, cost_h, cost_i, cost_j, cost_k])


        # import pdb;pdb.set_trace()
        # Run cost function with each window of data

    # Visualize window of data and cost function
    with open('/home/mateo/Data/SARA/TartanCost/cost_functions.csv', 'w', encoding='UTF8', newline='') as f:
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