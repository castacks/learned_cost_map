#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from learned_cost_map.msg import FloatStamped, BoolStamped
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

def badgr_bumpiness_score(data):
    '''Average magnitude of angular velocities

    Args:
        - data:
            Input signal to be analyzed: In this case, it is a numpy 3D vector containing angular velocities.
    '''
    cost = np.linalg.norm(data)

    return cost

def lin_acc_mag_xyz_score(data):
    '''Average magnitude of linear acceleration in the z axis

    Args:
        - data:
            Input signal to be analyzed: In this case, it is a numpy 3D vector containing linear acceleration in the xyz axes.
    '''
    cost = np.linalg.norm(data)

    return cost

def lin_acc_mag_z_score(data):
    '''Average magnitude of linear acceleration in the z axis

    Args:
        - data:
            Input signal to be analyzed: In this case, it is a numpy 1D vector containing linear acceleration in the z axis.
    '''
    cost = np.linalg.norm(data)

    return cost



class SmoothnessRecorder(object):
    def __init__(self, cost_stats_dir):
        
        # Set up subscribers
        rospy.Subscriber('/novatel/imu/data', Imu, self.handle_imu, queue_size=1)
        rospy.Subscriber('/mux/intervention', BoolStamped, self.handle_intervention, queue_size=1)

        # Set up publishers
        self.cost = None
        self.cost_publisher = rospy.Publisher('/smoothness_cost', FloatStamped, queue_size=10)

        # Set data buffer
        pad_val = Imu()
        pad_val.linear_acceleration.z = 9.81
        self.imu_freq = 100
        self.buffer_size = int(1*self.imu_freq)  # num_seconds*imu_freq
        self.buffer = Buffer(self.buffer_size, padded=True, pad_val=pad_val.linear_acceleration.z)

        # Load stats for different cost functions:
        self.cost_stats_dir = cost_stats_dir
        
        with open(cost_stats_dir, 'r') as f:
            self.all_costs_stats = yaml.safe_load(f)
        # Information about sensor and sensor frequency, Min and max frequencies set the band to be analyzed for the cost function.
        # self.cost_name = "freq_bins_5"
        self.cost_name = "freq_band_1_30"
        self.cost_stats = self.all_costs_stats[self.cost_name]
        self.sensor_name = "imu_z"
        self.sensor_freq = 100
        self.min_freq = 1
        self.max_freq = 30
        # self.num_bins = 5

        self.all_smoothness_values = []
        self.all_badgr_values = []
        self.all_linear_acc_z_magnitudes = []
        self.all_linear_acc_xyz_magnitudes = []
        self.all_intervention_flags = []

        self.currently_intervening = True

    def handle_intervention(self, msg):
        self.currently_intervening = bool(msg.data)

        self.all_intervention_flags.append(self.currently_intervening)


    def handle_imu(self, msg):
        print("-----")
        print("Received IMU message")
        self.buffer.insert(msg.linear_acceleration.z)
        # cost = cost_function(self.buffer.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=None, num_bins=self.num_bins)
        cost = cost_function(self.buffer.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[self.min_freq, self.max_freq], num_bins=None)

        linear_accelerations = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        angular_velocities = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        badgr_cost = badgr_bumpiness_score(angular_velocities)
        lin_acc_z_mag = lin_acc_mag_z_score(np.array([msg.linear_acceleration.z]))
        lin_acc_xyz_mag = lin_acc_mag_xyz_score(linear_accelerations)

        if not self.currently_intervening :
            self.all_smoothness_values.append(cost)
            self.all_badgr_values.append(badgr_cost)
            self.all_linear_acc_z_magnitudes.append(lin_acc_z_mag)
            self.all_linear_acc_xyz_magnitudes.append(lin_acc_xyz_mag)

            print(f"PSD cost: {cost}")
            print(f"BADGR cost: {badgr_cost}")
            print(f"Lin acc z cost: {lin_acc_z_mag}")
            print(f"Lin acc xyz cost: {lin_acc_xyz_mag}")

        else:
            print(f"Currently intervening")
            smoothness_array = np.array(self.all_smoothness_values)
            print(f"Final avg cost (so far): {np.mean(smoothness_array)}")
            print(f"--Final std cost (so far): {np.std(smoothness_array)}") 
            print(f"Final avg cost, nonzero (so far): {np.mean(smoothness_array[np.nonzero(smoothness_array)])}")
            print(f"--Final std cost, nonzero (so far): {np.std(smoothness_array[np.nonzero(smoothness_array)])}")
            print(f"Final avg badgr cost (so far): {np.mean(np.array(self.all_badgr_values))}")
            print(f"--Final std badgr cost (so far): {np.std(np.array(self.all_badgr_values))}") 
            print(f"Final avg z lin acc mag cost (so far): {np.mean(np.array(self.all_linear_acc_z_magnitudes))}")
            print(f"--Final std z lin acc mag cost (so far): {np.std(np.array(self.all_linear_acc_z_magnitudes))}")
            print(f"Final avg xyz lin acc mag cost (so far): {np.mean(np.array(self.all_linear_acc_xyz_magnitudes))}")
            print(f"Final std xyz lin acc mag cost (so far): {np.std(np.array(self.all_linear_acc_xyz_magnitudes))}")

        # print(f"Publishing cost: {cost}")
        # cost_msg = FloatStamped()
        # cost_msg.header = msg.header
        # cost_msg.data = cost
        # self.cost_publisher.publish(cost_msg)
        # print("Published cost!")


if __name__ == "__main__":
    rospy.init_node("autonomous_smoothness_node", log_level=rospy.INFO)
    rospy.loginfo("Initialized autonomous_smoothness_node node")
    cost_stats_dir = rospy.get_param("~cost_stats_dir")
    node = SmoothnessRecorder(cost_stats_dir)
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():

        rate.sleep()
