#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
import numpy as np


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

def cost_function(data, sensor_freq, cost_name, cost_stats, range=None, num_bins=None):
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
        assert range is not None, "range should not be None"
        bp_z = bandpower(data, sensor_freq, range, window_sec=None, relative=False)

        cost = bp_z

    else:
        raise NotImplementedError("cost_name needs to include bins or band")

    return cost

class TraversabilityCostNode(object):
    def __init__(self):
        
        # Set up subscribers
        rospy.Subscriber('/novatel/imu/data', Imu, self.handle_imu, queue_size=1)

        # Set up publishers
        self.cost = None
        self.cost_publisher = rospy.Publisher('/traversability_cost', Float32, queue_size=10)

        # Set data buffer
        pad_val = Imu()
        self.buffer = Buffer(250, padded=True, pad_val=pad_val.linear_acceleration.z)

        # Information about sensor and sensor frequency, Min and max frequencies set the band to be analyzed for the cost function.
        self.sensor_name = "imu_z"
        self.sensor_freq = 125
        self.min_freq = 5
        self.max_freq = 8

    def handle_imu(self, msg):
        self.buffer.insert(msg.linear_acceleration.z)
        cost = CostFunction(self.buffer.data, self.sensor_freq, min_freq=self.min_freq, max_freq=self.max_freq, sensor_name=self.sensor_name)

        self.cost_publisher.publish(cost)


if __name__ == "__main__":
    rospy.init_node("traversability_cost_node", log_level=rospy.INFO)
    rospy.loginfo("Initialized traversability_cost node")

    node = TraversabilityCostNode()
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        rate.sleep()