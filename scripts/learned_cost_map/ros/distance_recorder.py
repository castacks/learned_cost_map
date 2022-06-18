#!/usr/bin/python3
import rospy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from std_msgs.msg import Header, Float32
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
import time

class PathLengthRecorder(object):
    def __init__(self):
        self.cvbridge = CvBridge()

        rospy.Subscriber('/learned_costmap', OccupancyGrid, self.handle_learned_costmap, queue_size=1)

        rospy.Subscriber('/odometry/filtered_odom', Odometry, self.handle_odom, queue_size=1)
        self.learned_costmap = None
        self.last_odom = None
        self.distance_traveled = 0.0
        self.vel = None
        self.start_measuring = False
        self.stop_measuring = False
        self.num_poses = 0
        self.total_traversability_cost = 0.0
        self.total_costmap_energy = 0.0

        # Define map metadata so that we know how many cells we need to query to produce costmap
        map_height = 12.0 # [m]
        map_width  = 12.0 # [m]
        resolution = 0.04
        origin     = [-2.0, -6.0]
        self.map_metadata = {
            'height': map_height,
            'width': map_width,
            'resolution': resolution,
            'origin': origin
        }

        crop_width = 2.0  # in meters
        crop_size = [crop_width, crop_width]
        output_size = [64, 64]

        self.crop_params ={
            'crop_size': crop_size,
            'output_size': output_size
        }

        self.resolution = None
        self.origin_x = None
        self.origin_y = None

        # We will take the header of the rgbmap to populate the header of the output occupancy grid
        self.header = None 

        self.distance_pub = rospy.Publisher('/distance_traveled', Float32, )


    def handle_learned_costmap(self, msg):
        self.resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.learned_costmap = np.reshape(np.array(list(msg.data))/100.0, (width, height))

        # plt.imshow(self.learned_costmap)
        # plt.scatter(int((0-origin_x)/resolution), int((0-origin_y)/resolution), c='red')
        # plt.scatter(int((1-origin_x)/resolution), int((0-origin_y)/resolution), c='green')
        # plt.show()
        # Convert to color image
        # cm = plt.get_cmap('viridis')
        # self.learned_costmap = cm(self.learned_costmap)
        # print(self.learned_costmap)
    
    def handle_odom(self, msg):

        vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        vel_z = msg.twist.twist.linear.z

        if self.last_odom is None:
            self.last_odom = msg
            print(f"Distance traveled: {0.0}")
        else:
            dx = msg.pose.pose.position.x - self.last_odom.pose.pose.position.x
            dy = msg.pose.pose.position.y - self.last_odom.pose.pose.position.y
            d_dist = np.sqrt(dx**2 + dy**2)
            self.distance_traveled += d_dist
            print(f"Distance traveled: {self.distance_traveled}")
            self.last_odom = msg

        self.vel = float(np.linalg.norm([vel_x, vel_y, vel_z]))
        print(f"Velocity is {self.vel:.2f}")

        print(f"Start: {self.start_measuring}, stop: {self.stop_measuring}")
        if self.start_measuring and (not self.stop_measuring):
            self.num_poses += 1
            self.total_traversability_cost += self.learned_costmap[int((1-self.origin_x)/self.resolution), int((0-self.origin_y)/self.resolution)]
            self.total_costmap_energy += np.mean(self.learned_costmap)
            print(f"Number of poses: {self.num_poses}")
            print(f"Total traversability cost: {self.total_traversability_cost}")
            print(f"Total energy: {self.total_costmap_energy}")

        if self.vel > 0.1:
            self.start_measuring = True
        if self.distance_traveled >= 200:
            self.stop_measuring = True


if __name__ == '__main__':

    rospy.init_node("learned_costmap_visualizer", log_level=rospy.INFO)

    rospy.loginfo("learned_costmap_visualizer initialized")

    node = PathLengthRecorder()
    r = rospy.Rate(10)

    count = 0
    while not rospy.is_shutdown(): # loop just for visualization
        if node.learned_costmap is not None:
            # print(f"Saving image {count}.")
            # node.save_costmap_figs(count)
            # count += 1
            r.sleep()