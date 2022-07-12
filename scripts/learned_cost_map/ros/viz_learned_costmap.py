#!/usr/bin/python3
import rospy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
import time

class CostmapVizNode(object):
    def __init__(self):
        self.cvbridge = CvBridge()

        rospy.Subscriber('/learned_costmap', OccupancyGrid, self.handle_learned_costmap, queue_size=1)
        rospy.Subscriber('/multisense/left/image_rect_color', Image, self.handle_front_facing, queue_size=1)
        rospy.Subscriber('/local_rgb_map_inflate', Image, self.handle_rgb_inflate, queue_size=1)
        rospy.Subscriber('/odometry/filtered_odom', Odometry, self.handle_odom, queue_size=1)
        self.front_facing = None
        self.learned_costmap = None
        self.rgbmap_inflate = None
        self.vel = None


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

        # We will take the header of the rgbmap to populate the header of the output occupancy grid
        self.header = None 

        self.costmap_pub = rospy.Publisher('/learned_costmap', OccupancyGrid, queue_size=1, latch=False)
        self.costmap_img_pub = rospy.Publisher('learned_costmap_img', Image, queue_size=1)


    def handle_rgb_inflate(self, msg):
        self.rgbmap_inflate = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")

    def handle_front_facing(self, msg):
        self.front_facing = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")

    def handle_learned_costmap(self, msg):
        resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        self.learned_costmap = np.reshape(np.array(list(msg.data))/100.0, (width, height)).T
        
        # Convert to color image
        cm = plt.get_cmap('viridis')
        self.learned_costmap = cm(self.learned_costmap)
        # print(self.learned_costmap)
    
    def handle_odom(self, msg):
        vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        vel_z = msg.twist.twist.linear.z

        self.vel = float(np.linalg.norm([vel_x, vel_y, vel_z]))
        print(f"Velocity is {self.vel:.2f}")

    def save_costmap_figs(self, count):
        if self.learned_costmap is None:
            print("Haven't obtained learned costmap")
            return

        # axs.clear()

        # divider = make_axes_locatable(axs)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(self.learned_costmap, cax=cax, orientation='vertical')
        learned_costmap_dir = f"/home/mateo/corl_plots/data_collect1/learned_costmaps/frame_{count:08}.png"
        front_facing_dir = f"/home/mateo/corl_plots/data_collect1/front_facing/frame_{count:08}.png"
        rgb_map_dir = f"/home/mateo/corl_plots/data_collect1/rgb_maps/frame_{count:08}.png"

        plt.clf()
        plt.title("Learned Costmap")
        plt.imshow(self.learned_costmap, origin="lower", vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.savefig(learned_costmap_dir, dpi=300, bbox_inches="tight")
        plt.clf()


        plt.title("Front-Facing Image")
        plt.imshow(self.front_facing)
        plt.axis('off')
        plt.savefig(front_facing_dir, dpi=300, bbox_inches="tight")
        plt.clf()

        plt.title("RGB Map")
        plt.imshow(self.rgbmap_inflate, origin="lower")
        plt.colorbar()
        plt.savefig(rgb_map_dir, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == '__main__':

    rospy.init_node("learned_costmap_visualizer", log_level=rospy.INFO)

    rospy.loginfo("learned_costmap_visualizer initialized")

    node = CostmapVizNode()
    r = rospy.Rate(10)

    count = 0
    while not rospy.is_shutdown(): # loop just for visualization
        if node.learned_costmap is not None:
            # print(f"Saving image {count}.")
            # node.save_costmap_figs(count)
            # count += 1
            r.sleep()

