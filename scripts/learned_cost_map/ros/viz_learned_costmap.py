#!/usr/bin/python3
import rospy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2

from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
import time

class CostmapVizNode(object):
    def __init__(self, map_config):
        self.cvbridge = CvBridge()

        rospy.Subscriber('/learned_costmap', OccupancyGrid, self.handle_learned_costmap, queue_size=10)
        rospy.Subscriber('/warthog5/stereo_left/image_rect_color/compressed', CompressedImage, self.handle_front_facing, queue_size=10)
        rospy.Subscriber('/local_rgb_map_inflate', Image, self.handle_rgb_inflate, queue_size=10)
        rospy.Subscriber('/warthog5/odom', Odometry, self.handle_odom, queue_size=10)
        self.front_facing = None
        self.learned_costmap = None
        self.rgbmap_inflate = None
        self.vel = None


        # Define map metadata so that we know how many cells we need to query to produce costmap
        with open(map_config, "r") as file:
            map_info = yaml.safe_load(file)
        self.map_metadata = map_info["map_metadata"]
        self.crop_params = map_info["crop_params"]

        # We will take the header of the rgbmap to populate the header of the output occupancy grid
        self.header = None 

        self.costmap_pub = rospy.Publisher('/learned_costmap', OccupancyGrid, queue_size=1, latch=False)
        self.costmap_img_pub = rospy.Publisher('learned_costmap_img', Image, queue_size=1)


    def handle_rgb_inflate(self, msg):
        self.rgbmap_inflate = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")
        print("Obtained RGB map")

    def handle_front_facing(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np , cv2.COLOR_BGR2RGB)
        self.front_facing = image_np
        print("Obtained Front Facing")

    def handle_learned_costmap(self, msg):
        print("Obtained costmap")
        resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        self.learned_costmap = np.reshape(np.array(list(msg.data))/100.0, (width, height)).T
        
        # Convert to color image
        cm = plt.get_cmap('plasma')
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

        save_dir = "/home/mateo/icra_plots"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        learned_costmap_parent_dir = os.path.join(save_dir, "learned_costmaps")
        front_facing_parent_dir = os.path.join(save_dir, "front_facing")
        rgb_map_parent_dir = os.path.join(save_dir, "rgb_maps") 
        if not os.path.exists(learned_costmap_parent_dir):
            os.makedirs(learned_costmap_parent_dir)
        if not os.path.exists(front_facing_parent_dir):
            os.makedirs(front_facing_parent_dir)
        if not os.path.exists(rgb_map_parent_dir):
            os.makedirs(rgb_map_parent_dir)

        # divider = make_axes_locatable(axs)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(self.learned_costmap, cax=cax, orientation='vertical')
        learned_costmap_dir = os.path.join(learned_costmap_parent_dir, f"frame_{count:08}.png")
        front_facing_dir = os.path.join(front_facing_parent_dir, f"frame_{count:08}.png")
        rgb_map_dir = os.path.join(rgb_map_parent_dir, f"frame_{count:08}.png")

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

    map_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/map_params.yaml"

    node = CostmapVizNode(map_config)
    r = rospy.Rate(10)

    count = 0
    while not rospy.is_shutdown(): # loop just for visualization
        if (node.learned_costmap is not None) and (node.rgbmap_inflate is not None) and (node.front_facing is not None):
            print(f"Saving image {count}.")
            node.save_costmap_figs(count)
            count += 1
            r.sleep()
        else:
            print(f"learned_costmap is None: {node.learned_costmap is None}")
            print(f"rgbmap_inflate is None: {node.rgbmap_inflate is None}")
            print(f"front_facing is None: {node.front_facing is None}")
            r.sleep()