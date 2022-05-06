#!/usr/bin/env python
import cv2
import numpy as np

import rospy
import ros_numpy
import torch

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header

import time

from learned_cost_map.utils.costmap_utils import produce_costmap, rosmsgs_to_maps
from learned_cost_map.trainer.model import CostModel
# from learned_cost_map.trainer.utils import get_dataloaders


class CostmapNode(object):
    def __init__(self, saved_model):
        self.cvbridge = CvBridge()

        rospy.Subscriber('/local_height_map_inflate', Image, self.handle_height_inflate, queue_size=1)
        rospy.Subscriber('/local_rgb_map_inflate', Image, self.handle_rgb_inflate, queue_size=1)
        self.heightmap_inflate = None
        self.rgbmap_inflate = None

        # Load trained model to produce costmaps
        self.model = CostModel(input_channels=8, output_size=1).cuda()
        self.model.load_state_dict(torch.load(saved_model))
        self.model.eval()

        # Define map metadata so that we know how many cells we need to query to produce costmap
        map_height = 12.0 # [m]
        map_width  = 12.0 # [m]
        resolution = 0.02
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


    def handle_height(self, msg):
        self.heightmap = self.cvbridge.imgmsg_to_cv2(msg, "32FC4")
        print('Receive heightmap {}'.format(self.heightmap.shape))
        # import ipdb;ipdb.set_trace()

    def handle_rgb(self, msg):
        self.rgbmap = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")
        print('Receive rgbmap {}'.format(self.rgbmap.shape))

    def handle_height_inflate(self, msg):
        self.heightmap_inflate = self.cvbridge.imgmsg_to_cv2(msg, "32FC4")
        # print('Receive heightmap {}'.format(self.heightmap_inflate.shape))
        # import ipdb;ipdb.set_trace()

    def handle_rgb_inflate(self, msg):
        self.rgbmap_inflate = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")
        # print('Receive rgbmap {}'.format(self.rgbmap_inflate.shape))

    def publish_costmap(self):
        import pdb;pdb.set_trace()
        maps = rosmsgs_to_maps(self.rgbmap_inflate, self.heightmap_inflate)
        costmap = produce_costmap(self.model, maps, self.map_metadata, self.crop_params)



if __name__ == '__main__':

    rospy.init_node("learned_costmap_node", log_level=rospy.INFO)

    rospy.loginfo("learned_costmap_node initialized")
    saved_model = "/home/mateo/learned_cost_map/scripts/learned_cost_map/trainer/models/epoch_19.pt"
    node = CostmapNode(saved_model)
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): # loop just for visualization
        node.publish_costmap()


        


