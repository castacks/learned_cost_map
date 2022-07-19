#!/usr/bin/python3
import rospy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
import time

from learned_cost_map.utils.costmap_utils import produce_costmap, rosmsgs_to_maps
from learned_cost_map.trainer.model import CostModel, CostVelModel, CostFourierVelModel, CostFourierVelModelEfficientNet, CostModelEfficientNet, CostFourierVelModelSmall, CostFourierVelModelRGB


class CostmapNode(object):
    def __init__(self, model_name, saved_model, saved_freqs, map_config, costmap_config, height_map_topic, rgb_map_topic, odometry_topic):
        self.cvbridge = CvBridge()

        rospy.Subscriber(height_map_topic, Image, self.handle_height_inflate, queue_size=1)
        rospy.Subscriber(rgb_map_topic, Image, self.handle_rgb_inflate, queue_size=1)
        rospy.Subscriber(odometry_topic, Odometry, self.handle_odom, queue_size=1)
        self.heightmap_inflate = None
        self.rgbmap_inflate = None
        self.vel = None

        # Load trained model to produce costmaps
        # self.fourier_freqs = None
        # if model_name=="CostModel":
        #     self.model = CostModel(input_channels=8, output_size=1)
        # elif model_name=="CostVelModel":
        #     self.model = CostVelModel(input_channels=8, embedding_size=512, output_size=1)
        # elif model_name=="CostFourierVelModel":
        #     self.model = CostFourierVelModel(input_channels=8, ff_size=16, embedding_size=512, output_size=1)
        #     self.fourier_freqs = torch.load(saved_freqs)
        # elif model_name=="CostFourierVelModelEfficientNet":
        #     model = CostFourierVelModelEfficientNet(input_channels=8, ff_size=16, embedding_size=512, output_size=1)
        #     self.fourier_freqs = torch.load(saved_freqs)
        # else:
        #     raise NotImplementedError()

        pretrained=False
        embedding_size=512
        mlp_size=512
        self.fourier_freqs = None
        if model_name=="CostModel":
            self.model = CostModel(input_channels=8, output_size=1)
        elif model_name=="CostVelModel":
            self.model = CostVelModel(input_channels=8, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1)
        elif model_name=="CostFourierVelModel":
            self.model = CostFourierVelModel(input_channels=8, ff_size=16, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1, pretrained=pretrained)
            self.fourier_freqs = torch.load(saved_freqs)
        elif model_name=="CostModelEfficientNet":
            self.model = CostModelEfficientNet(input_channels=8, output_size=1)
        elif model_name=="CostFourierVelModelEfficientNet":
            self.model = CostFourierVelModelEfficientNet(input_channels=8, ff_size=16, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1)
            self.fourier_freqs = torch.load(saved_freqs)
        elif model_name=="CostFourierVelModelSmall":
            self.model = CostFourierVelModelSmall(input_channels=8, ff_size=16, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1)
            self.fourier_freqs = torch.load(saved_freqs)
        elif model_name=="CostFourierVelModelRGB":
            self.model = CostFourierVelModelRGB(input_channels=3, ff_size=16, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1)
            self.fourier_freqs = torch.load(saved_freqs)
        else:
            raise NotImplementedError()

        # model = CostModel(input_channels=8, output_size=1).cuda()
        self.model.load_state_dict(torch.load(saved_model))
        self.model.cuda()
        self.model.eval()


        # Define map metadata so that we know how many cells we need to query to produce costmap
        with open(map_config, "r") as file:
            map_info = yaml.safe_load(file)
        self.map_metadata = map_info["map_metadata"]
        self.crop_params = map_info["crop_params"]

        with open(costmap_config, "r") as file:
            costmap_params = yaml.safe_load(file)
        self.costmap_batch_size = costmap_params["batch_size"]
        self.costmap_stride = costmap_params["stride"]

        # We will take the header of the rgbmap to populate the header of the output occupancy grid
        self.header = None 

        self.costmap_pub = rospy.Publisher('/learned_costmap', OccupancyGrid, queue_size=1, latch=False)
        self.costmap_img_pub = rospy.Publisher('learned_costmap_img', Image, queue_size=1)


    def handle_height(self, msg):
        self.heightmap = self.cvbridge.imgmsg_to_cv2(msg, "32FC4")
        print('Receive heightmap {}'.format(self.heightmap.shape))
        # import ipdb;ipdb.set_trace()

    def handle_rgb(self, msg):
        self.rgbmap = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")
        print('Receive rgbmap {}'.format(self.rgbmap.shape))

    def handle_height_inflate(self, msg):
        self.heightmap_inflate = self.cvbridge.imgmsg_to_cv2(msg, "32FC4")
        self.header = msg.header
        # print('Receive heightmap {}'.format(self.heightmap_inflate.shape))
        # import ipdb;ipdb.set_trace()

    def handle_rgb_inflate(self, msg):
        self.rgbmap_inflate = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")
        # print('Receive rgbmap {}'.format(self.rgbmap_inflate.shape))

    def handle_odom(self, msg):
        vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        vel_z = msg.twist.twist.linear.z

        self.vel = float(np.linalg.norm([vel_x, vel_y, vel_z]))

    def publish_costmap(self):
        # import pdb;pdb.set_trace()
        if (self.rgbmap_inflate is None) or (self.heightmap_inflate is None) or (self.vel is None):
            print("Maps and vel not available yet. Check topic names.")
            return 
        maps = rosmsgs_to_maps(self.rgbmap_inflate, self.heightmap_inflate)
        before = time.time()
        # import pdb;pdb.set_trace()
        costmap = produce_costmap(self.model, maps, self.map_metadata, self.crop_params, costmap_batch_size=self.costmap_batch_size, costmap_stride=self.costmap_stride, vel=self.vel, fourier_freqs=self.fourier_freqs)
        print(f"Takes {time.time()-before} seconds to produce a costmap")

        costmap_img = Image()
        costmap_img.header = self.header
        costmap_img.height = int(costmap.shape[0])
        costmap_img.width = int(costmap.shape[1])
        costmap_img.encoding = "mono8"
        costmap_img.data = (costmap*100).astype(np.uint8).flatten().tolist()

        costmap_grid = OccupancyGrid()
        costmap_grid.header = self.header
        costmap_grid.info.map_load_time = costmap_grid.header.stamp
        costmap_grid.info.resolution = self.map_metadata['resolution']
        costmap_grid.info.width = int(self.map_metadata['width']*1/self.map_metadata['resolution'])
        costmap_grid.info.height = int(self.map_metadata['height']*1/self.map_metadata['resolution'])
        costmap_grid.info.origin.position.x = self.map_metadata['origin'][0]
        costmap_grid.info.origin.position.y = self.map_metadata['origin'][1]
        costmap_grid.info.origin.orientation.x = 0.0
        costmap_grid.info.origin.orientation.y = 0.0
        costmap_grid.info.origin.orientation.z = 0.0
        costmap_grid.info.origin.orientation.w = 1.0

        costmap_grid.data = (costmap.T*100).astype(np.int8).flatten().tolist()

        self.costmap_pub.publish(costmap_grid)
        self.costmap_img_pub.publish(costmap_img)


if __name__ == '__main__':

    rospy.init_node("learned_costmap_node", log_level=rospy.INFO)

    rospy.loginfo("learned_costmap_node initialized")
    model_dir = rospy.get_param("~model_dir")
    model_name = rospy.get_param("~model_name")
    map_config = rospy.get_param("~map_config")
    costmap_config = rospy.get_param("~costmap_config")
    height_map_topic = rospy.get_param("~height_map_topic")
    rgb_map_topic = rospy.get_param("~rgb_map_topic")
    odometry_topic = rospy.get_param("~odometry_topic")
    if (model_dir is None) or (model_name is None) or (map_config is None):
        raise NotImplementedError()
    saved_model = os.path.join(model_dir, 'epoch_50.pt')
    saved_freqs = os.path.join(model_dir, 'fourier_freqs.pt')
    node = CostmapNode(model_name, saved_model, saved_freqs, map_config, costmap_config)
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): # loop just for visualization
        node.publish_costmap()