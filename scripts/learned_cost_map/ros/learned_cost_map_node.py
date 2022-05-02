#!/usr/bin/env python3
import rospy
import rospkg
import torch
import os
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from learned_cost_map.knn import KNNRegression
from learned_cost_map.terrain_map import TerrainMap
from learned_cost_map.extract_patch import PatchTransform
from learned_cost_map.feature_extractor import FeatureExtractor

from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge

# Learned cost map node

class LearnedCostMapNode:
    '''Class that produces a costmap from a current local map (RGB + Height maps) observation.

    Args:
        - model_fp:
            Filepath to the trained model that will be used to generate the cost map
        - device:
            Device that will be used to run inference. 
    '''
    def __init__(self, model_fp, device="cpu"):

        self.device = device

        self.bridge = CvBridge()
        self.terrain_map = None

        self.got_rgb_map = False
        self.got_height_map_low = False
        self.got_height_map_high = False

        map_height = 10.0
        map_width = 10.0
        resolution = 0.05

        crop_width = 2  # in meters
        crop_size = [crop_width, crop_width]
        output_size = [244, 244]

         # TODO. Make sure the two dicts below are populated using rosparams
        self.map_metadata = {
            'height': map_height,
            'width': map_width,
            'resolution': resolution
        }

        self.crop_params ={
            'crop_size': crop_size,
            'output_size': output_size
        }
        
        ## Set up subscribers
        input_height_map_low_topic = rospy.get_param('~input_height_map_low_topic', '/local_height_map_low')
        input_height_map_high_topic = rospy.get_param('~input_height_map_high_topic', '/local_height_map_high')
        input_rgb_map_topic = rospy.get_param('~input_rgb_map_topic', '/local_rgb_map_inflate')


        height_map_low_sub = rospy.Subscriber(input_height_map_low_topic, Image, callback=self.height_map_low_cb, queue_size=100)
        height_map_high_sub = rospy.Subscriber(input_height_map_high_topic, Image, callback=self.height_map_high_cb, queue_size=100)
        rgb_map_sub = rospy.Subscriber(input_rgb_map_topic, Image, callback=self.rgb_map_cb, queue_size=100)

        ## Set up publishers
        self.cost_map_publisher = rospy.Publisher('/learned_cost_map', Image, queue_size=10)


        ## Load model
        model = torch.load(model_fp)
        features = model['features']
        costs = model['costs']

        ## Build kNN
        feature_dims = features.shape[1]
        max_datapoints = features.shape[0]
        K = 3
        sigma = 1.0
        self.knn = KNNRegression(feature_dims, max_datapoints, K, sigma=sigma, device=self.device)

        self.knn.insert(features, costs)
        self.knn.to(self.device)
        print("kNN built! Ready to produce costmaps")

    def height_map_low_cb(self, msg):
        '''height_map_low callback'''
        self.height_map_low = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.height_map_low_tensor = torch.unsqueeze(torch.Tensor(np.copy(self.height_map_low)), 0).permute(0,2,1)
        self.got_height_map_low = True

    def height_map_high_cb(self, msg):
        '''height_map_high callback'''
        self.height_map_high = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.height_map_high_tensor = torch.unsqueeze(torch.Tensor(np.copy(self.height_map_high)),0).permute(0,2,1)
        self.got_height_map_high = True

    def rgb_map_cb(self, msg):
        '''rgb_map callback'''
        self.rgb_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.rgb_map_tensor = T.ToTensor()(np.copy(self.rgb_map)).permute(0,2,1)
        self.got_rgb_map = True

    def publish_cost_map(self):
        '''Publishes a costmap based on input local map'''

        print(f"Got rgb_map?: {self.got_rgb_map}")
        print(f"Got height_map_low?: {self.got_height_map_low}")
        print(f"Got height_map_high?: {self.got_height_map_high}")

        if self.got_rgb_map and self.got_height_map_low and self.got_height_map_high:
            ## TODO: Get map metadata directly from rosbag to dataset

            print("=====")
            print("Obtained all maps, with these sizes: ")
            print(f"RGB Map:\n\tType: {type(self.rgb_map_tensor)}\n\tShape: {self.rgb_map_tensor.shape}\n\tMin: {torch.min(self.rgb_map_tensor)}\n\tMax:{torch.max(self.rgb_map_tensor)}")
            print(f"Height Map High:\n\tType: {type(self.height_map_high_tensor)}\n\tShape: {self.height_map_high_tensor.shape}\n\tMin: {torch.min(self.height_map_high_tensor)}\n\tMax:{torch.max(self.height_map_high_tensor)}")
            print(f"Height Map Low:\n\tType: {type(self.height_map_low_tensor)}\n\tShape: {self.height_map_low_tensor.shape}\n\tMin: {torch.min(self.height_map_low_tensor)}\n\tMax:{torch.max(self.height_map_low_tensor)}")

            maps = {
                'rgb_map': self.rgb_map_tensor,
                'height_map_low': self.height_map_low_tensor,
                'height_map_high': self.height_map_high_tensor
            }

            # Create TerrainMap with initial maps
            terrain_map = TerrainMap(map_metadata=self.map_metadata, maps=maps, device=self.device)

            # Initialize empty costmap
            costmap = torch.zeros(int(self.map_metadata['height']/self.map_metadata['resolution']), int(self.map_metadata['width']/self.map_metadata['resolution']))

            # Get coordinates for all points
            x_coords = torch.arange(0, int(self.map_metadata['height']/self.map_metadata['resolution']))

            y_coords = torch.arange(0, int(self.map_metadata['width']/self.map_metadata['resolution']))

            grid_x, grid_y = torch.meshgrid(x_coords, y_coords)

            print("Producing cost map now")
            for i in range(grid_x.shape[0]):
                print(f"Populating i={i}")
                for j in range(grid_x.shape[1]):
                    # print(f"Populating i={i}, j={j}")
                    position = [grid_x[i, j].item(), grid_y[i, j].item()]
                    yaw = 0
                    crop_size = [int(self.crop_params['crop_size'][0]/self.map_metadata['resolution']), int(self.crop_params['crop_size'][1]/self.map_metadata['resolution'])]
                    output_size = self.crop_params['output_size']
                    
                    pt = PatchTransform(position, yaw, crop_size, output_size, on_top=True)

                    # import pdb;pdb.set_trace()
                    # TODO. This should be called using terrain map methods
                    patch = pt(terrain_map.get_rgb_map())
                    patch = patch.expand(1, *patch.shape)

                    fe = FeatureExtractor(patch)
                    features = fe.get_features()

                    # print(f"Features shape: {features.shape}")

                    knn_sol = self.knn.forward(features).item()

                    # print(f"knn_sol: {knn_sol}")

                    costmap[position[0], position[1]] = knn_sol
            costmap = costmap.numpy()
            print("Costmap obtained, with these characteristics:")
            print(f"Costmap:\n\tType: {type(costmap)}\n\tShape: {costmap.shape}\n\tMin: {np.min(costmap)}\n\tMax:{np.max(costmap)}")
            plt.imshow(costmap)
            plt.colorbar()
            plt.show()
            costmap_msg = self.bridge.cv2_to_imgmsg(costmap, encoding="passthrough")
            print("Costmap message: ")
            print(costmap_msg.data)
            self.cost_map_publisher.publish(costmap_msg)

        else:
            print("Haven't received all required messages yet.")




    # def handle_imu(self, msg):
    #     self.buffer.insert(msg.linear_acceleration.z)
    #     cost = CostFunction(self.buffer.data, self.sensor_freq, min_freq=self.min_freq, max_freq=self.max_freq, sensor_name=self.sensor_name)

    #     self.cost_publisher.publish(cost)


if __name__ == "__main__":
    rospy.init_node("learned_cost_map_node", log_level=rospy.INFO)
    rospy.loginfo("Initialized learned_cost_map_node")

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('learned_cost_map')
    device = 'cpu'
    model_fp = os.path.join(package_path, "models", "kNN_Data.pt")
    print(f"Loading model from {model_fp}")
    node = LearnedCostMapNode(model_fp, device)
    rate = rospy.Rate(125)

    while not rospy.is_shutdown():
        node.publish_cost_map()
        rate.sleep()
