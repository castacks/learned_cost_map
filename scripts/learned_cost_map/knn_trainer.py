import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from learned_cost_map.terrain_map import TerrainMap, get_local_path
from learned_cost_map.path_transform import TrajLoader
from learned_cost_map.feature_extractor import FeatureExtractor
from learned_cost_map.extract_patch import PatchTransform
from learned_cost_map.knn import KNNRegression
from learned_cost_map.util import dict_map


class kNNTrainer:
    '''Trains a kNN by loading trajectories from a directory, filtering the input data to generate self-supervised cost labels, and building the kNN object. 

    - Args:
        - directory:
            Directory where .pt trajectories are saved. For more info on what a trajectory entails, look at rosbag-to-dataset.
        - device:
            'cpu' if running on cpu, 'cuda' if running on GPU.
        - map_metadata:
            Dictionary containing the map metadata:

            map_metadata = {
                'height': map_height -> Float (in meters),
                'width': map_width -> Float (in meters),
                'resolution': resolution -> Float (in meters)
            }
        - crop_params:
            Dictionary containing the crop parameters:

            crop_params ={
                'crop_size': crop_size -> [Float, Float] (in meters),
                'output_size': output_size -> [Int, Int] (in pixels)
            }
        - save_data:
            Boolean, whether to save features and costs tensors into a file as a .pt dictionary of tensors.
        - save_fp"
            String, file path where the data will be saved if save_data is True

    '''
    def __init__(self, 
                 directory, 
                 device, 
                 map_metadata,
                 crop_params,
                 save_data=True, 
                 save_fp='/home/mateo/SARA/kNN_Data.pt'):

        self.directory = directory
        self.device = device
        self.save_data = save_data
        self.save_fp = save_fp

        self.map_metadata = map_metadata  # TODO: Get rid of this eventually
        self.crop_params = crop_params   # TODO: Is there a better way to load this in?

        self.knn = None

        btime1 = time.time()
        self.tl = TrajLoader(self.directory, self.device)
        self.trajectories = self.tl.getTrajectory() 
        print(f"Getting trajectories from TrajLoader takes: {time.time()-btime1} seconds")

        btime2 = time.time()
        # Sample trajectories (obtain list of filtered subtrajectories)
        self.sampled_trajectories = self.sample_trajectories()
        print(f"Generating subtrajectories takes: {time.time()-btime2} seconds")

        # Create TerrainMap for each sampled trajectory, get features/cost
        #   Parse initial local maps from trajectory
        #   Create TerrainMap with initial maps
        #   Obtain trajectory's patches and costs from whole subtrajectory
        #   Process patches into features
        #   Aggregate features and cost into single tensor
        self.features = []
        self.costs = []
        

        btime3 = time.time()
        for traj in self.sampled_trajectories:
            # Parse initial local maps from trajectory
            # NOTE: Permutation step is really important to have x and y coordinates in the right coordinate frame
            rgb_map = traj['observation']['rgb_map_inflate'][0].permute(0,2,1)
            height_map_low = traj['observation']['heightmap_low'][0].permute(0,2,1)
            height_map_high = traj['observation']['heightmap_high'][0].permute(0,2,1)

            # TODO: Get map metadata directly from rosbag to dataset
            map_metadata = self.map_metadata
            maps = {
                'rgb_map': rgb_map,
                'height_map_low': height_map_low,
                'height_map_high':height_map_high
            }

            # Create TerrainMap with initial maps
            tm = TerrainMap(map_metadata=map_metadata, maps=maps, device=self.device)

            # Obtain trajectory's patches and costs from whole subtrajectory
            traj_crops, traj_costs = tm.get_labeled_crops(traj, self.crop_params)

            # Process patches into features
            fe = FeatureExtractor(traj_crops)
            traj_features = fe.get_features()

            # Aggregate features and cost into single tensor
            self.features.append(traj_features)
            self.costs.append(traj_costs)

        # Concatenate self.features and self.costs into single tensor
        self.features = torch.cat(self.features, dim=0)
        self.costs = torch.cat(self.costs, dim=0)
        print(f"Concatenating all features takes: {time.time()-btime3} seconds")

        # Save aggregated features/cost tensor in given filepath if save_data
        if self.save_data:
            print(f"Saving data to: {self.save_fp}")
            torch.save({'features':self.features.cpu(), 'costs':self.costs.cpu()}, self.save_fp)


        btime4 = time.time()
        # build kNN with given data
        feature_dims = self.features.shape[1]
        max_datapoints = self.features.shape[0]
        K = 3
        sigma = 1.0
        self.knn = KNNRegression(feature_dims, max_datapoints, K, sigma=sigma, device=self.device)

        self.knn.insert(self.features, self.costs)
        self.knn.to(self.device)
        print(f"Building kNN takes: {time.time()-btime4} seconds")
        

    def sample_trajectories(self):
        # Filter trajectories in some way. Create windows? Run SVD to maximize the novelty in the features?

        ## Initial version returns input list of trajectories, i.e., it assumes that each of the individual trajectories in the given directory are of the right length and non-overlapping

        sampled_trajectories = []

        self.sub_traj_length = 70

        for k, trajectory in enumerate(self.trajectories):
            # print("---")
            traj_len = trajectory['observation']['state'].shape[0]
            # print(f"Looking at trajectory {k}, which has length {traj_len}")
            for i in range(0,traj_len,self.sub_traj_length):
                slicer = lambda x: x[i:i+self.sub_traj_length]
                sub_traj = dict_map(trajectory, slicer) # Get k-step trajectory for data collection
                # print(f"sub_traj_length is {self.sub_traj_length}. Looking at interval {i}-{i+self.sub_traj_length}")
                sub_traj_size = sub_traj['observation']['state'].shape[0]
                # print(f"This subtrajectory has length {sub_traj_size}")
                sampled_trajectories.append(sub_traj)

        print(f"At the end, we have {len(sampled_trajectories)} sampled subtrajectories, and {len(sampled_trajectories)*self.sub_traj_length} sampled crops")
        return sampled_trajectories

    def visualize_trajectories(self):
        print(f"Size of sampled_trajectories: {len(self.sampled_trajectories)}")
        fig = plt.figure()
        # fig.suptitle('Cost visualizer')
        img_viewer = fig.add_subplot(121)
        patch_viewer = fig.add_subplot(122)
        for i, traj in enumerate(self.sampled_trajectories):
            print(f"Size of sampled_trajectories: {len(self.sampled_trajectories)}")
            rgb_map = traj['observation']['rgb_map_inflate'][0].permute(0,2,1)
            height_map_low = traj['observation']['heightmap_low'][0].permute(0,2,1)
            height_map_high = traj['observation']['heightmap_high'][0].permute(0,2,1)

            # TODO: Get map metadata directly from rosbag to dataset
            map_metadata = self.map_metadata
            maps = {
                'rgb_map': rgb_map,
                'height_map_low': height_map_low,
                'height_map_high':height_map_high
            }

            # Create TerrainMap with initial maps
            tm = TerrainMap(map_metadata=map_metadata, maps=maps, device=self.device)

            traj_crops, traj_costs = tm.get_labeled_crops(traj, self.crop_params)

            print(f"traj_crops: {traj_crops.shape}")
            print(f"traj_costs: {traj_costs.shape}")

            print(f"traj_crops after squeeze: {traj_crops.shape}")

            # if traj_crops.shape[0] == 1:
            #     batch = torch.squeeze(traj_crops)[:3].permute(1,2,0).cpu().numpy()
            #     plt.figure(1);plt.clf()
            #     plt.title(f"Step {i} in trajectory. Cost: {traj_costs[0].item():.4f}")
            #     plt.imshow(batch)
            #     plt.pause(0.1)

            original = tm.maps_tensor.permute(1,2,0).cpu().numpy()[:,:,:3]

            crop_width = crop_params['crop_size'][0] # Assume crop width and height are equal
            # Extract path from trajectory -> use odom to get path
            # NOTE: Path coordinates are in robot centric FLU metric coordinates
            state = traj['observation']['state']
            next_state = traj['next_observation']['state']

            cost = traj['observation']['traversability_cost']
            local_path = get_local_path(state, next_state)
    
            batch = traj_crops[:,:3]
            print(f"batch after getting rid of heightmaps: {batch.shape}")
            batch = batch.permute(0,2,3,1).cpu().numpy()
            batch_size = batch.shape[0]
            for b in range(batch_size):
                img_viewer.clear()
                patch_viewer.clear()
                
                x = local_path[b,0]
                y = local_path[b,1]
                theta = local_path[b,2]
                theta_deg = theta*180/np.pi
                pixel_x = int((x/tm.resolution))
                pixel_y = int((y - tm.origin[1])/tm.resolution)

                pixel_dx = int(crop_width/tm.resolution)*np.cos(theta)
                pixel_dy = int(crop_width/tm.resolution)*np.sin(theta)

                img_viewer.plot(pixel_x, pixel_y, marker='o', color='red')
                
                img_viewer.arrow(pixel_x-0.5*pixel_dx, pixel_y-0.5*pixel_dy, pixel_dx, pixel_dy, color="red")

                pixel_crop_width = int(crop_width/tm.resolution)
                corners = 0.5 * np.array([[pixel_crop_width, pixel_crop_width],
                                        [pixel_crop_width, -pixel_crop_width], [-pixel_crop_width, -pixel_crop_width], [-pixel_crop_width, pixel_crop_width], [pixel_crop_width, pixel_crop_width]])
                rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                square = np.matmul(rot, np.expand_dims(corners, axis=2))[:, :, 0] + np.array([pixel_x, pixel_y])
                img_viewer.plot(square[:, 0], square[:, 1], c='r')

                img_viewer.set_title(f"Current pose: x={pixel_x} y={pixel_y}, yaw={(theta_deg%360.):.2f}")
                img_viewer.set_xlabel("X axis")
                img_viewer.set_ylabel("Y axis")
                img_viewer.imshow(original, origin="lower")

                patch_viewer.set_title(f"Trajectory {i}, sub trajectory step: {b}/{batch_size}. Cost: {traj_costs[b].item():.4f}")
                patch_viewer.imshow(batch[b])
                fig.suptitle(f'Cost visualizer. Sub trajectory size: {self.sub_traj_length}')
                if i == 0 and b == 0:
                    plt.pause(15)
                plt.pause(0.1)


    def solve(self, random=True):

        fig = plt.figure()
        # fig.suptitle('Cost visualizer')
        image_viewer = fig.add_subplot(121)
        cost_viewer = fig.add_subplot(122)

        print(f"Length of sampled_trajectories: {len(self.sampled_trajectories)}")
        traj = self.sampled_trajectories[10]
        rgb_map = traj['observation']['rgb_map_inflate'][0].permute(0,2,1)
        height_map_low = traj['observation']['heightmap_low'][0].permute(0,2,1)
        height_map_high = traj['observation']['heightmap_high'][0].permute(0,2,1)

        # TODO: Get map metadata directly from rosbag to dataset
        map_metadata = self.map_metadata
        maps = {
            'rgb_map': rgb_map,
            'height_map_low': height_map_low,
            'height_map_high':height_map_high
        }

        # Create TerrainMap with initial maps
        tm = TerrainMap(map_metadata=map_metadata, maps=maps, device=self.device)

        print("Visualize chosen terrain map: ")
        rgb_map_viz = traj['observation']['rgb_map_inflate'][0].permute(1, 2, 0)
        plt.imshow(rgb_map_viz, origin="lower")
        plt.show()

        costmap = torch.zeros(int(self.map_metadata['height']/self.map_metadata['resolution']), int(self.map_metadata['width']/self.map_metadata['resolution']))

        x_coords = torch.arange(0, int(self.map_metadata['height']/self.map_metadata['resolution']))

        y_coords = torch.arange(0, int(self.map_metadata['width']/self.map_metadata['resolution']))

        print(f"Shape of costmap: {costmap.shape}")
        print(f"X_coords: {x_coords}")
        print(f"Y coords: {y_coords}")

        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')

        print(f"grid_x: {grid_x}")
        print(f"grid_y: {grid_y}")

    
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):

                position = [grid_x[i, j].item(), grid_y[i, j].item()]
                yaw = 0
                crop_size = [int(self.crop_params['crop_size'][0]/self.map_metadata['resolution']), int(self.crop_params['crop_size'][1]/self.map_metadata['resolution'])]
                # print(f"CROP SIZE: {crop_size}")
                output_size = self.crop_params['output_size']
                
                pt = PatchTransform(position, yaw, crop_size, output_size, on_top=True)

                patch = pt(rgb_map)
                # print(f"patch shape before: {patch.shape}")
                patch = patch.expand(1, *patch.shape)

                # print(f"patch shape after: {patch.shape}")
                fe = FeatureExtractor(patch)
                features = fe.get_features()

                # print(f"Features shape: {features.shape}")

                knn_sol = self.knn.forward(features).item()

                # print(f"knn_sol: {knn_sol}")

                costmap[position[0], position[1]] = knn_sol

        # costmap = costmap.cpu().numpy()
        # image_viewer.imshow(rgb_map_viz, origin="lower")
        # image_viewer.set_title("Original RGB map")
        # # p1 = cost_viewer.imshow(costmap, origin='lower')
        # # plt.colorbar(p1, ax=cost_viewer)
        # cost_viewer.set_title("Cost map")
        # plt.show()

        try:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(rgb_map_viz, origin='lower')
            axs[1].imshow(costmap, origin='lower')
            axs[0].set_title('RGB Map')
            axs[1].set_title('Cost')
            plt.show()
        except:
            import pdb;pdb.set_trace()
        # return self.knn.forward(features)


if __name__=="__main__":

    directory = "/home/mateo/Data/SARA/CostBags"
    device = 'cpu'


    map_height = 10.0
    map_width = 10.0
    resolution = 0.05

    map_metadata = {
        'height': map_height,
        'width': map_width,
        'resolution': resolution
    }

    crop_width = 2  # in meters
    crop_size = [crop_width, crop_width]
    output_size = [224, 224]
    
    crop_params ={
        'crop_size': crop_size,
        'output_size': output_size
    }


    save_data=True, 
    save_fp='/home/mateo/Data/SARA/knn/model.pt'

    trainer = kNNTrainer(directory, 
                         device, 
                         map_metadata,
                         crop_params,
                         save_data=True, 
                         save_fp=save_fp)

    # trainer.visualize_trajectories()

    # trainer.solve()
