import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

import time

from learned_cost_map.utils.costmap_utils import produce_costmap, rosmsgs_to_maps
from learned_cost_map.trainer.model import CostModel, CostVelModel, CostFourierVelModel, CostFourierVelModelEfficientNet

class CostmapGenerator(object):
    def __init__(self, model_name, saved_model, saved_freqs, data_dir, vel):
        self.data_dir = data_dir
        self.vel = vel

        # Load trained model to produce costmaps
        self.fourier_freqs = None
        if model_name=="CostModel":
            self.model = CostModel(input_channels=8, output_size=1)
        elif model_name=="CostVelModel":
            self.model = CostVelModel(input_channels=8, embedding_size=512, output_size=1)
        elif model_name=="CostFourierVelModel":
            self.model = CostFourierVelModel(input_channels=8, ff_size=16, embedding_size=512, output_size=1)
            self.fourier_freqs = torch.load(saved_freqs)
        elif model_name=="CostFourierVelModelEfficientNet":
            model = CostFourierVelModelEfficientNet(input_channels=8, ff_size=16, embedding_size=512, output_size=1)
            self.fourier_freqs = torch.load(saved_freqs)
        else:
            raise NotImplementedError()

        self.model.load_state_dict(torch.load(saved_model))
        self.model.cuda()
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


    def generate_costmaps(self):
        traj_dirs = list(filter(os.path.isdir, [os.path.join(self.data_dir,x) for x in sorted(os.listdir(self.data_dir))]))

        for i, d in enumerate(traj_dirs):
            print("-----")
            print(f"Evaluating trajectory {d}")
            ## Load height_map and rgb_map data
            rgbmap_dir = os.path.join(d, "rgb_map_vo")
            heightmap_dir = os.path.join(d, "height_map_vo")

            rgbmap_fps = list(filter(lambda fpath: ".npy" in fpath, [os.path.join(rgbmap_dir,x) for x in sorted(os.listdir(rgbmap_dir))]))

            heightmap_fps = list(filter(lambda fpath: ".npy" in fpath, [os.path.join(heightmap_dir,x) for x in sorted(os.listdir(heightmap_dir))]))

            learned_costmap_dir = os.path.join(d, f"learned_cost_map_vel_{self.vel}")
            if not os.path.exists(learned_costmap_dir):
                os.makedirs(learned_costmap_dir)
            for j, (rgb_map_path, height_map_path) in enumerate(zip(rgbmap_fps, heightmap_fps)):
                print(f"Evaluating map number: {j}")
                print(f"rgb_map_path: {rgb_map_path}")
                print(f"height_map_path: {height_map_path}")
                # import pdb;pdb.set_trace()
                rgb_map = np.load(rgb_map_path)
                height_map = np.load(height_map_path)
                maps = rosmsgs_to_maps(rgb_map, height_map)
                costmap = produce_costmap(self.model, maps, self.map_metadata, self.crop_params, vel=self.vel, fourier_freqs=self.fourier_freqs)
                
                costmap_fp = os.path.join(learned_costmap_dir, f"{j:06}.npy")
                costmap_prev_fp = os.path.join(learned_costmap_dir, f"{j:06}.png")
                np.save(costmap_fp, costmap)

                plt.imshow(costmap, origin="lower", vmin=0.0, vmax=1.0, cmap="plasma")
                plt.savefig(costmap_fp, dpi=300, bbox_inches="tight")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['CostModel', 'CostVelModel', 'CostFourierVelModel', 'CostModelEfficientNet', 'CostFourierVelModelEfficientNet', 'CostFourierVelModelSmall', 'CostFourierVelModelRGB'], default='CostModel')
    parser.add_argument('--saved_model', type=str, help='String for where the saved model that will be used for fine tuning is located.')
    parser.add_argument('--saved_freqs', type=str, help='String for where the saved Fourier frequencies that will be used for fine tuning are located.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--vel', type=float, default=5.0, help="Velocity at which to generate costmaps")
    parser.add_argument('--use_real_vel', action='store_true', help="If set, uses the recorded velocity instead of input velocity")

    parser.set_defaults(use_real_vel=False)

    args = parser.parse_args()

    model_name = args.model
    saved_model = args.saved_model
    saved_freqs = args.saved_freqs
    data_dir = args.data_dir
    vel = args.vel

    if (saved_model is None) or (model_name is None) or (saved_freqs is None):
        raise NotImplementedError()
    cg = CostmapGenerator(model_name, saved_model, saved_freqs, data_dir, vel)
    cg.generate_costmaps()