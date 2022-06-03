# learned_cost_map

This repo contains code to train and deploy a learned traversability costmap. 

## Requirements

This code was tested with the following package versions:
- Python 3.8
- PyTorch 1.10 (CUDA 11.3)
- NumPy 1.22

## Entry Points

In order to run the ROS node that generates the learned costmap, we first need to make sure that TartanVO is running. To run TartanVO, run the ```multisense_register_localmapping.launch``` inside the ```physics_atv_deep_stereo_vo``` package.

Once TartanVO is running, we can run the ```learned_cost_map.launch``` launch file in this package, which will publish an occupancy grid with the costmap.

The current dimensions of the costmap are 12m x 12m with a resolution of 0.02 m.

The trained models are saved inside ```learned_cost_map/scripts/learned_cost_map/trainer/models```.