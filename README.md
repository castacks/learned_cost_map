# learned_cost_map

This repo contains code to train and deploy a learned traversability costmap. 

## Requirements

This code was tested with the following package versions:
- Python 3.8
- PyTorch 1.10 (CUDA 11.3)
- NumPy 1.22

## Deployment

In order to run the ROS node that generates the learned costmap, we first need to make sure that we have three topics being published: 
- a top-down heightmap (in our case, it is /local_height_map_inflate)
- a top-down rgbmap (in our case, it is /local_rgb_map_inflate)
- an odometry (in our case, it is /integrated_to_init)

The top down maps should follow the dimensions set in learned_cost_map/configs/*map_params.yaml. If the top-down maps are obtained using TartanVO, then we need to make sure TartanVO is running. To run TartanVO, run the ```multisense_register_localmapping.launch``` inside the ```physics_atv_deep_stereo_vo``` package.

Once the necessary topics are published (or TartanVO is running), we can run the ```learned_cost_map.launch``` launch file in this package, which will publish an occupancy grid with the costmap.

The current dimensions of the costmap are 12m x 12m with a resolution of 0.02 m and the origin at (-2, -6).

The trained models are saved inside ```learned_cost_map/models```. The output costmap will be published to '/learned_costmap'

## Training

### Processing Rosbags

TODO

### Labeling with pseudo ground-truth traversability costs

TODO

#### Labeling with IMU cost function

TODO

#### Collecting statistics of dataset

TODO

### Data Balancing

TODO

### Training

TODO

### Visalization

TODO