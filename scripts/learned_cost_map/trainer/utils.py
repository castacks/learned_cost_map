import numpy as np


from torch.utils.data import DataLoader
from torchvision import transforms as T
from learned_cost_map.terrain_utils.terrain_map_tartandrive import TerrainMap, get_local_path
from learned_cost_map.dataloader.TartanDriveDataset import DatasetBase, data_transform

def get_dataloaders(batch_size, seq_length):
    data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    train_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    val_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'

    datatypes = "img0,img1,imgc,disp0,heightmap,rgbmap,cmd,odom,cost,patches,imu"
    base_mod_lengths = [1,1,1,1,1,1,1,1,1,1,10]
    modality_lengths = [seq_length*l for l in base_mod_lengths]

    train_set = DatasetBase(train_split,
                            dataroot= data_root_dir,
                            datatypes = datatypes,
                            modalitylens = modality_lengths,
                            transform=data_transform,
                            imu_freq = 10,
                            frame_skip = 0, 
                            frame_stride=5)

    val_set = DatasetBase(val_split,
                          dataroot= data_root_dir,
                          datatypes = datatypes,
                          modalitylens = modality_lengths,
                          transform=data_transform,
                          imu_freq = 10,
                          frame_skip = 0, 
                          frame_stride=5)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

def preprocess_data(input_dict):
    x = input_dict["patches"].view(-1, *input_dict["patches"].shape[-3:])
    y = input_dict["cost"].view(-1)
    return x.to('cuda'), y.to('cuda')

def avg_dict(all_metrics):
    keys = all_metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = np.mean([all_metrics[i][key].cpu().detach().numpy() for i in range(len(all_metrics))])
    return avg_metrics

# Maybe write vis_trajectory function