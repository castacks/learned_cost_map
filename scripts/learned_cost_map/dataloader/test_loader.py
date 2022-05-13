from learned_cost_map.dataloader.TartanDriveDataset import DatasetBase, data_transform

def main():
    batch_size = 1
    seq_length = 1
    data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    train_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    val_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'

    datatypes = "imgc,heightmap,rgbmap,odom,cost,patches"
    base_mod_lengths = [1,1,1,10,10,1]
    modality_lengths = [seq_length*l for l in base_mod_lengths]

    train_set = DatasetBase(train_split,
                            dataroot= data_root_dir,
                            datatypes = datatypes,
                            modalitylens = modality_lengths,
                            transform=data_transform,
                            imu_freq = 10,
                            frame_skip = 0, 
                            frame_stride=5)

    for i in range(5):
        import pdb;pdb.set_trace()
        data_dict = train_set[i]
        patches = data_dict["patches"]

if __name__=="__main__":
    main()