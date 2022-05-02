from os.path import isfile, join, isdir
from os import listdir
import os
import numpy as np

# == Generate txt file for tartan dataset ===

def process_traj(trajdir, folderstr = 'image_left'):
    imglist = listdir(join(trajdir, folderstr))
    imglist = [ff for ff in imglist if ff[-3:]=='png']
    imglist.sort()
    imgnum = len(imglist)

    lastfileind = -1
    outlist = []
    framelist = []
    for k in range(imgnum):
        filename = imglist[k]
        framestr = filename.split('_')[0].split('.')[0]
        frameind = int(framestr)

        if frameind==lastfileind+1: # assume the index are continuous
            framelist.append(framestr)
        else:
            if len(framelist) > 0:
                outlist.append(framelist)
                framelist = []
        lastfileind = frameind

    if len(framelist) > 0:
        outlist.append(framelist)
        framelist = []
    print('Find {} trajs, traj len {}'.format(len(outlist), [len(ll) for ll in outlist]))

    return outlist 


def enumerate_trajs(data_root_dir):
    trajfolders = listdir(data_root_dir)    
    trajfolders = [ee for ee in trajfolders if isdir(data_root_dir+'/'+ee)]
    trajfolders.sort()
    print('Detected {} trajs'.format(len(trajfolders)))
    return trajfolders


if __name__=="__main__":
    dataset_dir = "/home/mateo/Data/SARA/TartanDriveCost"
    trajectories_dir = os.path.join(dataset_dir, "Trajectories")
    splits_dir       = os.path.join(dataset_dir, "Splits")

    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)
    train_split_dp = os.path.join(splits_dir, "train.txt")

    # data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    # outfile = '/home/mateo/Data/SARA/TartanDriveCost/Splits/tartandrive_train.txt'
    f = open(train_split_dp, 'w')

    trajlist = enumerate_trajs(trajectories_dir)
    for trajdir in trajlist:
        trajindlist = process_traj(os.path.join(trajectories_dir, trajdir))
        for trajinds in trajindlist:
            f.write(trajdir)
            f.write(' ')
            f.write(str(len(trajinds)))
            f.write('\n')
            for ind in trajinds:
                f.write(ind)
                f.write('\n')
    f.close()