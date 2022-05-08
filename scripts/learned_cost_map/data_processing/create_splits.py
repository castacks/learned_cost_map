from os.path import isfile, join, isdir
from os import listdir
import os
import numpy as np
import argparse

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
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where the training split will be saved.')
    args = parser.parse_args()

    data_root_dir = args.data_dir #'/project/learningphysics/tartandrive_trajs'
    output_dir = args.output_dir #'/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outfile = os.path.join(output_dir, 'tartandrive_train.txt')
    f = open(outfile, 'w')

    trajlist = enumerate_trajs(data_root_dir)
    for trajdir in trajlist:
        trajindlist = process_traj(data_root_dir + '/' +trajdir)
        for trajinds in trajindlist:
            f.write(trajdir)
            f.write(' ')
            f.write(str(len(trajinds)))
            f.write('\n')
            for ind in trajinds:
                f.write(ind)
                f.write('\n')
    f.close()