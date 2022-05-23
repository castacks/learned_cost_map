import argparse
import os
import numpy as np

def create_split(num_train, num_val, all_train_fp, all_val_fp, output_dir):
    '''Creates two text files for a data split of a given size.

    The splits will be saved in output_dir under the names train{num_train}.txt, val{num_val}.txt. For example, if num_train=50 and num_val=10, the resulting filenames will be train50.txt and val10.txt, respectively.

    Args:
        - num_train:
            Integer, number of trajectories to be used for training
        - num_val:
            Integer, number of trajectories to be used for validation
        - all_train_fp:
            String, filepath where all the training trajectories are located
        - all_val_fp:
            String, filepath where all the validation trajectories are located
        - output_dir:
            String, directory where the splits will be saved.
    '''

    ## Read in full list of training and validation trajectories
    
    # Read in training trajectories
    train_names = []
    train_lengths = []
    train_dict = {}
    with open(all_train_fp,'r') as f:
        train_lines = f.readlines()
    # Sort training trajectories by length
    ind = 0
    while ind<len(train_lines):
        line = train_lines[ind].strip()
        traj_name, traj_len = line.split(' ')
        traj_len = int(traj_len)
        ind += 1
        frames = []
        for k in range(traj_len):
            line = train_lines[ind].strip()
            frames.append(line)
            ind += 1

        train_names.append(traj_name)
        train_lengths.append(traj_len)
        train_dict[traj_name] = frames
    sorted_train_names = [x for _, x in sorted(zip(train_lengths, train_names))]
    sorted_train_lengths = sorted(train_lengths)

    # Read in validation trajectories
    val_names = []
    val_lengths = []
    val_dict = {}
    with open(all_val_fp,'r') as f:
        val_lines = f.readlines()
    # Sort validation trajectories by length
    ind = 0
    while ind<len(val_lines):
        line = val_lines[ind].strip()
        traj_name, traj_len = line.split(' ')
        traj_len = int(traj_len)
        ind += 1
        frames = []
        for k in range(traj_len):
            line = val_lines[ind].strip()
            frames.append(line)
            ind += 1

        val_names.append(traj_name)
        val_lengths.append(traj_len)
        val_dict[traj_name] = frames
    sorted_val_names = [x for _, x in sorted(zip(val_lengths, val_names))]
    sorted_val_lengths = sorted(val_lengths)
    # import pdb;pdb.set_trace()

    ## Choose split from within the training trajectories
    if num_train > len(train_names) or num_val > len(val_names):
        raise NotImplementedError()
    num_train = min(num_train, len(train_names))
    num_val = min(num_val, len(val_names))

    ## Write text files in the given location

    # Write training split
    train_output_names = sorted_train_names[:num_train]
    
    train_split_fp = os.path.join(output_dir, f"train{num_train}.txt")
    f = open(train_split_fp, 'w')

    for i,traj_name in enumerate(train_output_names):
        f.write(traj_name)
        f.write(' ')
        f.write(str(sorted_train_lengths[i]))
        f.write('\n')
        for ind in train_dict[traj_name]:
            f.write(ind)
            f.write('\n')
    f.close()

    # Write val split
    val_output_names = sorted_val_names[:num_val]
    
    val_split_fp = os.path.join(output_dir, f"val{num_val}.txt")
    f = open(val_split_fp, 'w')

    for i,traj_name in enumerate(val_output_names):
        f.write(traj_name)
        f.write(' ')
        f.write(str(sorted_val_lengths[i]))
        f.write('\n')
        for ind in val_dict[traj_name]:
            f.write(ind)
            f.write('\n')
    f.close()

def train_val_split(all_trajs_fp, num_train, num_val, output_dir):
    '''Creates a data split by uniformly sampling num_train and num_val trajectories from all the trajectories.

    Args:
        - all_trajs_fp:
            String, filepath where all the trajectories are located
        - num_train:
            Int, number of trajectories to use for training
        - num_val:
            Int, number of trajectories to use for validation
        - output_dir:
            String, directory where the splits will be saved.
    '''
    # Read in all trajectories
    traj_names = []
    traj_lengths = []
    traj_dict = {}
    with open(all_trajs_fp,'r') as f:
        traj_lines = f.readlines()
    
    ind = 0
    while ind<len(traj_lines):
        line = traj_lines[ind].strip()
        traj_name, traj_len = line.split(' ')
        traj_len = int(traj_len)
        ind += 1
        frames = []
        for k in range(traj_len):
            line = traj_lines[ind].strip()
            frames.append(line)
            ind += 1

        traj_names.append(traj_name)
        traj_lengths.append(traj_len)
        traj_dict[traj_name] = frames

    # Sort  trajectories by length
    sorted_traj_names = [x for _, x in sorted(zip(traj_lengths, traj_names))]
    sorted_traj_lengths = sorted(traj_lengths)
    
    assert (num_train+num_val) <= len(traj_names), f"num_train ({num_train}) + num_val ({num_val}) can't be greater than the total number of trajectories ({len(traj_names)})"

    sorted_traj_names_array = np.array(sorted_traj_names)
    np.random.shuffle(sorted_traj_names_array)
    random_order_names = sorted_traj_names_array.tolist()

    train_traj_names = random_order_names[:num_train]
    val_traj_names = random_order_names[-num_val:]

    ## Write text files in the given location

    train_split_fp = os.path.join(output_dir, f"train_uniform.txt")
    f = open(train_split_fp, 'w')

    # Write train split
    for i,traj_name in enumerate(train_traj_names):
        f.write(traj_name)
        f.write(' ')
        num_frames = len(traj_dict[traj_name])
        f.write(str(num_frames))
        f.write('\n')
        for ind in traj_dict[traj_name]:
            f.write(ind)
            f.write('\n')
    f.close()

    # Write val split
    val_split_fp = os.path.join(output_dir, f"val_uniform.txt")
    f = open(val_split_fp, 'w')

    for i,traj_name in enumerate(val_traj_names):
        f.write(traj_name)
        f.write(' ')
        num_frames = len(traj_dict[traj_name])
        f.write(str(num_frames))
        f.write('\n')
        for ind in traj_dict[traj_name]:
            f.write(ind)
            f.write('\n')
    f.close()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_trajs_fp', type=str, required=True, help='Filepath that contains all training trajectories.')
    parser.add_argument("--num_train", type=int, default=50, help="Number of trajectories used for training.")
    parser.add_argument("--num_val", type=int, default=50, help="Number of trajectories used for validation.")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the splits will be stored')
    args = parser.parse_args()

    # create_split(args.num_train, args.num_val, args.all_train_fp, args.all_val_fp, args.output_dir)

    train_val_split(args.all_trajs_fp, args.num_train, args.num_val, args.output_dir)