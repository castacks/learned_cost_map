import argparse

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
    with open(all_train_fp,'r') as f:
        train_lines = f.readlines()
    import pdb;pdb.set_trace()

    ## Sort trajectories by length

    ## Choose split from within the training trajectories, try to make them balanced by using sorted list

    ## Write text files in the given location

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=50, help="Number of trajectories used for training.")
    parser.add_argument("--num_val", type=int, default=50, help="Number of trajectories used for validation.")
    parser.add_argument('--all_train_fp', type=str, required=True, help='Filepath that contains all training trajectories.')
    parser.add_argument('--all_val_fp', type=str, required=True, help='Filepath that contains all validation trajectories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the splits will be stored')
    args = parser.parse_args()

    create_split(args.num_train, args.num_val, args.all_train_fp, args.all_val_fp, args.output_dir)