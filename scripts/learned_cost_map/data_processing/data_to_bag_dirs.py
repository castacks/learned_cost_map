import argparse
import os
import shutil
from pathlib import Path

def main(data_dir, output_dir):
    bag_paths = []
    bag_dir_names = []
    other_paths = []

    output_dirs = []
    output_bag_paths = []

    print(f"data_dir is: {data_dir}")
    print(f"output_dir is: {output_dir}")

    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        print("=====")
        print("dirpath:")
        print(dirpath)
        print("-----")
        print("dirnames:")
        print(dirnames)
        print("-----")
        print("filenames:")
        print(filenames)

        # Get all full filepaths
        full_paths = [os.path.join(dirpath, filename) for filename in filenames]

        for k, path in enumerate(full_paths):
            if path.endswith(".bag"):
                bag_paths.append(path)
                bag_dir_names.append(Path(filenames[k]).stem)
            else:
                other_paths.append(path)
    
    print("+++++")
    print("bag_paths:")
    print(bag_paths)
    print("-----")
    print("bag_dir_names:")
    print(bag_dir_names)
    print("-----")
    print("other_paths:")
    print(other_paths)

    # import pdb;pdb.set_trace()
    output_dirs = [os.path.join(output_dir, bag_dir_name) for bag_dir_name in bag_dir_names]
    # output_bag_paths = [os.path.join(output_dir, bag_path) for bag_path in bag_paths]

    output_bag_paths = [os.path.join(output_dirs[k], os.path.split(bag_paths[k])[1]) for k in range(len(output_dirs))]



    # output_other_paths = [os.path.join(output_dir, other_path) for other_path in other_paths]
    output_other_paths = []

    print("-----")
    print("output_dirs:")
    print(output_dirs)
    print("-----")
    print("output_bag_paths:")
    print(output_bag_paths)
    print("-----")
    print("output_other_paths:")
    print(output_other_paths)

    for dir in output_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Move files from input to output dirs:
    for i, (src, dst) in enumerate(zip(bag_paths, output_bag_paths)):
        print(f"Moving file {i}/{len(bag_paths)}")
        shutil.move(src, dst)


    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains all the bags. Bags can be inside subdirectories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory that will contain all the bags inside bag-specific subdirectories.')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    main(data_dir, output_dir)


    # Was called with the command below:
    # python data_to_bag_dirs.py --data_dir /media/mateo/Extreme\ SSD/arl_dataset/ --output_dir /media/mateo/Extreme\ SSD/arl_dataset/bags/

