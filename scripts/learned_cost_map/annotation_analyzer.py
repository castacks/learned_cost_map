import numpy as np
import os
import yaml

def main():
    dataset_dir = '/home/mateo/Data/SARA/TartanCost'
    trajectories_dir = os.path.join(dataset_dir, "Trajectories")
    annotations_dir  = os.path.join(dataset_dir, "Annotations")
    dir_names = sorted(os.listdir(trajectories_dir))

    high_speed = []
    slow_speed = []
    up_slope   = []
    down_slope = []
    low_grass  = []
    tall_grass = []
    
    for dir in dir_names:
        trajectory_dir = os.path.join(trajectories_dir, dir)
        annotation_dir = os.path.join(annotations_dir, dir+".yaml")

        # Load annotation
        with open(annotation_dir, 'r') as f:
            annotation = yaml.safe_load(f)


        if annotation["labels"][11] == 1:
            high_speed.append(dir)

        if annotation["labels"][9] == 1:
            slow_speed.append(dir)

        if annotation["labels"][1] == 1:
            up_slope.append(dir)

        if annotation["labels"][2] == 1:
            down_slope.append(dir)

        if annotation["labels"][7] == 1:
            low_grass.append(dir)

        if annotation["labels"][8] == 1:
            tall_grass.append(dir)

    print("High speed: ")
    print(high_speed)
    print("Slow speed: ")
    print(slow_speed)
    print("Up slope: " )
    print(up_slope)
    print("Down slope: ")
    print(down_slope)
    print("Low grass: ")
    print(low_grass)
    print("Tall grass: ")
    print(tall_grass)

if __name__=="__main__":
    main()