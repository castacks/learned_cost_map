import argparse
import os
import rosbag

def main(bags_dir, required_topics):
    full_paths = []
    for (dirpath, dirnames, filenames) in os.walk(bags_dir):
        full_paths.extend([os.path.join(dirpath, filename) for filename in filenames])

    bags_to_keep = []
    bags_to_discard = []

    for bag_path in full_paths:
        has_left = False
        has_right = False
        has_odometry = False
        has_imu = False

        bag = rosbag.Bag(bag_path)
        topics = bag.get_type_and_topic_info()[1]
        topic_keys = bag.get_type_and_topic_info()[1].keys()
        for topic in topic_keys:
            if ('stereo_right' in topic) and ('Image' in topics[topic].msg_type):
                has_right=True
            if ('stereo_left' in topic) and ('Image' in topics[topic].msg_type):
                has_left=True
            if ('odom' in topic) and ('Odometry' in topics[topic].msg_type):
                has_odometry=True
            if ('imu' in topic) and ('Imu' in topics[topic].msg_type):
                has_imu=True
        
        if has_left and has_right and has_odometry and has_imu:
            bags_to_keep.append(bag_path)
        else:
            bags_to_discard.append(bag_path)
    
    print("-----")
    print("bags_to_keep:")
    print(bags_to_keep)
    print("\n\n\n")
    print("-----")
    print("bags_to_discard:")
    print(bags_to_discard)




if __name__ == "__main__":
    bags_dir = "/media/mateo/Extreme SSD/arl_dataset/bags/"
    required_topics = None

    main(bags_dir, required_topics)
