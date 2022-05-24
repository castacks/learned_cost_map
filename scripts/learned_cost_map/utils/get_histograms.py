import argparse
import numpy as np
from learned_cost_map.trainer.utils import get_dataloaders, preprocess_data
import matplotlib.pyplot as plt
import os

def main(data_dir, train_split, val_split, num_bins, output_dir):
    batch_size = 1
    seq_length = 1

    num_workers = 1
    shuffle_train = False
    shuffle_val = False
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val)

    train_costs = []
    train_speeds = []
    val_costs = []
    val_speeds = []

    ## Iterate over all training patches and collect all costs and speeds

    for i, data_dict in enumerate(train_loader):
        print(f"{i}/{len(train_loader)} Training Batch")
        input, labels = preprocess_data(data_dict, None)

        costs = data_dict['cost'][0].tolist() # Check if the [0] is needed
        vels = (20.0*input["vels"].squeeze()).tolist()

        train_costs.extend(costs)
        train_speeds.extend(vels)

    # Set directories to save cost data dictionary and histogram plot
    train_cost_plt_dir = os.path.join(output_dir, "train_cost_hist.png")
    train_cost_hist_dir = os.path.join(output_dir, "train_cost_hist.npy")
    
    
    plt.hist(x=train_costs, bins=num_bins, range=(0.0, 1.0))
    plt.savefig(train_cost_plt_dir, dpi=300)
    plt.close()

    # Get histogram data and save as a dictionary
    train_cost_counts, train_cost_bins = np.histogram(train_costs, range=(0.0, 1.0), bins=num_bins)
    train_costs_dict = {}
    train_costs_dict["data"] = train_costs
    train_costs_dict["counts"] = train_cost_counts
    train_costs_dict["bins"] = train_cost_bins
    np.save(train_cost_hist_dir, train_costs_dict)

    # Set directories to save speed data dictionary and histogram plot
    train_speed_plt_dir = os.path.join(output_dir, "train_speed_hist.png")
    train_speed_hist_dir = os.path.join(output_dir, "train_speed_hist.npy")

    # Plot histogram and save figure
    plt.hist(x=train_speeds, bins=num_bins, range=(0.0, 20.0))
    plt.savefig(train_speed_plt_dir, dpi=300)
    plt.close()

    # Get histogram data and save as a dictionary
    train_speed_counts, train_speed_bins = np.histogram(train_speeds, range=(0.0, 20.0), bins=num_bins)
    train_speeds_dict = {}
    train_speeds_dict["data"] = train_speeds
    train_speeds_dict["counts"] = train_speed_counts
    train_speeds_dict["bins"] = train_speed_bins

    np.save(train_speed_hist_dir, train_speeds_dict)


    ## Iterate over all validation patches and collect all costs and speeds

    for i, data_dict in enumerate(val_loader):
        print(f"{i}/{len(val_loader)} Validation Batch")
        # Get cost from neural net
        # x, y = preprocess_data(data_dict)
        input, labels = preprocess_data(data_dict, None)

        costs = data_dict['cost'][0].tolist() # Check if the [0] is needed
        vels = (20.0*input["vels"].squeeze()).tolist()

        val_costs.extend(costs)
        val_speeds.extend(vels)

    # Set directories to save cost data dictionary and histogram plot
    val_cost_plt_dir = os.path.join(output_dir, "val_cost_hist.png")
    val_cost_hist_dir = os.path.join(output_dir, "val_cost_hist.npy")
    
    
    plt.hist(x=val_costs, bins=num_bins, range=(0.0, 1.0))
    plt.savefig(val_cost_plt_dir, dpi=300)
    plt.close()

    # Get histogram data and save as a dictionary
    val_cost_counts, val_cost_bins = np.histogram(val_costs, range=(0.0, 1.0), bins=num_bins)
    val_costs_dict = {}
    val_costs_dict["data"] = val_costs
    val_costs_dict["counts"] = val_cost_counts
    val_costs_dict["bins"] = val_cost_bins
    np.save(val_cost_hist_dir, val_costs_dict)

    # Set directories to save speed data dictionary and histogram plot
    val_speed_plt_dir = os.path.join(output_dir, "val_speed_hist.png")
    val_speed_hist_dir = os.path.join(output_dir, "val_speed_hist.npy")

    # Plot histogram and save figure
    plt.hist(x=val_speeds, bins=num_bins, range=(0.0, 20.0))
    plt.savefig(val_speed_plt_dir, dpi=300)
    plt.close()

    # Get histogram data and save as a dictionary
    val_speed_counts, val_speed_bins = np.histogram(val_speeds, range=(0.0, 20.0), bins=num_bins)
    val_speeds_dict = {}
    val_speeds_dict["data"] = val_speeds
    val_speeds_dict["counts"] = val_speed_counts
    val_speeds_dict["bins"] = val_speed_bins

    np.save(val_speed_hist_dir, val_speeds_dict)


    # import pdb;pdb.set_trace()
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--train_split', type=str, required=True, help='Path to the file that contains the training split text file.')
    parser.add_argument('--val_split', type=str, required=True, help='Path to the file that contains the validation split text file.')
    parser.add_argument("--num_bins", type=int, default=20, help="How many bins to use for the cost and velocity histograms.")
    parser.add_argument('--output_dir', type=str, required=True, help='Path to directory where all the histogram data will be saved.')
    

    args = parser.parse_args()

    main(args.data_dir, args.train_split, args.val_split, args.num_bins, args.output_dir)