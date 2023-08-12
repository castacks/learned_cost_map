import numpy as np

hc_data_cost_fp = "/home/mateo/Data/SARA/tartancost_data/highcost_10k/costs.npy"
lc_data_cost_fp = "/home/mateo/Data/SARA/tartancost_data/lowcost_5k/costs.npy"

hc_data = np.load(hc_data_cost_fp)
lc_data = np.load(lc_data_cost_fp)

all_costs = np.concatenate([hc_data, lc_data], axis=0)

mean_cost = np.mean(all_costs)

mse = (np.linalg.norm(all_costs-mean_cost)**2)/len(all_costs)

print(f"Mean cost: {mean_cost}")
print(f"MSE: {mse}")


hc_data_cost_val_fp = "/home/mateo/Data/SARA/tartancost_data/highcost_val_2k/costs.npy"
lc_data_cost_val_fp = "/home/mateo/Data/SARA/tartancost_data/lowcost_val_1k/costs.npy"

hc_val_data = np.load(hc_data_cost_val_fp)
lc_val_data = np.load(lc_data_cost_val_fp)

all_val_costs = np.concatenate([hc_val_data, lc_val_data], axis=0)

val_mse = (np.linalg.norm(all_val_costs-mean_cost)**2)/len(all_val_costs)

print(f"Validation MSE: {val_mse}")