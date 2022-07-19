import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml

sns.set_style("whitegrid")


# Define CSV filepaths:
train_dict = {
    "corl": "/home/mateo/phoenix_ws/src/learned_cost_map/data/corl_train.csv",
    # "patch": "/home/mateo/phoenix_ws/src/learned_cost_map/data/patch_train.csv",
    "patch_vel": "/home/mateo/phoenix_ws/src/learned_cost_map/data/patch_vel_train.csv",
    "mlp32": "/home/mateo/phoenix_ws/src/learned_cost_map/data/mlp32_train.csv",
    # "small": "/home/mateo/phoenix_ws/src/learned_cost_map/data/resnet10_mlp512_train.csv",
    # "rgb": "/home/mateo/phoenix_ws/src/learned_cost_map/data/only_rgb_train.csv"
}

val_dict = {
    "corl": "/home/mateo/phoenix_ws/src/learned_cost_map/data/corl_val.csv",
    # "patch": "/home/mateo/phoenix_ws/src/learned_cost_map/data/patch_val.csv",
    "patch_vel": "/home/mateo/phoenix_ws/src/learned_cost_map/data/patch_vel_val.csv",
    "mlp32": "/home/mateo/phoenix_ws/src/learned_cost_map/data/mlp32_val.csv",
    # "small": "/home/mateo/phoenix_ws/src/learned_cost_map/data/resnet10_mlp512_val.csv",
    # "rgb": "/home/mateo/phoenix_ws/src/learned_cost_map/data/only_rgb_val.csv"
}

# train_dfs = [pd.read_csv(train_dict[k]) for k in train_dict.keys()]
# val_dfs   = [pd.read_csv(val_dict[k]) for k in val_dict.keys()]

# Add name of the model to each of the data frames
train_dfs = []
for k,v in train_dict.items():
    df = pd.read_csv(v)
    df.insert(loc=1, column="Model", value=k)
    train_dfs.append(df)

val_dfs = []
for k,v in val_dict.items():
    df = pd.read_csv(v)
    df.insert(loc=1, column="Model", value=k)
    val_dfs.append(df)

# import pdb;pdb.set_trace()

# Concatenate all training and validation data frames
train_df = pd.concat(train_dfs)
val_df = pd.concat(val_dfs)

train_cols_to_remove = []
val_cols_to_remove = []

for col in train_df.columns:
    if ("MAX" in col) or ("MIN" in col):
        train_cols_to_remove.append(col)

for col in val_df.columns:
    if ("MAX" in col) or ("MIN" in col):
        val_cols_to_remove.append(col)

train_df = train_df.drop(columns=train_cols_to_remove)
val_df = val_df.drop(columns=val_cols_to_remove)

# import pdb;pdb.set_trace()

stats = {}

train_stats = {}
for model in train_df.Model.unique():
    data_only_df = train_df.loc[train_df["Model"] == model].dropna(axis=1).drop(columns=["Step", "Model"])
    avg_vals = np.array(data_only_df.mean(axis=1))
    std_vals = np.array(data_only_df.std(axis=1))
    metrics = {}
    metrics['last_mean'] = float(avg_vals[-1]) # Average across trials of the last value obtained (after num_epochs)
    metrics['last_std'] = float(std_vals[-1])  # Std across trials of the last value obtained (after num_epochs)
    metrics['min_mean'] = float(avg_vals.min())
    metrics['min_std'] = float(std_vals[avg_vals.argmin()])
    metrics['min_step'] = int(avg_vals.argmin())
    metrics['num_epochs'] = int(data_only_df.shape[0])
    metrics['num_trials'] = int(data_only_df.shape[1])
    train_stats[model] = metrics

val_stats = {}
for model in val_df.Model.unique():
    data_only_df = val_df.loc[val_df["Model"] == model].dropna(axis=1).drop(columns=["Step", "Model"])
    avg_vals = np.array(data_only_df.mean(axis=1))
    std_vals = np.array(data_only_df.std(axis=1))
    metrics = {}
    metrics['last_mean'] = float(avg_vals[-1]) # Average across trials of the last value obtained (after num_epochs)
    metrics['last_std'] = float(std_vals[-1])  # Std across trials of the last value obtained (after num_epochs)
    metrics['min_mean'] = float(avg_vals.min())
    metrics['min_std'] = float(std_vals[avg_vals.argmin()])
    metrics['min_step'] = int(avg_vals.argmin())
    metrics['num_epochs'] = int(data_only_df.shape[0])
    metrics['num_trials'] = int(data_only_df.shape[1])
    val_stats[model] = metrics


stats = {
    'train': train_stats,
    'val': val_stats
}

print(yaml.dump(stats, sort_keys=False, default_flow_style=False))

# TODO: Modify here to save the stats somewhere if needed

train_df = train_df.melt(id_vars=["Step", "Model"],
             var_name="Seed",
             value_name="Loss")

val_df = val_df.melt(id_vars=["Step", "Model"],
             var_name="Seed",
             value_name="Loss")


g_train = sns.relplot(x="Step", y="Loss", hue="Model", kind="line", ci="sd", data=train_df)
g_train.fig.suptitle("Training Loss")
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()
plt.close()

g_val = sns.relplot(x="Step", y="Loss", hue="Model", kind="line", ci="sd", data=val_df)
g_val.fig.suptitle("Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()
plt.close()

# TODO: Modify here to save the plots if needed

# g = sns.relplot(x="Step", y="Loss", hue="Model", kind="line", ci="sd", data=df, legend=False)
# plt.legend(['patch', '_nolegend_', 'patch-vel', '_nolegend_', 'patch-Fourier-vel', '_nolegend_'])