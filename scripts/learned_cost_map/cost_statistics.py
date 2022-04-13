# Cost function mean evaluator
import yaml
import numpy as np
import os
import pandas as pd

cost_functions = {
    "freq_band_1_30": {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0
    },
    "freq_band_3_30": {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0
    },
    "freq_band_5_15": {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0
    },
    "freq_bins_3": {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0
    },
    "freq_bins_5": {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0
    },
    "freq_bins_15": {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0
    },
}

def main():
    # 1. Obtain statistics for each of the cost functions in the dictionary. This will be a bit hard coded.

    freq_band_costs = pd.read_csv("/home/mateo/Data/SARA/TartanCost/freq_cost_functions.csv")

    freq_band_costs = freq_band_costs.rename(columns={
                   'cost_1' :'1_3_Hz',
                   'cost_2' :'1_5_Hz',
                   'cost_3' :'1_8_Hz',
                   'cost_4' :'1_12_Hz',
                   'cost_5' :'1_15_Hz',
                   'cost_6' :'1_20_Hz',
                   'cost_7' :'1_30_Hz',
                   'cost_8' :'3_5_Hz',
                   'cost_9' :'3_8_Hz',
                   'cost_10':'3_12_Hz',
                   'cost_11':'3_15_Hz',
                   'cost_12':'3_20_Hz',
                   'cost_13':'3_30_Hz',
                   'cost_14':'5_8_Hz',
                   'cost_15':'5_12_Hz',
                   'cost_16':'5_15_Hz',
                   'cost_17':'5_20_Hz',
                   'cost_18':'5_30_Hz',
                   'cost_19':'8_12_Hz',
                   'cost_20':'8_15_Hz',
                   'cost_21':'8_20_Hz',
                   'cost_22':'8_30_Hz',
                   'cost_23':'12_15_Hz',
                   'cost_24':'12_20_Hz',
                   'cost_25':'12_30_Hz',
                   'cost_26':'15_20_Hz',
                   'cost_27':'15_30_Hz',
                   'cost_28':'20_30_Hz',
                   })

    freq_bins_costs = pd.read_csv("/home/mateo/Data/SARA/TartanCost/freq_bins_cost_functions.csv")


    cost_functions["freq_band_1_30"]["mean"] = float(freq_band_costs["1_30_Hz"].mean())
    cost_functions["freq_band_1_30"]["std"]  = float(freq_band_costs["1_30_Hz"].std())
    cost_functions["freq_band_1_30"]["min"]  = float(freq_band_costs["1_30_Hz"].min())
    cost_functions["freq_band_1_30"]["max"]  = float(freq_band_costs["1_30_Hz"].max())

    cost_functions["freq_band_3_30"]["mean"] = float(freq_band_costs["3_30_Hz"].mean())
    cost_functions["freq_band_3_30"]["std"]  = float(freq_band_costs["3_30_Hz"].std())
    cost_functions["freq_band_3_30"]["min"]  = float(freq_band_costs["3_30_Hz"].min())
    cost_functions["freq_band_3_30"]["max"]  = float(freq_band_costs["3_30_Hz"].max())

    cost_functions["freq_band_5_15"]["mean"] = float(freq_band_costs["5_15_Hz"].mean())
    cost_functions["freq_band_5_15"]["std"]  = float(freq_band_costs["5_15_Hz"].std())
    cost_functions["freq_band_5_15"]["min"]  = float(freq_band_costs["5_15_Hz"].min())
    cost_functions["freq_band_5_15"]["max"]  = float(freq_band_costs["5_15_Hz"].max())

    cost_functions["freq_bins_3"]["mean"] = float(freq_bins_costs["3_bins"].mean())
    cost_functions["freq_bins_3"]["std"]  = float(freq_bins_costs["3_bins"].std())
    cost_functions["freq_bins_3"]["min"]  = float(freq_bins_costs["3_bins"].min())
    cost_functions["freq_bins_3"]["max"]  = float(freq_bins_costs["3_bins"].max())

    cost_functions["freq_bins_5"]["mean"] = float(freq_bins_costs["5_bins"].mean())
    cost_functions["freq_bins_5"]["std"]  = float(freq_bins_costs["5_bins"].std())
    cost_functions["freq_bins_5"]["min"]  = float(freq_bins_costs["5_bins"].min())
    cost_functions["freq_bins_5"]["max"]  = float(freq_bins_costs["5_bins"].max())

    cost_functions["freq_bins_15"]["mean"] = float(freq_bins_costs["15_bins"].mean())
    cost_functions["freq_bins_15"]["std"]  = float(freq_bins_costs["15_bins"].std())
    cost_functions["freq_bins_15"]["min"]  = float(freq_bins_costs["15_bins"].min())
    cost_functions["freq_bins_15"]["max"]  = float(freq_bins_costs["15_bins"].max())

    # import pdb;pdb.set_trace()
    
    # 2. Dump cost function statistics into YAML file. This will be used to load the right statistics for the right cost function

    cost_statistics_fp = "/home/mateo/Data/SARA/TartanCost/cost_statistics.yaml"

    with open(cost_statistics_fp, 'w') as outfile:
        yaml.safe_dump(cost_functions, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()