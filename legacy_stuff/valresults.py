
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import yaml

df = pd.read_csv("/home/alexdenker/pet/PETRIC-UCL-EWS/runs/ews/2024-09-06_14-43-11/mMR_NEMA/results.csv")
metric_keys = list(df.keys())[2:]

base_path = "/home/alexdenker/pet/PETRIC-UCL-EWS/runs_validate4/cursed_bsrem"
files = [os.path.join(base_path, f, "mMR_NEMA") for f in os.listdir(base_path)]

metrics = {}
for key in metric_keys:
    if "AEM" in key or "RMSE" in key:
        metrics[key] = [] 

#print(metrics)

best_metrics = None 
best_cfg = None 
best_val = 1

for f_idx, f in enumerate(files):
    try:
        df = pd.read_csv(os.path.join(f, "results.csv"))
        with open(os.path.join(f, "config.yaml")) as file_:
            cfgdict = yaml.safe_load(file_)

        final_results = df.iloc[-1]
        if final_results["RMSE_whole_object"] < best_val:
            best_metrics = cfgdict
            best_metrics = df.iloc[-1]
            best_val = final_results["RMSE_whole_object"]
        #print(cfgdict)
        #print(df.iloc[-1])
    except FileNotFoundError:
        continue

print(best_val)