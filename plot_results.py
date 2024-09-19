
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import yaml

def cfg_to_name(cfg, file_id):
    try:
        method = cfg["method"]
    except KeyError:
        method = "osem"
    if method == "bsrem":
        name = f"{method}, subsets={cfg["num_subsets"]}, gamma={cfg["gamma"]} [{file_id}]" # {cfg["num_subsets"]}
        return name     
    if method == "ews":
        name = f"{method}, (ews={cfg["ews_method"]}) [{file_id}]" # {cfg["num_subsets"]}
        return name     
        
    if method == "bsrem_bb": 
        try:
            name = f"{method}, {cfg["num_subsets"]} ({cfg["mode"]}), beta={cfg["beta"]}, bb init {cfg["bb_init_mode"]} [{file_id}]" 
        except KeyError:
            name = f"{method}, ({cfg["mode"]}), beta={cfg["beta"]}, bb init {cfg["bb_init_mode"]} [{file_id}]" 

        return name 
    if method == "adam":
        name = f"{method}, {cfg["num_subsets"]} ({cfg["mode"]}), init lr={cfg["initial_step_size"]} [{file_id}]" 
        return name 
    if method == "adadelta":
        name = f"{method}, {cfg["num_subsets"]} ({cfg["mode"]}), init lr={cfg["initial_step_size"]} [{file_id}]" 
        return name 
    return f"{method} [{file_id}]"  


methods = ["Mediso_NEMA", "mMR_NEMA", "NeuroLF_Hoffman", "Siemens_mMR_ACR", "Vision600_thorax"]

for method in methods:
    print(method)
    df = pd.read_csv(f"/home/alexdenker/pet/PETRIC-UCL-EWS/logs/osem/2024-09-16_09-06-06/{method}/results.csv")
    metric_keys = list(df.keys())[2:]

    print(metric_keys)

    base_path = "/home/alexdenker/pet/PETRIC-UCL-EWS/logs/osem"
    files = [os.path.join(base_path, f, method) for f in os.listdir(base_path)]

    base_path_bsrem = "/home/alexdenker/pet/PETRIC-UCL-EWS/logs/bsrem"
    files = files + [os.path.join(base_path_bsrem, f, method) for f in os.listdir(base_path_bsrem)]

    fig, axes = plt.subplots(3, 4, figsize=(16,8))
    ax_ = list(axes.ravel())

    for f_idx, f in enumerate(files):
        try:
            df = pd.read_csv(os.path.join(f, "results.csv"))
            with open(os.path.join(f, "config.yaml")) as file_:
                cfgdict = yaml.safe_load(file_)

            
            for idx in range(len(metric_keys)):
                
                ax = ax_[idx]
                ax.set_title(metric_keys[idx])

                label_name = cfg_to_name(cfgdict, file_id=f.split("/")[-2])
                #ax.plot(df["time"], df[metric_keys[idx]], label=label_name)
                try:
                    if "step" in metric_keys[idx] or "loss" in metric_keys[idx]:
                        ax.plot(df["time"], df[metric_keys[idx]], "-", label=label_name)
                        #ax.plot(df[metric_keys[idx]], label=label_name)

                    else:
                        ax.semilogy(df["time"], df[metric_keys[idx]], "-", label=label_name)
                        
                        #ax.semilogy(df[metric_keys[idx]], label=label_name)

                    if "AEM" in metric_keys[idx]:
                        ax.hlines(0.005, 0, df["time"].max(), color="k")
                        ax.set_ylim(0.0001, 0.6)
                    if "RMSE_whole_object" in metric_keys[idx]:
                        ax.hlines(0.01, 0, df["time"].max(), color="k")
                        ax.set_ylim(0.001, 0.9)
                    if "RMSE_background" in metric_keys[idx]:
                        ax.hlines(0.01, 0, df["time"].max(), color="k")       
                        ax.set_ylim(0.001, 0.9)
                    if "change" in metric_keys[idx]:
                        ax.set_ylim(0.0001, 10.0)
        
                except KeyError:
                    if "step" in metric_keys[idx] or "loss" in metric_keys[idx]:
                        ax.plot(np.nan)
                    else:
                        ax.semilogy(np.nan)

                ax.legend(fontsize=7)
        except FileNotFoundError:
            pass

    ax_ = list(axes.ravel())
    ax_[-1].axis("off")
    #ax_[-2].axis("off")


    #for ax in axes:
    #    ax.set_xlim([0, 500])
    fig.suptitle(method)
    fig.tight_layout()
    plt.show()

