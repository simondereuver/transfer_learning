import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

directory = 'results'
subdir = 'NVIDIA_A100_80GB_PCIe_MIG_1g.10gb'

for csv_file in glob.glob(os.path.join(directory, subdir, "*.csv")):
    data = pd.read_csv(csv_file)

    data.columns = data.columns.str.strip()

    data['Epoch'] = pd.to_numeric(data['Epoch'], errors='coerce')

    data['Val Loss'] = data['Val Loss'] / data['Val Loss'].max()
    data['Test Loss'] = data['Test Loss'] / data['Test Loss'].max()

    data.rename(columns={'Val Loss': 'Validation Loss', 'Val Acc': 'Validation Accuracy', 'Test Acc': 'Test Accuracy'}, inplace=True)    
    melted = pd.melt(
        data, 
        id_vars=["Epoch"], 
        value_vars=["Validation Loss", "Validation Accuracy", "Test Loss", "Test Accuracy"],
        var_name="Metric", 
        value_name="Value"
    )
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    if '_' in base_name:
        base_name = base_name.split('_')[0]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted, x="Epoch", y="Value", hue="Metric", marker="o")
    if base_name == "cvd":
        fancy_base_name = "CvD"
    elif base_name == "stanford":
        fancy_base_name == "Stanford"
    elif "2" in base_name:
        fancy_base_name = "Experiment 2"
    elif "3" in base_name:
        fancy_base_name = "Experiment 3"
    elif "4" in base_name:
        fancy_base_name = "Experiment 4"

    plt.title(f"Metrics over Epochs for {fancy_base_name} ({subdir})")
    plt.ylabel("Value (Scaled 0-1)")
    plt.xlabel("Epoch")
    plt.tight_layout()

    save_dir = "images"
    plot_filename = os.path.join(save_dir, base_name + ".png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot for {csv_file} as {plot_filename}")
