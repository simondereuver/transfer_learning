import os
import glob
import csv

directory = 'results/NVIDIA_A100_80GB_PCIe_MIG_1g.10gb'

for txt_file in glob.glob(os.path.join(directory, "*.txt")):
    base_name = os.path.splitext(os.path.basename(txt_file))[0]
    csv_file = os.path.join(directory, base_name + ".csv")
    
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if line.startswith("│"):
            row = [cell.strip() for cell in line.strip().strip("│").split("│")]
            data.append(row)

    with open(csv_file, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerows(data)
    
    print(f"Converted {txt_file} to {csv_file}")
