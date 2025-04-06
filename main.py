import os
import tensorflow
import tensorflow.keras as keras
from experiments import tl_ex1, tl_ex2, tl_ex3, tl_ex4
from datamodule import make_table
from settings import EPOCHS, LR, BATCH_SIZE

def main():
    """Main loop handles execution of all experiments. Each experiment is divided into a separate function returning the best model and its results."""
    ##### EXPERIMENTS #####
    # pre
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy('mixed_float16')

    # experiment 1
    print("Starting experiment 1 ...")
    data_path_catsvdogs = "data/PetImages"
    data_path_stanford = "data/stanford_dogs/Images"

    stanford_model, datasets, results1 = tl_ex1(data_path_catsvdogs, data_path_stanford, EPOCHS, LR, BATCH_SIZE)
    results_cvd = results1["cvd"]
    results_stanford = results1["stanford"]
    print("CVD model:")
    for key, value in results_cvd.items():
        if "final" in key:
            print(f"{key}: {value}")
    print("Stanford model:")
    for key, value in results_stanford.items():
        if "final" in key:
            print(f"{key}: {value}")
    stanford_model.summary()
    print("Experiment 1 finished.")

    # experiment 2
    print("Starting experiment 2 ...")
    
    results2 = tl_ex2(stanford_model, EPOCHS, LR, *datasets)
    for key, value in results2.items():
        if "final" in key:
            print(f"{key}: {value}")

    print("Experiment 2 finished.")

    # experiment 3
    print("Starting experiment 3 ...")

    results3 = tl_ex3(stanford_model, EPOCHS, LR, *datasets)
    for key, value in results3.items():
        if "final" in key:
            print(f"{key}: {value}")

    print("Experiment 3 finished.")

    # experiment 4
    print("Starting experiment 4 ...")

    results4 = tl_ex4(stanford_model, EPOCHS, LR, *datasets)
    for key, value in results4.items():
        if "final" in key:
            print(f"{key}: {value}")

    print("Experiment 4 finished.")

    # combine all results from each experiment and print in a tabular table with tablefmt="mixed_outline"
    # and put each experiment result on a new row, and the val/test accuracy and loss in the columns
    # where if a results dict does not contain test or val put NaN instead. Also save to results/results.txt

    table_cvd      = make_table("Cats vs Dogs Base Model", results_cvd, EPOCHS)
    table_stanford = make_table("Stanford", results_stanford, EPOCHS)
    table_ex2      = make_table("Experiment 2", results2, EPOCHS)
    table_ex3      = make_table("Experiment 3", results3, EPOCHS)
    table_ex4      = make_table("Experiment 4", results4, EPOCHS)

    gpus = tensorflow.config.list_physical_devices('GPU')
    if gpus:
        gpu_details = tensorflow.config.experimental.get_device_details(gpus[0])
        gpu_name = gpu_details.get("device_name", "UnknownGPU")
    else:
        gpu_name = "NoGPU"

    gpu_name_path = gpu_name.replace(" ", "_")
    results_path = "results/" + gpu_name_path
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(f"{results_path}/cvd_results.txt", "w") as f:
        f.write(table_cvd)
    with open(f"{results_path}/stanford_results.txt", "w") as f:
        f.write(table_stanford)
    with open(f"{results_path}/experiment2_results.txt", "w") as f:
        f.write(table_ex2)
    with open(f"{results_path}/experiment3_results.txt", "w") as f:
        f.write(table_ex3)
    with open(f"{results_path}/experiment4_results.txt", "w") as f:
        f.write(table_ex4)

    print("All experiments finished, saved results to results/results.txt")

if __name__ == "__main__":
    main()
