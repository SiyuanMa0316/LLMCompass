import subprocess
import pathlib
import os
from pathlib import Path
import csv
import json
from hardware_model.device import Device

def run_gemm(device:Device, M,K,N, input_acc, accumulate_acc, compute_only=False, debug=False):
    if M==1:
        wkld = "gemv"
    else:
        wkld = "gemm"
    # input_acc = 4
    # accumulate_acc = 16
    array_count=device.compute_module.tile.arr_count
    array_cols=device.compute_module.tile.arr_cols
    array_rows=device.compute_module.tile.arr_rows

    d = os.path.dirname(os.path.abspath(__file__))
    path = Path(d)
    #read path from json
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
        PIMSAB_PATH = config["PIMSAB_PATH"]

    # PIMSAB_PATH = "/home/siyuan/pim-simulator"
    CFG_PATH = f"{PIMSAB_PATH}/configs/DMesh12x10_TileSize{array_count}_CramSize{array_rows}x{array_cols}.cfg"
        
    
    EXEC_PATH = f"{PIMSAB_PATH}/build/PIM_simulator"
    print(f"{CFG_PATH}, {EXEC_PATH}")
    if compute_only:
        OUTPUT_PATH = os.path.join(path, f"output_M{M}_K{K}_N{N}_i{input_acc}_a{accumulate_acc}_compute_only_{array_count}_{array_cols}_{array_rows}")
    else:
        OUTPUT_PATH = os.path.join(path, f"output_M{M}_K{K}_N{N}_i{input_acc}_a{accumulate_acc}_{array_count}_{array_cols}_{array_rows}")
    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True) 

    
    #check if log file already exists
    if not os.path.exists(f"{OUTPUT_PATH}/{wkld}_tiled_M{M}_K{K}_N{N}_i{input_acc}_a{accumulate_acc}.log"):
        #create param file based on M,K,N
        param_file = f"{wkld}_tiled_M{M}_K{K}_N{N}_i{input_acc}_a{accumulate_acc}.param"
        #write param file
        with open(os.path.join(path,wkld, param_file), "w") as f:
            f.write(f"M {M}\nK {K}\nN {N}\ninput_acc {input_acc}\naccumulate_acc {accumulate_acc}")
            f.close()
        if compute_only:
            task = f'{wkld}_tiled_compute_only'
        else:
            task = f'{wkld}_tiled'
        if debug:
            print(f"Running {task} with {param_file}")
        subprocess.run([EXEC_PATH, "-c", CFG_PATH, "-m", task, "-p", f"{str(path)}/{wkld}/{param_file}", "-l", f"{OUTPUT_PATH}/{wkld}_tiled_M{M}_K{K}_N{N}_i{input_acc}_a{accumulate_acc}.log"])

        if debug:
            print(f"Done running {task} with {param_file}")

    if debug:
        print(f"extracting results from {OUTPUT_PATH}/{wkld}_tiled_M{M}_K{K}_N{N}_i{input_acc}_a{accumulate_acc}.log")
    #extract results
    with open(f"{OUTPUT_PATH}/{wkld}_tiled_M{M}_K{K}_N{N}_i{input_acc}_a{accumulate_acc}.log", 'r') as f:
        for line in f:
            if "clocks" in line:
                words = line.split()
                cycle = int(words[4])
                break
    if debug:
        print("total cycles: "+str(cycle))
    #energy
    energy = 0
    with open(f"{OUTPUT_PATH}/{wkld}_tiled_M{M}_K{K}_N{N}_i{input_acc}_a{accumulate_acc}.log.energy.csv", 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            #print(f'file {filename}:')
            energy = float(row["Total_Dynamic_Energy"]) + float(row["Total_Static_Energy"])
            if debug:
                print(f'\t{row["WorkloadName"]} has total energy = {energy}, dynamic energy = {row["Total_Dynamic_Energy"]}, and total static energy = {row["Total_Static_Energy"]}.')
    latency = cycle/device.compute_module.clock_freq
    if debug:
        print(f"latency: {latency}")
    return latency, energy

# run gemm at ae/... instead of here
# time, energy = run_gemm(2048, 3072, 4096, debug=True)
# print(f"Total time: {time}, Total energy: {energy}")