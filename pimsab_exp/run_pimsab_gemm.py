import subprocess
import pathlib
import os
from pathlib import Path
import csv
import json

def run_gemm(M,K,N, compute_only=False, debug=False):
    d = os.path.dirname(os.path.abspath(__file__))
    path = Path(d)
    #read path from json
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
        PIMSAB_PATH = config["PIMSAB_PATH"]

    # PIMSAB_PATH = "/home/siyuan/pim-simulator"

    CFG_PATH = f"{PIMSAB_PATH}/configs/DMesh12x10_TileSize256_CramSize256x256.cfg"
    EXEC_PATH = f"{PIMSAB_PATH}/build/PIM_simulator"

    if compute_only:
        OUTPUT_PATH = os.path.join(path, f"output_M{M}_K{K}_N{N}_compute_only")
    else:
        OUTPUT_PATH = os.path.join(path, f"output_M{M}_K{K}_N{N}")
    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True) 

    #check if log file already exists
    if not os.path.exists(f"{OUTPUT_PATH}/gemm_tiled_M{M}_K{K}_N{N}.log"):
        #create param file based on M,K,N
        param_file = f"gemm_tiled_M{M}_K{K}_N{N}.param"
        #write param file
        with open(os.path.join(path,"gemm", param_file), "w") as f:
            f.write(f"M {M}\nK {K}\nN {N}")
            f.close()
        if compute_only:
            task = 'gemm_tiled_compute_only'
        else:
            task = 'gemm_tiled'
        if debug:
            print(f"Running {task} with {param_file}")
        subprocess.run([EXEC_PATH, "-c", CFG_PATH, "-m", task, "-p", f"{str(path)}/gemm/{param_file}", "-l", f"{OUTPUT_PATH}/gemm_tiled_M{M}_K{K}_N{N}.log"])

        if debug:
            print(f"Done running {task} with {param_file}")

    if debug:
        print(f"extracting results from {OUTPUT_PATH}/gemm_tiled_M{M}_K{K}_N{N}.log")
    #extract results
    with open(f"{OUTPUT_PATH}/gemm_tiled_M{M}_K{K}_N{N}.log", 'r') as f:
        for line in f:
            if "clocks" in line:
                words = line.split()
                cycle = int(words[4])
                break
    if debug:
        print("total cycles: "+str(cycle))
    #energy
    energy = 0
    with open(f"{OUTPUT_PATH}/gemm_tiled_M{M}_K{K}_N{N}.log.energy.csv", 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            #print(f'file {filename}:')
            energy = float(row["Total_Dynamic_Energy"]) + float(row["Total_Static_Energy"])
            if debug:
                print(f'\t{row["WorkloadName"]} has total energy = {energy}, dynamic energy = {row["Total_Dynamic_Energy"]}, and total static energy = {row["Total_Static_Energy"]}.')
    if debug:
        print(f"latency: {cycle*1E-9/1.5}")
    return cycle*1E-9/1.5, energy


# time, energy = run_gemm(2048, 3072, 4096, debug=True)
# print(f"Total time: {time}, Total energy: {energy}")