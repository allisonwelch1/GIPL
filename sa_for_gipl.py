import os
from pathlib import Path
import numpy as np
from pyDOE import lhs
import shutil
from typing import List, Tuple
from multiprocessing import Pool
import subprocess
import yaml


"""
# How to use this script

1) Make sure to have the Fortan compiler and Git. Run `gfortran --version` and `git --version` to make sure they are installed.
2) Clone GIPL to the home directory: `cd $HOME && git clone https://github.com/Elchin/GIPL.git`
3) Move into the directory via `cd GIPL` and build the project via `make`
4) Install the dependencies for the script: `pip install pyDOE numpy pyyaml`
5) Copy the script to your home directory and run the script via `python3 sa_for_gipl.py`

The script creates a directory called `gipl_experiments` in the home directory.
5 sub-folders are created in this directory.

A sub-folder contains 
* the gipl binary (`gipl`)
* the configuration file (`gipl_config.cfg`)
* input files (`in/`)
* output files (`out/`)

config.yaml is the configuration file for the script. The explanation on how to change the variables are
explained in the file.
"""


#GIPL_PATH = Path(f"{os.getenv('HOME')}/GIPL")
GIPL_PATH = Path(f"{os.getcwd()}")
GIPL_INPUT_PATH = GIPL_PATH / "in"
GIPL_OUTPUT_PATH = GIPL_PATH / "out"


def create_experiment_directories(n_samples) -> List[Path]:
    root_output_dir = GIPL_PATH / "sa_runs"
    # Path(f"{os.getenv('HOME')}/gipl_experiments")

    if root_output_dir.exists():
        shutil.rmtree(root_output_dir)

    # create directories
    root_output_dir.mkdir(exist_ok=True, parents=True)
    experiment_paths = []
    for i in range(n_samples):
        experiment_dir_path = root_output_dir / f"gipl_{i}"
        experiment_dir_path.mkdir()
        experiment_paths.append(experiment_dir_path)

        shutil.copy(GIPL_PATH / "gipl", experiment_dir_path)
        shutil.copy(GIPL_PATH / "gipl_config.cfg", experiment_dir_path)
        shutil.copytree(GIPL_INPUT_PATH, experiment_dir_path / "in")
        Path(experiment_dir_path / "out").mkdir(exist_ok=True)

    return experiment_paths


def read_mineral_file(path) -> Tuple[List[str], List[float]]:
    with open(path) as file:
        data = file.readlines()

    input_details = data[:2]
    data_matrix = []
    for line in data[2:]:
        row = [float(num) for num in line.strip().split("\t")]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    return input_details, data_matrix


def get_mineral_data(path, data):
    orig_header, orig_data = read_mineral_file(path)
    matrix = np.zeros_like(orig_data)

    for row_i, row in data.items():
        for col_i, elem in enumerate(row):
            if elem == 1 or isinstance(elem, list):
                matrix[row_i, col_i] = orig_data[row_i, col_i]

    return orig_header, orig_data, matrix


def update_mineral_file(path: str, orig_header, orig_data, mineral_data) -> None:
    for (i, j), elem in np.ndenumerate(mineral_data):
        if elem == 0:
            mineral_data[i, j] = orig_data[i, j]
        else:
            mineral_data[i, j] = round(mineral_data[i, j], 2)

    with open(path, "w") as file:
        for line in orig_header:
            file.write(line)

        for row in mineral_data:
            row_str = "\t".join(map(str, row))
            file.write(row_str + "\n")


def get_perturbation_ranges(data: dict, default_range):
    ranges = []
    for _, col in data.items():
        temp_row = []
        for elem in col:
            if elem == 1:
                temp_row.append([-default_range/100, default_range/100])
            elif isinstance(elem, list):
                temp_row.append(elem)
            else:
                temp_row.append([0, 0])

        ranges.append(temp_row)

    return np.array(ranges)


def run_gipl(path):
    print("Running GIPL...")
    os.chdir(path.parent)
    subprocess.run([path])
    print("Done running GIPL...")


def read_yaml():
    with open("config.yaml") as config_file:
        try:
            config = yaml.safe_load(config_file)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def parse_config(config_data):
    perturbation_range = config_data.get("perturbation_range")
    n_samples = config_data.get("n_samples")
    data_matrix = config_data.get("data")[0]

    return perturbation_range, n_samples, data_matrix


def apply_lhs(data, perturbation_ranges, n_samples=5):
    n_rows, n_cols = data.shape

    perturbed_samples = np.zeros((n_samples, n_rows, n_cols))

    for sample in range(n_samples):
        perturbed_data = np.zeros_like(data)
        for i in range(n_rows):
            for j in range(n_cols):
                min_perturb, max_perturb = perturbation_ranges[i, j]
                if data[i, j] != 0:
                    lhs_sample = lhs(1, samples=1)
                    lhs_sample = lhs_sample * (max_perturb - min_perturb) + min_perturb
                    perturbed_data[i, j] = data[i, j] * (1 + lhs_sample)
                else:
                    perturbed_data[i, j] = data[i, j]

        perturbed_samples[sample, :, :] = perturbed_data

    return perturbed_samples


if __name__ == "__main__":
    perturbation_range, n_samples, data = parse_config(read_yaml())
    orig_header, orig_data, mineral_data = get_mineral_data(GIPL_INPUT_PATH / "mineral.txt", data)
    perturbation_ranges = get_perturbation_ranges(data, perturbation_range)
    samples = apply_lhs(mineral_data, perturbation_ranges, n_samples)
    experiment_paths = create_experiment_directories(n_samples)
    for index, path in enumerate(experiment_paths):
        mineral_file_path = path / "in" / "mineral.txt"
        update_mineral_file(mineral_file_path, orig_header, orig_data, samples[index])

    gipl_paths = [path / "gipl" for path in experiment_paths]
    with Pool() as pool:
        result = pool.map(run_gipl, gipl_paths)
