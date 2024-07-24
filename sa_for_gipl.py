import os
from pathlib import Path
import csv
import numpy as np
from pyDOE import lhs
import shutil
from typing import List, Tuple
from multiprocessing import Pool
import subprocess
import yaml
from typing import Tuple, List, Set, Union


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


ROOT_OUTPUT_PATH = Path(f"{os.getenv('HOME')}/gipl_experiments")
GIPL_PATH = Path(f"{os.getenv('HOME')}/GIPL")
GIPL_INPUT_PATH = GIPL_PATH / "in"
GIPL_OUTPUT_PATH = GIPL_PATH / "out"

MINERAL_VARIABLE_MAPPING = {
    0: "vwc",
    1: "a",
    2: "b",
    3: "cap_th",
    4: "cap_fr",
    5: "k_th",
    6: "k_fr",
    7: "depth",
}


def create_experiment_directories(n_samples: int) -> List[Path]:
    if ROOT_OUTPUT_PATH.exists():
        shutil.rmtree(ROOT_OUTPUT_PATH)

    # create directories
    ROOT_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    experiment_paths = []
    for i in range(n_samples):
        experiment_dir_path = ROOT_OUTPUT_PATH / f"gipl_{i}"
        experiment_dir_path.mkdir()
        experiment_paths.append(experiment_dir_path)

        shutil.copy(GIPL_PATH / "gipl", experiment_dir_path)
        shutil.copy(GIPL_PATH / "gipl_config.cfg", experiment_dir_path)
        shutil.copytree(GIPL_INPUT_PATH, experiment_dir_path / "in")
        Path(experiment_dir_path / "out").mkdir(exist_ok=True)

    return experiment_paths


def read_mineral_file(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Reads a mineral data file and returns the header and data matrix.

    Args:
        path (str): The file path to the mineral data file.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple containing:
            - A list of strings representing the header information from the first two lines of the file.
            - A numpy array representing the data matrix extracted from the subsequent lines of the file.
    """
    with open(path) as file:
        data = file.readlines()

    data_matrix = []
    # The reason we are skipping the first two lines is because mineral.txt
    # contains input details. We return the first two lines at the end
    # because we will need it when we want to create another mineral.txt.
    for line in data[2:]:
        row = [float(num) for num in line.strip().split("\t")]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    return data[:2], data_matrix


def get_mineral_values_to_perturb(path: str, data: dict) -> Tuple[List[str], np.ndarray, np.ndarray]:
    orig_header, orig_data = read_mineral_file(path)
    matrix = np.zeros_like(orig_data, dtype=np.ndarray)

    for row_i, row in data.items():
        for col_i, elem in enumerate(row):
            if elem == 1:
                matrix[row_i, col_i] = orig_data[row_i, col_i]
            elif isinstance(elem, list):
                matrix[row_i, col_i] = elem

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


def get_sample_data(data):
    header = []
    for (_, col_j), elem in np.ndenumerate(data[0]):
        if elem != 0:
            header.append(MINERAL_VARIABLE_MAPPING.get(col_j))

    sample_data = []
    for sample_i in range(len(data)):
        temp = []
        for (_, _), elem in np.ndenumerate(data[sample_i]):
            if elem != 0:
                temp.append(round(elem, 2))

        sample_data.append(temp)

    return header, sample_data


def write_sample_csv(path, header, data):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


def run_gipl(path):
    print("Running GIPL...")
    os.chdir(path.parent)
    subprocess.run([path])
    print("Done running GIPL...")


def read_yaml() -> dict:
    with open("config.yaml") as config_file:
        try:
            config = yaml.safe_load(config_file)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return {}


def parse_config(config_data: dict) -> Tuple[int, int, dict]:
    perturbation_range = config_data.get("perturbation_range")
    n_samples = config_data.get("n_samples")
    perturbation_matrix = config_data.get("data")[0]

    return perturbation_range, n_samples, perturbation_matrix


def get_lhs_value(interval: Union[Set[float], List[float]]) -> float:
    """
    Generates a single sample from a given interval using Latin Hypercube Sampling.

    Parameters:
    interval (tuple): A tuple representing the (min, max) of the interval.

    Returns:
    float: A sampled value.
    """
    min_val, max_val = interval
    sample = lhs(n=1, samples=1)

    # Scale the sample to the interval
    sample = min_val + sample * (max_val - min_val)

    return sample.flatten()[0]


def perturb_value(value: float, percentage: int) -> float:
    """
    Perturbs a given float value by a specified percentage using Latin Hypercube Sampling.

    Parameters:
    value (float): The original value to be perturbed.
    percentage (int): The percentage by which to perturb the value.

    Returns:
    float: The perturbed value.
    """
    # Calculate the interval for perturbation
    perturbation_range = value * (percentage / 100)
    interval = (value - perturbation_range, value + perturbation_range)

    return get_lhs_value(interval)


def apply_lhs(data, n_samples=5, perturbation_range=0.1) -> np.ndarray:
    """
    Applies Latin Hypercube Sampling to the given matrix for each of the samples.

    Args:
        data (np.ndarray): The matrix that holds values to be perturbed.
        n_samples (int): The number of samples to be generated
        perturbation_range (float): Perturbation percentage. If given, values will be
            perturbed within that range. For instance, if `perturbation_range` is 10,
            the values will perturbed within -10% and +10%.

    Returns:
        np.ndarray: A three-dimensional numpy array. In other words, a list of matrices.
    """
    n_rows, n_cols = data.shape

    perturbed_samples = np.zeros((n_samples, n_rows, n_cols))

    for sample in range(n_samples):
        perturbed_data = np.zeros_like(data)
        for i in range(n_rows):
            for j in range(n_cols):
                if isinstance(data[i, j], list):
                    # data[i, j] in this case contains the interval
                    perturbed_data[i, j] = get_lhs_value(data[i, j])
                elif data[i, j] != 0:
                    perturbed_data[i, j] = perturb_value(data[i, j], perturbation_range)
                else:
                    perturbed_data[i, j] = 0

        perturbed_samples[sample, :, :] = perturbed_data

    return perturbed_samples


if __name__ == "__main__":
    perturbation_range, n_samples, perturbation_matrix = parse_config(read_yaml())
    orig_header, orig_data, mineral_values = get_mineral_values_to_perturb(GIPL_INPUT_PATH / "mineral.txt", perturbation_matrix)
    samples = apply_lhs(mineral_values, n_samples, perturbation_range)

    sample_csv_path = ROOT_OUTPUT_PATH / "sample.csv"
    sample_header, sample_data = get_sample_data(samples)
    experiment_paths = create_experiment_directories(n_samples)
    write_sample_csv(sample_csv_path, sample_header, sample_data)

    for index, path in enumerate(experiment_paths):
        mineral_file_path = path / "in" / "mineral.txt"
        update_mineral_file(mineral_file_path, orig_header, orig_data, samples[index])

    gipl_paths = [path / "gipl" for path in experiment_paths]
    with Pool() as pool:
        result = pool.map(run_gipl, gipl_paths)
