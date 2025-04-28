import pathlib
import os
from preprocess import preprocess_pipeline
import numpy as np
import yaml

def dataset():
    # Define the path to the dataset
    dataset_path = pathlib.Path("Speaker Identification_Dataset-2021")
    
    # Check if the path exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"The dataset path {dataset_path} does not exist.")
    
    # List all files in the directory
    x = []
    y = []
    for spk in dataset_path.iterdir():
        for wave_file in spk.iterdir():
            wav, label = preprocess_pipeline(wave_file, str(spk.name))
            x.append(wav)
            y.append(label)

    np_x = np.array(x).reshape(130,-1)

    np_y = np.array(y)
    return np_x, np_y

if __name__ == "__main__":
    dataset()