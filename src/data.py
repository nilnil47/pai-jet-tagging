import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

def generate_combained_constituent_dataset(files: dict[str, int], data_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    generate merged np array of the constituent for all files in the list
    """
    traks = []
    high_level_data = []
    labels = []
    for file, label in files.items():
        high_level_values_df = pd.read_hdf(f"data/{file}_small.h5")
        with h5py.File(f"data/{file}_const.h5", 'r') as f:
            consts = f['constituents'][:data_size]
            consts = normalize_constituents(consts, high_level_values_df, data_size)
            labels.append(label * np.ones((consts.shape[0], 1)))
            high_level_data.append(high_level_values_df.to_numpy()[:data_size])
            traks.append(consts)

    return np.concatenate(traks, axis=0), np.concatenate(labels, axis=0), np.concatenate(high_level_data, axis=0)

def generate_combined_high_level_dataset(files: dict[str, int]) -> pd.DataFrame:
    """
    crete merged dataset of the high level features (p, eta, phi, mass) with the labels in the dataframe
    """
    df = pd.DataFrame()
    for file, label in files.items():
        single_file_df = pd.read_hdf(f"data/{file}_small.h5")
        single_file_df['label'] = label
        df = pd.concat([df, single_file_df])
    return df

def load_jets_data(data_size: int = -1, multi_class=False) -> tuple:
    """Load the jests data and perform all the steps needed for training."""
    
    # each file contains the constituent of different jet tpye
    files : dict = {
        'qcd': 0,
        'top': 1,
        'wz': 2 if multi_class else 1,
    }


    tracks, labels, high_level_data = generate_combained_constituent_dataset(files, data_size)
    x_train, x_test_validation, h_train, h_test_validation, y_train, y_test_validation = train_test_split(tracks, high_level_data, labels, test_size=0.1)
    x_valid, x_test, h_valid, h_test, y_valid, y_test = train_test_split(x_test_validation, h_test_validation, y_test_validation, test_size=0.5)

    return (x_train, h_train, y_train), (x_valid, h_valid, y_valid), (x_test, h_test, y_test)

def normalize_constituents(constituents, high_level_df, data_size):
    """
    Normalize the eta and phi values of the constituents
    """
    returned_constituents = constituents.copy()
    returned_constituents[:,:,1] = high_level_df.eta.to_numpy()[:data_size,np.newaxis] - returned_constituents[:,:,1]
    returned_constituents[:,:,2] = high_level_df.phi.to_numpy()[:data_size,np.newaxis] - returned_constituents[:,:,2]
    return returned_constituents

def plot_jet_map(constituents, wights, title=None, show_colorbar=True):
    """
    Plot the jet map of the constituents
    """
    eta = constituents[:,1]
    phi = constituents[:,2]
    plt.hist2d(eta,phi,weights=wights,bins=(np.arange(-0.975,0.975,0.1),np.arange(-0.975,0.975,0.1)), norm=matplotlib.colors.LogNorm())
    
    if title:
        plt.title(title)

    if show_colorbar:
        plt.colorbar()

def plot_histogram(df: pd.DataFrame, var: str, labels: dict[str, int]):
    """
    Plot the histogram of the variable for each label
    labels: a dictionary of the labels names and their value represention in the dataframe
    
    for example:
    labels = {
        'QCD': 0,
        'Top': 1
    }
    """
    for label, value in labels.items():
        plt.hist(df[df['label'] == value][var], bins=50, alpha=0.5, label=f'{label}')
    plt.xlabel(var)
    plt.title(f'Histogram of {var}')
    plt.legend()