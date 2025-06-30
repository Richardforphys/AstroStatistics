import h5py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def load_data(data_path):
    print('loading data')
    with h5py.File(data_path, 'r') as f:
        y = f['det'][()]
        keys = [key for key in f.keys() if key != 'det']
        arrays = [f[key][()].reshape(len(f[key]), -1) for key in keys]
        data = np.concatenate(arrays, axis=1)
    nan_mask = ~np.isnan(data).any(axis=1)
    data = data[nan_mask]
    y = y[nan_mask]
    
    return y, data, keys
    
def downsample_uniform(data, y, factor=10):
    print('Uniform Downsample')
    return y[::factor], data[::factor]

def downsample(data, y, ratio=1.0):
    print('Downsampling')
    # Trova indici positivi e negativi
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    # Quanti negativi vogliamo tenere?
    n_pos = len(idx_pos)
    n_neg_keep = int(n_pos * ratio)

    # Se vuoi sottocampionare negativi casualmente
    np.random.seed(42)  # Per riproducibilità
    idx_neg_down = np.random.choice(idx_neg, size=n_neg_keep, replace=False)

    # Unisci gli indici e mescola
    idx_final = np.concatenate([idx_pos, idx_neg_down])
    np.random.shuffle(idx_final)

    return y[idx_final], data[idx_final]

def downsample_unbalanced(X, y, n_samples=20000, random_state=None):
    """
    Randomly downsamples the dataset to n_samples, preserving the original class imbalance.

    Parameters:
        X (np.ndarray): Feature array of shape (n_samples_original, n_features).
        y (np.ndarray): Label array of shape (n_samples_original,).
        n_samples (int): Total number of samples to keep after downsampling.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        y_down (np.ndarray): Downsampled labels.
        X_down (np.ndarray): Downsampled features.
    """
    assert len(X) == len(y), "X and y must have the same length."
    assert n_samples <= len(y), "n_samples must be less than or equal to original dataset size."

    print('Downsampling (unbalanced)...')
    
    rng = np.random.default_rng(random_state)
    
    # Randomly sample indices
    idx_sampled = rng.choice(len(y), size=n_samples, replace=False)
    
    return y[idx_sampled], X[idx_sampled]


def downsample_balanced(X, y, n_samples=20000, random_state=None):
    """
    Downsamples the dataset to n_samples while keeping the classes balanced.

    Parameters:
        X (np.ndarray): Feature array of shape (n_samples_original, n_features).
        y (np.ndarray): Label array of shape (n_samples_original,), must be binary (0 or 1).
        n_samples (int): Total number of samples to keep after downsampling.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        X_down (np.ndarray): Downsampled features.
        y_down (np.ndarray): Downsampled labels.
    """
    assert len(X) == len(y), "X and y must have the same length."
    assert n_samples <= len(y), "n_samples must be less than or equal to original dataset size."
    
    print('Downsampling...')
    
    rng = np.random.default_rng(random_state)

    # Find indices for each class
    idx_class0 = np.where(y == 0)[0]
    idx_class1 = np.where(y == 1)[0]

    # Number of samples per class
    n_per_class = n_samples // 2

    if len(idx_class0) < n_per_class or len(idx_class1) < n_per_class:
        raise ValueError("Not enough samples in one of the classes to perform balanced downsampling.")

    # Randomly sample from each class
    idx_sampled_0 = rng.choice(idx_class0, size=n_per_class, replace=False)
    idx_sampled_1 = rng.choice(idx_class1, size=n_per_class, replace=False)

    # Concatenate and shuffle
    idx_total = np.concatenate([idx_sampled_0, idx_sampled_1])
    rng.shuffle(idx_total)

    return y[idx_total], X[idx_total]

