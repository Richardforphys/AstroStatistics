import h5py
import numpy as np
import sys
sys.path.append(r"C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\Notebooks\Exam\LIGO")
from LIGO_Utils.LigoUtils import load_data, downsample_balanced, downsample_unbalanced

data_path = r"C:\Users\ricca\OneDrive\Documents\LIGO.h5"

y, data, keys = load_data(data_path)
y, data = downsample_unbalanced(data, y, 20000, 42)

print('Saving DS data...')
np.save('y_ds.npy', y)
np.save('data_ds.npy', data)
np.save('keys.npy', keys)