import h5py
import numpy as np
import sys
sys.path.append(r"C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\Notebooks\Exam\LIGO")
from LIGO_Utils.LigoUtils import load_data, downsample_balanced

data_path = r"C:\Users\ricca\OneDrive\Documents\LIGO.h5"

y, data, keys = load_data(data_path)

print('Saving data...')
np.save('data_100k', data)
np.save('labels_100k', y)
np.save('keys_100k', keys)

y, data = downsample_balanced(data, y, 50000, 42)
print('Saving DS data...')
np.save('y_ds_100k.npy', y)
np.save('data_ds_100k.npy', data)