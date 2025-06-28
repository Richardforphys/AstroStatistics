import h5py
import numpy as np
import sys
sys.path.append(r"C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\Notebooks\venv\Exam\LIGO\LIGO_Utils")
from LIGO_Utils.LIGOUTILS import load_data, downsample_balanced

data_path = r"C:\Users\ricca\OneDrive\Documents\LIGO.h5"

y, data, keys = load_data(data_path)

print('Saving data...')
np.save('data', data)
np.save('labels', y)
np.save('keys', keys)

y, data = downsample_balanced(data, y, 200000, 42)
print('Saving DS data...')
np.save('y_ds.npy', y)
np.save('data_ds.npy', data)