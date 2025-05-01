import numpy as np
from utils import read_data, Gauss_peaks, GM_fit
from sklearn.mixture import GaussianMixture
#Load Data
url = 'https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt'
raw, names = read_data(url)
print('Data imported...')

#Clean Data
T90, flux = zip(*[(float(a), float(b)) for a, b in zip(raw[6], raw[9]) if a != '-999' and b != '-999' and float(b)!=0])
T90  = np.array(T90)
flux = np.array(flux)
T90 = np.log10(T90)
flux= np.log10(flux)

#Organize data in matrix
X = np.vstack([T90, flux])
X = X.T

# Decide how many clusters to use
n_best , bic = Gauss_peaks(X, 2, 10, True)
print(f'n_best = {n_best}')
print(f'bic_score = {bic}')

labels, instance = GM_fit(X, n_best, plot=True)