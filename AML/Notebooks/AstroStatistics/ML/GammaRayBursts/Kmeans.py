import numpy as np
from sklearn.metrics import silhouette_score
from utils import read_data, elbow_cluster, KMeans_fit

#Load Data
url = 'https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt'
raw, names = read_data(url)
print('DAta imported...')

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
n_best = elbow_cluster(X, 2, 10, plot=True)
print(f'n_best = {n_best}')

labels, centers, instance = KMeans_fit(X, n_best, plot=True)

SS = silhouette_score(X, labels)
print(f'Mean Silhouette score : {SS:.2f}')