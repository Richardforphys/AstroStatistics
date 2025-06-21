import sys
sys.path.append(r"C:\Users\ricca\Documents\Unimib-Code\Astrostatistics\Notebooks\venv\Utilities\utils.py")
from Utilities import plot_settings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests

# Download file
r = requests.get('https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt')
with open("Summary_table.txt", 'wb') as f:
    f.write(r.content)

# Read content
data = np.loadtxt("Summary_table.txt", dtype='str',unpack='True')

# Read headers
with open("Summary_table.txt",'r') as f:
    names = np.array([n.strip().replace(" ","_") for n in f.readlines()[1].replace("#","").replace("\n","").lstrip().split('    ') if n.strip()!=''])
    
    
T90  = data[6]
T100 = data[12]
F    = data[9]
R    = data[11] 

# Assume T90, T100, F, R are arrays of strings
X = np.vstack([T90, T100, F, R]).T  # Shape: (n_samples, 4)

# Mask rows that do NOT contain the string '-999'
mask = ~np.any(X == '-999', axis=1)

# Keep only clean rows
Y = X[mask]

Y = Y.astype(float)


Y_train, Y_test = train_test_split(Y, test_size=0.3)

scaler = StandardScaler()
scaler.fit(Y_train)
Y_train_transformed = scaler.transform(Y_train)
Y_test_transformed = scaler.transform(Y_test)

pca = PCA(n_components=4)
pca.fit(Y_train_transformed)

ratios = pca.explained_variance_ratio_

A_train = pca.transform(Y_train_transformed)[:, :3]  # oppure [:2] per visualizzazione
A_test  = pca.transform(Y_test_transformed)[:, :3]

scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(A_train)
    labels = kmeans.predict(A_test)
    score = silhouette_score(A_test, labels)
    scores.append(score)

best_k = k_range[np.argmax(scores)]
print("Best k by silhouette score:", best_k)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(A_train)
labels = kmeans.predict(A_test)

mean = scaler.mean_[:3]
scale = scaler.scale_[:3]

# Manual inverse transform
original_A_test = A_test * scale + mean
original_A_test_red = A_test[labels==0] * scale + mean
original_A_test_blue = A_test[labels==1] * scale + mean


plt.scatter(original_A_test.T[0], original_A_test.T[1], color='grey', marker='.', label='raw data')
plt.scatter(original_A_test_red.T[0], original_A_test_red.T[1], color='red', marker='.')
plt.scatter(original_A_test_blue.T[0], original_A_test_blue.T[1], color='blue', marker='.')