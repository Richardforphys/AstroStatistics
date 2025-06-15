import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

def read_data(url):
    """
    Download file from url
    ______________________
    Returns, names and raw data
    """
    r = requests.get(url)
    with open("Summary_table.txt", 'wb') as f:
        f.write(r.content)

    # Read content
    raw = np.loadtxt("Summary_table.txt", dtype='str',unpack='True')

    # Read headers
    with open("Summary_table.txt",'r') as f:
        names = np.array([n.strip().replace(" ","_") for n in f.readlines()[1].replace("#","").replace("\n","").lstrip().split('    ') if n.strip()!=''])
        
    return raw, names

def Gauss_peaks(X, minr, maxr, plot=False):
    """
    Determine optimal number of Gaussian components using BIC.
    ----------------------------------------------------------
    Parameters:
        X : array-like, shape (n_samples, n_features)
            Input data.
        minr : int
            Minimum number of components to test.
        maxr : int
            Maximum number of components to test (exclusive).
        plot : bool
            If True, plot the BIC scores.

    Returns:
        best_n : int
            Number of components with lowest BIC.
        best_bic : float
            Corresponding BIC score.
    """
    n_range = list(range(minr, maxr))
    bic_scores = []

    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))

    bic_scores = np.array(bic_scores)
    best_index = np.argmin(bic_scores)
    best_n = n_range[best_index]

    if plot:
        plt.plot(n_range, bic_scores, 'o-')
        plt.xlabel('Number of components')
        plt.ylabel('BIC')
        plt.title('BIC for GMM')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_n, bic_scores[best_index]

    

def elbow_cluster(X, minr=1, maxr=10, plot=False):
    """
    Decide how many clusters to use with the elbow method based on inertia.
    ___________________________________________________________________
    Parameters:
        X : array-like, shape (n_samples, n_features)
            Data to cluster.
        minr : int
            Minimum number of clusters to test.
        maxr : int
            Maximum number of clusters to test.
        plot : bool
            Whether to plot the inertia curve.
    Returns:
        int: Optimal number of clusters (elbow point).
    """
    k_range = np.arange(minr, maxr + 1)
    inertia = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    inertia = np.array(inertia)
    gradients = np.gradient(inertia)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(k_range, inertia, 'o-', label='Inertia')
        plt.plot(k_range, gradients, 'o--', label='Gradient')
        plt.xlabel('Number of clusters k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.xticks(k_range)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Find the elbow point using the point of maximum curvature (or simplest drop detection)
    deltas = np.diff(inertia)
    delta_ratios = np.abs(np.diff(deltas))
    elbow_index = np.argmax(delta_ratios) + 1  # +1 because of np.diff
    
    return k_range[elbow_index]

def KMEans_fit(X, n_clusters, plot=False):
    """
    Perform KMeans clustering.
    --------------------------
    Parameters:
        X : array-like, shape (n_samples, n_features)
            Data to cluster.
        n_clusters : int
            Number of clusters.
        plot : bool
            If True, plots the 2D clustered data.

    Returns:
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point.
        centers : ndarray of shape (n_clusters, n_features)
            Coordinates of cluster centers.
        instance : KMeans
            Fitted KMeans object.
    """
    instance = KMeans(n_clusters=n_clusters, tol=1e-6, algorithm='elkan', random_state=0)
    instance.fit(X)
    centers = instance.cluster_centers_
    labels = instance.labels_
    
    if plot:
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data.")
        for i in np.unique(labels):
            plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', label='Centroids')
        plt.title("KMeans Clustering")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return labels, centers, instance

def GM_fit(X, n, plot=False):
    """
    Fit a Gaussian Mixture Model and optionally plot the result.
    ------------------------------------------------------------
    Parameters:
        X : array-like, shape (n_samples, n_features)
            Input data.
        n : int
            Number of Gaussian components.
        plot : bool
            If True, plot the clustering result (only for 2D data).

    Returns:
        labels : ndarray of shape (n_samples,)
            Cluster labels for each data point.
        model : GaussianMixture
            Fitted GaussianMixture model.
    """
    model = GaussianMixture(n_components=n, random_state=0)
    labels = model.fit_predict(X)
    
    if plot:
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data.")
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
        plt.xlabel('T90 (s)')
        plt.ylabel('Flux (keV/cm²)')
        plt.title(f'Gaussian Mixture Clustering (k={n})')
        plt.colorbar(label='Cluster label')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return labels, model


def cross_validation_kde_MISE(data, bandwidths, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []

    for bandwidth in bandwidths:
        mse_fold = []
        for train_idx, test_idx in kf.split(data):
            X_train, X_test = data[train_idx].reshape(-1, 1), data[test_idx].reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(X_train)
            log_dens = kde.score_samples(X_test)
            mse = mean_squared_error(np.zeros_like(log_dens), log_dens)
            mse_fold.append(mse)
        
        mse_list.append(np.mean(mse_fold))
    
    return mse_list

from scipy.spatial.distance import cdist
from scipy.stats import norm

def gaussian_kernel(u):
    """Univariate Gaussian kernel."""
    return norm.pdf(u)

def KDE_LOO(X, h):
    """
    Computes the leave-one-out log-likelihood CV score for KDE using Gaussian kernel.
    
    Parameters:
        X: numpy array of shape (N, D)
        h: bandwidth (float)
    
    Returns:
        CV_l(h): float
    """
    N, D = X.shape
    log_likelihood = 0.0
    const = 1 / ((N - 1) * (h ** D))
    norm_const = 1 / ((2 * np.pi) ** (D / 2))

    for i in range(N):
        X_loo = np.delete(X, i, axis=0)
        dists = cdist([X[i]], X_loo)[0] / h  # Now dists has shape (N-1,)
        
        kernel_vals = np.exp(-0.5 * dists ** 2)
        
        f_hat = const * norm_const * np.sum(kernel_vals)
        
        log_likelihood += np.log(f_hat + 1e-12)  # To avoid log(0)

    return log_likelihood / N


def cross_validation_LOO(X, bandwidths, plot=False):
    
    scores = [KDE_LOO(X, bw) for bw in bandwidths]
    
    if plot:
        plt.plot(bandwidths, scores, '-o', color='blue', linewidth=2)
        plt.title('Cross Validation LOO score')
        plt.ylabel('CV score')
        plt.xlabel('Bandwidth')
        
    return np.array(scores), max(scores), np.argmax(scores)

def kde_sklearn(data, xgrid, bandwidth, kernel="gaussian"):
    kde_skl = KernelDensity(bandwidth = bandwidth, kernel=kernel)
    kde_skl.fit(data[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xgrid[:, np.newaxis]) # sklearn returns log(density)
    return np.exp(log_pdf)
    