import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib as mtl

def batch_downsample(X, y, batch_size, f=2.0):
    """
    Downsample majority class within each batch by a factor f.

    Parameters:
    - X: ndarray, features
    - y: ndarray, target (binary: 0 = majority, 1 = minority)
    - batch_size: int, size of each batch
    - f: float > 1, downsampling factor (e.g., f=2 keeps 1/2 of majority class per batch)

    Returns:
    - X_downsampled, y_downsampled: concatenated downsampled arrays
    """

    assert f > 1, "Downsampling factor f must be > 1"

    downsampled_X_batches = []
    downsampled_y_batches = []

    n_samples = X.shape[0]

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # Separate classes
        X_majority = X_batch[y_batch == 0]
        X_minority = X_batch[y_batch == 1]

        y_majority = y_batch[y_batch == 0]
        y_minority = y_batch[y_batch == 1]

        # Downsample majority class by factor f
        n_majority = len(y_majority)
        n_keep = max(1, int(n_majority / f))

        if n_majority > n_keep:
            idx = np.random.choice(n_majority, size=n_keep, replace=False)
            X_majority_down = X_majority[idx]
            y_majority_down = y_majority[idx]
        else:
            X_majority_down = X_majority
            y_majority_down = y_majority

        # Keep all minority samples (unchanged)
        X_downsampled_batch = np.vstack((X_majority_down, X_minority))
        y_downsampled_batch = np.concatenate((y_majority_down, y_minority))

        downsampled_X_batches.append(X_downsampled_batch)
        downsampled_y_batches.append(y_downsampled_batch)

    # Combine all batches
    X_downsampled = np.vstack(downsampled_X_batches)
    y_downsampled = np.concatenate(downsampled_y_batches)

    return X_downsampled, y_downsampled

def downsample_by_factor(X, y, f=2.0):
    """
    Downsample majority class (label 0) globally by a factor f.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
    - y: ndarray, shape (n_samples,), binary labels (0 = majority, 1 = minority)
    - f: float > 1, keep 1/f of the majority class

    Returns:
    - X_downsampled: ndarray, features after downsampling
    - y_downsampled: ndarray, labels after downsampling
    """
    assert f > 1, "Downsampling factor f must be > 1"

    # Split majority and minority
    X_majority = X[y == 0]
    X_minority = X[y == 1]
    y_majority = y[y == 0]
    y_minority = y[y == 1]

    # Downsample majority
    n_majority = len(y_majority)
    n_keep = max(1, int(n_majority / f))

    indices = np.random.choice(n_majority, size=n_keep, replace=False)
    X_majority_down = X_majority[indices]
    y_majority_down = y_majority[indices]

    # Combine
    X_downsampled = np.vstack((X_majority_down, X_minority))
    y_downsampled = np.concatenate((y_majority_down, y_minority))

    # Optional: shuffle
    perm = np.random.permutation(len(y_downsampled))
    return X_downsampled[perm], y_downsampled[perm]

def compute_completeness_contamination(predictions, y_true):
    """
    Compute completeness (recall) and contamination (false discovery rate).
    """
    completeness, contamination = [], []
    for y_pred in predictions:
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        comp = TP / (TP + FN) if (TP + FN) > 0 else 0
        cont = FP / (TP + FP) if (TP + FP) > 0 else 0
        completeness.append(comp)
        contamination.append(cont)
    return np.array(completeness), np.array(contamination)

def evaluate_classifier_over_features(X_train, X_test, y_train, y_test, clf, ROC=False):
    """
    Train a classifier using an increasing number of features and compute performance metrics.
    """
    predictions, classifiers, y_prob = [], [], []
    Ncolors = np.arange(1, X_train.shape[1] + 1)

    for nc in Ncolors:
        print('Computing predictions on feature ' , nc)
        clf_i = clone(clf)
        clf_i.fit(X_train[:, :nc], y_train)
        y_pred = clf_i.predict(X_test[:, :nc])
        y_prob.append(clf_i.predict_proba(X_test[:, :nc])[:, 1])
        predictions.append(y_pred)
        classifiers.append(clf_i)
        
        
    print('Computing completeness, contamination...')
    completeness, contamination = compute_completeness_contamination(predictions, y_test)

    result = {
        'completeness': completeness,
        'contamination': contamination,
        'classifiers': classifiers,
        'predictions': predictions,
        'proba': y_prob
    }

    if ROC:
        print('Plotting ROC...')
        plot_roc_curve(y_test, classifiers[-1].predict_proba(X_test)[:, 1])

    return result

def cross_validate_gmm_components(X, y, classifier_class, n_components_list, n_splits=5):
    """
    Perform K-fold cross-validation to select optimal GMM components.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for n in n_components_list:
        print(f'Performing CV on {n} components')
        acc = []
        for train_idx, test_idx in kf.split(X):
            clf = classifier_class(n_components=n, tol=1e-5, covariance_type='full')
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            acc.append(accuracy_score(y[test_idx], y_pred))
        scores.append(np.mean(acc))

    best_n = n_components_list[np.argmax(scores)]
    print(f"Best n_components: {best_n}")
    return best_n, scores

def evaluate_gmm_classifier(X_train, X_test, y_train, y_test, classifier_class, n_components, ROC=False):
    """
    Evaluate a GMM classifier over increasing number of features.
    """
    classifiers, predictions = [], []
    Ncolors = np.arange(1, X_train.shape[1] + 1)

    for nc in Ncolors:
        clf = classifier_class(n_components=n_components, tol=1e-5, covariance_type='full')
        clf.fit(X_train[:, :nc], y_train)
        y_pred = clf.predict(X_test[:, :nc])
        classifiers.append(clf)
        predictions.append(y_pred)

    completeness, contamination = compute_completeness_contamination(predictions, y_test)

    if ROC:
        plot_roc_curve(y_test, classifiers[-1].predict_proba(X_test)[:, 1])

    return {
        'completeness': completeness,
        'contamination': contamination,
        'classifiers': classifiers,
        'predictions': predictions
    }

def plot_roc_curve(y_true, y_score):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_classification(S, y_ds, clf, completeness, contamination, f=10, a=0, b=1):
    """
    Visualize decision boundary, completeness, and contamination.

    Parameters:
    - S: 2D array of shape (n_samples, n_features)
    - y_ds: 1D array of true labels
    - clf: trained classifier with predict_proba()
    - completeness: array of completeness scores
    - contamination: array of contamination scores
    - f: int, undersampling factor for scatter plot (default: 10)
    - a: int, feature index for y-axis
    - b: int, feature index for x-axis
    """
    Ncolors = np.arange(1, S.shape[1] + 1)
    
    # Undersample
    X_sub = S[::f, [a, b]]
    y_sub = y_ds[::f]

    # Build grid
    padding_x = 0.05 * np.ptp(S[:, b])
    padding_y = 0.05 * np.ptp(S[:, a])
    xlim = (S[:, b].min() - padding_x, S[:, b].max() + padding_x)
    ylim = (S[:, a].min() - padding_y, S[:, a].max() + padding_y)
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                            np.linspace(ylim[0], ylim[1], 200))

    grid = np.c_[yy.ravel(), xx.ravel()]  # shape (N, 2) with columns: [a, b]
    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(hspace=0.0, wspace=0.2)

    ax = fig.add_subplot(121)
    ax.scatter(X_sub[:, b], X_sub[:, a], c=y_sub, s=4, lw=0, cmap='coolwarm', zorder=2)
    img = ax.imshow(Z, origin='lower', aspect='auto', cmap='gray',
                    extent=xlim + ylim, zorder=1)
    plt.colorbar(img, ax=ax, label='P(Class 1)')
    ax.contour(xx, yy, Z, [0.5], colors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(222)
    ax.plot(Ncolors, completeness, 'o-k')
    ax.set_ylabel('Completeness')
    ax.grid(True)

    ax = plt.subplot(224)
    ax.plot(Ncolors, contamination, 'o-k')
    ax.set_xlabel('N Features')
    ax.set_ylabel('Contamination')
    ax.grid(True)

    plt.show()