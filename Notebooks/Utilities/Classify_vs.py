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


def evaluate_classifier_over_features(X_train, X_test, y_train, y_test, clf):
    """
    Train a classifier using an increasing number of features and compute performance metrics.
    Returns the results dict plus the classifier with highest ROC AUC.
    """
    predictions, classifiers, y_prob = [], [], []
    auc_scores = []
    Ncolors = np.arange(1, X_train.shape[1] + 1)

    plt.figure(figsize=(6, 5))

    for nc in Ncolors:
        print('Computing predictions using first', nc, 'features...')
        clf_i = clone(clf)
        clf_i.fit(X_train[:, :nc], y_train)
        
        y_pred = clf_i.predict(X_test[:, :nc])
        prob = clf_i.predict_proba(X_test[:, :nc])[:, 1]

        predictions.append(y_pred)
        y_prob.append(prob)
        classifiers.append(clf_i)

        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

        if nc%2==0:
            plt.plot(fpr, tpr, label=f'{nc} feat. (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve by Number of Features')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    completeness, contamination = compute_completeness_contamination(predictions, y_test)

    best_index = np.argmax(auc_scores)
    best_classifier = classifiers[best_index]

    result = {
        'completeness': completeness,
        'contamination': contamination,
        'classifiers': classifiers,
        'predictions': predictions,
        'proba': y_prob,
        'auc_scores': auc_scores,
        'best_classifier': best_classifier,
        'best_n_features': Ncolors[best_index],
        'best_auc': auc_scores[best_index]
    }

    return result

def evaluate_classifier_over_features_all(X_train, X_test, y_train, y_test, clf, use_all_features=True):
    """
    Train classifier(s) using an increasing number of features (or all features at once)
    and compute performance metrics.

    Parameters:
    - use_all_features: if True, train only on all features at once.
                        if False, train progressively from 1 to all features.

    Returns:
    - dict with completeness, contamination, classifiers, predictions, proba, best_classifier, best_n_features
    """
    predictions, classifiers, y_prob = [], [], []
    
    if use_all_features:
        feature_counts = [X_train.shape[1]]
    else:
        feature_counts = np.arange(1, X_train.shape[1] + 1)
    
    plt.figure(figsize=(6, 5))
    
    for nc in feature_counts:
        print('Computing predictions on feature count:', nc)
        clf_i = clone(clf)
        clf_i.fit(X_train[:, :nc], y_train)
        y_pred = clf_i.predict(X_test[:, :nc])
        proba = clf_i.predict_proba(X_test[:, :nc])[:, 1]
        
        y_prob.append(proba)
        predictions.append(y_pred)
        classifiers.append(clf_i)
        
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC with {nc} features (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Assuming compute_completeness_contamination is defined elsewhere
    completeness, contamination = compute_completeness_contamination(predictions, y_test)
    
    # Find best classifier (max completeness - contamination or max AUC?)
    # Here, simplest: max completeness - contamination
    scores = np.array(completeness) - np.array(contamination)
    best_idx = np.argmax(scores)
    
    result = {
        'completeness': completeness,
        'contamination': contamination,
        'classifiers': classifiers,
        'predictions': predictions,
        'proba': y_prob,
        'best_classifier': classifiers[best_idx],
        'best_n_features': feature_counts[best_idx]
    }
    
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
    

def visualize_classification_generic(S, y_ds, clf, completeness, contamination, f=10, a=0, b=1):
    """
    Visualize decision boundary on features a and b, completeness, and contamination.

    Parameters:
    - S: 2D array (n_samples, n_features)
    - y_ds: 1D array of true labels
    - clf: trained classifier (expects input with clf.n_features_in_)
    - completeness: array of completeness scores
    - contamination: array of contamination scores
    - f: int, undersampling factor for scatter plot (default 10)
    - a: int, feature index for y-axis
    - b: int, feature index for x-axis
    """
    Ncolors = np.arange(1, S.shape[1] + 1)

    # Undersample for scatter plot
    X_sub = S[::f, [b, a]]  # order x, y
    y_sub = y_ds[::f]

    # Define plot limits with padding
    padding_x = 0.05 * np.ptp(S[:, b])
    padding_y = 0.05 * np.ptp(S[:, a])
    xlim = (S[:, b].min() - padding_x, S[:, b].max() + padding_x)
    ylim = (S[:, a].min() - padding_y, S[:, a].max() + padding_y)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                         np.linspace(ylim[0], ylim[1], 200))

    # Prepare grid data for prediction
    n_grid_points = xx.size
    n_features = clf.n_features_in_

    # Use mean values for all features as baseline
    baseline = np.mean(S, axis=0)

    # Create grid with shape (n_grid_points, n_features)
    grid = np.tile(baseline, (n_grid_points, 1))

    # Replace columns a and b with meshgrid values
    # Make sure to assign in correct order (features indices)
    grid[:, b] = xx.ravel()
    grid[:, a] = yy.ravel()

    # Predict probabilities for class 1
    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(hspace=0.0, wspace=0.2)

    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_sub[:, 0], X_sub[:, 1], c=y_sub, s=4, lw=0, cmap='coolwarm', zorder=2)
    img = ax.imshow(Z, origin='lower', aspect='auto', cmap='gray',
                    extent=xlim + ylim, zorder=1)
    plt.colorbar(img, ax=ax, label='P(Class 1)')
    ax.contour(xx, yy, Z, levels=[0.5], colors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f'Feature {b}')
    ax.set_ylabel(f'Feature {a}')
    ax.set_title('Decision Boundary (P(Class 1))')

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

def visualize_classification_generic_all(S, y_ds, clf, completeness, contamination, f=10, a=0, b=1):
    """
    Visualize decision boundary, completeness, and contamination.

    Parameters:
    - S: 2D array of shape (n_samples, n_features)
    - y_ds: 1D array of true labels
    - clf: trained classifier with predict_proba()
    - completeness: list or array of completeness scores
    - contamination: list or array of contamination scores
    - f: int, undersampling factor for scatter plot
    - a: int, feature index for y-axis (must be < n_features)
    - b: int, feature index for x-axis (must be < n_features)
    """
    best_n_features = S.shape[1]
    completeness = completeness[:best_n_features]
    contamination = contamination[:best_n_features]
    Ncolors = np.arange(1, best_n_features + 1)

    # Undersample for scatter
    X_sub = S[::f, [a, b]]
    y_sub = y_ds[::f]

    padding_x = 0.05 * np.ptp(S[:, b])
    padding_y = 0.05 * np.ptp(S[:, a])
    xlim = (S[:, b].min() - padding_x, S[:, b].max() + padding_x)
    ylim = (S[:, a].min() - padding_y, S[:, a].max() + padding_y)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                         np.linspace(ylim[0], ylim[1], 200))

    # Grid points must be in same feature order as classifier expects:
    # Create full feature vectors filled with zeros
    grid = np.zeros((xx.size, best_n_features))
    grid[:, a] = yy.ravel()
    grid[:, b] = xx.ravel()

    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(hspace=0.0, wspace=0.2)

    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_sub[:, 1], X_sub[:, 0], c=y_sub, s=4, lw=0, cmap='coolwarm', zorder=2)
    img = ax.imshow(Z, origin='lower', aspect='auto', cmap='gray',
                    extent=xlim + ylim, zorder=1)
    plt.colorbar(img, ax=ax, label='P(Class 1)')
    ax.contour(xx, yy, Z, [0.5], colors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title('Decision Boundary (P(Class 1))')

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
    
def plot_learning_curve(train_sizes, train_scores, test_scores):

    # Media e deviazione standard dei punteggi
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 🔧 Plot
    plt.figure(figsize=(8, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color="blue")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2, color="orange")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="orange", label="Cross-validation score")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()