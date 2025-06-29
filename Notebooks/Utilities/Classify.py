from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib as mtl
import numpy as np
import matplotlib.pyplot as plt

def completeness_contamination(predictions, y_true):
    completeness = []
    contamination = []
    for y_pred in predictions:
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        comp = TP / (TP + FN) if (TP + FN) > 0 else 0
        cont = FP / (TP + FP) if (TP + FP) > 0 else 0
        completeness.append(comp)
        contamination.append(cont)
    return np.array(completeness), np.array(contamination)

def compute_cc(X, X_train, X_test, y_train, y_test, clf, ROC=False):
    
    classifiers, predictions, y_prob = [], [], []
    Ncolors = np.arange(1, X.shape[1] + 1)

    for nc in Ncolors:
        clf_i = clone(clf)
        clf_i.fit(X_train[:, :nc], y_train)
        y_pred = clf_i.predict(X_test[:, :nc])
        y_prob.append(clf_i.predict_proba(X_test[:, :nc])[:, 1])

        classifiers.append(clf_i)
        predictions.append(y_pred)

    completeness, contamination = completeness_contamination(predictions, y_test)

    print("completeness", completeness)
    print("contamination", contamination)
    
    d = {
        'completeness': completeness,
        'contamination': contamination,
        'classifiers': classifiers,
        'predictions': predictions,
        'proba': y_prob
    }
    
    if ROC:
        y_score = d['classifiers'][-1].predict_proba(X_test)[:, 1]

        # ----------------------------
        # Step 4: Compute ROC and AUC
        # ----------------------------
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # ----------------------------
        # Step 5: Plot ROC Curve
        # ----------------------------
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
    
    return d 

def cross_validate_gmm_components_kfold(X, y, classifier, n_components_list, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for n in n_components_list:
        accs = []
        for train_idx, test_idx in kf.split(X):
            clf = classifier(n, tol=1e-5, covariance_type='full')
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], y_pred))
        scores.append(np.mean(accs))

    best_n = n_components_list[np.argmax(scores)]
    print(f"Best n_components (via KFold): {best_n}")
    return best_n, scores

from sklearn.metrics import roc_curve, auc

def compute_GMMbayes(X_train, X_test, y_train, y_test, classifier, n_components, ROC):
    classifiers = []
    predictions = []

    Ncolors = np.arange(1, X_train.shape[1] + 1)

    for nc in Ncolors:
        clf = classifier(n_components, tol=1E-5, covariance_type='full')
        clf.fit(X_train[:, :nc], y_train)
        y_pred = clf.predict(X_test[:, :nc])

        classifiers.append(clf)
        predictions.append(y_pred)

    completeness, contamination = completeness_contamination(predictions, y_test)

    print("completeness", completeness)
    print("contamination", contamination)

    d = {
        'completeness': completeness,
        'contamination': contamination,
        'classifiers': classifiers,
        'predictions': predictions,
    }
    
    if ROC:
        y_score = d['classifiers'][-1].predict_proba(X_test)[:, 1]  # Probability of class 1
        mtl.rcdefaults()  # Resets rcParams to default values

        # ----------------------------
        # Step 4: Compute ROC and AUC
        # ----------------------------
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # ----------------------------
        # Step 5: Plot ROC Curve
        # ----------------------------
        fig = plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return d


def classify(classifier, X, X_train, X_test, y_train, y_test, y, cc_func, GM=False, n_components=0, ROC=False, k=None):
    
    """
    Perform classification using a specified classifier and visualize results.

    This function applies a classification method (via a callable `cc_func`) on the given dataset,
    trains the classifier, predicts class probabilities over a grid for visualization,
    and plots:
      - The data points and classifier decision boundary in feature space,
      - Completeness vs. number of colors (features),
      - Contamination vs. number of colors (features).

    Parameters
    ----------
    classifier : object
        The classifier instance or identifier to be used for training.
    X : ndarray, shape (n_samples, n_features)
        The full feature dataset.
    X_train : ndarray
        Training set features.
    X_test : ndarray
        Test set features.
    y_train : ndarray
        Training set labels.
    y_test : ndarray
        Test set labels.
    y : ndarray
        Labels for the full dataset.
    cc_func : callable
        A function that performs classification and returns a dictionary containing
        classifiers, completeness, and contamination metrics.
        Expected to be called as either:
          - cc_func(X, X_train, X_test, y_train, y_test, classifier, ROC)
          - or cc_func(X_train, X_test, y_train, y_test, classifier, n_components, ROC)
    GM : bool, optional (default=False)
        Flag controlling behavior in `cc_func` (not used directly here).
    n_components : int, optional (default=0)
        Number of components used in classification, passed to `cc_func`.
    ROC : bool, optional (default=False)
        Whether to compute ROC curve metrics, passed to `cc_func`.
    k : int, optional
        Additional number of points to include in the plot from the end of the dataset.
        Used to adjust the number of plotted samples.

    Returns
    -------
    None

    Notes
    -----
    - The function assumes the first two features (columns) of X correspond to
      meaningful dimensions for plotting decision boundaries.
    - It visualizes predicted probabilities on a grid spanning the feature space.
    - Completeness and contamination curves are plotted with respect to
      the number of colors (features).
    - The function displays plots but does not return any objects.
    """

    N_tot = len(y)
    N_st = np.sum(y == 0)
    N_rr = N_tot - N_st
    N_train = len(y_train)
    N_test = len(y_test)
    N_plot = k + N_rr
    
    try:
        dic = cc_func(X, X_train, X_test, y_train, y_test,classifier, ROC)
    except:
        dic = cc_func(X_train, X_test, y_train, y_test,classifier, n_components, ROC)
        
    clf = dic['classifiers'][1]
    
    Ncolors = np.arange(1, X.shape[1] + 1)

    padding_x = 0.05 * (X[:, 0].max() - X[:, 0].min())
    padding_y = 0.05 * (X[:, 1].max() - X[:, 1].min())

    xlim = (X[:, 0].min() - padding_x, X[:, 0].max() + padding_x)
    ylim = (X[:, 1].min() - padding_y, X[:, 1].max() + padding_y)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                        np.linspace(ylim[0], ylim[1], 200))
    
    grid = np.c_[yy.ravel(), xx.ravel()]  # Note the order to match X[:, :2]
    
    Z = clf.predict_proba(grid)
    Z = Z[:, 1].reshape(xx.shape)
    
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                        left=0.1, right=0.95, wspace=0.2)

    # Left plot: data and decision boundary
    ax = fig.add_subplot(121)
    im = ax.scatter(X[-N_plot:, 1], X[-N_plot:, 0], c=y[-N_plot:],
                    s=4, lw=0, cmap=plt.cm.Oranges, zorder=2)
    im.set_clim(-0.5, 1)

    im = ax.imshow(Z, origin='lower', aspect='auto',
                cmap=plt.cm.binary, zorder=1,
                extent=xlim + ylim)
    im.set_clim(0, 1.5)
    ax.contour(xx, yy, Z, [0.5], colors='k')
    cbar = plt.colorbar(im, ax=ax, label='P(QSO)')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)   

    # Top-right: Completeness vs Ncolors
    ax = plt.subplot(222)
    ax.plot(Ncolors, dic['completeness'], 'o-k', ms=6)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylabel('completeness')
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)

    # Bottom-right: Contamination vs Ncolors
    ax = plt.subplot(224)
    ax.plot(Ncolors, dic['contamination'], 'o-k', ms=6)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel('N colors')
    ax.set_ylabel('contamination')
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)

    plt.show()
