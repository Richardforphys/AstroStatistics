import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

class LR:
    """
    Class for performing polynomial regression of arbitrary degree.
    """

    def __init__(self, xvalues, yvalues, errors, degree=1, test_pc=0.15, k_fold=5):
        """
        Initialize the Regression class.
        """
        self._X = np.asarray(xvalues)
        self._Y = np.asarray(yvalues)
        self._errors = np.asarray(errors)
        self.degree = degree
        self.status = 'train'
        self.M = None
        self.C = None
        
        self.test_pc = test_pc
        self.X_test = None
        self.Y_test = None
        self.errors_test = None
        
        self.X_train = None
        self.Y_train = None
        self.errors_train = None
        
        self.k_fold = k_fold
        
        self.Theta = None
        self.Sigma = None


    def set_test_data(self): 
        """
        Split the data into training and testing sets.
        """
        if self.test_pc < 0 or self.test_pc > 1:
            raise ValueError("test_pc must be between 0 and 1.")

        indices = np.arange(len(self._X))
        np.random.shuffle(indices)

        test_end = int(len(self._X) * self.test_pc)

        test_idx = indices[:test_end]
        train_idx = indices[test_end:]

        self.X_test = self._X[test_idx]
        self.Y_test = self._Y[test_idx]
        self.errors_test = self._errors[test_idx]

        self.X_train = self._X[train_idx]
        self.Y_train = self._Y[train_idx]
        self.errors_train = self._errors[train_idx]       

    def _build_M(self):
        if self.status == 'train':
            self.M = np.vander(self.X_train, N=self.degree + 1, increasing=True)
        elif self.status == 'test':
            self.M = np.vander(self.X_test, N=self.degree + 1, increasing=True)

    def _build_C(self):
        if self.status == 'train':
            self.C = np.diag(self.errors_train ** 2)
        elif self.status == 'test':
            self.C = np.diag(self.errors_test ** 2)

    
    def train(self, plot=False):
        
        if self.X_train is None or self.Y_train is None:
            self.set_test_data()
        self.status = 'train'
        self._build_M()
        self._build_C()

        C_inv = np.linalg.inv(self.C)
        MT_Cinv = self.M.T @ C_inv
        self.Theta = np.linalg.inv(MT_Cinv @ self.M) @ (MT_Cinv @ self.Y_train)
        self.Sigma = np.linalg.inv(MT_Cinv @ self.M)

        if plot:
            fig, ax = self.plot()
            ax.scatter(self.X_train, self.Y_train, color='blue', label='Train data', zorder=5)
            ax.legend()
            plt.show()

        return self.Theta, self.Sigma
    
    
    def train2(self):
        
        self.status = 'train'
        self._build_M()
        self._build_C()

        C_inv = np.linalg.inv(self.C)
        MT_Cinv = self.M.T @ C_inv
        self.Theta = np.linalg.inv(MT_Cinv @ self.M) @ (MT_Cinv @ self.Y_train)
        self.Sigma = np.linalg.inv(MT_Cinv @ self.M)
        return self.Theta, self.Sigma
        
    def test(self, plot=False):
        if self.Theta is None:
            raise ValueError("Model must be trained before testing.")
        
        self.status = 'test'
        self._build_M()
        self._build_C()

        residuals = self.Y_test - self.M @ self.Theta
        chi2 = residuals.T @ np.linalg.inv(self.C) @ residuals
        self.Sigma = np.linalg.inv(self.M.T @ np.linalg.inv(self.C) @ self.M)

        if plot:
            _, ax = self.plot()
            ax.scatter(self.X_test, self.Y_test, color='green', label='Test data')
            plt.show()

        return self.Theta, self.Sigma

        
    def compute_chi2(self):
        if self.status == 'train':
            residuals = self.Y_train - self.M @ self.Theta
        elif self.status == 'test':
            residuals = self.Y_test - self.M @ self.Theta
        else:
            raise ValueError("Unknown status")
        return residuals.T @ np.linalg.inv(self.C) @ residuals

    def compute_reduced_chi2(self):
        if self.status == 'train':
            dof = len(self.Y_train) - (self.degree + 1)
        elif self.status == 'test':
            dof = len(self.Y_test) - (self.degree + 1)
        else:
            raise ValueError("Unknown status")
        
        chi2 = self.compute_chi2()
        return chi2 / dof if dof > 0 else np.inf


    def plot(self, resolution=1000, ax=None):
        """
        Plot the data and the polynomial fit.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Prediction line
        x_fit = np.linspace(np.min(self._X), np.max(self._X), resolution)
        X_fit = np.vander(x_fit, N=self.degree + 1, increasing=True)
        y_fit = X_fit @ self.Theta

        ax.plot(x_fit, y_fit, color='red', alpha=0.7, label=f'Fit (deg {self.degree})')

        # Add base data
        ax.errorbar(self._X, self._Y, yerr=self._errors, fmt='o', color='grey', alpha=0.3, label='All data')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Polynomial Fit')
        ax.grid(True)
        ax.text(0.05, 0.95, f'$\chi^2_r$ = {self.compute_reduced_chi2():.2f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        return fig, ax

    def k_fold_cross_validation(self):
        """
        Perform k-fold cross-validation and return mean RMSE over folds
        for both training and validation sets.
        """
        if self.k_fold < 2:
            raise ValueError("k_fold must be at least 2.")
        
        indices = np.arange(len(self._X))
        np.random.shuffle(indices)

        fold_size = len(self._X) // self.k_fold
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(self.k_fold - 1)]
        folds.append(indices[(self.k_fold - 1) * fold_size:])  # Last fold gets remainder

        train_rmse_list = []
        val_rmse_list = []

        for i in range(self.k_fold):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.k_fold) if j != i])
            
            X_train = self._X[train_idx]
            Y_train = self._Y[train_idx]
            errors_train = self._errors[train_idx]

            X_test = self._X[test_idx]
            Y_test = self._Y[test_idx]

            # Fit the model on the training fold
            M_train = np.vander(X_train, N=self.degree + 1, increasing=True)
            C_train = np.diag(errors_train**2)
            C_inv = np.linalg.inv(C_train)
            MT_Cinv = M_train.T @ C_inv
            #Comment the following line if you want to NOT use regularization
            Theta = np.linalg.inv(MT_Cinv @ M_train) @ (MT_Cinv @ Y_train)
            #lambda_reg = 1e-6  # You can tune this value
            #I = np.eye(M_train.shape[1])
            #Theta = np.linalg.inv(MT_Cinv @ M_train + lambda_reg * I) @ (MT_Cinv @ Y_train)

            # Predict on training and test sets
            Y_train_pred = M_train @ Theta
            M_test = np.vander(X_test, N=self.degree + 1, increasing=True)
            Y_test_pred = M_test @ Theta

            # RMSE calculations
            train_rmse = np.sqrt(np.mean((Y_train - Y_train_pred) ** 2))
            val_rmse = np.sqrt(np.mean((Y_test - Y_test_pred) ** 2))

            train_rmse_list.append(train_rmse)
            val_rmse_list.append(val_rmse)

        mean_train_rmse = np.mean(train_rmse_list)
        mean_val_rmse = np.mean(val_rmse_list)
        return mean_train_rmse, mean_val_rmse