import numpy as np
import matplotlib.pyplot as plt

class LR:
    """
    Class for performing polynomial regression of arbitrary degree.
    """

    def __init__(self, xvalues, yvalues, errors, degree=1):
        """
        Initialize the Regression class.
        """
        self._X = np.asarray(xvalues)
        self._Y = np.asarray(yvalues)
        self._errors = np.asarray(errors)
        self.degree = degree
        
        self.M = self._build_design_matrix()
        self.C = self._build_covariance_matrix()
        self.Theta = None
        self.Sigma = None

    def _build_design_matrix(self):
        """
        Build the design matrix for polynomial regression.
        """
        M = np.vander(self._X, N=self.degree + 1, increasing=True)
        return M

    def _build_covariance_matrix(self):
        """
        Build the diagonal covariance matrix from errors.
        """
        return np.diag(self._errors**2)

    def linear_fit(self):
        """
        Fit the polynomial regression model.
        """
        C_inv = np.linalg.inv(self.C)
        MT_Cinv = self.M.T @ C_inv
        self.Theta = np.linalg.inv(MT_Cinv @ self.M) @ (MT_Cinv @ self._Y)
        self.Sigma = np.linalg.inv(MT_Cinv @ self.M)
        
    def compute_chi2(self):
        """
        Compute the chi-squared statistic.
        """
        residuals = self._Y - self.M @ self.Theta
        chi2 = residuals.T @ np.linalg.inv(self.C) @ residuals
        return chi2
    
    def compute_reduced_chi2(self):
        """
        Compute the reduced chi-squared statistic.
        """
        dof = len(self._Y) - (self.degree + 1)
        chi2 = self.compute_chi2()
        return chi2 / dof if dof > 0 else np.inf

    def plot_fit(self, resolution=1000):
        """
        Plot the data and the polynomial fit.
        """
        plt.figure(figsize=(10, 6))
        plt.errorbar(self._X, self._Y, yerr=self._errors, fmt='o', label='Data')

        x_fit = np.linspace(np.min(self._X), np.max(self._X), resolution)
        X_fit = np.vander(x_fit, N=self.degree + 1, increasing=True)
        y_fit = X_fit @ self.Theta

        plt.plot(x_fit, y_fit, color='red', label=f'Polynomial fit (degree {self.degree})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Polynomial Regression (degree {self.degree})')
        plt.legend()
        plt.grid(True)
        plt.text(0.05, 0.95, f'$\chi^2_r$ = {self.compute_reduced_chi2():.2f}',
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.show()
