import numpy as np
import matplotlib.pyplot as plt
import sys
path = r"C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroStatistics\ML\GammaRayBursts\Utilities"
sys.path.append(path)
import plot_settings
from astroML.datasets import generate_mu_z
from sklearn.model_selection import ShuffleSplit
from astroML.linear_model import PolynomialRegression

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234) # YOU CANNOT CHANGE THIS

X = np.vstack((z_sample, mu_sample, dmu)).T

CV = ShuffleSplit(n_splits=5, test_size=0.4, random_state=1234)

for train_index, test_index in CV.split(X):
    x_train, x_test = z_sample[train_index], z_sample[test_index]
    y_train, y_test = mu_sample[train_index], mu_sample[test_index]
    y_err_train, y_err_test = dmu[train_index], dmu[test_index]
    
degree = 3
model = PolynomialRegression(degree) # fit 1rd degree polynomial
model.fit(x_train.reshape((-1,1)), y_train, y_error=1/y_err_train**2)

y_pred = model.predict(x_test.reshape((-1,1)))

plt.errorbar(x_train, y_train, y_err_train, fmt='.k', ecolor='gray', lw=1,label='data')
plt.errorbar(x_test, y_test, y_err_test, fmt='.k', ecolor='red', lw=1,label='data')
plt.plot(x_train, y_train, 'o', label='Training data', color='gray', alpha=0.5)
plt.plot(x_test, y_test, 'o', label='Test data', color='red', alpha=0.5)
plt.plot(x_test, y_pred, 'o', label='Prediction', color='blue', alpha=0.5)
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.title('Prediction: %dth degree polynomial' % degree)
plt.show()
plt.savefig('SN.png', dpi=300, bbox_inches='tight')