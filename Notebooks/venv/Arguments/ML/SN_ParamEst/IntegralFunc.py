import numpy as np
import matplotlib.pyplot as plt
import emcee

data = np.genfromtxt(r'C:\Users\ricca\Documents\Unimib-Code\Astrostatistics\Notebooks\Data\mu_z.txt', delimiter=' ')
z_sample  = data[:, 0]
mu_sample = data[:, 1]
dmu       = data[:, 2]


y_true = mu_sample / 5
sigma = dmu

def f(z, B):
    return  1/np.sqrt(B*z**3 + 1)

def compute_y_pred(x):
    A, B = x
    zi = np.random.uniform(0, max(y_true), size=10000)
    return np.log10(A) + np.log10(zi*(1+zi)) + np.log10(np.mean(f(zi, B)))   

def log_likelihood(x):
    A, B = x
    N = y_true.shape[0]
    y_pred = compute_y_pred(x)
    return -N/2 * np.log(2 * np.pi) - np.sum(np.log(sigma)) - 0.5 * np.sum((y_true - y_pred)**2/(sigma**2))

def prior(x):
    
    A, B = x

    if not (0 <= A <= 1 and 0 <= B <= 1):
        return -np.inf  # outside the prior bounds

    log_prior_A     = np.log(1.0)   # uniform(0, 1)
    log_prior_B     = np.log(1.0)

    return log_prior_A + log_prior_B

def posterior(x):
    
    log_prior_value = prior(x)
    if not np.isfinite(log_prior_value):
        return -np.inf

    log_like = log_likelihood(x)
    if not np.isfinite(log_like):
        return -np.inf

    return log_like + log_prior_value

ndim     = 2  # number of parameters in the model
nwalkers = 10  # number of MCMC walkers
nsteps   = int(1e5)  # number of MCMC steps to take **for each walker**

guess_parameters = np.array([0.1, 0.3])

sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior)

sampler.run_mcmc(guess_parameters, nsteps)