import numpy as np

re = 79.4

def distance(beta, theta_lim_1, theta_lim_2, d, D):
    if ((beta<theta_lim_1 and beta>theta_lim_2) or beta>-theta_lim_1 and beta<-theta_lim_2):
        return D*np.sqrt(1+np.tan(beta))
    elif (beta>-theta_lim_2 and beta<theta_lim_2):
        a = d - D/(2 * np.tan(beta))
        b = D/2 - d*np.tan(beta)
        return np.sqrt(a**2 + b**2) 
    else:
        return 0
    
def interaction_prob(x, xs):
    return 1 - np.exp(-x * xs)

def lambda_over_lambdap(Energy, theta):
    mec2 = 0.511  # MeV
    return 1 / (1 + (Energy / mec2) * (1 - np.cos(theta)))

def KN(Energy, theta):
    ll = lambda_over_lambdap(Energy, theta)
    return 0.5 * re**2 * ll**2 * (ll + 1/ll - np.sin(theta)**2)

def sample_theta_from_KN(Energy, E0, compton_number=1):
    samples, scattered, deposited = [], [], []
    theta_vals = np.linspace(0, np.pi, int(1e6))
    KN_vals = KN(Energy, theta_vals)
    KN_max = KN_vals.max()

    while len(samples) < compton_number:
        theta_candidate = np.random.uniform(0, np.pi)
        y = np.random.uniform(0, KN_max)
        if y < KN(Energy, theta_candidate):
            samples.append(theta_candidate)
            scattered.append(E0/(1 + E0 * (1 - np.cos(theta_candidate))))
            deposited.append(E0 - E0/(1 + E0 * (1 - np.cos(theta_candidate))) )
        

    return np.array(samples), scattered, deposited

def interaction_prob_compton(x, xs_compton):  # Usa xs_compton
    return 1 - np.exp(-x * xs_compton)