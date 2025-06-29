import numpy as np
import matplotlib.pyplot as plt
import sys

d = float(sys.argv[1])  # Semidistanza tra NaI in cm
D = 5 # Lato degli NaI in cm

theta_lim_1 = np.arctan(D/(2*d))   # angolo massimo in radianti per faccia verticale
theta_lim_2 = np.arctan(D/(2*D+d)) # angolo massimo in radianti per faccia orizzontale

beta = np.random.uniform(-np.pi/2, np.pi/2, size=int(1e5))

re = 79.4

Coeff_int = 1.6*1e-2 # Coefficiente d'interazione a 511 KeV in cm2/g
densità   = 3.67 #Densità NaI in g/cm3 
xs = Coeff_int * densità

abs_compton = 4.8*1e-2
xs_compton = abs_compton * densità

E0 = 1274

def distance(beta):
    if ((beta<theta_lim_1 and beta>theta_lim_2) or beta>-theta_lim_1 and beta<-theta_lim_2):
        return D*np.sqrt(1+np.tan(beta))
    elif (beta>-theta_lim_2 and beta<theta_lim_2):
        a = d - D/(2 * np.tan(beta))
        b = D/2 - d*np.tan(beta)
        return np.sqrt(a**2 + b**2) 
    else:
        return 0
    
def interaction_prob(x):
    return 1 - np.exp(-x * xs)

def lambda_over_lambdap(Energy, theta):
    mec2 = 0.511  # MeV
    return 1 / (1 + (Energy / mec2) * (1 - np.cos(theta)))

def KN(Energy, theta):
    ll = lambda_over_lambdap(Energy, theta)
    return 0.5 * re**2 * ll**2 * (ll + 1/ll - np.sin(theta)**2)

def sample_theta_from_KN(Energy, compton_number=1):
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


def interaction_prob_compton(x):  # Usa xs_compton
    return 1 - np.exp(-x * xs_compton)

distances = np.array([distance(b) for b in beta])

print('FE distances simulated')

probs = interaction_prob(distances)

energies = np.where(np.random.uniform(0, 1, size=probs.size) < probs, 511, 0)  # in keV

N_1274 = int(1e5)
beta_1274 = np.random.uniform(-np.pi/2, np.pi/2, size=N_1274)
dist_1274 = np.array([distance(b) for b in beta_1274])

prob_compton = interaction_prob_compton(dist_1274)

print('Computed Compton probs')

# Campiona chi interagisce con Compton
compton_interactions = np.random.uniform(0, 1, size=N_1274) < prob_compton
n_comptons = compton_interactions.sum()

# Ora chiama la funzione
theta_compton, energy_scattered, energy_deposited = sample_theta_from_KN(Energy=E0, compton_number=n_comptons)
print('KN simulation, done')
fwhm = 0.07 * 511  # stima media
sigma = fwhm / 2.355

n_bg_flat = 1000  # numero di eventi ambientali a bassa energia
bg_flat = np.random.uniform(0, 200, size=n_bg_flat)  # tra 0 e 200 keV

n_bg_wide = 2000  # eventi su tutta la gamma energetica
bg_wide = np.random.uniform(0, 1400, size=n_bg_wide)

n_k40 = 300
bg_k40 = np.random.normal(1460, 50, size=n_k40)  # fwhm simulato ~117 keV

energies_all = np.concatenate((energies, energy_deposited))
energies_all = energies_all[energies_all > 0]
energies_smeared = energies_all + np.random.normal(0, sigma, size=energies_all.size)

# Aggiungi le componenti di background
total_spectrum = np.concatenate((energies_smeared, bg_flat, bg_wide, bg_k40))  # bg_k40 se lo includi

bins = np.linspace(0, 1400, 700)  # bin per coprire anche i 1274 keV
plt.hist(total_spectrum, bins=bins, histtype='step', color='blue', label='511 keV + Compton 1274 keV + Fondo')
plt.xlabel("Energia [keV]")
plt.ylabel("Conteggi")
plt.title("Spettro simulato")
plt.grid(True)
plt.savefig(f'spettro_totale_{d}_cm.png')