import numpy as np
import matplotlib.pyplot as plt
import sys
from Rivelatori.RivUtils import Uts

####################################### Coefficienti Geometrici ########################################


d = float(sys.argv[1])  # Semidistanza tra NaI in cm
D = 5 # Lato degli NaI in cm

theta_lim_1 = np.arctan(D/(2*d))   # angolo massimo in radianti per faccia verticale
theta_lim_2 = np.arctan(D/(2*D+d)) # angolo massimo in radianti per faccia orizzontale

####################################### Parte temporale ########################################

N_events = int(1e6)         
Observation_time = 10       
Rate = N_events / Observation_time
tau = 5e-5                  # tempo discesa: 50 µs
tau_rise = 1e-6             # tempo salita: 1 µs


###################################### Efficienza geometrica ###########################################

beta = np.random.uniform(-np.pi/2, np.pi/2, size=N_events)

Coeff_int = 1.6*1e-2 # Coefficiente d'interazione a 511 KeV in cm2/g
densità   = 3.67 #Densità NaI in g/cm3 
xs = Coeff_int * densità
abs_compton = 4.8*1e-2
xs_compton = abs_compton * densità
E0 = 1274

distances = np.array([Uts.distance(b, theta_lim_1, theta_lim_2, d, D) for b in beta])
print('FE distances simulated')
probs = Uts.interaction_prob(distances, xs)
energies = np.where(np.random.uniform(0, 1, size=probs.size) < probs, 511, 0)  # Effetto fotoelettrico

N_1274 = N_events
beta_1274 = np.random.uniform(-np.pi/2, np.pi/2, size=N_1274)
dist_1274 = np.array([Uts.distance(b, theta_lim_1, theta_lim_2, d, D) for b in beta_1274])

prob_compton = Uts.interaction_prob_compton(dist_1274, xs_compton)

print('Computed Compton probs')

# Campiona chi interagisce con Compton
compton_interactions = np.random.uniform(0, 1, size=N_1274) < prob_compton # Compton
n_comptons = compton_interactions.sum()

# Ora chiama la funzione
theta_compton, energy_scattered, energy_deposited = Uts.sample_theta_from_KN(Energy=E0, E0=E0, compton_number=n_comptons)
print('KN simulation, done')

fwhm = 0.07 * 511  # stima media
sigma = fwhm / 2.355

# Solo gli eventi con tempi (511 keV + Compton depositati)
energies_all = np.concatenate((energies, energy_deposited))
energies_all = energies_all[energies_all > 0]

# Applichiamo lo smear (risoluzione energetica)
energies_smeared = energies_all + np.random.normal(0, sigma, size=energies_all.size)

print('Parte temporale...')

# Tempo tra eventi (distribuzione esponenziale)
inter_arrival_times = np.random.exponential(scale=1/Rate, size=energies_smeared.shape[0])

# Differenze tra tempi consecutivi (quanto sono vicini gli eventi)
arrival_times = np.cumsum(inter_arrival_times)
time_diffs = np.diff(arrival_times)

# Individua i pile-up
desc_pile_up_idx = np.where((time_diffs < tau) & (time_diffs > tau_rise))[0]
# Identifica pile-up a tempo di salita
rise_pile_up_idx = np.where(time_diffs < tau_rise)[0]

# Verifica coerenza
assert energies_smeared.shape[0] == arrival_times.shape[0], "Energie e tempi non allineati"

print('Disegno Istogramma')

# Somma energie dei segnali sovrapposti
fused_energies = energies_smeared[rise_pile_up_idx] + energies_smeared[rise_pile_up_idx + 1]
fused_times = arrival_times[rise_pile_up_idx]  # tempo del primo evento

# Escludi gli eventi coinvolti nel pile-up
to_exclude = np.unique(np.concatenate([rise_pile_up_idx, rise_pile_up_idx + 1]))
valid_mask = np.ones_like(energies_smeared, dtype=bool)
valid_mask[to_exclude] = False

remaining_energies = energies_smeared[valid_mask]
remaining_times = arrival_times[valid_mask]

# Unione di tutti gli eventi
final_energies = np.concatenate([remaining_energies, fused_energies])
final_times = np.concatenate([remaining_times, fused_times])

# Ordina in base al tempo
sorted_idx = np.argsort(final_times)
final_energies = final_energies[sorted_idx]

bins = np.linspace(0, 1400, 700)  # bin energetici

print(f"Totale eventi simulati: {energies_smeared.shape[0]}")
print(f"Eventi in pile-up (tempo discesa): {len(desc_pile_up_idx)}")
print(f"Eventi in pile-up (tempo salita): {len(rise_pile_up_idx)}")
print(f"Tempo medio tra eventi: {np.mean(time_diffs):.2e}")


plt.figure(figsize=(10,6))
plt.hist(final_energies, bins=bins, histtype='step', color='blue', label='511 keV + Compton 1274 keV + Fondo')
plt.xlabel("Energia [keV]")
plt.ylabel("Conteggi")
plt.title("Spettro simulato")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'spettro_totale_{d}_cm.png')