import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import h5py
import sys
sys.path.append(r'C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroStatistics\ML\GammaRayBursts\Utilities')
import plot_settings
sys.path.append(r'C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroGW\GWUtilities')
from GWUtils import compute_PSD, compute_OF, Normalize, zeropad


#=================================================== Load data and templates ==============================================================


file_path = r"C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroGW\GW-signals\data_w_signal.hdf5"

# -- Read the data file (16 seconds, sampled at 4096 Hz)
fs = 4096
dataFile = h5py.File(file_path, 'r')
data = dataFile['strain/Strain'][...]
dataFile.close()
time = np.arange(0, 16, 1./fs)

template1 = np.loadtxt(r"C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroGW\GW-signals\GW150914_4_NR_waveform.txt").T[1]
temp_time1 = np.arange(0, template1.size / (1.0*fs), 1./fs)

template_path = r"C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroGW\GW-signals\template.hdf5"
templateFile = h5py.File(template_path, 'r')
template2 = templateFile['strain/Strain'][...]
temp_time2 = np.arange(0, template2.size / (1.0*fs), 1./fs)
templateFile.close()

template_dic = {
    'templates': [template1, template2],
    'times': [temp_time1, temp_time2]
}


#=================================================== Compute OF  ==============================================================

for time, template in zip(template_dic['times'], template_dic['templates']):
    power_vec = compute_PSD(data, fs, True)
    padded_template = zeropad(data, template)
    template_fft = np.fft.fft(padded_template)
    optimal, optimal_time = compute_OF(data, padded_template, power_vec)
    SNR = Normalize(data, template_fft, fs, optimal_time, power_vec, True)
