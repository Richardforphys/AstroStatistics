import numpy as np
import matplotlib.pyplot as plt


def zeropad(data, template):
    """
    Returns zero-padded template data
    ----------------------------------
    Parameters: 
        - data : shape(N,0)
        - template: shape(M,0) with M<N
    Returns:
        - Zeropadded array
    """
    zero_pad = np.zeros(abs(data.size - template.size))
    return  np.append(template, zero_pad)

def compute_PSD(data, fs, plot):
    
    """
    Compute PSD of input data
    -------------------------------------
    Parameters:
        - data (shape (N, 0))
        - template (shape(N,0))
        - fs - sampling frequency
    Returns:
        - power vec    
    
    """    
    # -- Calculate the PSD of the data
    if plot:
        power_data, freq_psd = plt.psd(data[12*fs:], Fs=fs, NFFT=fs, visible=True)
        plt.show()
    else:
        power_data, freq_psd = plt.psd(data[12*fs:], Fs=fs, NFFT=fs, visible=False)

    # -- Interpolate to get the PSD values at the needed frequencies
    datafreq = np.fft.fftfreq(data.size)*fs
    power_vec = np.interp(datafreq, freq_psd, power_data)
    
    return power_vec
    
def compute_OF(data, template, power_vec):
    
    """
    Computes Optimum filter for input data, given template and power vec
    -----------------------------
    Parameters:
        - data (shape (N, 0))
        - template (shape(N,0))
    
    """
    data_fft = np.fft.fft(data)
    template_fft = np.fft.fft(template)
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)
    
    return optimal, optimal_time

def Normalize(data, template_fft, fs, optimal_time, power_vec, plot):

    datafreq = np.fft.fftfreq(data.size)*fs
    df = np.abs(datafreq[1] - datafreq[0])
    sigmasq = 2*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = abs(optimal_time) / (sigma)
    
    if plot:
        plt.figure()
        plt.plot(SNR, label='SNR', color='blue', alpha=1)
        plt.title('Signal')
        plt.xlabel('Offset time')
        plt.ylabel('Normalized filter output')
        #plt.savefig(r'C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroGW\PNGs\Template0.png')
        plt.show()
    return SNR

