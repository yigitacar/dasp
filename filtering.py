import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Define filter design functions
def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def band_pass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def low_shelf_filter(data, cutoff, fs, gain_db, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.iirfilter(order, normal_cutoff, btype='low', ftype='butter', rs=1, rp=0.5, output='ba')
    sos = signal.iirfilter(order, normal_cutoff, btype='low', ftype='butter', rs=1, rp=0.5, output='sos')
    # Apply gain
    gain = 10**(gain_db / 20.0)
    b *= gain
    y = signal.sosfilt(sos, data)
    return y

def high_shelf_filter(data, cutoff, fs, gain_db, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.iirfilter(order, normal_cutoff, btype='high', ftype='butter', rs=1, rp=0.5, output='ba')
    sos = signal.iirfilter(order, normal_cutoff, btype='high', ftype='butter', rs=1, rp=0.5, output='sos')
    # Apply gain
    gain = 10**(gain_db / 20.0)
    b *= gain
    y = signal.sosfilt(sos, data)
    return y

def plot_fourier(signal, fs, title):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)
    fft_values = np.fft.fft(signal)
    plt.plot(freqs[:n//2], np.abs(fft_values)[:n//2])
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

# Load an example sound file (change the path to your file)
fs, data = wavfile.read('StarWars3.wav')

# If stereo, convert to mono by averaging the channels
if len(data.shape) == 2:
    data = data.mean(axis=1)

# Define filter parameters
lp_cutoff = 1000  # Low-pass filter cutoff frequency (Hz)
hp_cutoff = 1000  # High-pass filter cutoff frequency (Hz)
bp_lowcut = 500   # Band-pass filter low cutoff frequency (Hz)
bp_highcut = 1500 # Band-pass filter high cutoff frequency (Hz)
shelf_gain_db = 10  # Gain in decibels for shelving filters
shelf_cutoff = 1000  # Cutoff frequency for shelving filters

# Apply filters
lp_filtered_data = low_pass_filter(data, lp_cutoff, fs)
hp_filtered_data = high_pass_filter(data, hp_cutoff, fs)
bp_filtered_data = band_pass_filter(data, bp_lowcut, bp_highcut, fs)
low_shelf_filtered_data = low_shelf_filter(data, shelf_cutoff, fs, shelf_gain_db)
high_shelf_filtered_data = high_shelf_filter(data, shelf_cutoff, fs, shelf_gain_db)

# Save the filtered audio
wavfile.write('low_pass_filtered.wav', fs, lp_filtered_data.astype(np.int16))
wavfile.write('high_pass_filtered.wav', fs, hp_filtered_data.astype(np.int16))
wavfile.write('band_pass_filtered.wav', fs, bp_filtered_data.astype(np.int16))
wavfile.write('low_shelf_filtered.wav', fs, low_shelf_filtered_data.astype(np.int16))
wavfile.write('high_shelf_filtered.wav', fs, high_shelf_filtered_data.astype(np.int16))

# Plot the results
plt.figure(figsize=(15, 25))

# Plot time domain signals
plt.subplot(6, 2, 1)
plt.plot(data, label='Original Signal')
plt.title('Original Signal (Time Domain)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(6, 2, 2)
plot_fourier(data, fs, 'Original Signal (Frequency Domain)')

plt.subplot(6, 2, 3)
plt.plot(lp_filtered_data, label='Low-Pass Filtered Signal', color='r')
plt.title('Low-Pass Filtered Signal (Time Domain)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(6, 2, 4)
plot_fourier(lp_filtered_data, fs, 'Low-Pass Filtered Signal (Frequency Domain)')

plt.subplot(6, 2, 5)
plt.plot(hp_filtered_data, label='High-Pass Filtered Signal', color='g')
plt.title('High-Pass Filtered Signal (Time Domain)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(6, 2, 6)
plot_fourier(hp_filtered_data, fs, 'High-Pass Filtered Signal (Frequency Domain)')

plt.subplot(6, 2, 7)
plt.plot(bp_filtered_data, label='Band-Pass Filtered Signal', color='b')
plt.title('Band-Pass Filtered Signal (Time Domain)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(6, 2, 8)
plot_fourier(bp_filtered_data, fs, 'Band-Pass Filtered Signal (Frequency Domain)')

plt.subplot(6, 2, 9)
plt.plot(low_shelf_filtered_data, label='Low-Shelf Filtered Signal', color='m')
plt.title('Low-Shelf Filtered Signal (Time Domain)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(6, 2, 10)
plot_fourier(low_shelf_filtered_data, fs, 'Low-Shelf Filtered Signal (Frequency Domain)')

plt.subplot(6, 2, 11)
plt.plot(high_shelf_filtered_data, label='High-Shelf Filtered Signal', color='c')
plt.title('High-Shelf Filtered Signal (Time Domain)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(6, 2, 12)
plot_fourier(high_shelf_filtered_data, fs, 'High-Shelf Filtered Signal (Frequency Domain)')

plt.tight_layout()
plt.show()
