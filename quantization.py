import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
from dither import signal_dither
# TODO: read a wav file

# Create signal
frequency = 1000
num_samples = 500
sampling_rate = 48000.0
amplitude = 16
quantization_step_size = 4

def quantize_rounding(signal, step_size):
    """Quantize the signal using rounding."""
    return np.round(signal / step_size) * step_size

def quantize_truncation(signal, step_size):
    """Quantize the signal using truncation."""
    return np.floor(signal / step_size) * step_size


# Generate signal
t = np.arange(num_samples)
original_signal = amplitude * np.array([np.sin(2 * np.pi * frequency * t / sampling_rate) for t in range(num_samples)])

signal_dither_rect, signal_dither_tri, signal_dither_hp = signal_dither(original_signal, quantization_step_size)

# Quantize the signal using rounding
quantized_signal_rounding = quantize_rounding(original_signal, quantization_step_size)
quantized_signal_rounding_dither_rect = quantize_rounding(signal_dither_rect, quantization_step_size)
quantized_signal_rounding_dither_tri = quantize_rounding(signal_dither_tri, quantization_step_size)
quantized_signal_rounding_dither_hp = quantize_rounding(signal_dither_hp, quantization_step_size)

# Quantize the signal using truncation
quantized_signal_truncation = quantize_truncation(original_signal, quantization_step_size)
quantized_signal_truncation_dither_rect = quantize_truncation(signal_dither_rect, quantization_step_size)
quantized_signal_truncation_dither_tri = quantize_truncation(signal_dither_tri, quantization_step_size)
quantized_signal_truncation_dither_hp = quantize_truncation(signal_dither_hp, quantization_step_size)

# Calculate quantization errors
quantization_error_rounding = original_signal - quantized_signal_rounding
quantization_error_truncation = original_signal - quantized_signal_truncation

# Plot the results
plt.figure(figsize=(12, 8))

# Plot original and quantized signals
plt.subplot(2, 1, 1)
plt.plot(t, original_signal, label='Original Signal')
plt.plot(t, quantized_signal_rounding, 'o-', label='Quantized Signal (Rounding)')
plt.plot(t, quantized_signal_truncation, 'x-', label='Quantized Signal (Truncation)')
plt.legend()
plt.title('Original and Quantized Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# Plot quantization errors
plt.subplot(2, 1, 2)
plt.plot(t, quantization_error_rounding, 'o-', label='Quantization Error (Rounding)')
plt.plot(t, quantization_error_truncation, 'x-', label='Quantization Error (Truncation)')
plt.legend()
plt.title('Quantization Errors')
plt.xlabel('Sample Index')
plt.ylabel('Error')

# Plot quantized signals with rounding and truncation, with and without dithering
plt.figure(figsize=(12, 16))

# Original signal
plt.subplot(4, 2, 1)
plt.plot(t, original_signal, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Quantized signal with rounding (no dithering)
plt.subplot(4, 2, 2)
plt.plot(t, quantized_signal_rounding, label='Quantized Signal (Rounding)')
plt.title('Quantized Signal with Rounding (No Dithering)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Quantized signal with truncation (no dithering)
plt.subplot(4, 2, 3)
plt.plot(t, quantized_signal_truncation, label='Quantized Signal (Truncation)')
plt.title('Quantized Signal with Truncation (No Dithering)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Quantized signal with rounding (dithering: rectangular)
plt.subplot(4, 2, 4)
plt.plot(t, quantized_signal_rounding_dither_rect, label='Quantized Signal (Rounding, Dither: Rectangular)')
plt.title('Quantized Signal with Rounding (Dither: Rectangular)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Quantized signal with truncation (dithering: rectangular)
plt.subplot(4, 2, 5)
plt.plot(t, quantized_signal_truncation_dither_rect, label='Quantized Signal (Truncation, Dither: Rectangular)')
plt.title('Quantized Signal with Truncation (Dither: Rectangular)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Quantized signal with rounding (dithering: triangular)
plt.subplot(4, 2, 6)
plt.plot(t, quantized_signal_rounding_dither_tri, label='Quantized Signal (Rounding, Dither: Triangular)')
plt.title('Quantized Signal with Rounding (Dither: Triangular)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Quantized signal with truncation (dithering: triangular)
plt.subplot(4, 2, 7)
plt.plot(t, quantized_signal_truncation_dither_tri, label='Quantized Signal (Truncation, Dither: Triangular)')
plt.title('Quantized Signal with Truncation (Dither: Triangular)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Quantized signal with rounding (dithering: high-pass)
plt.subplot(4, 2, 8)
plt.plot(t, quantized_signal_rounding_dither_hp, label='Quantized Signal (Rounding, Dither: High-Pass)')
plt.title('Quantized Signal with Rounding (Dither: High-Pass)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()


plt.tight_layout()
plt.show()

