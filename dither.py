import numpy as np
import wave
import struct
import matplotlib.pyplot as plt

def shift_signal(signal, shift):
    shifted_signal = np.zeros_like(signal)
    if shift > 0:
        # Right shift
        shifted_signal[shift:] = signal[:-shift]
    elif shift < 0:
        # Left shift
        shifted_signal[:shift] = signal[-shift:]
    else:
        # No shift
        shifted_signal = signal
    return shifted_signal


def generate_dither(signal, step_size):
    # Generate dither noise (white noise)
    dither1 = np.random.uniform(-step_size / 2, step_size / 2, size=signal.shape)
    dither2 = np.random.uniform(-step_size / 2, step_size / 2, size=signal.shape)

    shift_amount = len(dither1)  # Shifting to n-1
    shifted_dither1 = shift_signal(dither1, shift_amount)

    # Add dither to the signal
    d_rect = dither1
    d_tri = dither1 + dither2
    d_hp = dither1 + shifted_dither1

    return d_rect, d_tri, d_hp


def signal_dither(signal, step_size):
    d_rect, d_tri, d_hp = generate_dither(signal, step_size)

    signal_dither_rect = signal + d_rect
    signal_dither_tri = signal + d_tri
    signal_dither_hp = signal + d_hp

    return signal_dither_rect, signal_dither_tri, signal_dither_hp