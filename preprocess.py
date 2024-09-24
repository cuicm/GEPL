from scipy.signal import resample
import numpy as np
from scipy.fft import fft

def resample_signals(signals, original_sample_rate, target_sample_rate=200):
    if original_sample_rate != target_sample_rate:
        num_samples = int(target_sample_rate * int(signals.shape[1] / original_sample_rate))
        resampled_signals = resample(signals, num=num_samples, axis=1)
        return resampled_signals
    return signals

def segment_signals_list(signals, window_length=10000):
    num_segments = signals.shape[1] // window_length
    segmented_signals = []
    for i in range(num_segments):
        start_idx = i * window_length
        end_idx = start_idx + window_length
        segment = signals[:, start_idx:end_idx]
        segmented_signals.append(segment)
    return segmented_signals


def segment_signals(signals, window_length=10000):
    signal_length = signals.shape[1]
    if signal_length < window_length:
        padded_signals = np.zeros((signals.shape[0], window_length))
        padded_signals[:, :signal_length] = signals
        return padded_signals
    else:
        start_idx = np.random.randint(0, signal_length - window_length + 1)
        end_idx = start_idx + window_length
        segmented_signal = signals[:, start_idx:end_idx]
        return segmented_signal

# Z-score
def z_score_standardization(signals):
    mean = np.mean(signals, axis=1, keepdims=True)
    std = np.std(signals, axis=1, keepdims=True)
    standardized_data = (signals - mean) / std
    return standardized_data

def FFT(signals, n):

    transformed_signal = fft(signals, n=n, axis=-1)

    positive_freq_index = int(np.floor(n / 2))
    transformed_signal = transformed_signal[:, :positive_freq_index]
    
    amplitude = np.abs(transformed_signal)
    amplitude[amplitude == 0.0] = 1e-8

    FT = np.log(amplitude)
    P = np.angle(transformed_signal)

    return FT, P


def preprocess_eeg(signals,original_sample_rate, target_sample_rate,window_length=10000):
    
    # 1. resample
    signals = resample_signals(signals,original_sample_rate,target_sample_rate)
    
    # 2. Intercepting signals of the same length
    signals = segment_signals(signals,window_length)

    # 3. FFT
    FT_signals,_ = FFT(signals,window_length)
     
    return FT_signals
    