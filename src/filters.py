import tdt 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, iirnotch, filtfilt, resample
from scipy.io import savemat

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def bandpass_neural(neural, fs):
    return butter_bandpass_filter(neural, 250, 3000, fs)
    
def notch_filter(neural, fs):
    f0 = 60.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    b_notch, a_notch = iirnotch(f0, Q, fs)
    
    return filtfilt(b_notch, a_notch, neural)

    

