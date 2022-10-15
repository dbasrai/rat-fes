import tdt
import math
import numpy as np
import pandas as pd
from src.tdt_support import *
from src.filters import *

#i initially wrote these thinking neural is channels x samples, but more
# standard is samples x channels. I just transpose all functions. its easier.

def threshold_crossings(neural, threshold): #this finds all upwards threshold crossings, no artifact detection

    neural = neural.T

    polarity=True #if threshold is positive
    if threshold < 0:
        polarity=False
    if polarity:
        crossings = np.diff(neural > threshold, prepend=False) #we get a 1
        #right when it crosses. then 0s until it comes back/crosses again
    else:
        crossings = np.diff(neural < threshold, prepend=False) 
    output = np.zeros((crossings.shape[0], crossings.shape[1]))
    upward_indices_list = []
    for i, channel in enumerate(crossings):
        upward_indices = np.argwhere(channel)[::2,0]
        upward_indices_list.append(upward_indices)
        np.put(output[i,:], upward_indices, np.ones(upward_indices.shape))
    
    return output.T, upward_indices_list

def autothreshold_crossings(neural, multiplier): #this finds all upwards threshold crossings, no artifact detection
    neural = neural.T

    polarity=True #if threshold is positive
    if multiplier < 0:
        polarity=False
    spikes = []
    
    
    for channel in neural:
        std = np.std(channel)
        if polarity:
            crossings = np.diff(channel > multiplier * std, prepend=0) #we get a 1
            #right when it crosses. then 0s until it comes back/crosses again
        else:
            crossings = np.diff(channel < multiplier*std, prepend=0)

        np.put(crossings, np.where(crossings==-1), 0) #only catch crossing
        spikes.append(crossings)

    return np.array(spikes).T #samples x spikes

def spike_binner(spikes, fs, binsize=0.05):
    spikes = spikes.T

    time = 1/fs
    output_list=[]
    bin_length = int(binsize / time)
    for idx, channel in enumerate(spikes):
        binned_spikes = []
        for i in range(0,channel.size,bin_length):
            binned_spikes.append(np.count_nonzero(channel[i:i+bin_length]) / binsize)
        output_list.append(binned_spikes)
        
    return np.array(output_list).T

def bandpass_neural(neural,fs):
    neural = neural.T

    # return butter_bandpass_filter(neural, 250, 3000, fs).T
    return butter_bandpass_filter(neural, 250, 6000, fs).T

# def bandpass_neural_test(neural,fs):
#     neural = neural.T

#     # return butter_bandpass_filter(neural, 250, 3000, fs).T
#     return butter_bandpass_filter(neural, 10, 6000, fs).T

def filter_neural(neural, fs):
    neural = neural.T

    return notch_filter(bandpass_neural(neural, fs), fs).T
    
def average_neural(neural):
    neural = neural.T

    return np.divide(np.sum(neural, 0), neural.shape[0]).T

def remove_artifacts(neural, fs):
    neural = neural.T

    sum_neural = np.sum(np.abs(neural), 0)
    binsize = math.floor(0.001 * fs)
    binned_sum_neural = []

    for i in range(0,sum_neural.shape[0],binsize):
        binned_sum_neural.append(np.average(sum_neural[i:i+binsize]))
        
    std = np.std(np.array(binned_sum_neural))
    bad_bins = []
    for index, bin_value in enumerate(binned_sum_neural):
        if bin_value > (5*std):
            bad_bins.append(index*binsize)
            
    new_neural = []

    for channel in neural:
        std = np.std(channel)
        for artifacts in bad_bins:
            channel[artifacts:artifacts + binsize] = std
            
        new_neural.append(channel)
        
    updated_neural = np.array(new_neural)
    return updated_neural.T
