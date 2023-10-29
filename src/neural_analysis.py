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

# def autothreshold_crossings(neural, multiplier): #this finds all upwards threshold crossings, no artifact detection
#     neural = neural.T

#     polarity=True #if threshold is positive
#     if multiplier < 0:
#         polarity=False
#     spikes = []
    
    
#     for channel in neural:
#         std = np.std(channel)
#         if polarity:
#             crossings = np.diff(channel > multiplier * std, prepend=0) #we get a 1
#             #right when it crosses. then 0s until it comes back/crosses again
#         else:
#             crossings = np.diff(channel < multiplier*std, prepend=0)

#         np.put(crossings, np.where(crossings==-1), 0) #only catch crossing
#         spikes.append(crossings)

#     return np.array(spikes).T #samples x spikes


def static_threshold_crossings(neural, thresholds, multiplier): #this finds all upwards threshold crossings, no artifact detection
    neural = neural.T

    polarity=True #if threshold is positive
    if multiplier < 0:
        polarity=False
    spikes = []
    
    for i in range(neural.shape[0]):
        if polarity:
            crossings = np.diff(neural[i,:] > thresholds[i], prepend=0) #we get a 1
            #right when it crosses. then 0s until it comes back/crosses again
        else:
            crossings = np.diff(neural[i,:] < thresholds[i], prepend=0)

        np.put(crossings, np.where(crossings==-1), 0) #only catch crossing
        spikes.append(crossings)

    return np.array(spikes).T #samples x spikes

def get_thresholds(neural, multiplier):
    neural = neural.T
    thresholds = []
    
    for channel in neural:
        std = np.std(channel)
        threshold = multiplier * std
        thresholds.append(threshold)

    return np.array(thresholds).T #samples x spikes

def get_autothresholds(neural, fs, multiplier = -2.1, window = 5):
    neural = neural.T
    thresholds = []
    multi = multiplier
    for channel in neural:
        threshold = np.zeros_like(channel)
        segment_number = np.int(channel.shape[0]/(fs*window))
        segment_length = np.int(channel.shape[0]/segment_number)
        for i in range(0,segment_number):
            if i < (segment_number-1):
                std_noise = np.median(abs(channel[i*segment_length:(i+1)*segment_length]))/0.6745
                thresh = multi*std_noise
                threshold[i*segment_length:(i+1)*segment_length] = thresh
            else:
                std_noise = np.median(abs(channel[i*segment_length:]))/0.6745
                thresh = multi*std_noise
                threshold[i*segment_length:] = thresh
        thresholds.append(threshold)

    return np.array(thresholds).T

def autothreshold_crossings(neural, thresholds):  # (negative peaks only)
    neural = neural.T
    thresholds =thresholds.T
    spikes_tmp = []
    for i in range(neural.shape[0]):
        crossings = np.diff(neural[i,:] < thresholds[i,:], prepend=0)
        np.put(crossings, np.where(crossings==-1), 0) #only catch crossing
        spikes_tmp.append(crossings)

    return np.array(spikes_tmp).T #samples x spikes



def rolling_threshold_crossings(neural, fs, multiplier, wind):  # (negative peaks only)
    neural = neural.T
    refrac = 0.0005
    spikes_tmp = []
    rolling_window = int(fs*wind)
    for channel in neural:
        rolling_std = np.squeeze(np.array(pd.DataFrame(channel).rolling(window=rolling_window).std()))
        for i in range(0,rolling_window-1):
            rolling_std[i] = rolling_std[rolling_window-1]
        threshold = multiplier*rolling_std
        crossings = np.diff(channel < threshold, prepend=0)
        np.put(crossings, np.where(crossings==-1), 0) #only catch crossing
        spikes_tmp.append(crossings)

    return np.array(spikes_tmp).T #samples x spikes

def refractory_limit(spikes_tmp, fs, refrac = 0.001):  #accounts for an absolute refractory period to prevent jagged peaks from counting as multiple events
    spikes = []
    spikes_tmp = spikes_tmp.T
    length = spikes_tmp.shape[1]
    for channel in spikes_tmp:
        spikeindex = np.where(channel == 1)[0]
        spikeindex = spikeindex.astype('float64')
        for i in range(spikeindex[:-1].shape[0]):
            j = 1
            nextiter = False
            if np.isnan(spikeindex[i]):
                nextiter = True
            while not nextiter:
                try:
                    if (spikeindex[i+j]-spikeindex[i])/fs < refrac:
                        spikeindex[i+j] = np.nan
                        j = j+1
                    else:
                        nextiter = True
                except:
                    nextiter = True
        clean_spikeindex = spikeindex[~np.isnan(spikeindex)]
        clean_spikeindex = clean_spikeindex.astype('int64')
        channel_spikes = np.zeros(length)
        for i in clean_spikeindex:
            channel_spikes[i] = 1
        spikes.append(channel_spikes)
    spikes = np.array(spikes, dtype = "int64").T
    return spikes

def spike_binner(spikes, fs, binsize):
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

    return butter_bandpass_filter(neural, 250, 3000, fs).T
    # return butter_bandpass_filter(neural, 350, 8000, fs).T

# def bandpass_neural_test(neural,fs):
#     neural = neural.T

#     # return butter_bandpass_filter(neural, 250, 3000, fs).T
#     return butter_bandpass_filter(neural, 10, 6000, fs).T

def filter_neural(neural, fs):
    neural = neural.T

    return notch_filter(bandpass_neural(neural, fs), fs).T
    # return bandpass_neural(neural, fs).T

    
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
