import tdt
import time
import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from scipy.signal import resample, find_peaks
from src.filters import *
from src.tdt_support import *
from src.neural_analysis import *
from src.wiener_filter import *

def process_neural_kinangles(tdt, kin, np_ts, threshold_multiplier,
        crop=(0,0), binsize=0.05):
    tdt_data = extract_tdt(tdt)
    bp_list, kinematics = extract_anipose_angles(kin)
    
    tdt_data, kinematics = crop_data(tdt_data, kinematics, np_ts, crop)

    fs = tdt_data['fs']
    tdt_data['neural'] = filter_neural(tdt_data['neural'], fs) #bandpass
    tdt_data['neural'] = remove_artifacts(tdt_data['neural'], fs)
    
    spikes = autothreshold_crossings(tdt_data['neural'], threshold_multiplier)
    firing_rates = spike_binner(spikes, fs, binsize)

    resampled_angles = resample(kinematics, firing_rates.shape[1], axis=1)

    return firing_rates, resampled_angles

def linear_decoder(firing_rates, kinematics, n=10, l2=0):
    rates_format, angles_format = format_data(firing_rates.T, kinematics.T, n)
    h = train_wiener_filter(rates_format, angles_format, l2)

    return h

def stitch_data(rates1, rates2, kin1, kin2):
    rates = np.hstack((rates1, rates2))
    kin = np.hstack((kin1, kin2))
    return rates, kin

def extract_peaks(angles, thres_height=115):
    peaks, nada = find_peaks(angles, height=thres_height)
    return peaks

def find_bad_gaits(peaks):
    average_gait_samples = np.average(np.diff(peaks))
    above = 1.25 * average_gait_samples
    below = .8 * average_gait_samples

    bad_above = np.argwhere(np.diff(peaks) > above)
    bad_below = np.argwhere(np.diff(peaks) < below)

    bads = np.squeeze(np.concatenate((bad_above, bad_below)))

    return bads.tolist()

def remove_bad_gaits(rates, angles):
    peaks = extract_peaks(angles[1,:], 80)
    bads = find_bad_gaits(peaks)

    rates_list = []
    angles_list = []

    for i in range(np.size(peaks)-1):
        if i in bads:
            continue
        first = peaks[i]
        last=peaks[i+1]
        gait_rates = rates[:, first:last]
        gait_angles = angles[:, first:last]

        rates_list.append(gait_rates)
        angles_list.append(gait_angles)

    rebuilt_rates = np.hstack(rates_list)
    rebuilt_angles = np.hstack(angles_list)

    return rebuilt_rates, rebuilt_angles 

def extract_gait(rates, angles, thres, bool_resample=False):
    peaks = extract_peaks(angles,thres)
    rates_list = []
    angles_list = []

    avg_samples = int(np.round(np.average(np.diff(peaks))))

    for i in range(np.size(peaks)-1):
        first = peaks[i]
        last = peaks[i+1]
        gait_rates = rates[:, first:last]
        gait_angles = angles[first:last]
        if bool_resample:
            gait_rates = resample(gait_rates, avg_samples, axis=1)
            gait_angles = resample(gait_angles, avg_samples)
        rates_list.append(gait_rates)
        angles_list.append(gait_angles)

    return rates_list, angles_list, peaks

def vid_from_gait(crop, angles_list, gait_number, video, peaks, filename, binsize=0.05,
        framerate=1):
    bin_number = np.size(np.hstack(angles_list[0:gait_number]))

    frame_iter = int(binsize*200)
    start_frame = crop[0]*200 + ((peaks[0]+bin_number) * frame_iter)
    end_frame = start_frame + len(angles_list[gait_number]) * frame_iter

    directory = '/home/diya/Documents/rat-fes/results/movies/{}-{}/'.format(filename, gait_number)
    os.mkdir(directory)
    
    angles = angles_list[gait_number]

    fig0 = plt.figure(figsize = (720/96, 440/96), dpi=96)
    ax0 = fig0.add_subplot(111)
    ax0.set_xlim(0, np.size(angles)-1)
    ax0.set_ylim(np.min(angles), np.max(angles))
    
    img_list = []

    for i in range(np.size(angles)):
        y = angles[0:i+1]
        x = np.arange(0,i+1)
        degree_num = int(angles[i])
        ax0.plot(x, y, c='blue')
        plt.savefig(directory + 'degree{}_{}'.format(i, degree_num), )
        time.sleep(.1)
        img_list.append(cv2.imread(directory +
            'degree{}_{}.png'.format(i, degree_num)))

    height = img_list[0].shape[0]
    width = img_list[0].shape[1]
    out = cv2.VideoWriter(directory + 'kin.mp4',
            cv2.VideoWriter_fourcc(*'DIVX'), framerate,(width, height))

    for img in img_list:
        out.write(np.array(img))

    out.release()
    

    cap = cv2.VideoCapture(video)
    img_list = []
    
    j=0
    for k in range(start_frame, end_frame, frame_iter):
        cap.set(1, k)
        ret, frame = cap.read()
        cv2.imwrite(directory + 'live_frame{}_{}.png'.format(k, int(angles[j])), frame)
        img_list.append(frame)
        j=j+1
    
    height = img_list[0].shape[0]
    width = img_list[0].shape[1]

    out = cv2.VideoWriter(directory + 'live_video.mp4',
            cv2.VideoWriter_fourcc(*'DIVX'), framerate,(width, height))
    for img in img_list:
        out.write(np.array(img))

    out.release()



