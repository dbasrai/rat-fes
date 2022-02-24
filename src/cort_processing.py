import tdt
import math
import numpy as np
import pandas as pd
from scipy.signal import resample
from src.filters import *
from src.tdt_support import *
from src.neural_analysis import *
from src.wiener_filter import *

def process_neural_kinangles(tdt, kin, np_ts, threshold_multiplier, crop=(0,0)):
    tdt_data = extract_tdt(tdt)
    bp_list, kinematics = extract_anipose_angles(kin)
    
    tdt_data, kinematics = crop_data(tdt_data, kinematics, np_ts, crop)

    fs = tdt_data['fs']
    tdt_data['neural'] = filter_neural(tdt_data['neural'], fs) #bandpass
    tdt_data['neural'] = remove_artifacts(tdt_data['neural'], fs)
    
    spikes = autothreshold_crossings(tdt_data['neural'], threshold_multiplier)
    firing_rates = spike_binner(spikes, fs)

    resampled_angles = resample(kinematics, firing_rates.shape[1], axis=1)

    return firing_rates, resampled_angles

def linear_decoder(firing_rates, kinematics, n=10):
    rates_format, angles_format = format_data(firing_rates.T, kinematics.T, n)
    h = train_wiener_filter(rates_format, angles_format)

    return h

