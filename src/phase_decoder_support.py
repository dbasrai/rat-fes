from src.folder_handler import *
from src.tdt_support import *
import math
import pickle
import scipy as spicy
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
from src.wiener_filter import *
from matplotlib.pyplot import cm
from scipy import signal
import scipy.stats as stats

def arctan_fn(predicted_sin, predicted_cos):   
    arctan = np.arctan2(predicted_sin,predicted_cos)
    arctan_angles = np.degrees(arctan) + 180
    return arctan_angles

def sine_and_cosine(phase_list):
    phase_list = np.radians(phase_list - 180)
    sin_array = np.sin(phase_list)
    cos_array = np.cos(phase_list)
    return sin_array, cos_array
    

def predicted_lines(actual, H):
    holding_array = []
    for ii in range(H.shape[1]):
        temp1 = test_wiener_filter(actual, H[:,ii])
        holding_array.append(temp1)
    holding_array = np.array(holding_array).T
    return holding_array
    
    
def to_phasex(peaks, angles):
    for i in range(0,peaks.shape[0]-1):
        for j in range(0, peaks[i+1]-peaks[i]):
            angles[peaks[i]+j] = j*360/(peaks[i+1]-peaks[i])
        angles[-1] = 0
    return angles

#do this before stitch and format is called to ensure no weird behaviors occur at the splice. 
def ss_cleaner(unrect_phase, significant_hesitation):
    phase = np.copy(unrect_phase)
    #first, check for the rare cases where a value of 0 or 1 is sandwiched between two values of 2, 
    #and rectify this to be entirley swing
    for i in range(1,phase.shape[0]-1):
        if (phase[i] == 0) or (phase[i] == 1):
            if (phase[i-1] == 2) and (phase[i+1] == 2):
                phase[i] = 2
    #scrub out any solitary 2's
    # for i in range(1,phase.shape[0]-1):
    #     if (phase[i] == 2) and (phase[i-1] != 2) and (phase[i+1] != 2):
    #         phase[i] = 1
    #next, bias 0 - 1 jittering in favor of 1
    for i in range(1,phase.shape[0]-1):
        if (phase[i] == 0):
            if (phase[i-1] == 1) and (phase[i+1] == 1):
                phase[i] = 1
    #second pass at 0 - 1 jitter
    for i in range(1,phase.shape[0]-1):
        if (phase[i] == 1):
            if (phase[i-1] == 0) and (phase[i+1] == 0):
                phase[i] = 0                 
    #this logic forces the indicies immediately before and after swing to be stance, 
    #which prevents swing phases from running together when 0 becomes null
    for i in range(1,phase.shape[0]-10):
        if (phase[i] == 2) and (phase[i-1] != 2):
            index_list = []
            index_list.append(i)
            tick = 0
            while phase[i+1+tick] == 2:
                index_list.append(i+1+tick)
                tick = tick+1
            if (phase[i-1] != 1):
                phase[i-1] = 1
            if (phase[i+1+tick] != 1):
                phase[i+1+tick] = 1  
    #reject all de-jittered runs of hesitance under some threshold defined in the function call
    for i in range(1,phase.shape[0]-10):
        if (phase[i] == 0) and (phase[i-1] != 0):
            index_list = []
            index_list.append(i)
            tick = 0
            while phase[i+1+tick] == 0:
                index_list.append(i+1+tick)
                tick = tick+1
            if len(index_list) < significant_hesitation:
                if (phase[i-1] == 1) or (phase[i+1+tick] == 1):
                    for j in index_list:
                        phase[j] = 1
                elif (phase[i-1] == 2) and (phase[i+1+tick] == 2):
                    for j in index_list:
                        phase[j] = 2
    return phase

def get_phase_angles(phase):
    angle_array = np.copy(phase)
    phase_inits = []
    swing_inits = []
    for i in range(angle_array.shape[0]):
        if (angle_array[i] == 2) and (angle_array[i-1] == 1):
            swing_inits.append(i)
    if angle_array[0] == 1:
        phase_inits.append(0)
    else:
        index_list = []
        index_list.append(0)
        tick = 0
        while (angle_array[tick+1] == 2):
            index_list.append(1+tick)
            tick = tick+1
        denom = len(index_list)
        step = 180/denom
        for index in (index_list):
            angle = 180 + step*index
            angle_array[index_list] = angle
    for i in range(angle_array.shape[0]):    
        if angle_array[i-1] !=1 and angle_array[i] ==1 and angle_array[i] ==1 :
            phase_inits.append(i)
    for init in phase_inits:
        angle_array[init] = None
    for i in range(angle_array.shape[0]):
        if (np.isnan(angle_array[i]) == True):
            index_list = []
            index_list.append(i)
            tick = 0
            while (np.isnan(angle_array[i+1+tick]) != True):
                index_list.append(i+1+tick)
                tick = tick+1
                if (i+1+tick >= angle_array.shape[0]):
                    break
            if angle_array[index_list[-1]] == 2:
                denom = len(index_list)
                step = 360/denom
                for i in range(len(index_list)):
                    angle = step*i
                    angle_array[index_list[i]] = angle
            elif angle_array[index_list[-1]] == 1:
                denom = len(index_list)
                step = 180/denom
                for i in range(len(index_list)):
                    angle = step*i
                    angle_array[index_list[i]] = angle     
    counter = 0
    for init in swing_inits:
        counter = counter + angle_array[init]
    mean_swing = counter/len(swing_inits)
    return angle_array, mean_swing

def phase_sychrony(crossings, crossings2):
    new_index_list = []
    assoc_delay = []
    tally_spacing = []
    for i in range(crossings.shape[0]):
        if crossings[i] == 1:
            tick = 0
            neg_index_list = []
            neg_index_list.append(i)
            if i+1 < crossings.shape[0]:
                while (crossings[i+1+tick] == 0):
                    neg_index_list.append(i+1+tick)
                    tick = tick+1
                    if (i+1+tick >= crossings.shape[0]):
                        break
            tally_spacing.append(len(neg_index_list))
    mean_spacing = np.mean(np.array(tally_spacing))
    for i in range(crossings2.shape[0]):
        if crossings2[i] == 1:
            if crossings[i] == 1:
                new_index_list.append(i)
                assoc_delay.append(0)
            else:
                tick = 0
                neg_index_list = []
                neg_index_list.append(i)
                if i-1 >= 0:
                    while (crossings[i-1-tick] == 0):
                        neg_index_list.append(i-1-tick)
                        tick = tick+1
                        if (i-1-tick <= 0):
                            break
                tock = 0
                pos_index_list = []
                pos_index_list.append(i)
                if i+1 < crossings.shape[0]:
                    while (crossings[i+1+tock] == 0):
                        pos_index_list.append(i+1+tock)
                        tock = tock+1
                        if (i+1+tock >= crossings.shape[0]):
                            break
                if 0 in neg_index_list:
                    new_index_list.append(pos_index_list[-1]+1)
                    assoc_delay.append(len(pos_index_list))
                elif crossings.shape[0] in pos_index_list:
                    new_index_list.append(neg_index_list[-1]-1)
                    assoc_delay.append(-len(neg_index_list))
                else:
                    if tick >= tock:
                        new_index_list.append(pos_index_list[-1]+1)
                        assoc_delay.append(len(pos_index_list))
                    elif tick < tock:
                        new_index_list.append(neg_index_list[-1]-1)
                        assoc_delay.append(-len(neg_index_list))
    true_index = np.where(crossings ==1)[0]                    
    new_index = np.array(new_index_list)
    delays = np.array(assoc_delay)
    trigger_list = []
    delay_array_array = []
    for i in range(true_index.shape[0]):
        trig = np.where(new_index == true_index[i])[0]
        trigger_count = trig.shape[0]
        trigger_list.append(trigger_count)
        delay_array = []

        for tri in trig:
            delay_array.append(delays[tri])
        delay_array_array.append(np.array(delay_array))
    triggers = np.array(trigger_list)
    return true_index, new_index, delay_array_array, mean_spacing

def phase_diagnositc(predicted_arctans_nl, test_arctans, swing_mean, true_index, new_index, delay_array_array, mean_spacing, bounds, plotting = False):
    if plotting == True:
        fig444, ax0= plt.subplots(1, 1, figsize=(9,6), sharex = True)
        tstest = np.linspace(0, (test_arctans.shape[0]*50)/1000,test_arctans.shape[0]) 
        ax0.plot(tstest, test_arctans, alpha = 0.5, c = 'k', label = 'actual')
        ax0.plot(tstest, predicted_arctans_nl, alpha = 0.5, c = 'r' , label = 'predicted')
        ax0.axhline(y=swing_mean, color='b', linestyle='--')
        ax0.set_ylabel('Phase (Degrees)')
        ax0.set_xlabel('Time (seconds)')
        ax0.set_title('phase predictions')
        ax0.legend()

        fig555, ax1= plt.subplots(1, 1, figsize=(9,6), sharex = True)
        for i in range(len(delay_array_array)):
            stims = delay_array_array[i].shape[0]
            if stims == 0:
                ax1.scatter(x = i, y = 0, c = 'k', alpha = 1)
                continue
            else:
                for ii in range(delay_array_array[i].shape[0]):
                    if delay_array_array[i][ii] > 0:
                        ax1.bar(x = i, height = delay_array_array[i][ii], color = 'y', alpha = 0.5)
                    if delay_array_array[i][ii] < 0:
                        ax1.bar(x = i, height = delay_array_array[i][ii], color = 'r', alpha = 0.5)
                    if (delay_array_array[i][ii] == 0):
                        ax1.scatter(x = i, y = delay_array_array[i][ii], c = 'g', alpha = 1)
        ax1.set_ylabel('Trigger displacment from real threshold crossing (50 ms)')
        ax1.set_xlabel('Test Phase Number')
        ax1.set_title('correctly triggered stimulation events')
        ax1.axhline(y = mean_spacing, c='y', linestyle = '--', label = 'average time since last phase')
        ax1.axhline(y = -mean_spacing, c='r', linestyle = '--', label = 'average time until next phase')
        ax1.axhline(y = bounds[0], c='g', linestyle = '--', label = 'acceptable range')
        ax1.legend()
        ax1.axhline(y = bounds[-1], c='g', linestyle = '--')
    
    tallywhacker = 0
    for i in range(len(delay_array_array)):
        if (delay_array_array[i].shape[0] == 1):
            if (delay_array_array[i][0] >= bounds[0]) and (delay_array_array[i][0] <= bounds[-1]):
                tallywhacker = tallywhacker +1
    score = tallywhacker/len(delay_array_array)

    return score

def stim_cooldown(crossingsB, refractory_tics):
    frac = np.copy(crossingsB)
    for i in range(frac.shape[0]):
        if frac[i] == 1:
            for j in range(refractory_tics):
                if i+1+j >= frac.shape[0]:
                    continue
                if frac[i+1+j] != 0:
                    frac[i+1+j] = 0
    return frac
    
    
    

# some commented functions serve as multi-dinemsional alternatives to thoe currently i use (should one ever wish to use fore and hind limb phase)


def impulse_response(AOI, H_mat, phase_list, plotting = False):
    column_response = []
    for i in range(0,32):
            product_list = []
            for j in range(0,10):
                dummyarray = np.zeros((10,32))
                dummyarray[-1,i] = 1
                dummyarray = np.roll(dummyarray,-j,axis = 0)
                dummyarray = dummyarray.flatten()
                dummyarray = np.insert(dummyarray,0,1)
                dummyarray = dummyarray.reshape(321,1)
                dummyarray = dummyarray.T
                product = np.dot(dummyarray, H_mat)
                product_list.append(product[0])
            product_list = np.array(product_list)
            column_response.append(product_list)
    av_gait_bins = average_gait_bins(phase_list, AOI)
    padded_column_response = []
    if (av_gait_bins - 10) >= 1:
        n = av_gait_bins-10
        for i in range(len(column_response)):
            padded_column = np.append(column_response[i][:,AOI], np.full((n,1), np.mean(column_response[i][:,AOI])))
            padded_column_response.append(padded_column)
        padded_column_response = np.array(padded_column_response)
    if plotting == True:
        fig1, ax1 = plt.subplots()
        x = np.arange(0,10,1)        
        for i in range(len(column_response)):
            ax1.plot(x, column_response[i][:,AOI])
        ax1.set_xticks(x)
    return padded_column_response

def rates_by_gait(full_phase_list, rates_array, AOI, count_thresh, plotting = False):
    phase_list = full_phase_list[:,AOI]
    unique_val, unique_first_index, unique_count = np.unique(phase_list, return_index=True, return_counts=True)
    unique_range = len(unique_count)                  
    rates_by_phase = []
    phase_tracker = []
    for i in range(1, unique_range):
        if unique_count[i] >= count_thresh:
            phase_tracker = np.append(phase_tracker, unique_val[i])
            hodl = np.where(phase_list==unique_val[i])
            hold = hodl[0]
            hold_rates = []
            for i in range(len(hold)):
                rates_row = rates_array[hold[i],-32:]
                hold_rates.append(rates_row)
            hold_rates = np.mean(hold_rates, axis = 0)
            rates_by_phase.append(hold_rates)
    rates_by_phase = np.array(rates_by_phase)
    if plotting == True:
        fig1, ax1 = plt.subplots()
        x = phase_tracker       
        for i in range(rates_by_phase.shape[1]):
            ax1.plot(x, rates_by_phase[:,i], alpha = 0.5)
    return rates_by_phase, phase_tracker


# def arctan_fn(predicted_sin, predicted_cos):   
#     arctans = []
#     for i in range(predicted_sin.shape[1]):
#         arctan = np.arctan2(predicted_sin[:,i],predicted_cos[:,i])
#         arctan_angles = np.degrees(arctan) + 180
#         arctans.append(arctan_angles)
#     arctans = np.array(arctans).T
#     return arctans



# def sine_and_cosine(phase_list):
#     phase_list = np.radians(phase_list - 180)
#     sin_array = []
#     cos_array = []
#     for i in range(phase_list.shape[1]):
#         sin = np.sin(phase_list[:,i])
#         cos = np.cos(phase_list[:,i])
#         sin_array.append(sin)
#         cos_array.append(cos)
#     sin_array = np.array(sin_array).T
#     cos_array = np.array(cos_array).T
#     return sin_array, cos_array




# def tailored_peaks(angles, index, name):
#     peak_dict = {
#             'foot' : {
#                 'signal': -(angles[:,index]),
#                 'window': 60,
#                 'prominence': 10,
#                 'distance': 6,
#                 'height' : -np.mean(angles[:,index])
#             },
#             'knee' : {
#                 'signal': angles[:,index],
#                 'window': 60,
#                 'prominence': 10,
#                 'distance': None,
#                 'height' : np.mean(angles[:,index])
#             },
#             'hip' : {
#                 'signal': angles[:,index],
#                 'window': 60,
#                 'prominence': 3,
#                 'distance': 5,
#                 'height' : None
#             },
#             'limbfoot' : {
#                 'signal': angles[:,index],
#                 'window': 60,
#                 'prominence': 10,
#                 'distance': None,
#                 'height' : np.mean(angles[:,index])
#             },
#             'hand' : {
#                 'signal': -(angles[:,index]),
#                 'window': 60,
#                 'prominence': 10,
#                 'distance': 6,
#                 'height' : -np.mean(angles[:,index])
#                 },
#             'elbow' : {
#                 'signal': angles[:,index],
#                 'window': 60,
#                 'prominence': 6.5,
#                 'distance': 5,
#                 'height' : None
#             },
#             'shoulder' : {
#                 'signal': -(angles[:,index]),
#                 'window': 60,
#                 'prominence': 5,
#                 'distance': None,
#                 'height' : -1.1*np.mean(angles[:,index])
#             },
#             'forelimb' : {
#                 'signal': angles[:,index],
#                 'window': 60,
#                 'prominence': 9,
#                 'distance': 5,
#                 'height' : None
#             },
#             'ankle' : {
#                     'signal': -(angles[:,index]),
#                     'window': 60,
#                     'prominence': 10,
#                     'distance': 6,
#                     'height' : -np.mean(angles[:,index])
#             }
#         }
#     peaks, _ = spicy.signal.find_peaks(peak_dict[name]['signal'], wlen = peak_dict[name]['window'], prominence=peak_dict[name]['prominence'], distance =peak_dict[name]['distance'], height =peak_dict[name]['height'])    
#     peaks = np.concatenate([[0],peaks,[np.shape(angles[:,index])[0]-1]])
#     return peaks

# def align_to_hind(phases, target):
#     temp_shift = []
#     ts = np.linspace(0, (phases.shape[0]*50)/1000,phases.shape[0])
#     dx = np.mean(np.diff(ts))
#     for i in range(phases.shape[1]):
#         if i != target:
#             shift = (np.argmax(signal.correlate(phases[:,i], phases[:,target])) - phases[:,target].shape[0]) * dx
#             shift = int(-shift*1000/50)
#             temp_shift = np.append(temp_shift, shift)
#         else:
#             temp_shift = np.append(temp_shift, 0)
#     return temp_shift

# def best_hindlimb_match(phase_list, roll, AOI):
#     rank_list = []
#     for i in range(4,7):
#         mate = np.roll(phase_list[:,i],int(roll[i]))
#         target_phase = phase_list[:,AOI]
#         r, p = stats.pearsonr(mate, target_phase)
#         # phase_synchrony = 1-np.sin(np.abs(target_phase-mate)/2)
#         # rank = np.mean(phase_synchrony)
#         # rank_list.append(rank)
#         rank_list.append(r)
#     best_index = max(range(len(rank_list)), key=rank_list.__getitem__)+4
#     return best_index



# def average_gait_bins(phase_list, AOI):
#     gait_lengths = []
#     gait_indicies = np.where(phase_list[:,AOI]==0)[0]
#     for i in range(1, gait_indicies.shape[0]):
#         gait_length = gait_indicies[i]-gait_indicies[i-1]
#         gait_lengths = np.append(gait_lengths, gait_length)
#     av_gait_length = math.ceil(np.average(gait_lengths))
#     return av_gait_length
