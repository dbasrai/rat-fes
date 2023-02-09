import tdt
import numpy as np
import pandas as pd
import copy

def deprec_extract_tdt(tdt_file):
    tdt_dict = {}

    data = tdt.read_block(tdt_file)
    tdt_dict['neural'] = data.streams.Wav1.data*1000000 #in microvolts
    tdt_dict['fs'] = data.streams.Wav1.fs
    tdt_dict['ts'] = np.arange(0, tdt_dict['neural'].shape[1] / tdt_dict['fs'], 
            1/tdt_dict['fs'])
    tdt_dict['pulse_time'] = data.epocs.U11_.onset[0]
    
    return tdt_dict

def extract_tdt(tdt_file, npts_file):
    tdt_dict = {}

    data = tdt.read_block(tdt_file)
    tdt_dict['neural'] = (data.streams.Wav1.data*1000000).T #in microvolts
    tdt_dict['fs'] = data.streams.Wav1.fs
    tdt_dict['ts'] = np.arange(0, tdt_dict['neural'].shape[0] / tdt_dict['fs'], 
            1/tdt_dict['fs'])
    # Use camera trigger pulse onset time from Port-C.1 on RZ5 Gizmo in Synapse, 
    tdt_dict['pulse_time'] = data.epocs.PtC1.onset[0] 
    
    tdt_dict['cam_timestamps'] = np.load(npts_file)

    return tdt_dict



def extract_kin_data(coords_csv, angles_csv):
    kin_data = {}
    bp_list, coords = extract_anipose_3d(coords_csv)
    angles_list, angles = extract_anipose_angles(angles_csv)
    fnum = extract_fnum(angles_csv)
    kin_data['bodyparts'] = bp_list
    kin_data['coords'] = coords
    kin_data['angles_list'] = angles_list
    kin_data['angles'] = angles
    kin_data['fnum'] = fnum

    return kin_data

def get_sync_sample(tdt_data):
    cam_ts = tdt_data['cam_timestamps']
    # There is a delay after cameras receive pulse, hence why timestamps are normalized to 2nd element
    # there is also a 3 sample delay before trigger is actually sent out from Port-C.1 and Port-C.3 to cameras (read IODelays.pdf)
    delay = cam_ts[1]-cam_ts[0] + 3*(1/24414.0625)
    
    # If the RZ5 and PZA have the same sampling rate (and data is extracted from stream gizmo) there is a 24 sample delay. 
    # ie. if a spike shows up at t = 10 in data, it's actually a spike that happened 24 samples ago, so the real start time of the data
    # is much later in the data
    RZ_to_PZ_delay = 24*(1/24414.0625)
    start_time = tdt_data['pulse_time'] + delay + RZ_to_PZ_delay
    sample_number = start_time * tdt_data['fs']
    return round(sample_number)

def deprec_get_sync_sample(np_ts, tdt_data):
    cam_ts = np.load(np_ts)
    delay = cam_ts[1]-cam_ts[0]
    start_time = tdt_data['pulse_time'] + delay
    sample_number = start_time * tdt_data['fs']
    return round(sample_number)

def extract_anipose_3d(csv):
    df = pd.read_csv(csv)
    long_bp_list = df.columns
    bp_list = []
    for bodypart in long_bp_list:
        temp = bodypart.split('_')[0]
        stop_here = 'M'
        if temp is stop_here:
            break
        if temp not in bp_list:
            bp_list.append(temp)
    
    ret_arr = np.empty((df.index.size, len(bp_list), 3))
    
    for idx, bodypart in enumerate(bp_list):
        ret_arr[:, idx, 0] = df[bodypart + '_x']
        ret_arr[:, idx, 1] = df[bodypart + '_y']
        ret_arr[:, idx, 2] = df[bodypart + '_z']
    #if filtering==True:
    #    clear_list = df.loc[df['ScaRot_ncams'].isnull()].index.tolist()
    #    for element in clear_list:
    #        ret_arr[:, element, :] = np.nan
    
    return bp_list, ret_arr
            
def extract_anipose_angles(csv):
    df = pd.read_csv(csv)
    df = df.iloc[:,0:df.columns.get_loc('fnum')]
    bp_list = df.columns.to_list()
    angles_list = []
    for column in df:
        angles_list.append(df[column].to_numpy())

    return bp_list, np.array(angles_list).T

def extract_fnum(csv):
    df = pd.read_csv(csv)
    df = df.iloc[:,df.columns.get_loc('fnum')]
    frame_list = np.array(df).T

    return frame_list

def deprec_crop_data(tdt_data, kinematics, np_ts, crop=(0,70)):
    #add in start_time/end_time in video, output cropped cortical/kin data
    start_time = crop[0]
    end_time = crop[1]

    kin_start = start_time*200
    kin_end = end_time*200

    init_start_sample = get_sync_sample(np_ts, tdt_data)

    start_sample = round(init_start_sample + (start_time * tdt_data['fs']))
    end_sample = round(init_start_sample + (end_time * tdt_data['fs']))


    temp_neural = tdt_data['neural'] #going to slice variable
    temp_ts = tdt_data['ts']
    tdt_data['neural'] = temp_neural[:,start_sample:end_sample]
    tdt_data['ts'] = temp_ts[start_sample:end_sample]

    if kinematics.ndim==3:
        kinematics = kinematics[:, kin_start:kin_end,:]
    elif kinematics.ndim==2:
        kinematics = kinematics[:, kin_start:kin_end]
    elif kinematics.ndim==1:
        kinematics = kinematics[kin_start:kin_end]

    return tdt_data, kinematics

def crop_data_tdt(tdt_file, np_ts, start_time, end_time):

    tdt_data = extract_tdt(tdt_file)

    init_start_sample = get_sync_sample(np_ts, tdt_data)

    start_sample = round(init_start_sample + (start_time * tdt_data['fs']))
    end_sample = round(init_start_sample + (end_time * tdt_data['fs']))

    temp_neural = tdt_data['neural'] #going to slice variable
    tdt_data['neural'] = temp_neural[:,start_sample:end_sample]


    return tdt_data



