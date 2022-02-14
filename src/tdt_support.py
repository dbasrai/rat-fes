import tdt
import numpy as np
import pandas as pd
from src.xray import * 

def extract_tdt(tdt_file):
    tdt_dict = {}

    data = tdt.read_block(tdt_file)
    tdt_dict['neural'] = data.streams.Wav1.data*1000000 #in microvolts
    tdt_dict['fs'] = data.streams.Wav1.fs
    tdt_dict['ts'] = np.arange(0, tdt_dict['neural'].shape[1] / tdt_dict['fs'], 
            1/tdt_dict['fs'])
    tdt_dict['pulse_time'] = data.epocs.U11_.onset[0]

    return tdt_dict

def get_sync_sample(np_ts, tdt_data):
    cam_ts = np.load(np_ts)
    delay = cam_ts[1]-cam_ts[0]
    start_time = tdt_data['pulse_time'] + delay
    sample_number = start_time * tdt_data['fs']
    return round(sample_number)

def crop_data(anipose_csv, tdt_file, np_ts, start_time, end_time):
    #add in start_time/end_time in video, output cropped cortical/kin data
    tdt_data = extract_tdt(tdt_file)

    kin_start = start_time*200
    kin_end = end_time*200

    init_start_sample = get_sync_sample(np_ts, tdt_data)

    start_sample = round(init_start_sample + (start_time * tdt_data['fs']))
    end_sample = round(init_start_sample + (end_time * tdt_data['fs']))

    print(start_sample, end_sample)

    temp_neural = tdt_data['neural'] #going to slice variable
    tdt_data['neural'] = temp_neural[:,start_sample:end_sample]

    bp_list, kinematics = extract_anipose_3d(anipose_csv)
    kinematics = kinematics[:, kin_start:kin_end,:]

    return tdt_data, bp_list, kinematics
