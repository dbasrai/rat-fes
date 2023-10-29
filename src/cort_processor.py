import traceback
import yaml
import gc

from src.neural_analysis import *
from src.wiener_filter import *
from src.filters import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *
from src.plotter import *
from src.phase_decoder_support import *
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from scipy.signal import resample, find_peaks
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import numpy as np
import pandas as pd
import cv2
import copy

class CortProcessor:
    '''
    class that can handle neural + kinematic/EMG data simultaneously
    upon initialization, extracts data from TDT and anipose file
    see cort_processor.md in 'docs' for more information
    '''    
    def __init__(self, folder_path):
        if os.path.isdir(folder_path):
            #see docs/data_folder_layout.md for how to structure folder_path
            self.handler = FolderHandler(folder_path)
            self.tdt_data, self.kin_data = self.extract_data()
            
            self.crop_list = None
            crop = self.parse_config()
            if isinstance(crop, list):
                self.crop_list = crop
            
            self.data={}
            self.data['rates'] = 'run_process_first'
            self.data['coords'] = 'run_process_first'
            self.data['angles'] = 'run_process_first'
            self.data['toe_height'] = 'run process first, then run process\
                    toehight'

            self.gait_indices = 'run get_gait_indices first'
            self.avg_gait_samples = 'run get_gait_indices first'




    def parse_config(self):
        '''
        this loads config.yaml file
        '''
        try:
            path = self.handler.folder_path
            with open(f"{path}/config.yaml", "r") as stream:
                yaml_stuff = yaml.safe_load(stream)
                crop = 'crop'

                if crop in yaml_stuff:
                    return yaml_stuff[crop]
                else:
                    print('no crop')
                    return 0
        except yaml.YAMLError as exc:
            print(exc)

    def extract_data(self):
        '''
        this is called when a CortProcessor is initialized.
        It extracts raw neural,raw angles, and raw coords
        neural data is saved under CortProcessr.tdt_data
        kinematic data is saved under CortProcessor.kin_data

        If you use process cort script, all these attriibutes are overwritten
        since raw data isnt saved after processing.
        '''
        tdt_data_list = []
        raw_ts_list = self.handler.ts_list
        raw_tdt_list = self.handler.tdt_list
        for i in range(len(raw_tdt_list)):
            tdt_data_list.append(extract_tdt(raw_tdt_list[i], raw_ts_list[i]))
        
        kin_data_list = []

        raw_coords_list = self.handler.coords_list
        raw_angles_list = self.handler.angles_list
        raw_phase_list = self.handler.phase_list
        for i in range(len(raw_coords_list)):
            kin_data_list.append(extract_kin_data(raw_coords_list[i],
                raw_angles_list[i], raw_phase_list[i]))

        return tdt_data_list, kin_data_list


    def process(self, manual_crop_list=None, binsize = 0.05, multi = -2.1, override = True):

        if self.crop_list is not None:
            crop_list = self.crop_list
        elif manual_crop_list is not None:
            crop_list = manual_crop_list
        else:
            print('need to make it so that it is uncropped')

        tdt_data_list = self.tdt_data
        kin_data_list = self.kin_data

        if override == True:
            self.binsize = binsize
            self.data = {}
            self.data['bodyparts'] = kin_data_list[0]['bodyparts']
            self.data['angle_names'] = kin_data_list[0]['angles_list']
            self.data['rates']=[]
            self.data['coords']=[]
            self.data['angles']=[]
            self.data['phase'] = []
            self.data['fnum'] = []
        else:
            data = {}
            data['bodyparts'] = kin_data_list[0]['bodyparts']
            data['angle_names'] = kin_data_list[0]['angles_list']
            data['rates']=[]
            data['coords']=[]
            data['angles']=[]
            data['phase'] = []
            data['fnum'] = []
            data['crossings']= []
            
        for i in range(len(tdt_data_list)):
            self.tdt_data[i], self.kin_data[i] = self.crop_data(tdt_data_list[i], kin_data_list[i], crop_list[i])
            
            fs = self.tdt_data[i]['fs']
            neural = self.tdt_data[i]['neural']
        
            half_filtered_neural = fresh_filt(neural, 300, 8000, fs, order = 4)
            filtered_neural = notch_filter(half_filtered_neural, fs)
            
            thresholds = get_autothresholds(filtered_neural, fs, multi)
            clean_filtered_neural = remove_artifacts(filtered_neural, fs)
            spikes= autothreshold_crossings(clean_filtered_neural, thresholds)
            firing_rates = spike_binner(spikes, fs, binsize)
            
            if override == True:                    
                self.data['rates'].append(firing_rates)
            else:
                data['rates'].append(firing_rates)

            temp_angles = self.kin_data[i]['angles']
            temp_coords = self.kin_data[i]['coords'] 
            temp_phase = self.kin_data[i]['phase'] 
            temp_frames = self.kin_data[i]['fnum'] 

            resampled_angles = resample(temp_angles, firing_rates.shape[0], axis=0)
            resampled_coords = resample(temp_coords, firing_rates.shape[0], axis=0)
            resampled_phase = self.resample_mode(temp_phase, firing_rates.shape[0])
            resampled_frames = self.resample_mode(temp_frames, firing_rates.shape[0])

            if override == True:
                self.data['angles'].append(resampled_angles)
                self.data['coords'].append(resampled_coords)
                self.data['phase'].append(resampled_phase)
                self.data['fnum'].append(resampled_frames)
                self.tdt_data[i] = 0
                self.kin_data[i] = 0 
            else:
                data['angles'].append(resampled_angles)
                data['coords'].append(resampled_coords)
                data['phase'].append(resampled_phase)
                data['fnum'].append(resampled_frames)
                data['crossings'].append(spikes)

            gc.collect()
            
        if override == True:
            return np.vstack(self.data['rates']), np.vstack(self.data['angles']), np.hstack(self.data['phase'])
        else:
            return np.vstack(data['rates']), np.vstack(data['angles']), np.hstack(data['phase'])#, np.vstack(data['crossings'])

    
    def resample_mode(self, phase, shape):
        re = resample(phase, shape, axis=0)
        for i in range(re.shape[0]):
            re[i] = int(np.round(re[i]))
        return re

    def process_toe_height(self):
        """
        extracts toe height from data['coords'], then scales such that lowest
        toe height is 0.

        toe_num is which bodypart in data['coords'] is the toe. default is
        bodypart 0
        """
        try:
            toe_num = self.bodypart_helper('toe')
            temp_list = []
            self.data['toe_height'] = []
            minimum_toe = 1000 #arbitrary high number to be beat
            for i in range(len(self.data['coords'])):
                #find y-value of toe
                toe_height = self.data['coords'][i][:, toe_num, 1]
                if np.min(toe_height) < minimum_toe:
                    minimum_toe = np.min(toe_height)
                temp_list.append(toe_height)
            for i in range(len(temp_list)):
                norm_toe_height = temp_list[i] - minimum_toe
                self.data['toe_height'].append(norm_toe_height)

            return np.hstack(self.data['toe_height'])
        except Exception as e: 
            print('failed!! did you run process first')
            print(e)
            
    def marker_position(self, bodypart, dimension):
        """
        updated version of process_toe_height that 
        a) takes any bodypart
        b) accounts for reformatting splices 
        """
        mark_num = self.bodypart_helper(bodypart)
        mark_stack = []
        for i in range(len(self.data['coords'])):
            mark_stack.append(self.data['coords'][i][9:, mark_num, dimension])
        mark_y = np.hstack(mark_stack)
        mark_min = np.min(mark_y)
        mark_y_norm = mark_y - mark_min
        return mark_y_norm.T

        
    def crop_data(self, tdt_data, kin_data, crop):
        '''
        helper function that both syncs and crops neural/kinematic data
        see docs/cam_something.odg for how it syncs em.
        '''

        crop_tdt_datafile=tdt_data
        crop_kin_datafile=kin_data

        start_time = crop[0]
        end_time = crop[1]

        kin_start = start_time*200
        kin_end = end_time * 200

        init_start_sample = get_sync_sample(tdt_data)
        
        start_sample = round(init_start_sample + (start_time * tdt_data['fs']))
        end_sample = round(init_start_sample + (end_time * tdt_data['fs']))

        temp_neural = tdt_data['neural'] #going to slice variable
        temp_ts = tdt_data['ts']

        crop_tdt_datafile['neural'] = temp_neural[start_sample:end_sample,:]
        crop_tdt_datafile['ts'] = temp_ts[start_sample:end_sample]
        
        del(temp_neural)
        del(temp_ts)

        gc.collect()

        temp_coords = kin_data['coords']
        temp_angles = kin_data['angles']
        temp_phase = kin_data['phase']
        temp_frames = kin_data['fnum']

        crop_kin_datafile['coords'] = temp_coords[kin_start:kin_end,:,:]
        crop_kin_datafile['angles'] = temp_angles[kin_start:kin_end,:]
        crop_kin_datafile['phase'] = temp_phase[kin_start:kin_end]        
        crop_kin_datafile['fnum'] = temp_frames[kin_start:kin_end]
        
        #maybe need edge-case if only single angle/bodypart
        
        return crop_tdt_datafile, crop_kin_datafile




    def stitch_and_format(self, firing_rates_list=None,
            resampled_angles_list=None, N = 10):
        """
        takes list of rates, list of angles, then converts them into lags of 10
        using format rate in wiener_filter.py, and then stitches them into one
        big array

        both lists must have same # of elements, and each array inside list
        must have the same size as the corresponding array in the other list.
        """
        if firing_rates_list == None and resampled_angles_list == None:
            firing_rates_list = self.data['rates']
            resampled_angles_list = self.data['angles']
        
        
        assert isinstance(firing_rates_list, list), 'rates must be list'
        assert isinstance(resampled_angles_list, list), 'angles must be list'
        formatted_rates = []
        formatted_angles = []

        for i in range(len(firing_rates_list)):
            f_rate, f_angle = format_data(firing_rates_list[i],
                    resampled_angles_list[i], N)
            formatted_rates.append(f_rate)
            formatted_angles.append(f_angle)


        if len(formatted_rates)==1: #check if just single array in list
            rates = np.array(formatted_rates)
        else: #if multiple, stitch into single array
            rates = np.vstack(formatted_rates)

        if len(formatted_angles)==1: #check if single array
            kin = np.array(formatted_angles)
        elif formatted_angles[0].ndim > 1: #check if multiple angles
            kin = np.vstack(formatted_angles)
        else:
            kin = np.hstack(formatted_angles)
        
        return np.squeeze(rates), np.squeeze(kin)

    def stitch_data(self, firing_rates_list, resampled_angles_list):
        '''
        deprecated. you can just use np.vstack instead of this
        WARNING: VSTACK WILL NOT BE INDEXED AT PARITY TO STITCH AND FORMAT UNLESS THE LAST 10 ENTRIES OF EACH LIST IS OMITTED
        '''
        rates = np.vstack(firing_rates_list)
        kin = np.vstack(resampled_angles_list)

        return rates, kin
   


    def subsample(self, percent, X=None, Y=None):
        '''
        function to subsample RAW data (not gait aligned). 
        we keep it in order since we go by folds.
        '''
        if X is None:
            X = self.data['rates']
        if Y is None:
            Y = self.data['angles']

        if percent==1.0:
            return X, Y

        new_x=[]
        new_y=[]

        for i in range(len(X)):
            subsample = int(np.round(percent * X[i].shape[0]))
            new_x.append(X[i][:subsample,:])
            new_y.append(Y[i][:subsample,:])

        return new_x, new_y
        

    def decode_angles(self, X=None, Y=None, metric_angle='limbfoot', scale=False):
        """
        takes list of rates, angles, then using a wiener filter to decode. 
        if no parameters are passed, uses data['rates'] and data['angles']

        returns best_h, vaf (array of all angles and all folds), test_x (the
        x test set that had best performance, and test_y (the y test set that
        had best formance)
        """
        try:
            if X is None:
                X = self.data['rates']
            if Y is None:
                Y = self.data['angles']
            
            if scale is True:
                X = self.apply_scaler(X)
            X, Y = self.stitch_and_format(X,Y)

            metric = self.angle_name_helper(metric_angle)

            h_angle, vaf_array, final_test_x, final_test_y, _ = decode_kfolds(X,Y,
                    metric_angle=metric)
            
            self.h_angle = h_angle
            return h_angle, vaf_array, final_test_x, final_test_y
        except Exception as e:
            print(e)
            print('did you run process() first.')

    

    def decode_toe_height(self):
        '''
        a PIPELINE code, which you can you only after you run
        process_toe_height. You cannot pass any custom parameters in this. it
        uses a wiener filter to create a decoder just for toe height
        '''
        try: 
            X,Y = self.stitch_and_format(self.data['rates'],
                    self.data['toe_height'])
            h_toe, vaf_array, final_test_x, final_test_y = decode_kfolds_single(X, Y)
            self.h_toe = h_toe
            return h_toe, vaf_array, final_test_x, final_test_y
        except:
            print('did you run process_toe_height() yet?????')


    def phase_train(self, upper_limit = 10, lower_limit = 10, angles = None, phase = None, rates = None,  forcible_test_index = None):     
        if angles is None:
            angles = self.data['angles']
        if phase is None:
            phase = self.data['phase']
        if rates is None:
            rates = self.data['rates']
        
        rect_phase_list = []
        for i in range(len(phase)):
            phase_copy = np.copy(phase[i])
            phase_copy = phase_copy + 1 
            phase_copy[0] = 0 
            phase_copy[-1] = 0
            rect_phase_list.append(phase_copy)
        form_rates, preform_phase = self.stitch_and_format(rates, rect_phase_list)      
        form_phase = drop_phase(preform_phase, upper_limit, lower_limit)
        phase_12 = form_phase[np.nonzero(form_phase)]
        rates_12 = form_rates[np.nonzero(form_phase),:][0]
        phase_angles, swing_mean = get_phase_angles(phase_12)    
        sin_arr, cos_arr = sine_and_cosine(phase_angles)    
        _, form_angles = self.stitch_and_format(rates,angles)
        angles_12 = form_angles[np.nonzero(form_phase),:][0]
        metric = self.angle_name_helper('limbfoot')
        phase_score, h_sin, h_cos, test_rates, test_arctans, predicted_arctans, final_test_index = parallel_decoder(X=rates_12, Y1=sin_arr, Y2=cos_arr, forced_test_index=forcible_test_index, printing = False)
        h_angle, angle_scores, _, test_angle, _ = decode_kfolds(rates_12,angles_12,metric_angle=metric, forced_test_index = final_test_index)
    
        self.predicted_arctans = predicted_arctans
        self.test_arctans = test_arctans
        self.swing_mean = swing_mean
        self.h_sin_nl = h_sin
        self.h_cos_nl = h_cos
        
        return phase_score, angle_scores, h_sin, h_cos, predicted_arctans, test_arctans, test_rates, swing_mean, h_angle, test_angle, final_test_index  
    
    def null_test(self, boots = 100):     
        angles = self.data['angles']
        phase = self.data['phase']
        rates = self.data['rates']
        rect_phase_list = []
        for i in range(len(phase)):
            phase_copy = np.copy(phase[i])
            for j in range(phase_copy.shape[0]):
                phase_copy[j] = phase_copy[j] + 1
            phase_copy[0] = 0 
            phase_copy[-1] = 0
            rect_phase_list.append(phase_copy)
        form_rates, preform_phase = self.stitch_and_format(rates, rect_phase_list)
        form_phase = drop_phase(preform_phase, 10, 10)
        phase_12 = form_phase[np.nonzero(form_phase)]
        rates_12 = form_rates[np.nonzero(form_phase),:][0]
        phase_angles, swing_mean = get_phase_angles(phase_12)    
        sin_arr, cos_arr = sine_and_cosine(phase_angles)    
        _, form_angles = self.stitch_and_format(rates,angles)
        angles_12 = form_angles[np.nonzero(form_phase),:][0]
        metric = self.angle_name_helper('limbfoot')
        phase_score, h_sin, h_cos, test_rates, test_arctans, predicted_arctans, final_test_index = parallel_decoder(X=rates_12, Y1=sin_arr, Y2=cos_arr, forced_test_index=None, printing = False)
        h_angle, angle_scores, _, test_angle, _ = decode_kfolds(rates_12,angles_12,metric_angle=metric, forced_test_index = final_test_index)

        null_test_alpha = null_hyopthesis_test_a(X=rates_12, Y1=sin_arr, Y2=cos_arr, boots = boots)
        null_test_beta = null_hyopthesis_test_b(rates_12,angles_12,metric_angle=metric, boots = boots)
        
        self.predicted_arctans = predicted_arctans
        self.test_arctans = test_arctans
        self.swing_mean = swing_mean
        self.h_sin_nl = h_sin
        self.h_cos_nl = h_cos
        
        return phase_score, null_test_alpha, angle_scores, null_test_beta, phase_angles
    
    
    def exclusion_train(self, upper_limit, lower_limit, exclusion_index):
       
        angles = self.data['angles']
        phase = self.data['phase']
        rates = self.data['rates']
        
        rect_phase_list = []
        for i in range(len(phase)):
            phase_copy = np.copy(phase[i])
            for j in range(phase_copy.shape[0]):
                phase_copy[j] = phase_copy[j] + 1
            phase_copy[0] = 0 
            phase_copy[-1] = 0
            rect_phase_list.append(phase_copy)
        temp_rates, pre_temp_phase = self.stitch_and_format(rates, rect_phase_list)        
        temp_phase = drop_phase(pre_temp_phase, 3.1, 1)
                
        phase_21 = temp_phase[np.nonzero(temp_phase)]
        rates_21 = temp_rates[np.nonzero(temp_phase),:][0]
        
        test_rates = rates_21[exclusion_index, :]
        form_rates = np.delete(rates_21, exclusion_index, axis=0)
        test_phase = phase_21[exclusion_index]
        preform_phase = np.delete(phase_21, exclusion_index)  
        form_phase = drop_phase(preform_phase, upper_limit, lower_limit)
    
        phase_12 = form_phase[np.nonzero(form_phase)]
        rates_12 = form_rates[np.nonzero(form_phase),:][0]
        test_arctans, _ = get_phase_angles(test_phase)
        phase_angles, swing_mean = get_phase_angles(phase_12)
        sin_arr, cos_arr = sine_and_cosine(phase_angles)    
        score, h_sin, h_cos, _, _, _, _ = parallel_decoder(X=rates_12, Y1=sin_arr, Y2=cos_arr, k=8, printing = False)
        
        
        pred_sin = test_wiener_filter(test_rates, h_sin)
        pred_cos = test_wiener_filter(test_rates, h_cos)        
        predicted_arctans = arctan_fn(pred_sin, pred_cos)
        
        # _, form_angles = self.stitch_and_format(rates,angles)
        # angles_12 = form_angles[np.nonzero(form_phase),:][0]
        # metric = self.angle_name_helper('limbfoot')
        # h_angle, vaf_array, test_angle_rates, test_angle_angle, _ = decode_kfolds(rates_12,angles_12,metric_angle=metric, vaf_scoring = False, forced_test_index = final_test_index)
        # , vaf_array, h_angle, test_angle_rates, test_angle_angle
        
        return score, h_sin, h_cos, predicted_arctans, test_arctans, test_rates, swing_mean
    
    
    def phase_evaluate(self, manual_threshold = None, refractory_tics = 0, bounds = [-4, 4], plotting = False):
        if manual_threshold == None: 
            threshold = self.swing_mean
        else:
            threshold = manual_threshold
        crossingsA = np.diff(self.test_arctans >threshold, prepend=0)
        np.put(crossingsA, np.where(crossingsA==-1), 0)
        crossingsB = np.diff(self.predicted_arctans >threshold, prepend=0)
        np.put(crossingsB, np.where(crossingsB==-1), 0)
        crossingsC = stim_cooldown(crossingsB, refractory_tics)
        true_indicies, new_indicies, delay_array_list, spacing_mean = phase_sychrony(crossingsA, crossingsC)
        true_score, effective_score = phase_diagnositc(self.predicted_arctans, self.test_arctans, threshold, true_indicies, new_indicies, delay_array_list, spacing_mean, bounds, plotting)
        return true_score, effective_score
    
    def get_H(self, H):
        if H == 'toe':
            H_mat = self.h_toe
        if H == 'angle':
            H_mat = self.h_angle
        if H == 'cos':
            H_mat = self.h_cos
        if H == 'sin':
            H_mat = self.h_sin
        return H_mat
    
    def impulse_response(self, AOI, H = None, plotting = True):
        phase_list = self.phase_list
        if H == None:
            H = 'sin'
        h_mat = self.get_H(H)
        response = impulse_response(AOI, h_mat, phase_list, plotting)
        return response


    def neuron_tuning(self, rates_gait=None):
        '''
        this takes in neural data that is divided into gaits, and sorts each
        channel by where in the gait cycle each channel is most active

        feed the output of this into plot_raster!
        '''
        try:
            if rates_gait is None:
                rates_gait = self.rates_gait
            rates_gait = np.vstack(rates_gait)
            gait_array_avg = np.average(rates_gait, axis=0)
            df = pd.DataFrame(gait_array_avg)
            temp = df.iloc[:, df.idxmax(axis=0).argsort()]
            self.norm_sorted_neurons=((temp-temp.min())/(temp.max()-temp.min()))
            return self.norm_sorted_neurons

        except Exception as e:
            print(e)
            print('make sure you run divide into gaits first')

    

    def apply_PCA(self, dims=None, X=None, transformer=None):
        '''
        this works now?
        '''
        if X is None:
            X=self.data['rates']
        if dims is None:
            dims=.95
        X_full = np.vstack(X)
        if transformer is None:
            pca_object = PCA(n_components = dims)
            pca_object.fit(X_full)
        else:
            pca_object = transformer

        x_pca = []

        for X_recording in X:
            x_pca.append(pca_object.transform(X_recording))
        
        self.num_components = x_pca[0].shape[1]
        self.pca_object = pca_object
        return x_pca


    def apply_scaler(self, X=None, scaler=None):
        if X is None:
            X=self.data['rates']
        if scaler is None:
            my_scaler=StandardScaler()
            X_full = np.vstack(X)
            my_scaler.fit(X_full)
        else:
            my_scaler = scaler
        
        scaled_X = []
        for X_recording in X:
            scaled_X.append(my_scaler.transform(X_recording))
        
        self.scaler = my_scaler
        return scaled_X


    def display_frame(self, video_path, recording_number, sample_number):
        cap = cv2. VideoCapture(video_path)
        crop_times = self.crop_list[recording_number]
        first_frame = crop_times[0] * 200
        my_frame = first_frame + (sample_number * 4)
        cap.set(1, my_frame)
        ret, frame = cap.read()

        video_time = my_frame/200
        print(f'video_time is: {video_time}')

        return frame
    
    def predicted_lines_malleable(self, h_sin, h_cos):
        try:  
            full_rates, full_angles = self.stitch_and_format(self.data['rates'], 
                            self.data['angles'])
            phase_list = self.phase_list
            sin_array, cos_array = sine_and_cosine(phase_list)
            predicted_sin = predicted_lines(full_rates, h_sin)
            predicted_cos = predicted_lines(full_rates, h_cos)
            arctans = arctan_fn(predicted_sin, predicted_cos)
            r2_array = []
            for i in range(sin_array.shape[1]):
                r2_sin = r2_score(sin_array[:,i], predicted_sin[:,i])
                r2_cos = r2_score(cos_array[:,i], predicted_cos[:,i])
                r2_array.append(np.mean((r2_sin,r2_cos)))
            return arctans, phase_list, r2_array
        except:
            print('error lol')
            print('Feed in sin and cos H matricies from another session to test gernealizability')

    def angle_name_helper(self, angle_name):
        return self.data['angle_names'].index(angle_name)

    def bodypart_helper(self, bodypart):
        return self.data['bodyparts'].index(bodypart)
    
    def DOM(self, upper_limit, lower_limit, legend=False):
        phase = self.data['phase']
        rates = self.data['rates']
        rect_phase_list = []
        for i in range(len(phase)):
            phase_copy = np.copy(phase[i])
            for j in range(phase_copy.shape[0]):
                phase_copy[j] = phase_copy[j] + 1
            phase_copy[0] = 0 
            phase_copy[-1] = 0
            rect_phase_list.append(phase_copy)
        form_rates, preform_phase = self.stitch_and_format(rates, rect_phase_list)
        form_phase = drop_phase(preform_phase, upper_limit, lower_limit)
        phase_12 = form_phase[np.nonzero(form_phase)]
        rates_rs = form_rates[np.nonzero(form_phase),:][0]
        phase_angles, swing_mean = get_phase_angles(phase_12)
        phase_rs = np.radians(phase_angles-180)
        magn = []
        heading = []
        for j in range(rates_rs.shape[1]):
            sum1 = 0
            sum_sin = 0
            sum_cos = 0
            for i in range(rates_rs.shape[0]):
                sum1 = sum1 + rates_rs[i,j]
                sum_sin = sum_sin + rates_rs[i,j]*np.sin(phase_rs[i])
                sum_cos = sum_cos + rates_rs[i,j]*np.cos(phase_rs[i])
            sin_bar = sum_sin/sum1
            cos_bar = sum_cos/sum1
            r = (sin_bar**2 + cos_bar**2)**(1/2)
            theta = np.arctan2(sin_bar, cos_bar)
            magn.append(r)
            heading.append(theta)
        magn = np.array(magn)
        heading = np.array(heading)
        compass(heading, magn, swing_mean, legend)
        return