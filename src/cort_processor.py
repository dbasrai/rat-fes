import traceback
import yaml
import gc

from src.neural_analysis import *
from src.wiener_filter import *
from src.filters import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *
from src.phase_decoder_support import *
from sklearn.metrics import r2_score
from scipy.signal import resample, find_peaks

class CortProcessor:
    '''
    class that can handle neural + kinematic/EMG data simultaneously
    upon initialization, extracts data from TDT and anipose file
    see cort_processor.md in 'docs' for more information
    '''
    
    
    def __init__(self, folder_path):
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
        It extracts raw neuiral,raw angles, and raw coords
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
        
        for i in range(len(raw_coords_list)):
            kin_data_list.append(extract_kin_data(raw_coords_list[i],
                raw_angles_list[i]))

        return tdt_data_list, kin_data_list

    def process(self, manual_crop_list=None, threshold_multiplier = -3.0, binsize = 0.05):
        """
        1) takes raw neural data, then bandpass+notch filters, then extracts
        spikes using threshold_multiplier, then bins into firing rates with bin
        size
        2) takes coordinates/angles, then resamples at 20 HZ
        3) saves all under CortHandler.data

        crop is automatically read from config.yaml file, unless manual crop
        list is set.
        """
        if self.crop_list is not None:
            crop_list = self.crop_list
        elif manual_crop_list is not None:
            crop_list = manual_crop_list
        else:
            #TODO
            print('need to make it so that it is uncropped')

        tdt_data_list = self.tdt_data
        kin_data_list = self.kin_data

        self.data = {}

        self.data['bodyparts'] = kin_data_list[0]['bodyparts']
        self.data['angle_names'] = kin_data_list[0]['angles_list']
        
        self.data['rates']=[]
        self.data['coords']=[]
        self.data['angles']=[]
        for i in range(len(tdt_data_list)):
            #syncs neural data/kinematic data, then crops
            self.tdt_data[i], self.kin_data[i] = self.crop_data(tdt_data_list[i], 
                    kin_data_list[i], crop_list[i])
            
            fs = self.tdt_data[i]['fs'] #quick accessible variable
            neural = self.tdt_data[i]['neural'] #quick accessible variable
            
            #notch and bandpass filter
            
            means = np.mean(neural, axis = 1)
            commonmode = []
            for j in range(neural.shape[1]):
                commonhold = neural[:,j] - means
                commonmode.append(commonhold)
            commonmode = np.array(commonmode).T
            
            clean_filtered_neural = fresh_filt(commonmode, 350, 8000, fs, order = 4)

            #extract spike and bin
            spikes_tmp = threshold_crossings_refrac(clean_filtered_neural,
                    threshold_multiplier)
            spikes = refractory_limit(spikes_tmp, fs)
            firing_rates = spike_binner(spikes, fs, binsize)
            
            self.data['rates'].append(firing_rates)

            temp_angles = self.kin_data[i]['angles'] #quick accessible variable
            temp_coords = self.kin_data[i]['coords'] 

            #resample at same frequency of binned spikes
            resampled_angles = resample(temp_angles, firing_rates.shape[0],
                    axis=0)
            resampled_coords = resample(temp_coords, firing_rates.shape[0],
                    axis=0)

            self.data['angles'].append(resampled_angles)
            self.data['coords'].append(resampled_coords)

            #remove raw data to save memory
            self.tdt_data[i] = 0
            self.kin_data[i] = 0 
            gc.collect()
        
        #returning stitched rates --- we don't directly use this for anything.     
        return np.vstack(self.data['rates']), np.vstack(self.data['angles'])

    def process_toe_height(self, toe_num=0):
        """
        extracts toe height from data['coords'], then scales such that lowest
        toe height is 0.

        toe_num is which bodypart in data['coords'] is the toe. default is
        bodypart 0
        """
        try:
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

        crop_kin_datafile['coords'] = temp_coords[kin_start:kin_end,:,:]
        crop_kin_datafile['angles'] = temp_angles[kin_start:kin_end,:]
        #maybe need edge-case if only single angle/bodypart
        
        return crop_tdt_datafile, crop_kin_datafile




    def stitch_and_format(self, firing_rates_list=None, resampled_angles_list=None):
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
                    resampled_angles_list[i])
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
        '''
        rates = np.vstack(firing_rates_list)
        kin = np.vstack(resampled_angles_list)

        return rates, kin

    def decode_angles(self, X=None, Y=None):
        """
        takes list of rates, angles, then using a wiener filter to decode. 
        if no parameters are passed, uses data['rates'] and data['angles']

        returns best_h, vaf (array of all angles and all folds), test_x (the
        x test set that had best performance, and test_y (the y test set that
        had best formance)
        """
        try:
            if X is None and Y is None:
                X, Y = self.stitch_and_format(self.data['rates'], 
                        self.data['angles'])

            else:
                X, Y = self.stitch_and_format(X, Y)
            h_angle, vaf_array, final_test_x, final_test_y = decode_kfolds(X,Y)
            self.h_angle = h_angle
            return h_angle, vaf_array, final_test_x, final_test_y
        except:
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
            
    def decode_phase(self, rates=None, angles=None):
        if rates is None and angles is None:
            full_rates, full_angles = self.stitch_and_format(self.data['rates'], 
                        self.data['angles'])

        else:
            full_rates, full_angles = self.stitch_and_format(rates, angles)
        phase_list = []
        for i in range(full_angles.shape[1]):
            peak_list = tailored_peaks(full_angles, i)
            phase_list_tmp = to_phasex(peak_list, full_angles[:,i])
            phase_list.append(phase_list_tmp)
        phase_list = np.array(phase_list).T
        sin_array, cos_array = sine_and_cosine(phase_list)
        h_sin, _, _, _ = decode_kfolds(X=full_rates, Y=sin_array)
        h_cos, _, _, _ = decode_kfolds(X=full_rates, Y=cos_array)
        predicted_sin = predicted_lines(full_rates, h_sin)
        predicted_cos = predicted_lines(full_rates, h_cos)
        arctans = arctan_fn(predicted_sin, predicted_cos)
        r2_array = []
        for i in range(sin_array.shape[1]):
            r2_sin = r2_score(sin_array[:,i], predicted_sin[:,i])
            r2_cos = r2_score(cos_array[:,i], predicted_cos[:,i])
            r2_array.append(np.mean((r2_sin,r2_cos)))
        self.phase_list = phase_list
        self.h_sin = h_sin
        self.h_cos = h_cos
        return arctans, phase_list, r2_array
    
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
        
    def get_gait_indices(self, Y=None):
        '''
        This takes a kinematic variable, and returns indices where each peak is
        found. It also returns the average number of samples between each
        peaks. 

        If passing in without parameter, it uses the 3rd angle measurement,
        which is usually the limbfoot angle. 

        This is mainly used as a starter method for other method.
        Divide_into_gaits for instance takes in these indices, and divides both
        the kinematics and rates into gait cycles.
        '''
        limbfoot_angles = []
        if Y is None:
            for angles in self.data['angles']:
                limbfoot_angles.append(angles[:,3])
        else:
            assert isinstance(Y, list), 'Y must be a list'
            limbfoot_angles = Y

        gait_indices = []
        samples_list = []
        for angle in limbfoot_angles:
            peaks, nada = find_peaks(angle, prominence=10)
            peaks = np.append(peaks, np.size(angle))
            peaks = np.insert(peaks, 0, 0)
            gait_indices.append(peaks)
            samples_list.append(np.diff(peaks))
        
        if len(samples_list) > 1:
            samples = np.concatenate(samples_list)
        else:
            samples = samples_list[0]
        avg_gait_samples = int(np.round(np.average(samples)))
        
        if Y is None:
            self.gait_indices = gait_indices
            self.avg_gait_samples = avg_gait_samples

        return gait_indices, avg_gait_samples
    
    def divide_into_gaits(self, X=None, Y=None, gait_indices=None,
            avg_gait_samples=None,
            bool_resample=True):
        '''
        this takes in X, which is usually rates, Y which is usually some
        kinematic variable, and indices, which tell you how to divide up the
        data, and divides all the data up as lists. Since the originall X is
        already a list, it returns a list of list of lists. Confusing?
        
        if you don't pass any parameters, then get_gait_indices must be run
        first. This finds gait_indices/avg_gait_samples using limbfoot angle.

        If you don't pass in X/Y paramters, it by default uses rates and
        angles, and then divides em up.
        '''
       
        if gait_indices is None:
            gait_indices = self.gait_indices
        
        if avg_gait_samples is None:
            avg_gait_samples = self.avg_gait_samples 

        if X is None:
            rates = self.data['rates']
        else:
            assert isinstance(X, list), 'X must be a list'
            rates = X

        if Y is None:
            angles = self.data['angles']
        else:
            assert isinstance(Y, list), 'Y must be a list'
            angles = Y

        X_gait = []
        Y_gait = []

        for i, trial_gait_index in enumerate(gait_indices):
            trial_rate_gait = []
            trial_angle_gait = []
            for j in range(np.size(trial_gait_index)-1):
                end = trial_gait_index[j+1]
                start = trial_gait_index[j]

                temp_rate = rates[i][start:end,:]
                temp_angle = angles[i][start:end,:]

                if bool_resample:
                    temp_rate = resample(temp_rate, avg_gait_samples, axis=0)
                    temp_angle = resample(temp_angle, avg_gait_samples, axis=0)
                
                trial_rate_gait.append(temp_rate)
                trial_angle_gait.append(temp_angle)
            X_gait.append(np.array(trial_rate_gait))
            Y_gait.append(np.array(trial_angle_gait))

        if X is None:
            self.rates_gait = X_gait
        if Y is None:
            self.angles_gait = Y_gait 

        return X_gait, Y_gait #return list of list of lists lol

    def remove_bad_gaits(self, X=None, Y=None, gait_indices=None,
            avg_gait_samples = None, bool_resample=True):
        '''
        similar to divide into gaits, but instead of just dividing, it also
        removes any gait cycles that have a much smaller or much larger amount
        of samples. in a sense it divies up gaits and removes bad ones.0
        '''
        if gait_indices is None:
            gait_indices = self.gait_indices
        
        if avg_gait_samples is None:
            avg_gait_samples = self.avg_gait_samples
        
        
        above = 1.33 * avg_gait_samples
        below = .66 * avg_gait_samples
        bads_list = []
        for idx in gait_indices:

            bad_above = np.argwhere(np.diff(idx)>above)
            bad_below = np.argwhere(np.diff(idx)<below)

            bads_list.append(np.squeeze(np.concatenate((bad_above,
                bad_below))).tolist())

        if X is None:
            rates = self.data['rates']
        elif isinstance(X, list):
            rates = X
        else: 
            print('X must be list')
            return


        if Y is None:
            angles = self.data['angles']
        elif isinstance(Y, list):
            angles = Y
        else:
            print('Y must be list')
            return

        proc_rates = []
        proc_angles = []
        for i, trial_indices in enumerate(gait_indices):
            trial_rate_gait = []
            trial_angle_gait = []
            for j in range(np.size(trial_indices)-1):
                if j in bads_list[i]:
                    continue
                else:
                    end = trial_indices[j+1]
                    start = trial_indices[j]

                    temp_rate = rates[i][start:end,:]
                    temp_angle = angles[i][start:end,:]
                    if bool_resample:
                        temp_rate = resample(temp_rate, avg_gait_samples, axis=0)
                        temp_angle = resample(temp_angle, avg_gait_samples, axis=0)
                    trial_rate_gait.append(temp_rate)
                    trial_angle_gait.append(temp_angle)
 
            proc_rates.append(trial_rate_gait)
            proc_angles.append(trial_angle_gait)

        return np.vstack(proc_rates), np.vstack(proc_angles)
                
 

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

    def convert_to_phase(self, gait_indices = None):
        '''
        this is still TODO. might already work, but double check this
        '''
        #TODO
        phase_list = []
        
        if gait_indices is None:
            gait_indices = self.gait_indices
        for i in range(np.size(gait_indices)-1):
            end = gait_indices[i+1]
            start = gait_indices[i]
            phase = np.sin(np.linspace(0.0, 2.0*math.pi, num=end-start, 
                endpoint=False))
            phase_list.append(gait)
        
        return phase_list #use np.hstack on output to get continuous

    def with_PCA(self, dims):
        '''
        this is todo, probably doesn't work.
        '''
        #TODO
        temp_rates, nada = self.stitch_data(self.rate_list, self.angle_list)
        nada, pca_output = apply_PCA(temp_rates.T, dims)
        
        self.PCA_rate_list = []

        for rate in self.rate_list:
            temp_output = pca_output.transform(rate.T)
            self.PCA_rate_list.append(temp_output.T)

        return self.PCA_rate_list, pca_output

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
            
            