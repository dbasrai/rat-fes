import traceback
import yaml
import gc

from src.neural_analysis import *
from src.wiener_filter import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *

from scipy.signal import resample, find_peaks

class CortProcessor:
    def __init__(self, folder_path):
        self.handler = FolderHandler(folder_path)
        self.tdt_data, self.kin_data = self.extract_data()
        
        self.crop_list = None
        crop = self.parse_config()
        if isinstance(crop, list):
            self.crop_list = crop

        self.rate_list = None
        self.angle_list = None

        self.format_rates = None
        self.format_angles = None

        self.toe_height_list = None
        self.format_toe_height = None

        self.gait_indices = None
        self.avg_gait_samples = None

    def parse_config(self):
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

    def process(self, manual_crop_list=None, threshold_multiplier = 3.0, binsize = 0.05):
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
        
        self.data['rates']=[]
        self.data['coords']=[]
        self.data['angles']=[]
        for i in range(len(tdt_data_list)):
            self.tdt_data[i], self.kin_data[i] = self.crop_data(tdt_data_list[i], 
                    kin_data_list[i], crop_list[i])
            
            fs = self.tdt_data[i]['fs'] #quick accessible variable
            neural = self.tdt_data[i]['neural'] #quick accessible variable
            
            filtered_neural = filter_neural(neural, fs)
            clean_filtered_neural = remove_artifacts(filtered_neural, fs)

            spikes = autothreshold_crossings(clean_filtered_neural,
                    threshold_multiplier)
            firing_rates = spike_binner(spikes, fs, binsize)
            
            self.data['rates'].append(firing_rates)

            temp_angles = self.kin_data[i]['angles'] #quick accessible variable
            temp_coords = self.kin_data[i]['coords']

            resampled_angles = resample(temp_angles, firing_rates.shape[0],
                    axis=0)
            resampled_coords = resample(temp_coords, firing_rates.shape[0],
                    axis=0)

            self.data['angles'].append(resampled_angles)
            self.data['coords'].append(resampled_coords)

            self.tdt_data[i] = 0
            self.kin_data[i] = 0 
            
            gc.collect()
            #freeing memory
        
            
        return np.vstack(self.data['rates']), np.vstack(self.data['angles'])

    def process_toe_height(self, toe_num=0):
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
            traceback.print_exc()

    def crop_data(self, tdt_data, kin_data, crop):
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




    def stitch_and_format(self, firing_rates_list, resampled_angles_list):
        formatted_rates = []
        formatted_angles = []

        for i in range(len(firing_rates_list)):
            f_rate, f_angle = format_data(firing_rates_list[i],
                    resampled_angles_list[i])
            formatted_rates.append(f_rate)
            formatted_angles.append(f_angle)

        rates = np.vstack(formatted_rates)

        if formatted_angles[0].ndim > 1:
            kin = np.vstack(formatted_angles)
        else:
            kin = np.hstack(formatted_angles)
        return rates, kin

    def stitch_data(self, firing_rates_list, resampled_angles_list):
        rates = np.vstack(firing_rates_list)
        kin = np.vstack(resampled_angles_list)

        return rates, kin

    def decode_angles(self, X=None, Y=None):
        try:
            if X is None and Y is None:
                X, Y = self.stitch_and_format(self.data['rates'], 
                        self.data['angles'])

            else:
                X, Y = self.stitch_and_format(X, Y)
            h_angle, vaf_array, final_test_x, final_test_y = decode_kfolds(X,Y)
   
            return h_angle, vaf_array, final_test_x, final_test_y
        except:
            print('did you run process() first.')

    def decode_toe_height(self):
        try: 
            X,Y = self.stitch_and_format(self.data['rates'],
                    self.data['toe_height'])
            h_toe, vaf_array, final_test_x, final_test_y = decode_kfolds_single(X, Y)
            return h_toe, vaf_array, final_test_x, final_test_y
        except:
            print('did you run process_toe_height() yet?????')

    def get_gait_indices(self, Y=None):
        limbfoot_angles = []
        if Y is None:
            for angles in self.data['angles']:
                limbfoot_angles.append(angles[:,3])
        elif isinstance(Y, np.ndarray):
            limbfoot_angles.append(Y)
        else:
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
       
        if gait_indices is None:
            gait_indices = self.gait_indices
        
        if avg_gait_samples is None:
            avg_gait_samples = self.avg_gait_samples 

        if X is None:
            rates = self.data['rates']
        elif isinstance(X, np.ndarray):
            rates = []
            rates.append(X)
        else:
            rates = X

        if Y is None:
            angles = self.data['angles']
        elif isinstance(Y, np.ndarray):
            angles = []
            angles.append(Y)
        else:
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

    def remove_bad_gaits(self, X=None, Y=None, gait_indices=None):
        if gait_indices is None:
            gait_indices = self.gait_indices
            avg_gait_samples = self.avg_gait_samples
            above = 1.33 * self.avg_gait_samples
            below = .66 * self.avg_gait_samples
            bads_list = []
            for idx in self.gait_indices:

                bad_above = np.argwhere(np.diff(idx)>above)
                bad_below = np.argwhere(np.diff(idx)<below)

                bads_list.append(np.squeeze(np.concatenate((bad_above,
                    bad_below))).tolist())

        if X is None:
            rates = self.data['rates']
            angles = self.data['angles']
        proc_rates = []
        proc_angles = []
        for i, trial_indices in enumerate(gait_indices):
            temp_rates = []
            temp_angles = []
            for j in range(np.size(trial_indices)-1):
                if j in bads_list[i]:
                    continue
                else:
                    end = trial_indices[j+1]
                    start = trial_indices[j]

                    temp_rates.append(rates[i][start:end,:])
                    temp_angles.append(angles[i][start:end,:])

            proc_rates.append(np.vstack(temp_rates))
            proc_angles.append(np.vstack(temp_angles))

        return proc_rates, proc_angles
                
 

    def neuron_tuning(self, rates_gait=None):
        try:
            if rates_gait is None:
                temp_rates_gait = self.rates_gait

                rates_gait = np.vstack(temp_rates_gait)

            gait_array_avg = np.average(rates_gait, axis=0)
            df = pd.DataFrame(gait_array_avg)
            temp = df.iloc[:, df.idxmax(axis=0).argsort()]
            self.norm_sorted_neurons=((temp-temp.min())/(temp.max()-temp.min()))
            return self.norm_sorted_neurons

        except Exception as e:
            print(e)
            print('make sure you run divide into gaits first')

    def convert_to_phase(self, Y, gait_indices = None):
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
        temp_rates, nada = self.stitch_data(self.rate_list, self.angle_list)
        nada, pca_output = apply_PCA(temp_rates.T, dims)
        
        self.PCA_rate_list = []

        for rate in self.rate_list:
            temp_output = pca_output.transform(rate.T)
            self.PCA_rate_list.append(temp_output.T)

        return self.PCA_rate_list, pca_output

