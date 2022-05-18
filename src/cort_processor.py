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

    def process(self, crop_list, threshold_multiplier = 3.0, binsize = 0.05):
        
        tdt_data_list = self.tdt_data
        kin_data_list = self.kin_data

        self.rate_list = []
        self.angle_list = []

        self.crop_tdt_data = []
        self.crop_kin_data = []
        
        for i in range(len(tdt_data_list)):
            crop_tdt_datafile, crop_kin_datafile = self.crop_data(tdt_data_list[i], 
                    kin_data_list[i], crop_list[i])
            
            self.crop_tdt_data.append(crop_tdt_datafile)
            self.crop_kin_data.append(crop_kin_datafile)

            fs = crop_tdt_datafile['fs'] #quick accessible variable
            neural = crop_tdt_datafile['neural'] #quick accessible variable
            
            filtered_neural = filter_neural(neural, fs)
            clean_filtered_neural = remove_artifacts(filtered_neural, fs)

            spikes = autothreshold_crossings(clean_filtered_neural,
                    threshold_multiplier)
            firing_rates = spike_binner(spikes, fs, binsize)

            temp_angles = crop_kin_datafile['angles'] #quick accessible variable

            resampled_angles = resample(temp_angles, firing_rates.shape[0],
                    axis=0)
            
            self.rate_list.append(firing_rates)
            self.angle_list.append(resampled_angles)
            
    
        self.format_rate, self.format_angle = self.stitch_and_format(self.rate_list, 
                self.angle_list)

        return self.format_rate, self.format_angle

    def process_toe_height(self, toe_num=0):
        try:
            temp_list = []
            self.toe_height_list = []
            minimum_toe = 1000
            for i in range(len(self.crop_tdt_data)):
                toe_height = self.crop_kin_data[i]['coords'][toe_num, :, 1]
                if np.min(toe_height) < minimum_toe:
                    minimum_toe = np.min(toe_height)
                temp_list.append(toe_height)
            for i in range(len(temp_list)):
                resampled_toe = resample(temp_list[i]-minimum_toe,
                        self.rate_list[i].shape[1])
                self.toe_height_list.append(resampled_toe)

            nada, self.format_toe_height = self.stitch_and_format(self.rate_list,
                    self.toe_height_list)
        except: 
            print('failed!! did you run process first')
        return self.toe_height_list

    def crop_data(self, tdt_data, kin_data, crop):
        crop_tdt_datafile=copy.deepcopy(tdt_data)
        crop_kin_datafile=copy.deepcopy(kin_data)

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
            formatted_rates.append(f_rate.T)
            formatted_angles.append(f_angle.T)

        rates = np.hstack(formatted_rates)
        kin = np.hstack(formatted_angles)
        #all the transpositions since hstack needs first dim to match
        #then we flip it back around to return it
        return rates.T, kin.T

    def stitch_data(self, firing_rates_list, resampled_angles_list):
        rates = np.hstack(firing_rates_list)
        kin = np.hstack(resampled_angles_list)

        return rates, kin

    def decode_angles(self):
        try:
            X = self.format_rate
            Y = self.format_angle
            h_angle, vaf_array, final_test_x, final_test_y = decode_kfolds(X,Y)
   
            return h_angle, vaf_array, final_test_x, final_test_y
        except:
            print('did you run process first. looool')

    def decode_toe_height(self):
        if not hasattr(self, 'format_toe_height'):
            print('run process_toe_height_first')
        else:
            X = self.format_rate
            Y = self.format_toe_height

            h_toe, vaf_array, final_test_x, final_test_y = decode_kfolds_single(X, Y)
        return h_toe, vaf_array, final_test_x, final_test_y

    def get_gait_indices(self, limbfoot_angle=3):
        self.rate_gait = []
        self.angle_gait = []

        stitched_rates, stitched_angles = self.stitch_data(self.rate_list,
                self.angle_list)
        peaks, nada = find_peaks(stitched_angles[3,:], prominence=10)
        peaks = np.append(peaks, np.size(stitched_angles[3,:])) #append the end of the array
        peaks = np.insert(peaks, 0, 0) #and start it at the beginning
 
        self.gait_indices = peaks
        self.avg_gait_samples = int(np.round(np.average(np.diff(peaks))))

    
    def divide_into_gaits(self, X, Y, bool_resample=True):

        X_gait = []
        Y_gait = []

        for i in range(np.size(self.gait_indices)-1):
            end = self.gait_indices[i+1]
            start = self.gait_indices[i]

            temp_rate = X[:,start:end]
            temp_angle = Y[:,start:end]

            if bool_resample:
                temp_rate = resample(temp_rate, self.avg_gait_samples, axis=1)
                temp_angle = resample(temp_angle, self.avg_gait_samples, axis=1)

            X_gait.append(temp_rate)
            Y_gait.append(temp_angle)

        return X_gait, Y_gait


    def neuron_tuning(self, gait_rates):
        try:
            gait_array = np.array(gait_rates)
            gait_array_avg = np.average(gait_array, axis=0)
            
            channel_numbers = np.arange(0,32)
            
            df = pd.DataFrame(gait_array_avg, index=channel_numbers)
            self.sorted_neurons = df.iloc[df.idxmax(axis=1).argsort()]
            temp = self.sorted_neurons.transpose()
            self.norm_sorted_neurons=((temp-temp.min())/(temp.max()-temp.min())).transpose()
            return self.norm_sorted_neurons

        except:
            print('make sure you run divide into gaits first')


    def with_PCA(self, dims):
        temp_rates, nada = self.stitch_data(self.rate_list, self.angle_list)
        nada, pca_output = apply_PCA(temp_rates.T, dims)
        
        self.PCA_rate_list = []

        for rate in self.rate_list:
            temp_output = pca_output.transform(rate.T)
            self.PCA_rate_list.append(temp_output.T)

        return self.PCA_rate_list, pca_output

