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
        
        for i in range(len(tdt_data_list)):
            crop_tdt_data, crop_kin_data = crop_data(tdt_data_list[i], kin_data_list[i],
                    crop_list[i])
            fs = crop_tdt_data['fs'] #quick accessible variable
            neural = crop_tdt_data['neural'] #quick accessible variable
            
            filtered_neural = filter_neural(neural, fs)
            clean_filtered_neural = remove_artifacts(filtered_neural, fs)

            spikes = autothreshold_crossings(clean_filtered_neural,
                    threshold_multiplier)
            firing_rates = spike_binner(spikes, fs, binsize)

            temp_angles = crop_kin_data['angles'] #quick accessible variable

            resampled_angles = resample(temp_angles, firing_rates.shape[1], axis=1)
            
            self.rate_list.append(firing_rates)
            self.angle_list.append(resampled_angles)

    
        self.format_rate, self.format_angle = self.stitch_and_format(self.rate_list, 
                self.angle_list)

        return self.format_rate, self.format_angle

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

    def linear_decoder(self):
        if self.format_rate is None:
            print('run process first')
        else:
            X = self.format_rate
            Y = self.format_angle
            h, vaf_array, final_test_x, final_test_y = decode_kfolds(X,Y)
   
        return h, vaf_array, final_test_x, final_test_y

    def divide_into_gaits(self, limbfoot_angle=3, bool_resample=True):
        self.rate_gait = []
        self.angle_gait = []

        stitched_rates, stitched_angles = self.stitch_data(self.rate_list,
                self.angle_list)
        peaks, nada = find_peaks(stitched_angles[3,:], prominence=10)
        peaks = np.append(peaks, np.size(stitched_angles[3,:])) #append the end of the array
        peaks = np.insert(peaks, 0, 0) #and start it at the beginning
 
        avg_samples = int(np.round(np.average(np.diff(peaks))))
        for i in range(np.size(peaks)-1):
            end = peaks[i+1]
            start = peaks[i]

            temp_rate = stitched_rates[:,start:end]
            temp_angle = stitched_angles[:,start:end]

            if bool_resample:
                temp_rate = resample(temp_rate, avg_samples, axis=1)
                temp_angle = resample(temp_angle, avg_samples, axis=1)

            self.rate_gait.append(temp_rate)
            self.angle_gait.append(temp_angle)

        #return self.rate_gait, self.angle_gait





