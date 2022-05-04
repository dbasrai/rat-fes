from src.neural_analysis import *
from src.wiener_filter import *
from src.folder_handler import *
from src.tdt_support import *

class CortProcessor:
    def __init__(self, folder_path):
        self.handler = FolderHandler(folder_path)

    def extract_data(self):
        tdt_data_list = []
        raw_ts_list = self.handler.get_ts_list()
        raw_tdt_list = self.handler.get_tdt_list()
        for i in range(len(raw_tdt_list)):
            tdt_data_list.append(extract_tdt(raw_tdt_list[i], raw_ts_list[i]))
        
        kin_data_list = []

        raw_coords_list = self.handler.get_coords_list()
        raw_angles_list = self.handler.get_angles_list()
        
        for i in range(len(raw_coords_list)):
            kin_data_list.append(extract_kin_data(raw_coords_list[i],
                raw_angles_list[i]))

        return tdt_data_list, kin_data_list

    def process(self, crop_list, threshold_multiplier = 3.0, binsize = 0.05):
        tdt_data_list, kin_data_list = self.extract_data()

        firing_rates_list = []
        resampled_angles_list = []

        for i in range(len(tdt_data_list)):
            tdt_data, kin_data = crop_data(tdt_data_list[i], kin_data_list[i],
                    crop_list[i])

            fs = tdt_data['fs'] #quick accessible variable
            neural = tdt_data['neural'] #quick accessible variable
            
            filtered_neural = filter_neural(neural, fs)
            clean_filtered_neural = remove_artifacts(filtered_neural, fs)

            spikes = autothreshold_crossings(clean_filtered_neural,
                    threshold_multiplier)
            firing_rates = spike_binner(spikes, fs, binsize)

            angles = kin_data['angles'] #quick accessible variable

            resampled_angles = resample(angles, firing_rates.shape[1], axis=1)
            
            firing_rates_list.append(firing_rates)
            resampled_angles_list.append(resampled_angles)

        
        rates, kins = stitch_data(firing_rates_list, resampled_angles_list)

        return rates, kins

   # def decode
   
