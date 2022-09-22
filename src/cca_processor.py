from src.neural_analysis import *
from src.cort_processor import *
from src.wiener_filter import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

class CCAProcessor:
    def __init__(self, cp1, cp2, limbfoot_angle=1):
        self.cp1 = cp1
        self.cp1.get_gait_indices()
        
        self.cp2 = cp2
        self.cp2.get_gait_indices()
        self.limbfoot_angle=limbfoot_angle
        
        self.data = {}
        self.data['cp1'] = {}
        self.data['cp2'] = {}

        self.data['cp1']['proc_x'], self.data['cp1']['proc_y'],\
        self.data['cp2']['proc_x'], self.data['cp2']['proc_y'] =\
        self.process_and_align_kinematics()

        
        self.data['cp1']['h'], self.data['cp1']['proc_vaf'], nada, nada =\
        self.cp1.decode_angles(X=[self.data['cp1']['proc_x']],\
        Y=[self.data['cp1']['proc_y']])

        self.data['cp2']['h'], self.data['cp2']['proc_vaf'], nada, nada =\
        self.cp2.decode_angles(X=[self.data['cp2']['proc_x']],\
        Y=[self.data['cp2']['proc_y']])

        print(self.data['cp1']['proc_x'].shape)
        print(self.data['cp1']['proc_y'].shape)
        print(self.data['cp2']['proc_x'].shape)
        print(self.data['cp2']['proc_y'].shape)


    def get_better_decoder(self, limbfoot_angle=1):
        nada, vaf1, nada, nada = self.cp1.decode_angles()
        nada, vaf2, nada, nada = self.cp2.decode_angles()

        if np.average(vaf1,1)[limbfoot_angle] >= np.average(vaf2, 1)[limbfoot_angle]:
            print('cp1 is better')
        else:
            print('cp2 is better')

    def check_same_kinematics(self):
        if self.cp1.data['angle_names'] == self.cp2.data['angle_names']:
            output = 'should be good to align'
            return output
        else:
            output = 'kinematics are different'
            return output

    def apply_PCA(self, cp1_x=None, cp2_x=None, preset_num_components = None):

        if cp1_x is None:
            cp1_x = self.data['cp1']['proc_x']
        if cp2_x is None:
            cp2_x = self.data['cp2']['proc_x']

        if preset_num_components is None:
            pca_cp1 = PCA(n_components=.95)
            cp1_pca = pca_cp1.fit_transform(self.data['cp1']['proc_x'])

            pca_cp2 = PCA(n_components=.95)
            cp2_pca = pca_cp2.fit_transform(self.data['cp2']['proc_x'])

            num_components = min(cp2_pca.shape[1], cp1_pca.shape[1])

            self.num_components = num_components
        else:
            num_components = preset_num_components
            self.num_components = num_components

        pca_cp1 = PCA(n_components = num_components)
        pca_cp2 = PCA(n_components = num_components)

        self.data['cp1']['pca_x'] = pca_cp1.fit_transform(cp1_x)
        self.data['cp2']['pca_x'] = pca_cp2.fit_transform(cp2_x)

        return num_components, self.data['cp1']['pca_x'],\
                self.data['cp2']['pca_x']

    def CCA_cp2(self, cp1_x=None, cp2_x=None, preset_num_components=None):
        if preset_num_components is None:
            try:
                num_components = self.num_components
            except:
                print('set num copmoentns')
        else:
            num_components = preset_num_components

        if cp1_x is None:
            cp1_x = self.data['cp1']['pca_x']
        if cp2_x is None:
            cp2_x = self.data['cp2']['pca_x']
        cca_cp1cp2 = CCA(n_components = num_components, scale=False)
        x1_cca, x2_cca=cca_cp1cp2.fit_transform(cp1_x, cp2_x)

        self.cca = cca_cp1cp2
        self.x1_cca = x1_cca
        self.x2_cca = x2_cca

        x2_into_x1 = cca_cp1cp2.inverse_transform(x2_cca)

        return x2_into_x1

    def process_and_align_kinematics(self):

        print(self.check_same_kinematics())

        avg_samples = self.cp1.avg_gait_samples #arbitrarily grab from 1
        cp1_gait_x, cp1_gait_y = self.cp1.remove_bad_gaits()
        cp2_gait_x, cp2_gait_y = self.cp2.remove_bad_gaits(avg_gait_samples =
                avg_samples)

        
        if len(cp1_gait_x) >= len(cp2_gait_x):
            end_slice = len(cp2_gait_x)
            cp1_gait_x = cp1_gait_x[:end_slice]
            cp1_gait_y = cp1_gait_y[:end_slice]

        else:
            end_slice = len(cp1_gait_x)
            cp2_gait_x = cp2_gait_x[:end_slice]
            cp2_gait_y = cp2_gait_y[:end_slice]


        total_samples = cp1_gait_x.shape[0] * cp1_gait_x.shape[1]

        cp1_gait_x = np.reshape(cp1_gait_x, (total_samples,
            cp1_gait_x.shape[2]))
        cp1_gait_y = np.reshape(cp1_gait_y, (total_samples,
            cp1_gait_y.shape[2]))
        cp2_gait_x = np.reshape(cp2_gait_x, (total_samples, 
            cp2_gait_x.shape[2]))
        cp2_gait_y = np.reshape(cp2_gait_y, (total_samples,
            cp2_gait_y.shape[2]))
        
        return cp1_gait_x, cp1_gait_y, cp2_gait_x, cp2_gait_y 


    def back_to_gait(self, x, y=None, avg_gait_samples=None):
        if avg_gait_samples is None:
            avg_gait_samples = self.cp1.avg_gait_samples
        x_return = np.reshape(x, (int(x.shape[0]/avg_gait_samples), 
            avg_gait_samples, x.shape[1]), 'C')
        if y is not None:
            y_return = np.reshape(y, (int(y.shape[0]/avg_gait_samples), 
                avg_gait_samples, y.shape[1]), 'C')
            return x_return, y_return

        return x_return

    def subsample(self, percent, cp1_x=None, cp1_y=None, cp2_x=None,
            cp2_y=None):
        #perhaps i am writing somethign that is 3 lines of code in 576 lines
        #but do i care?


        avg_gait_samples = self.cp1.avg_gait_samples

        if cp1_x is None:
            cp1_x = self.data['cp1']['proc_x']
        if cp1_y is None:
            cp1_y = self.data['cp1']['proc_y']
        if cp2_x is None:
            cp2_x = self.data['cp2']['proc_x']
        if cp2_y is None:
            cp2_y = self.data['cp2']['proc_y']

        if percent==1.0:
            return cp1_x, cp1_y, cp2_x, cp2_y
        temp_x1, temp_y1 = self.back_to_gait(x=cp1_x, y=cp1_y)
        temp_x2, temp_y2 = self.back_to_gait(x=cp2_x, y=cp2_y)

        subsize = int(percent * temp_x1.shape[0])
        temp = np.random.choice(temp_x1.shape[0], size=subsize, replace=False)
        temp.sort()

        my_list = [temp_x1, temp_y1, temp_x2, temp_y2]
        new_array = []
        for array in my_list:
            temp_array = array[temp]
            total_samples = temp_array.shape[0] * temp_array.shape[1]
            new_array.append(np.reshape(temp_array, (total_samples,
                temp_array.shape[2])))
 
        return new_array[0], new_array[1], new_array[2], new_array[3]
