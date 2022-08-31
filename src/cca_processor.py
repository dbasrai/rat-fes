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

    def PCA_to_same_dimensions(self, preset_num_components = None):
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

        self.data['cp1']['pca_x'] = pca_cp1.fit_transform(self.data['cp1']['proc_x'])
        self.data['cp2']['pca_x'] = pca_cp2.fit_transform(self.data['cp2']['proc_x'])

        return num_components, self.data['cp1']['pca_x'],\
                self.data['cp2']['pca_x']

    def CCA_cp2(self, cp1_x, cp2_x, preset_num_components=None):
        if preset_num_components is None:
            try:
                num_components = self.num_components
            except:
                print('set num copmoentns')
        else:
            num_components = preset_num_components

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


