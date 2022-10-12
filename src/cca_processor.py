from src.neural_analysis import *
from src.cort_processor import *
from src.wiener_filter import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

class CCAProcessor:
    def __init__(self, cp1, cp2, limbfoot_angle=1, align=0):
        #align = 0 is sorting and stitching
        #align = 1 is resampling
        self.cp1 = cp1
        self.cp1.get_gait_indices(angle_number=limbfoot_angle)
        
        self.cp2 = cp2
        self.cp2.get_gait_indices(angle_number=limbfoot_angle)
        self.limbfoot_angle=limbfoot_angle
        
        self.data = {}
        self.data['cp1'] = {}
        self.data['cp2'] = {}

        if align==0:

            self.data['cp1']['proc_x'], self.data['cp1']['proc_y'],\
            self.data['cp2']['proc_x'], self.data['cp2']['proc_y'] =\
            self.sort_and_align()

        elif align==1:

            self.data['cp1']['proc_x'], self.data['cp1']['proc_y'],\
            self.data['cp2']['proc_x'], self.data['cp2']['proc_y'] =\
            self.process_and_align_kinematics()



        
       # self.data['cp1']['h'], self.data['cp1']['proc_vaf'], nada, nada =\
       # self.cp1.decode_angles(X=[self.data['cp1']['proc_x']],\
       # Y=[self.data['cp1']['proc_y']])

       # self.data['cp2']['h'], self.data['cp2']['proc_vaf'], nada, nada =\
       # self.cp2.decode_angles(X=[self.data['cp2']['proc_x']],\
       # Y=[self.data['cp2']['proc_y']])

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

        self.data['cp1']['pca_transformer'] = pca_cp1
        self.data['cp2']['pca_transformer'] = pca_cp2

        return self.data['cp1']['pca_x'], self.data['cp2']['pca_x']

    def apply_CCA_clean(self, cp1_x=None, cp2_x=None, transformer=None):
        if cp1_x is None:
            if transformer is None:
                cp1_x = self.data['cp1']['proc_x']        
        if cp2_x is None:
            cp2_x = self.data['cp2']['proc_x']

        num_components = cp2_x.shape[0]

        if transformer is None:
            cca_cp1cp2 = CCA(n_components = num_components, scale=False)
            x1_cca, x2_cca=cca_cp1cp2.fit_transform(cp1_x, cp2_x)
        else:
            cca_cp1cp2 = transformer
            nada, x2_cca = cca_cp1cp2.transform(cp2_x, cp2_x)

        self.cca = cca_cp1cp2
        self.x2_cca = x2_cca

        x2_into_x1 = cca_cp1cp2.inverse_transform(x2_cca)

        return cca_cp1cp2, x2_into_x1


    def apply_CCA(self, cp1_x=None, cp2_x=None, preset_num_components=None,
            transformer=None, no_pca=False):
        if preset_num_components is None:
            if no_pca:
                pass
            else:
                try:
                    num_components = self.num_components
                except:
                    print('set num copmoentns')
        else:
            num_components = preset_num_components

        if cp1_x is None:
            if transformer is None:
                if no_pca is True:
                    cp1_x = self.data['cp1']['proc_x']
                else:
                    cp1_x = self.data['cp1']['pca_x']
        if cp2_x is None:
            if no_pca is True:
                cp2_x = self.data['cp2']['proc_x']
            else:
                cp2_x = self.data['cp2']['pca_x']

        if transformer is None:
            cca_cp1cp2 = CCA(n_components = num_components, scale=False)
            x1_cca, x2_cca=cca_cp1cp2.fit_transform(cp1_x, cp2_x)
        else:
            cca_cp1cp2 = transformer
            nada, x2_cca = cca_cp1cp2.transform(cp2_x, cp2_x)

        self.cca = cca_cp1cp2
        self.x2_cca = x2_cca

        x2_into_x1 = cca_cp1cp2.inverse_transform(x2_cca)

        return cca_cp1cp2, x2_into_x1

    def process_and_align_kinematics(self):

        print(self.check_same_kinematics())

        avg_samples = self.cp2.avg_gait_samples #arbitrarily grab from 1
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

    def sort_and_align(self, sample_variance =3):

        print(self.check_same_kinematics())
        
        cp1_gait_x, cp1_gait_y =\
        self.cp1.divide_into_gaits(bool_resample=False)

        cp2_gait_x, cp2_gait_y =\
        self.cp2.divide_into_gaits(bool_resample=False)

        cp1_gait_x = cp1_gait_x[0]
        cp1_gait_y = cp1_gait_y[0]
        cp2_gait_x = cp2_gait_x[0]
        cp2_gait_y = cp2_gait_y[0]
        
        cp1_avg_gaits = self.cp2.avg_gait_samples

        cp1_x_sortdict={}
        cp1_y_sortdict={}
        cp2_x_sortdict={}
        cp2_y_sortdict={}

        var_range = range(cp1_avg_gaits-sample_variance,
        cp1_avg_gaits+sample_variance+1)

        for i in var_range:
            cp1_x_sortdict[i]=[]
            cp1_y_sortdict[i]=[]
            cp2_x_sortdict[i]=[]
            cp2_y_sortdict[i]=[]
            for idx in range(len(cp1_gait_x)):
                if len(cp1_gait_x[idx])==i:
                    cp1_x_sortdict[i].append(cp1_gait_x[idx])
                    cp1_y_sortdict[i].append(cp1_gait_y[idx])
            for idx in range(len(cp2_gait_x)):
                if len(cp2_gait_x[idx])==i:
                    cp2_x_sortdict[i].append(cp2_gait_x[idx])
                    cp2_y_sortdict[i].append(cp2_gait_y[idx])

        cp1_final_x = []
        cp1_final_y = []
        cp2_final_x = []
        cp2_final_y = []
        for i in var_range:
            num = min(len(cp1_x_sortdict[i]), len(cp2_x_sortdict[i]))

            for j in range(num):
                cp1_final_x.append(cp1_x_sortdict[i][j])
                cp1_final_y.append(cp1_y_sortdict[i][j])
                cp2_final_x.append(cp2_x_sortdict[i][j])
                cp2_final_y.append(cp2_y_sortdict[i][j])

        return np.concatenate(cp1_final_x), np.concatenate(cp1_final_y), np.concatenate(cp2_final_x), np.concatenate(cp2_final_y)

        

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

        subsize = int(percent * cp1_x.shape[0])

        my_list = [cp1_x, cp1_y, cp2_x, cp2_y]
        new_array = []
        for array in my_list:
            new_array.append(array[:subsize, :])
 
        return new_array[0], new_array[1], new_array[2], new_array[3]

    def apply_ridge(self, reduce_dims=False, angle=3):
        if reduce_dims:
            x1 = self.data['cp1']['pca_x']
            x2 = self.data['cp2']['pca_x']
        else:
            x1 = self.data['cp1']['proc_x']
            x2 = self.data['cp2']['proc_x']
        
        y1 = self.data['cp1']['proc_y']
        y2 = self.data['cp2']['proc_y']
        

        b0, nada, nada, nada = self.cp1.decode_angles(scale=True) 
        transformer, x2_cca = self.apply_CCA_clean(cp1_x = x1, cp2_x = x2)
        scaler=StandardScaler()
        x2_scca = scaler.fit_transform(x2_cca)
        
        x2_scca_format, y2_format = format_data(x2_scca, y2)

        wpost, ywpost = ridge_fit(b0, x2_scca_format, y2_format, my_alpha=100,
                angle=angle)

        return transformer, wpost, ywpost

    def quick_cca(self, x, transformer, scale=True):
        nada, temp=transformer.transform(x,x)
        temp2 = transformer.inverse_transform(temp)
        scaler = StandardScaler()
        return scaler.fit_transform(temp2)
