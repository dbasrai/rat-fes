
import os
import pandas as pd
import numpy as np
import random as rand
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import keras
# from keras.models import load_model
# from numba import cuda
# from keras import backend as K
import gc
import getpass


import os
import pandas as pd
import numpy as np
import random as rand
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import keras
# from keras.models import load_model
# from numba import cuda
# from keras import backend as K
import gc
import getpass


# tmp_folder = os.path.exists(f'./tmp')
# if not tmp_folder:
#     os.makedirs(f'./tmp')
#     os.makedirs(f'./tmp/angles')
#     os.makedirs(f'./tmp/pose-3d')
#     os.makedirs(f'./tmp/phase')

    
    
path_to_rat = input('path to rat: ')
folder_path = f'{path_to_rat}/'
path_real = os.path.exists(folder_path)
if not path_real:
        raise ValueError('path does not exist: make sure filesystem is mounted and spelling is correct')
trial_list = os.listdir(folder_path)
    
usr = getpass.getuser()    
pwd = getpass.getpass("{}'s password: ".format(usr))
for trial in trial_list:
    file_path = f'{path_to_rat}/{trial}'
    angles_path_list = get_angles_list(file_path)
    coords_path_list = get_coords_list(file_path)
    copy_angles = "echo '{0}\n' | sudo cp -kS -r {1}/{2}/angles/.' ./tmp/angles/ ".format(pwd,path,trial)
    # os.system(copy_angles)
    copy_phase = "echo '{0}\n' | sudo cp -kS -r f'{1}/{2}/phase/.' ./tmp/phase/ ".format(pwd,path,trial)
    # os.system(copy_phase)
    
    
    
    call_chown = "echo '{0}\n' | sudo chown -kS -R {1} ./tmp".format(pwd,usr)
#     os.system(copy_angles)
    
    print(copy_angles)
    print(copy_phase)
    print(call_chown)
    
    
#     phase_folder = os.path.exists(f'{folder_path}/{file}/phase')
#     if not phase_folder:
#         os.makedirs(f'{folder_path}/{file}/phase')
#     for i in range(len(angles_path_list)):
#         angle_path = angles_path_list[i]
#         coords_path = coords_path_list[i]

#         name = angles_path_list[i].split('/')[-1]
#         name_b = angles_path_list[i].split('/')[-1]
#         if name != name_b:
#             raise ValueError('phase/angle recording session mismatch')

#         df_1 = pd.read_csv(angle_path)
#         df_2 = pd.read_csv(coords_path)
#         df_1 = df_1.join(df_2.set_index('fnum'), on='fnum')
#         df_1 = df_1[columns]
#         df_1.to_csv(f'{folder_path}/{file}/phase/_{name}', index=False)
#         master_phase_list.append(f'{folder_path}/{file}/phase/_{name}')


#     cmd = "echo '{}\n' | sudo -kS ls".format(pwd)
#     os.system(cmd)
    

    



# print(path)


# os.system(cmd)

# echo 'password' | sudo -kS ls



# path_to_rat = folder_path