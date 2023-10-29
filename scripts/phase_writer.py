#!/usr/bin/env python
# coding: utf-8

# In[64]:


import os
import pandas as pd
import numpy as np
import random as rand
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras
from keras.models import load_model
from numba import cuda
from keras import backend as K
import gc

# In[62]:
usr = raw_input("user_ID:")


###MAKE SURE NUM MATCHES THE NUMBER USED TO TRAIN THE NETWORK######
Num = 100

stashed_model = './0_is_1_model.h5'


columns = ['elbow', 'shoulder', 'forelimb', 'finger_x', 'finger_y', 'finger_z', 
           'knuckle_x', 'knuckle_y', 'knuckle_z', 'wrist_x', 'wrist_y', 'wrist_z', 
           'elbow_x', 'elbow_y', 'elbow_z', 'shoulder_x', 'shoulder_y', 'shoulder_z', 
           'scapula_x', 'scapula_y', 'scapula_z', 'fnum']
###################################################################

def timeseries_dataset_from_array_malleable(
    data,
    offset_targets,
    start_positions,
    N,
    sampling_rate=1,
    batch_size=1,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None,
):
    if offset_targets is not None:
        targets = np.asarray(np.roll(offset_targets, -N, axis=0), dtype="int64")
    sequence_length = (N*2)+1

    start_index = 0
    end_index = len(data)
    
    index_dtype = "int64"

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()

    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)
    ).map(
        lambda i, positions: tf.range(
            positions[i],
            positions[i] + sequence_length * sampling_rate,
            sampling_rate,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)
        ).map(
            lambda i, positions: positions[i],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        target_ds = sequences_from_indices(
            targets, indices, start_index, end_index
        )
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset

def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset



def get_angles_list(file_path):
    angles_path = f'{file_path}/angles'
    unsorted = []
    time_list = []

    for file_name in os.listdir(angles_path):
        if file_name.endswith('.csv'):
            subject_name = file_name.split('_')[1]
            time_list.append(subject_name.split('-')[-1]) 
            unsorted.append(f'{angles_path}/{file_name}')

    angles_path_list = [unsorted for _, unsorted in
            sorted(zip(time_list, unsorted))]

    return angles_path_list

def get_coords_list(file_path):
    coords_path = f'{file_path}/pose-3d'
    unsorted = []
    time_list = []

    for file_name in os.listdir(coords_path):
        if file_name.endswith('.csv'):
            subject_name = file_name.split('_')[1]
            time_list.append(subject_name.split('-')[-1]) 
            unsorted.append(f'{coords_path}/{file_name}')

    coords_path_list = [unsorted for _, unsorted in
            sorted(zip(time_list, unsorted))]

    return coords_path_list




def analyzer(path):
    data = pd.read_csv(path, header=0)
    X = np.array(data[columns[:-1]])
    indexes = np.arrange(X.shape[0])[:-2*Num]
    phase_predictions = np.argmax(con.predict(timeseries_dataset_from_array_malleable(X, None, indexes, Num)), axis=1)
    classification = np.hstack([np.zeros(Num),phase_predictions,np.zeros(Num)])
    data['classification'] = classification
    data.to_csv(path, index=False)
    return




import getpass



path = input('path to rat:')
security_check = input('is this a secure filepath? (yes/no)')
if security_check in ['Yes', 'yes', 'Y', 'y', 'YES', 'yup', 'yeah', 'sure', 'you betcha']:
    usr = getpass.getuser()    
    pwd = getpass.getpass("{}'s password:".format(usr))
    cmd = "echo {} | iconv -f utf8 -t eucjp | kakasi -i euc -w | kakasi -i euc -Ha -Ka -Ja -Ea -ka".format(pwd)




print(path)


os.system(cmd)



path_to_rat = folder_path


folder_path = './rollie_polly'

file_list = ['0103','0112','0120','0203','0210','0216','0224','1129','1209','1216','0302','0313','tq0120','tq0210','tq0216','tq0224']


master_phase_list = []




trial_list = os.listdir(folder_path)


for trial in trial_list:
    file_path = f'{path_to_rat}/{trial}'
    angles_path_list = get_angles_list(file_path)
    coords_path_list = get_coords_list(file_path)
    phase_folder = os.path.exists(f'{folder_path}/{file}/phase')
    if not phase_folder:
        os.makedirs(f'{folder_path}/{file}/phase')
    for i in range(len(angles_path_list)):
        angle_path = angles_path_list[i]
        coords_path = coords_path_list[i]
        
        name = angles_path_list[i].split('/')[-1]
        name_b = angles_path_list[i].split('/')[-1]
        if name != name_b:
            raise ValueError('phase/angle recording session mismatch')
        
        df_1 = pd.read_csv(angle_path)
        df_2 = pd.read_csv(coords_path)
        df_1 = df_1.join(df_2.set_index('fnum'), on='fnum')
        df_1 = df_1[columns]
        df_1.to_csv(f'{folder_path}/{file}/phase/_{name}', index=False)
        master_phase_list.append(f'{folder_path}/{file}/phase/_{name}')
 



for i in range(len(master_phase_list)):
    con = load_model(stashed_model)
    analyzer(master_phase_list[i])
    K.clear_session()
    gc.collect()
    
    
    


# In[65]:

