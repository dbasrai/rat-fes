import numpy as np
from src.filters import *
from src.wiener_filter import *
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def decode_kfolds(X, Y, k=10, metric =3):
    kf = KFold(n_splits=k)

    h_list = []

    vaf_array = np.zeros((Y.shape[1], k))
    index=0
    best_vaf=-1
    for train_index, test_index in kf.split(X):


        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index, :], Y[test_index, :]

        h=train_wiener_filter(train_x, train_y)
        predic_y = test_wiener_filter(test_x, h)
        
        for j in range(predic_y.shape[1]):
            vaf_array[j, index] = vaf(test_y[:,j], predic_y[:,j])
            
        if vaf_array[3, index] > best_vaf:
            best_vaf = vaf_array[3, index]
            best_h = h
            final_test_x = test_x
            final_test_y = test_y

        index = index+1
    
    return best_h, vaf_array, final_test_x, final_test_y

def decode_kfolds_single(X, Y, k=10):
    kf = KFold(n_splits=k)
    index=0
    best_vaf=-1
    vaf_average = []
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index], Y[test_index]

        h=train_wiener_filter(train_x, train_y)
        predic_y = test_wiener_filter(test_x, h)
        vaf_average.append(vaf(test_y, predic_y))

        if vaf_average[-1] > best_vaf:
            final_test_x = test_x
            final_test_y = test_y
            best_h = h

    print(np.array(vaf_average))
    
    return best_h, np.average(np.array(vaf_average)), final_test_x, final_test_y

def apply_PCA(X, dims):
    pca_output = PCA(n_components=dims, random_state=2020)
    pca_output.fit(X)

    X_pca_output = pca_output.transform(X)

    return X_pca_output, pca_output

