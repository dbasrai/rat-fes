import numpy as np
from src.filters import *
from src.wiener_filter import *
from src.phase_decoder_support import *
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

def decode_kfolds(X, Y, metric_angle, k=10, preset_h=None, forced_test_index = None):
    '''
    01/08/23
    inputs and outputs reorganized
    updates made to correct hard coded metric angle
    option to overrwrite selected final_test_indicies to accomodate test cases where 
    multiple decoders are used for a single prediction 
    '''
    kf = KFold(n_splits=k)

    h_list = []

    scores = np.zeros((Y.shape[1], k))
    index=0
    best_score=-10000000
    for train_index, test_index in kf.split(X):


        train_x, test_x = X[train_index, :], X[test_index, :]
        train_y, test_y = Y[train_index, :], Y[test_index, :]
        if preset_h is None:
            h=train_wiener_filter(train_x, train_y)
        else:
            h=preset_h
        predic_y = test_wiener_filter(test_x, h)
        for j in range(predic_y.shape[1]):
            scores[j, index] = corrcoef(test_y[:,j], predic_y[:,j])
            
            
        if scores[metric_angle, index] > best_score:
            best_score = scores[metric_angle, index]
            best_h = h
            final_test_x = test_x
            final_test_y = test_y
            final_test_index = test_index

        index = index+1
    
    if forced_test_index is not None:
        final_test_x = X[forced_test_index, :]
        final_test_y = Y[forced_test_index, :]
        final_train_x = np.delete(X, forced_test_index, axis=0)
        final_train_y = np.delete(Y, forced_test_index, axis=0)
        final_test_y = Y[forced_test_index]
        best_h=train_wiener_filter(final_train_x, final_train_y)
        final_test_index = forced_test_index
        
    
    return best_h, np.average(scores, 1), final_test_x, final_test_y, final_test_index


def decode_kfolds_single(X, Y, k=10, preset_h=None, forced_test_index = None):
    kf = KFold(n_splits=k)
    best_score=-1000000
    scores = []
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index], Y[test_index]
        if preset_h is None:
            h=train_wiener_filter(train_x, train_y)
        else:
            h=preset_h
        predic_y = test_wiener_filter(test_x, h)
        scores.append(corrcoef(test_y, predic_y))
            
        if scores[-1] > best_score:
            final_test_x = test_x
            final_test_y = test_y
            best_h = h
            final_test_index = test_index
            best_score = scores[-1]
        if forced_test_index is not None:
            final_test_x = X[forced_test_index,:]
            final_test_y = Y[forced_test_index]
    
    return best_h, np.average(scores), final_test_x, final_test_y, final_test_index

def parallel_decoder(X, Y1, Y2, k=10, forced_test_index = None, printing = False):
    kf = KFold(n_splits=k)
    best_score=-1000000
    circorr_average = []

    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y1, test_y1 = Y1[train_index], Y1[test_index]
        train_y2, test_y2 = Y2[train_index], Y2[test_index]
        h1=train_wiener_filter(train_x, train_y1)
        h2=train_wiener_filter(train_x, train_y2)
        pred_y1 = test_wiener_filter(test_x, h1)
        pred_y2 = test_wiener_filter(test_x, h2)
        arctan_test = arctan_fn(test_y1, test_y2)
        arctan_pred = arctan_fn(pred_y1, pred_y2)
        pred_y1_res, pred_y2_res = sine_and_cosine(arctan_pred)
        circorr = corrcoef(arctan_test, arctan_pred)
        
        if printing == True:    
            print(score)
            
        circorr_average.append(circorr)

        if circorr > best_score:
            final_test_x = test_x
            final_test_y1 = test_y1
            final_test_y1 = test_y1
            best_h1 = h1
            best_h2 = h2
            final_testarc = arctan_test
            final_predarc = arctan_pred
            best_score = circorr
            final_test_index = test_index
    if forced_test_index is not None:
        final_test_x = X[forced_test_index, :]
        final_train_x = np.delete(X, forced_test_index, axis=0)
        final_train_y1 = np.delete(Y1, forced_test_index)
        final_test_y1 = Y1[forced_test_index]
        final_train_y2 = np.delete(Y2, forced_test_index)
        final_test_y2 = Y2[forced_test_index]
        best_h1=train_wiener_filter(final_train_x, final_train_y1)
        best_h2=train_wiener_filter(final_train_x, final_train_y2)
        ft_pred_y1 = test_wiener_filter(final_test_x, best_h1)
        ft_pred_y2 = test_wiener_filter(final_test_x, best_h2)
        final_testarc = arctan_fn(final_test_y1, final_test_y2)
        final_predarc = arctan_fn(ft_pred_y1, ft_pred_y2)
    return np.average(circorr_average), best_h1, best_h2, final_test_x, final_testarc, final_predarc, final_test_index


def null_hyopthesis_test_a(X, Y1, Y2, k=10, boots=10):  
    bootstrapped_avg = []
    for i in range(0,boots):
        X = shuffle(X)
        circorr_average = []
        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(X):
            train_x, test_x = X[train_index, :], X[test_index,:]
            train_y1, test_y1 = Y1[train_index], Y1[test_index]
            train_y2, test_y2 = Y2[train_index], Y2[test_index]
            h1=train_wiener_filter(train_x, train_y1)
            h2=train_wiener_filter(train_x, train_y2)
            pred_y1 = test_wiener_filter(test_x, h1)
            pred_y2 = test_wiener_filter(test_x, h2)
            arctan_test = arctan_fn(test_y1, test_y2)
            arctan_pred = arctan_fn(pred_y1, pred_y2)
            pred_y1_res, pred_y2_res = sine_and_cosine(arctan_pred)
            circorr = corrcoef(arctan_test, arctan_pred)
            circorr_average.append(circorr)
        bootstrapped_avg.append(np.average(circorr_average))

    return np.average(bootstrapped_avg)

def null_hyopthesis_test_b(X, Y, metric_angle, k=10, boots=10):
    bootstrapped_avg = []
    for i in range(0,boots):
        X = shuffle(X)
        kf = KFold(n_splits=k)
        scores = np.zeros((Y.shape[1], k))
        index=0
        for train_index, test_index in kf.split(X):
            train_x, test_x = X[train_index, :], X[test_index, :]
            train_y, test_y = Y[train_index, :], Y[test_index, :]
            h=train_wiener_filter(train_x, train_y)
            predic_y = test_wiener_filter(test_x, h)
            for j in range(predic_y.shape[1]):
                scores[j, index] = corrcoef(test_y[:,j], predic_y[:,j])
            index = index+1
        bootstrapped_avg.append(np.mean(scores, axis = 1))        

    return np.mean(bootstrapped_avg, axis = 0)





def decode_kfolds_single_nonlinear(X, Y, k=10, forced_test_index = None):
    kf = KFold(n_splits=k)
    best_score=-1000000
    scores = []
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index], Y[test_index]
        h, lsq=train_nonlinear_wiener_filter(train_x, train_y)
        predic_y = test_nonlinear_wiener_filter(test_x, h, lsq)
        
        scores.append(corrcoef(test_y, predic_y))
            
        if scores[-1] > best_score:
            final_test_x = test_x
            final_test_y = test_y
            best_h = h
            best_lsq = lsq
            final_test_index = test_index
            best_score = score_average[-1]
        if forced_test_index is not None:
            final_test_x = X[forced_test_index,:]
            final_test_y = Y[forced_test_index]
    
    return best_h, best_lsq, np.average(score_average), final_test_x, final_test_y, final_test_index










