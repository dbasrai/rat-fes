import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import numpy as np
from src.wiener_filter import *
from matplotlib.pyplot import cm

def plot_raster(df):
    fig=plt.figure()
    ax= fig.add_subplot(111)
    ax.set_axis_off()
    ax.table(cellColours=plt.cm.Greens(np.array(df)),rowLabels = df.index,
            colLabels = df.columns, cellLoc='center', loc = 'upper left')
    

def plot_gait_state_space_3D(list_of_array, subsample=5):
    #should be in gaits x gait samples x pca_dimensions
    #ONLY 2D FOR NOW
    #this is a little hard to explaijection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    average_gait = np.average(array, axis=0)
    random_sampling = np.random.randint(0, array.shape[0], subsample)
    gait_sampling = np.vstack(array[random_sampling,:,:])

    ax.scatter3D(average_gait[:,0], average_gait[:,1], average_gait[:,2], color='blue')
    ax.plot(average_gait[:,0], average_gait[:,1], average_gait[:,2], color='blue')

    ax.plot(gait_sampling[:,0], gait_sampling[:,1], gait_sampling[:,2], alpha=0.2, color='blue')
 
    return fig, ax

def plot_gait_state_space_2D(list_of_array, subsample=5):
    #should be in gaits x gait samples x pca_dimensions
    #ONLY 2D FOR NOW
    #this is a little hard to explain
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = iter(cm.rainbow(np.linspace(0, 1, len(list_of_array))))
    for array in list_of_array:
        c_current = next(color)
        average_gait = np.average(array, axis=0)
        random_sampling = np.random.randint(0, array.shape[0], subsample)
        gait_sampling = np.vstack(array[random_sampling,:,:])

        ax.scatter(average_gait[:,0], average_gait[:,1], color=c_current)
        ax.plot(average_gait[:,0], average_gait[:,1], color=c_current)

        ax.plot(gait_sampling[:,0], gait_sampling[:,1], alpha=0.2, color=c_current)
         

def plot_wiener_filter_predic(test_x, test_y, h):
    predic_y = test_wiener_filter(test_x, h)
    vaffy = vaf(test_y, predic_y)
    
    samples = np.shape(test_y)[0]

    ts = np.linspace(0, (samples*50)/1000,
            samples)

    fig, ax = plt.subplots()
    ax.set_title(f'vaf:{vaffy}')
    ax.plot(ts, test_y, c='black')
    ax.plot(ts, predic_y, c='red')

def plot_both(array1, array2): #stupid function to save time tonight
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #for gait in array:
    #    ax.plot3D(gait[0,:], gait[1,:], gait[2,:], color='lightsteelblue')
    avg1 = np.average(array1, axis=0)
    avg1 = np.vstack((avg1.T, avg1[:,0].T)).T

    avg2 = np.average(array2, axis=0)
    avg2 = np.vstack((avg2.T, avg2[:,0].T)).T

    ax.plot3D(avg1[0,:], avg1[1,:], avg1[2,:], color='blue')
    ax.scatter(avg1[0,:], avg1[1,:], avg1[2,:], color='blue')
    
    ax.plot3D(avg2[0,:], avg2[1,:], avg2[2,:], color='orange')
    ax.scatter(avg2[0,:], avg2[1,:], avg2[2,:], color='orange')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    




