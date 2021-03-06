import tdt
import time
import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def classif_accuracy(array1, array2):
    return np.divide(np.sum(array1==array2), np.size(array1))
