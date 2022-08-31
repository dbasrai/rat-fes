from src.neural_analysis import *
from src.cort_processor import *
from src.cca_processor import *
from src.wiener_filter import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *

import os

path = '/home/diya/Documents/rat-fes/data/filipe_data/N5'
filenames = os.listdir(path)

datasets = []
for file in filenames:
    datasets.append(path+file)

datasets.reverse()





