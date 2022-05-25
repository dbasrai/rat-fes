# describe each file

CLASS FILES:

folder_handler.py contains code for FolderHandler object. this looks through data folder and obtains 4 paths --- for TDT raw data, for camera numpy timestamps, for angles data, and for coords data.

cort_processor.py contains code for CortHandler object. this extracts data using paths giving to it by FolderHandler object, and then contains a variety of functions for manipulating data containing neural + kinematic data.

see cort_processor_description.md in docs for full info

MODULES:

plotter.py: plots a lot of generic plots I do a lot. 

neural_analysis.py: contains functions that work on 2D neural data in (samples x channels)structure. note: i initially wrote these with inverted dimensions, so thats why i transpose at the start of every function.

decoders.py: contains functions that train a variety of linear and non-linear decoding algortihms. Accepts generic [X: sample x features, Y: samples x outputs] data. I rarely call these functions directly, mostly use CortProcessor to first format data, then call it from there. 

filters.py: helper functions for generic filters. i mostly use to notch/bandpass filter neural data. 

wiener_filter.py: contains all functions for wiener_filter.

tdt_support.py: Contains functions for extracting raw TDT data, as well as for extracting anipose.csv files. more aptly named TDT_and_anipose_support.py 

utils.py: random helper functions
