# overview

cort processor is the main class we will directly work with. It directly receives a data folder (using FolderHandler class), extracts TDT and Anipose raw files, and call a variety of methods that simultaneously process neural/kinematic data and keep it synced.

The main idea is if you're working with any dataset that includes both neural and kinematic data, you will be mostly using Cort Processor. At some point this will include handling EMG as well.

# Initializing cort processor:

To create a CortProcessor class, you can init by passing in the path to a data folder.

```session = CortProcessor('INSERT_PATH_TO_DATA_FOLDER_HERE')``

see in docs data_folder_layout.md for information on how to structure your data folder

# using cort processor:

The first two things cort_processor can do is extract raw data, and then process into firing rates and kinematics, all sampled at 20 hz. I suggest not doing this in a Jupyter notebook since its memory intensive. 

Instead, use scripts/process_cort.py to run these first few steps, and save the CortProcessor object as a pickle. Then you can load the pickle.

See scripts/using process cort for an example.

# cort processor methods

Once you initialized your CortProcessor object and processed the data into rates/angles, you can now run a variety of methods.

# basic logic on using Cort Processor

CortProcessor methods frequently handle two different use cases. If


