# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""
# I/O modules
import glob
import os
import pandas as pd
import sys

# Data processing modules
import numpy as np
import lmfit

# Plotting
import matplotlib.pyplot as plt


# Misc modules
import traceback
import multiprocessing

# Custom module
# For localizing module
repo_dir = os.path.abspath('..')

# For data processing
sys.path.insert(0, repo_dir)
from functions import utils


#%% Define input files
# Input directory and the labelled protein fraction in each of them
in_dir_names= []
alpha_label = []
glob_dir = '/fs/pool/pool-schwille-spt/Experiment_analysis/20231117_JHK_NKaletta_ParM_oligomerization'


''' Labelled protein fraction'''
in_dir_names.extend([glob_dir + '/ParM_ATP_1'])
alpha_label.append(50 / 20000) # 50 nM in 20 µM

# Naming pattern for detecting correct files within subdirs of dir
# file_name_pattern = '_ACF_ch0_dt_ar_bg.csv', # ACF ch0
# file_name_pattern = '_ACF_ch1_dt_ar_bg.csv', # ACF ch1
file_name_pattern = '_CCF_symm_ch0_ch1_dt_ar_bg.csv' # CCF




#%% Define output directories/files

# Output dir for result file writing
save_path = glob_dir + '/Multicomponent1_Blink_FreeDye'

# .csv table for collecting fit results
results_table_path = os.path.join(save_path, f'__{n_components}comp_fit_param_table.csv')




#%% Input interpretation

# Automatic input file detection
in_dir_names, in_file_names, alpha_label = utils.detect_files(in_dir_names, file_name_pattern, alpha_label, save_path)


# Prepare output
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    

#%% Start processing
for i_file, file_name in enumerate(in_file_names):
    
    # Build path to file, but leaving out .csv ending for handling reasons
    in_path = os.path.join(in_dir_names[i_file], file_name)

    # Load and unpack Kristine format FCS data
    data = pd.read_csv(in_path + '.csv', header = None)
        
    lag_times = data.iloc[:,0].to_numpy()
    G = data.iloc[:, 1].to_numpy()
    sigma_G = data.iloc[:, 3].to_numpy()
    avg_count_rate = data.iloc[0:2,2].mean() 
        # In case of cross-correlation, we average count rates of both channels for simplicity - not entirely accurate, but good enough I guess

    lag_time_mask = np.logical_and(lag_times > tau_domain[0], 
                               lag_times < tau_domain[1])

    lag_times_masked = lag_times[lag_time_mask]
    G_masked = G[lag_time_mask]
    sigma_G_masked = sigma_G[lag_time_mask]