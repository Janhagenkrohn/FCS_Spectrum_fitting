# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""
# I/O modules
import glob
import os
import pandas as pd

# Data processing modules
import numpy as np
import lmfit

# Plotting
import matplotlib.pyplot as plt


# Misc modules
import traceback
import multiprocessing
import sys

# Custom module
# For localizing module
repo_dir = os.path.abspath('..')

# For data processing
sys.path.insert(0, repo_dir)
from functions import fitting
from functions import utils


#%% Define input files
# Input directory and the labelled protein fraction in each of them
in_dir_names= []
alpha_label = []
glob_dir = r'U:\Data\D044_MT200\20240307_JHK_NK_ParM_oligomerization\Data.sptw'

in_dir_names.extend([glob_dir + r'\Calibration_AF488_23C_1'])

alpha_label.append(1.) #

# Naming pattern for detecting correct files within subdirs of dir
file_name_pattern_PCH = '*PCMH1*' # CCF
file_name_pattern_FCS = '*CCF_symm_ch0_ch*' # CCF


#%% Metadata
FCS_psf_width_nm = 210. # Roughly
FCS_psf_aspect_ratio = 6. # Roughly

acquisition_time_s = 90.

PCH_Q = 8. # More calculation parameter than metadata, but whatever

#%% Define output directories/files

# Output dir for result file writing◘
save_path = glob_dir + '/PCH_testfit'

# .csv table for collecting fit results
results_table_path = os.path.join(save_path, f'_fit.csv')

#%% Input interpretation

# Automatic input file detection
_in_dir_names, in_file_names_PCH, _alpha_label = utils.detect_files(in_dir_names,
                                                                  file_name_pattern_PCH, 
                                                                  alpha_label, 
                                                                  save_path)

# Repeat for FCS
_, in_file_names_FCS, _ = utils.detect_files(in_dir_names, 
                                             file_name_pattern_FCS, 
                                             alpha_label,
                                             save_path)


# Prepare output
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    

#%% Start processing
for i_file, file_name_FCS in enumerate(in_file_names_FCS):
    


    # Load and unpack Kristine format FCS data
    in_path_FCS = os.path.join(_in_dir_names[i_file], file_name_FCS)
    data_FCS = pd.read_csv(in_path_FCS + '.csv', header = 0)
        
    lag_times = data_FCS.iloc[:,0].to_numpy()
    G = data_FCS.iloc[:, 1].to_numpy()
    sigma_G = data_FCS.iloc[:, 3].to_numpy()
    avg_count_rate = data_FCS.iloc[0:2,2].mean() 
        # In case of cross-correlation, we average count rates of both channels for simplicity - not entirely accurate, but good enough I guess

    # Load and unpack PCH data
    in_path_PCH = os.path.join(_in_dir_names[i_file], in_file_names_PCH[i_file])
    data_PCH = pd.read_csv(in_path_PCH + '.csv', header = 0)
    
    bin_times = np.array([float(bin_time) for bin_time in data_PCH.keys()])
    
    fluctuation_analysis = fitting.FCS_spectrum(FCS_psf_width_nm = FCS_psf_width_nm,
                                                FCS_psf_aspect_ratio = FCS_psf_aspect_ratio,
                                                PCH_Q = PCH_Q,
                                                acquisition_time_s = acquisition_time_s,
                                                data_FCS_tau_s = lag_times,
                                                data_FCS_G = G,
                                                data_FCS_sigma = sigma_G,
                                                data_PCH_bin_times = bin_times,
                                                data_PCH_hist = data_PCH.to_numpy()
                                                )
    
    fit_results = []
    for i_bin_time in np.arange(0, bin_times.shape[0]):
        fit_results.append(fluctuation_analysis.run_simple_PCH_fit(i_bin_time))
    
    
    
    
    
    
    