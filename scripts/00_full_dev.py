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
from functions import fitting


#%% Define input files
# Input directory and the labelled protein fraction in each of them
in_dir_names= []
alpha_label = []
# glob_dir = '/fs/pool/pool-schwille-spt/Experiment_analysis/20231117_JHK_NKaletta_ParM_oligomerization'
glob_dir = r'D:\Testdata'


''' Labelled protein fraction'''
in_dir_names.extend([os.path.join(glob_dir, 'Calibration_AF488_23C_1')])
alpha_label.append(1.) # 50 nM in 20 µM

# Naming pattern for detecting correct files within subdirs of each in_dir
file_name_pattern_PCH = '*PCMH_ch0_ch1*' # Dual-channel PCH
file_name_pattern_FCS = '*CCF_symm_ch0_ch1_bg*' # CCF

#%% Fit settings

### FCS settings

# Shortest and longest lag time to fit
FCS_min_lag_time = 1E-6
FCS_max_lag_time = 1. 


#%% Metadata/calibration data
FCS_psf_width_nm = 210. # Roughly
FCS_psf_aspect_ratio = 6. # Roughly

acquisition_time_s = 90.

PCH_Q = 8. # More calculation parameter than metadata, but whatever



#%% Define output directories/files

# Output dir for result file writing
save_path = os.path.join(glob_dir, 'TestDebugFits')

# .csv table for collecting fit results
results_table_path = os.path.join(save_path, 'fit_param_table.csv')




#%% Input interpretation

# Automatic input file detection

# Detect PCH files
in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(in_dir_names,
                                                                      file_name_pattern_PCH, 
                                                                      alpha_label, 
                                                                      save_path)

# Repeat for FCS
in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(in_dir_names, 
                                                            file_name_pattern_FCS, 
                                                            alpha_label,
                                                            save_path)

in_dir_names, in_file_names_FCS, in_file_names_PCH, alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
                                                                                               in_file_names_FCS,
                                                                                               alpha_label_FCS,
                                                                                               in_dir_names_PCH,
                                                                                               in_file_names_PCH,
                                                                                               alpha_label_PCH)
    
# Prepare output
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    

# #%% Start processing
# for i_file, dir_name in enumerate(in_dir_names):
    
#     data_FCS_tau_s, data_FCS_G, avg_count_rate, data_FCS_sigma, acquisition_time_s = utils.read_Kristine_FCS(dir_name, 
#                                                                                                              in_file_names_FCS[i_file],
#                                                                                                              FCS_min_lag_time,
#                                                                                                              FCS_max_lag_time)
    
#     data_PCH_bin_times, data_PCH_hist = utils.read_PCMH(dir_name,
#                                                         in_file_names_PCH[i_file])
    
#     fitter = fitting.FCS_spectrum(FCS_psf_width_nm = FCS_psf_width_nm,
#                                   FCS_psf_aspect_ratio = FCS_psf_aspect_ratio,
#                                   PCH_Q = PCH_Q,
#                                   acquisition_time_s = acquisition_time_s if acquisition_time_s > 0 else 90., # Dummy for debugging with old-format data
#                                   data_FCS_tau_s = data_FCS_tau_s,
#                                   data_FCS_G = data_FCS_G,
#                                   data_FCS_sigma = data_FCS_sigma,
#                                   data_PCH_bin_times = None,
#                                   data_PCH_hist = None,
#                                   labelling_efficiency = 1.,
#                                   numeric_precision = 1E-4
#                                   )
    
    