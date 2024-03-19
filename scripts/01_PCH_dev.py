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
# For localizing FCS_Fixer
repo_dir = os.path.abspath('..')

# For data processing
sys.path.insert(0, repo_dir)
from functions import fitting
from functions import utils


#%% Define input files
# Input directory and the labelled protein fraction in each of them
in_dir_names= []
alpha_label = []
glob_dir = r'Y:\Data\D044_MT200\20240307_JHK_NK_ParM_oligomerization\Data.sptw'

''' Labelled protein fraction'''
in_dir_names.extend([glob_dir + r'\Calibration_AF488_23C_1'])
alpha_label.append(50 / 20000) # 50 nM in 20 µM

# Naming pattern for detecting correct files within subdirs of dir
# file_name_pattern = '*PCMH*.csv' # CCF
file_name_pattern = '12_PCMH0.csv' # CCF

#%% Define output directories/files

# Output dir for result file writing
save_path = glob_dir + '/PCH_testfit'

# .csv table for collecting fit results
results_table_path = os.path.join(save_path, f'_fit.csv')

#%% Input interpretation

# Automatic input file detection
# in_dir_names, in_file_names, alpha_label = utils.detect_files(in_dir_names, file_name_pattern, alpha_label, save_path)
# Containers for storing results
_in_file_names=[]
_in_dir_names = [] 
_other_info = []

# Iteration over directories
for i_dir, directory in enumerate(in_dir_names):
    
    # Iteration over files: Find all that match pattern, and list
    search_pattern = os.path.join(directory, '**/*' + file_name_pattern)
    print(search_pattern)
    for name in glob.glob(search_pattern, recursive = True):
        
        head, tail = os.path.split(name)
        tail = tail.strip('.csv')
        
        if head != save_path:
            _in_file_names.extend([tail])
            _in_dir_names.extend([head])
            _other_info.append(alpha_label[i_dir])

if len(_in_dir_names) == 0:
    raise ValueError('Could not detect any files.')

in_file_names = _in_file_names
in_dir_names = _in_dir_names
alpha_label = _other_info

# Prepare output
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    

#%% Start processing
for i_file, file_name in enumerate(in_file_names):
    
    # Build path to file, but leaving out .csv ending for handling reasons
    in_path = os.path.join(in_dir_names[i_file], file_name)

    # Load and unpack Kristine format FCS data
    data = pd.read_csv(in_path + '.csv', header = 0)

    
    