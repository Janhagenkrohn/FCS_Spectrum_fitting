# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:56:42 2024

@author: Krohn
"""

# I/O modules
import os
import sys
import pandas as pd

# Data processing modules
import numpy as np

# Plotting
import matplotlib.pyplot as plt

# Misc
import datetime
from itertools import cycle # used only in plotting
import traceback # Crash handling
import multiprocessing 
import lmfit # only used for type recognition


# Custom module
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils
from functions import fitting



#%% Define input files and output dir
# Input directory and the labelled protein fraction in each of them
in_dir_names = []
in_file_names_FCS = []
in_file_names_PCH = []
alpha_label = []

glob_in_dir = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora'
# glob_in_dir = '/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Data/'


#%% 20240423 dataset - AF488-EGFP mixtures 
# Good for PCH proof of concept stuff
# ''' Labelled protein fraction'''
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, r'20240423_Test_data\Test_data.sptw')


# [_in_dir_names.extend([os.path.join(local_dir, f'AF488_1nM_power{x}')]) for x in [4000]]
# [_alpha_label.append(1.) for x in [4000]]
# [_in_dir_names.extend([os.path.join(local_dir, f'AF488_1nM_power{x}')]) for x in [50, 150, 450, 1350, 4000]]
# [_alpha_label.append(1.) for x in [50, 150, 450, 1350, 4000]]
# [_in_dir_names.extend([os.path.join(local_dir, f'EGFP_3nM_power{x}')]) for x in [50, 150, 450, 1350, 4000]]
# [_alpha_label.append(1.) for x in [50, 150, 450, 1350, 4000]]
# _in_dir_names.extend([os.path.join(local_dir, r'EGFP_AF488_Mix_Dil3\EGFP_AF488_Mix_Dil3_1_T0s_1_20240610_1336')])
# _alpha_label.append(1.)

# # Naming pattern for detecting correct files within subdirs of each in_dir
# file_name_pattern_PCH = '*08_PCMH_ch0*' # Dual-channel PCH
# file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*' # CCF

# # Detect PCH files
# in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(_in_dir_names,
#                                                                           file_name_pattern_PCH, 
#                                                                           _alpha_label, 
#                                                                           '')

# # Repeat for FCS
# in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(_in_dir_names, 
#                                                                           file_name_pattern_FCS, 
#                                                                           _alpha_label,
#                                                                           '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
#                                                                                                    in_file_names_FCS,
#                                                                                                    alpha_label_FCS,
#                                                                                                    in_dir_names_PCH,
#                                                                                                    in_file_names_PCH,
#                                                                                                    alpha_label_PCH)

# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]



# #%% 20240425 dataset 1 - AF488-labelled DNA origami
# # Let's see what it's good for...
# ''' Labelled protein fraction'''
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, '20240423_Test_data/Test_data.sptw')
# [_in_dir_names.extend([os.path.join(local_dir, f'DNA_Origami_NAO5{x}power500')]) for x in ['_', '_rep2_']]
# [_alpha_label.append(1.) for x in ['_', '_rep2_']] # N_labels_max = 12!
# [_in_dir_names.extend([os.path.join(local_dir, f'DNA_Origami_NAO6{x}power500')]) for x in ['_', '_rep2_']]
# [_alpha_label.append(0.5) for x in ['_', '_rep2_']] # N_labels_max = 12!
# [_in_dir_names.extend([os.path.join(local_dir, f'DNA_Origami_NAO7{x}power500')]) for x in ['_', '_rep2_']]
# [_alpha_label.append(0.2) for x in ['_', '_rep2_']] # N_labels_max = 12!
# [_in_dir_names.extend([os.path.join(local_dir, f'DNA_Origami_NAO8{x}power500')]) for x in ['_', '_rep2_']]
# [_alpha_label.append(1.) for x in ['_', '_rep2_']] # N_labels_max = 3!
# [_in_dir_names.extend([os.path.join(local_dir, f'DNA_Origami_NAO9{x}power500')]) for x in ['_', '_rep2_']]
# [_alpha_label.append(1.) for x in ['_', '_rep2_']] # N_labels_max = 1!

# # Naming pattern for detecting correct files within subdirs of each in_dir
# file_name_pattern_PCH = '*08_PCMH_ch0*' # Dual-channel PCH
# file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*' # CCF


# # Detect PCH files
# in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(_in_dir_names,
#                                                                           file_name_pattern_PCH, 
#                                                                           _alpha_label, 
#                                                                           '')
# # Repeat for FCS
# in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(_in_dir_names, 
#                                                                           file_name_pattern_FCS, 
#                                                                           _alpha_label,
#                                                                           '')
# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
#                                                                                                in_file_names_FCS,
#                                                                                                alpha_label_FCS,
#                                                                                                in_dir_names_PCH,
#                                                                                                in_file_names_PCH,
#                                                                                                alpha_label_PCH)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]

# #%% 20240425 dataset 2 - QD565
# # Should be good for incomplete sampling stuff
# ''' Labelled protein fraction'''
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, '20240423_Test_data/Test_data.sptw')
# _in_dir_names.extend([os.path.join(local_dir, 'QD565_50nM_power50')])
# _alpha_label.append(1.)
# _in_dir_names.extend([os.path.join(local_dir, 'QD565_50nM_power150')])
# _alpha_label.append(1.)
# _in_dir_names.extend([os.path.join(local_dir, 'QD565_US_50nM_power50')])
# _alpha_label.append(1.)


# # Naming pattern for detecting correct files within subdirs of each in_dir
# # Change to channel 1!!!
# file_name_pattern_PCH = '*08_PCMH_ch0*' # Dual-channel PCH
# file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*' # CCF


# # Detect PCH files
# in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(_in_dir_names,
#                                                                           file_name_pattern_PCH, 
#                                                                           _alpha_label, 
#                                                                           '')

# # Repeat for FCS
# in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(_in_dir_names, 
#                                                                           file_name_pattern_FCS, 
#                                                                           _alpha_label,
#                                                                           '')
# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
#                                                                                                in_file_names_FCS,
#                                                                                                alpha_label_FCS,
#                                                                                                in_dir_names_PCH,
#                                                                                                in_file_names_PCH,
#                                                                                                alpha_label_PCH)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]



# #%% 20240515 dataset 1 - AF488 with high SNR
# # SHould be generically usable
# ''' Labelled protein fraction'''
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, '20240515_Test_data\Data.sptw')
# [_in_dir_names.extend([os.path.join(local_dir, f'DNA_Origami_NAO{x}_power500')]) for x in [5, 6, 7, 8, 9]]
# [_alpha_label.append(1.) for x in [5, 6, 7, 8, 9]]
# [_in_dir_names.extend([os.path.join(local_dir, f'DNA_Origami_NAO{x}_rep2_power500')]) for x in [5, 6, 7, 8, 9]]
# [_alpha_label.append(1.) for x in [5, 6, 7, 8, 9]]

# # Naming pattern for detecting correct files within subdirs of each in_dir
# file_name_pattern_PCH = '*08_PCMH_ch0*' # Dual-channel PCH
# file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*' # CCF


# # Detect PCH files
# in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(_in_dir_names,
#                                                                           file_name_pattern_PCH, 
#                                                                           _alpha_label, 
#                                                                           '')

# # Repeat for FCS
# in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(_in_dir_names, 
#                                                                           file_name_pattern_FCS, 
#                                                                           _alpha_label,
#                                                                           '')
# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
#                                                                                                in_file_names_FCS,
#                                                                                                alpha_label_FCS,
#                                                                                                in_dir_names_PCH,
#                                                                                                in_file_names_PCH,
#                                                                                                alpha_label_PCH)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]

# #%% 20240521 dataset 1 - A488-labelled SUVs
# # SHould be good for incomplete-sampling stuff
# ''' Labelled protein fraction'''
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, '20240521_Test_data/20240521.sptw')
# [_in_dir_names.extend([os.path.join(local_dir, f'SUVs1_{x}_labelling_1')]) for x in ['1e-4', '2e-5', '5e-4']]
# [_alpha_label.append(x) for x in [1e-4, 2e-5, 5e-4]]

# # Naming pattern for detecting correct files within subdirs of each in_dir
# file_name_pattern_PCH = '*08_PCMH_ch0*' # Dual-channel PCH
# file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*' # CCF


# # Detect PCH files
# in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(_in_dir_names,
#                                                                           file_name_pattern_PCH, 
#                                                                           _alpha_label, 
#                                                                           '')

# # Repeat for FCS
# in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(_in_dir_names, 
#                                                                           file_name_pattern_FCS, 
#                                                                           _alpha_label,
#                                                                           '')
# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
#                                                                                                in_file_names_FCS,
#                                                                                                alpha_label_FCS,
#                                                                                                in_dir_names_PCH,
#                                                                                                in_file_names_PCH,
#                                                                                                alpha_label_PCH)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]


# #%% 20240604 - A488-labelled DNA
# #SHould be good for incomplete-sampling stuff
# ''' Labelled protein fraction'''
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240604_Test_data\20240604_data.sptw\Sample9_DNAs17_1')
# # local_dir = os.path.join(glob_in_dir, r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240604_Test_data\20240604_data.sptw\Sample9_DNAs157_1')
# _in_dir_names.extend([os.path.join(local_dir)])
# _alpha_label.append(1.) 

# # Naming pattern for detecting correct files within subdirs of each in_dir
# file_name_pattern_PCH = '*08_PCMH_ch0*' # Dual-channel PCH
# file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*' # CCF


# # Detect PCH files
# in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(_in_dir_names,
#                                                                           file_name_pattern_PCH, 
#                                                                           _alpha_label, 
#                                                                           '')

# # Repeat for FCS
# in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(_in_dir_names, 
#                                                                           file_name_pattern_FCS, 
#                                                                           _alpha_label,
#                                                                           '')
# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
#                                                                                                 in_file_names_FCS,
#                                                                                                 alpha_label_FCS,
#                                                                                                 in_dir_names_PCH,
#                                                                                                 in_file_names_PCH,
#                                                                                                 alpha_label_PCH)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]





# #%% 20240604 - A488-labelled ParM
# SHould be good for incomplete-sampling stuff
''' Labelled protein fraction'''
_in_dir_names = []
_alpha_label = []

local_dir = os.path.join(glob_in_dir, r'20240416_JHK_NK_New_ParM_data\20240416_data.sptw')
# _in_dir_names.extend([os.path.join(local_dir, r'10uM_ParM_2\ParM_10uM_1in10k1_T0s_1_20240613_1016')])
# _in_dir_names.extend([os.path.join(local_dir, r'10uM_ParM_3\ParM_10uM_1in401_T0s_1_20240613_1435')])
# _in_dir_names.extend([os.path.join(local_dir, r'10uM_ParM_2\ParM_10uM_1in10k1_T0s_1_20240613_1016')])
# _alpha_label.append(0.001) 
_in_dir_names.extend([os.path.join(local_dir, r'10uM_ParM_3\ParM_10uM_1in401_T0s_1_20240613_1435')])
_alpha_label.append(0.025) 

# Naming pattern for detecting correct files within subdirs of each in_dir
file_name_pattern_PCH = '*07_PCMH_ch0*' # Dual-channel PCH
file_name_pattern_FCS = '*06_ACF_ch0_dt_bg*' # CCF


# Detect PCH files
in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(_in_dir_names,
                                                                          file_name_pattern_PCH, 
                                                                          _alpha_label, 
                                                                          '')

# Repeat for FCS
in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(_in_dir_names, 
                                                                          file_name_pattern_FCS, 
                                                                          _alpha_label,
                                                                          '')

_in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
                                                                                                in_file_names_FCS,
                                                                                                alpha_label_FCS,
                                                                                                in_dir_names_PCH,
                                                                                                in_file_names_PCH,
                                                                                                alpha_label_PCH)

[in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
[in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
[in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
[alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]






#%% Fit settings
# Output dir for result file writing
glob_out_dir = r'C:\Users\Krohn\Desktop\20240611_tempfits\reg_par_sequence_test'
# glob_out_dir = r'C:\Users\Krohn\Desktop\20240611_tempfits\DNA157_fitting'

### General model settings

labelling_correction_list = [False, True]

n_species_list = [80]
reg_spectrum_type_list = ['reg_MEM'] # 'reg_MEM', 'reg_CONTIN'
par_spectrum_type_list = ['par_StrExp'] # 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
par_spectrum_parameter_list = ['N_monomers'] # 'Amplitude', 'N_monomers', 'N_oligomers',
oligomer_type_list = ['single_filament'] # 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'

use_blinking_list = [False]


### FCS settings
use_FCS_list = [True]

# Shortest and longest diffusion time to fit (parameter bounds)
reg_tau_diff_min_list = [1E-6]
par_tau_diff_min_list = [2.1E-4]
tau_diff_max_list = [1E0]

# Shortest and longest lag time to consider in fit (time axis clipping)
FCS_min_lag_time_list = [1E-5] # Use 0. to use full range of data in .csv file
FCS_max_lag_time_list = [np.inf]  # Use np.inf to use full range of data in .csv file


### PCH settings
use_PCH_list = [False]
time_resolved_PCH_list = [False]

# Shortest and longest bin times to consider
PCH_min_bin_time_list = [0.] # Use 0. to use full range of data in .csv file
PCH_max_bin_time_list = [5E-4] # Use np.inf to use full range of data in .csv file

# Calculation settings
NLL_funcs_accurate_list = [False, True] # Accurate MLE or faster least-squares approximation (affects some, not all, likelihood terms)?
numeric_precision_list = [np.array([1E-3, 1E-4, 1E-5])] # PCH requires numerical precision cutoff, which is set here


#%% Metadata/calibration data/settings that are global for all measurements
verbosity = 3 # How much do you want the software to talk?
FCS_psf_width_nm = 210. # Roughly
FCS_psf_aspect_ratio = 6. # Roughly

acquisition_time_s = 90.

PCH_Q = 10. # More calculation parameter than metadata, but whatever

# How many parallel processes?
# If mp_processes <= 1, we use multiprocessing WITHIN the fit which allows acceleration of multi-species PCH
# If mp_processes > 1, we run multiple fits simultaneously, each in single-thread calculation
# mp_processes = os.cpu_count()-1 
mp_processes = 1 # no multiprocessing

#%% Wrap all permutations for different fit settings and all files...Long list!


# Iterate over all settings and files
list_of_parameter_tuples = []

for use_FCS in use_FCS_list:
    for FCS_min_lag_time in FCS_min_lag_time_list:
        for FCS_max_lag_time in FCS_max_lag_time_list:
            for use_PCH in use_PCH_list:
                for PCH_min_bin_time in PCH_min_bin_time_list:
                    for PCH_max_bin_time in PCH_max_bin_time_list:
                        for time_resolved_PCH in time_resolved_PCH_list:
                            for NLL_funcs_accurate in NLL_funcs_accurate_list:
                                for n_species in n_species_list:
                                    for reg_tau_diff_min in reg_tau_diff_min_list:
                                        for par_tau_diff_min in par_tau_diff_min_list:
                                            for tau_diff_max in tau_diff_max_list:
                                                for use_blinking in use_blinking_list:
                                                    for reg_spectrum_type in reg_spectrum_type_list:
                                                        for par_spectrum_type in par_spectrum_type_list:
                                                            for par_spectrum_parameter in par_spectrum_parameter_list:
                                                                for oligomer_type in oligomer_type_list:
                                                                    for labelling_correction in labelling_correction_list:
                                                                        for numeric_precision in numeric_precision_list:
                                                                    
                                                                            # a number of sanity-checks to make sure we do not waste time planning fits that cannot work:
                                                                            if (type(use_FCS) == bool and
                                                                                FCS_min_lag_time >= 0 and
                                                                                FCS_max_lag_time > FCS_min_lag_time and
                                                                                type(use_PCH) == bool and
                                                                                PCH_min_bin_time >= 0 and
                                                                                PCH_max_bin_time >= PCH_min_bin_time and
                                                                                type(time_resolved_PCH) == bool and
                                                                                type(NLL_funcs_accurate) == bool and
                                                                                utils.isint(n_species) and n_species > 10 and
                                                                                reg_tau_diff_min > 0 and
                                                                                par_tau_diff_min >= reg_tau_diff_min and
                                                                                tau_diff_max >= par_tau_diff_min and
                                                                                type(use_blinking) == bool and
                                                                                reg_spectrum_type in ['reg_MEM', 'reg_CONTIN'] and
                                                                                par_spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'] and
                                                                                par_spectrum_parameter in ['Amplitude', 'N_monomers', 'N_oligomers'] and
                                                                                oligomer_type in ['naive', 'spherical_shell', 'sherical_dense', 'single_filament', 'double_filament'] and
                                                                                type(labelling_correction) == bool and
                                                                                ((utils.isfloat(numeric_precision) and 
                                                                                      numeric_precision < 1. and 
                                                                                      numeric_precision > 0.) or
                                                                                  (utils.isiterable(numeric_precision) and 
                                                                                      np.all(numeric_precision < 1.) and 
                                                                                      np.all(numeric_precision > 0.)))
                                                                                ):
                                                                            
                                                                                fit_settings_str = f'{par_spectrum_type}_{n_species}spec'
                                                                                fit_settings_str += f'_{oligomer_type}_{par_spectrum_parameter}'
                                                                                fit_settings_str += '_blink' if use_blinking else ''
                                                                                fit_settings_str += '_lblcr' if labelling_correction else ''
                                                                                fit_settings_str += '_FCS' if use_FCS else ''
                                                                                fit_settings_str += ('_PCMH' if time_resolved_PCH else '_PCH') if use_PCH else ''
                                                                                fit_settings_str += ('_MLE' if NLL_funcs_accurate else '_WLSQ') if use_PCH else ''
                                                                                
                                                                                for i_file, dir_name in enumerate(in_dir_names):
                                                                                    job_prefix = in_file_names_FCS[i_file] + '_' + fit_settings_str
                                                                                    save_path = os.path.join(glob_out_dir, fit_settings_str)
                                                                                    if not os.path.exists(save_path):
                                                                                        os.makedirs(save_path)
                                                                                    
                                                                                    fit_res_table_path = os.path.join(save_path, 'Fit_params_' + fit_settings_str)
                                                                                    
                                                                                    parameter_tuple = (job_prefix,
                                                                                                        save_path,
                                                                                                        fit_res_table_path,
                                                                                                        dir_name,
                                                                                                        in_file_names_FCS[i_file],
                                                                                                        in_file_names_PCH[i_file],
                                                                                                        use_FCS,
                                                                                                        FCS_min_lag_time,
                                                                                                        FCS_max_lag_time,
                                                                                                        FCS_psf_width_nm,
                                                                                                        FCS_psf_aspect_ratio,
                                                                                                        use_PCH,
                                                                                                        PCH_min_bin_time,
                                                                                                        PCH_max_bin_time,
                                                                                                        PCH_Q,
                                                                                                        NLL_funcs_accurate,
                                                                                                        time_resolved_PCH,
                                                                                                        n_species,
                                                                                                        reg_tau_diff_min,
                                                                                                        par_tau_diff_min,
                                                                                                        tau_diff_max,
                                                                                                        use_blinking,
                                                                                                        reg_spectrum_type,
                                                                                                        par_spectrum_type,
                                                                                                        par_spectrum_parameter,
                                                                                                        oligomer_type,
                                                                                                        alpha_label[i_file],
                                                                                                        labelling_correction,
                                                                                                        numeric_precision,
                                                                                                        verbosity)
                                                                                    list_of_parameter_tuples.extend((parameter_tuple,))
                                                                            
                                                                            
                                                                            
#%% Parallel processing function definition


def fitting_parfunc(job_prefix,
                    save_path,
                    fit_res_table_path,
                    dir_name,
                    in_file_name_FCS,
                    in_file_name_PCH,
                    use_FCS,
                    FCS_min_lag_time,
                    FCS_max_lag_time,
                    FCS_psf_width_nm,
                    FCS_psf_aspect_ratio,
                    use_PCH,
                    PCH_min_bin_time,
                    PCH_max_bin_time,
                    PCH_Q,
                    PCH_fitting_accurate,
                    time_resolved_PCH,
                    n_species,
                    reg_tau_diff_min,
                    par_tau_diff_min,
                    tau_diff_max,
                    use_blinking,
                    reg_spectrum_type,
                    par_spectrum_type,
                    par_spectrum_parameter,
                    oligomer_type,
                    labelling_efficiency,
                    labelling_correction,
                    numeric_precision,
                    verbosity,
                    ):
    
    
    # Command line message
    time_tag = datetime.datetime.now()
    message = f'[{job_prefix}] Fitting '
    message += in_file_name_FCS if use_FCS else ''
    message += ' and ' if (use_FCS and use_PCH) else ''
    message += in_file_name_PCH if use_PCH else ''
    message += ' globally:' if (use_FCS and use_PCH) else ''
    print('\n' + time_tag.strftime("%Y-%m-%d %H:%M:%S") + '\n' + message)
    
    try:
        data_FCS_tau_s, data_FCS_G, avg_count_rate, data_FCS_sigma, acquisition_time_s = utils.read_Kristine_FCS(dir_name, 
                                                                                                                  in_file_name_FCS,
                                                                                                                  FCS_min_lag_time,
                                                                                                                  FCS_max_lag_time
                                                                                                                  )
        if use_PCH:
            data_PCH_bin_times, data_PCH_hist = utils.read_PCMH(dir_name,
                                                                in_file_name_PCH,
                                                                PCH_min_bin_time,
                                                                PCH_max_bin_time)
            
        fitter = fitting.FCS_spectrum(FCS_psf_width_nm = FCS_psf_width_nm,
                                      FCS_psf_aspect_ratio = FCS_psf_aspect_ratio,
                                      PCH_Q = PCH_Q,
                                      acquisition_time_s = 90., # Dummy for debugging with old-format data
                                      data_FCS_tau_s = data_FCS_tau_s if use_FCS else None,
                                      data_FCS_G = data_FCS_G if use_FCS else None,
                                      data_FCS_sigma = data_FCS_sigma if use_FCS else None,
                                      data_PCH_bin_times = data_PCH_bin_times if use_PCH else None,
                                      data_PCH_hist = data_PCH_hist if use_PCH else None,
                                      labelling_efficiency = labelling_efficiency,
                                      numeric_precision = numeric_precision,
                                      NLL_funcs_accurate = NLL_funcs_accurate,
                                      verbosity = verbosity,
                                      job_prefix = job_prefix
                                      )
        
        reg_fit_result, N_pop_array, par_fit_result = fitter.reg_par_fit_sequence(use_FCS, # bool
                                                                                  use_PCH, # bool
                                                                                  time_resolved_PCH, # bool
                                                                                  reg_spectrum_type, # 'reg_MEM', 'reg_CONTIN'
                                                                                  par_spectrum_type, # 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
                                                                                  par_spectrum_parameter, # 'Amplitude', 'N_monomers', 'N_oligomers',
                                                                                  oligomer_type, # spherical_shell', 'sherical_dense', 'single_filament', 'double_filament'
                                                                                  labelling_correction, # bool
                                                                                  n_species, # int
                                                                                  reg_tau_diff_min, # float
                                                                                  par_tau_diff_min,  # float
                                                                                  tau_diff_max, # float
                                                                                  use_blinking, # bool
                                                                                  i_bin_time = 0, # int
                                                                                  use_parallel = mp_processes <= 1 # Bool
                                                                                  )


        
        # Par fit result
        if hasattr(par_fit_result, 'params') and hasattr(par_fit_result, 'covar'): 
            # dirty workaround for essentia.lly testing "if type(par_fit_result) == lmfit.MinimizerResult", as MinimizerResult class cannot be explicitly referenced
            # Unpack fit result
            fit_params = par_fit_result.params
            covar = par_fit_result.covar
            # Recalculate number of species, it is possible we lose some in between
            n_species = fitter.get_n_species(fit_params)
                            
        else:
            raise Exception(f'Got a fit_result output with unsupported type. Expected lmfit.MinimizerResult, got {type(par_fit_result)}')

        out_name = os.path.join(save_path,
                                time_tag.strftime("%Y%m%d-%H%M%S") + f'{in_file_name_FCS if use_FCS else in_file_name_PCH}_fit_{par_spectrum_type}_{n_species}spec')
        
        # Command line preview of fit results
        print(f' [{job_prefix}]   Fitted parameters:')
        [print(f'[{job_prefix}] {key}: {fit_params[key].value}') for key in fit_params.keys() if fit_params[key].vary]
        
        # Write fit results
        fit_result_dict = {}            
        fit_result_dict['file'] = in_file_name_FCS if use_FCS else in_file_name_PCH 
        
        has_covar = not covar == None
        if has_covar:
            uncertainty_array = np.sqrt(np.diag(covar))
            covar_pointer = 0
        
        for key in fit_params.keys():
            # Fit parameters
            fit_result_dict[key + '_val'] = fit_params[key].value
            fit_result_dict[key + '_vary'] = 'Vary' if fit_params[key].vary else 'Fix_Dep'
            if not fit_params[key].stderr == None:
                fit_result_dict[key + '_err'] = fit_params[key].stderr
            elif has_covar and fit_params[key].vary:
                fit_result_dict[key + '_err'] = uncertainty_array[covar_pointer]
                
                
        fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
        
        fit_res_table_path_full = fit_res_table_path + '.csv'
        if not os.path.isfile(fit_res_table_path_full):
            # Does not yet exist - create with header
            fit_result_df.to_csv(fit_res_table_path_full, 
                                  header = True, 
                                  index = False)
        else:
            # Exists - append
            fit_result_df.to_csv(fit_res_table_path_full, 
                                  mode = 'a', 
                                  header = False, 
                                  index = False)
            
        # Additional spreadsheets for N and diffusion time spectra
        if n_species > 1:
            fit_res_N_path = fit_res_table_path + '_N.csv' # N-only spreadsheet for convenience
            fit_res_tau_diff_path = fit_res_table_path + '_tau_diff.csv' # tau_diff-only spreadsheet for convenience
            N_result_dict = {}
            tau_diff_max_result_dict = {}
            
            # Metadata
            N_result_dict['file'] = in_file_name_FCS if use_FCS else in_file_name_PCH 
            tau_diff_max_result_dict['file'] = in_file_name_FCS if use_FCS else in_file_name_PCH 
                
            # Species parameters
            for i_spec in range(n_species):
                N_result_dict[f'N_avg_pop_{i_spec}'] = fit_params[f'N_avg_pop_{i_spec}'].value
            for i_spec in range(n_species):
                N_result_dict[f'N_avg_obs_{i_spec}'] = fit_params[f'N_avg_obs_{i_spec}'].value
            for i_spec in range(n_species):
                tau_diff_max_result_dict[f'tau_diff_{i_spec}'] = fit_params[f'tau_diff_{i_spec}'].value
            
            fit_result_N_df = pd.DataFrame(N_result_dict, index = [1]) 
            fit_result_tau_diff_df = pd.DataFrame(tau_diff_max_result_dict, index = [1]) 


            if not os.path.isfile(fit_res_N_path):
                # Does not yet exist - create with header
                fit_result_N_df.to_csv(fit_res_N_path, 
                                        header = True, 
                                        index = False)
            else:
                # Exists - append
                fit_result_N_df.to_csv(fit_res_N_path, 
                                        mode = 'a', 
                                        header = False, 
                                        index = False)

            if not os.path.isfile(fit_res_tau_diff_path):
                # Does not yet exist - create with header
                fit_result_tau_diff_df.to_csv(fit_res_tau_diff_path, 
                                              header = True, 
                                              index = False)
            else:
                # Exists - append
                fit_result_tau_diff_df.to_csv(fit_res_tau_diff_path, 
                                              mode = 'a', 
                                              header = False, 
                                              index = False)

        # Show and write fits themselves
        if use_FCS:
            if not labelling_correction:
                model_FCS = fitter.get_acf_full_labelling_par(fit_params)

            else:
                model_FCS = fitter.get_acf_partial_labelling_par_reg(fit_params)

            # Plot FCS fit
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.semilogx(data_FCS_tau_s,
                        data_FCS_G, 
                        'dk')
            ax.semilogx(data_FCS_tau_s, 
                        data_FCS_G + data_FCS_sigma,
                        '-k', 
                        alpha = 0.7)
            ax.semilogx(data_FCS_tau_s, 
                        data_FCS_G - data_FCS_sigma,
                        '-k', 
                        alpha = 0.7)
            ax.semilogx(data_FCS_tau_s,
                        model_FCS, 
                        marker = '',
                        linestyle = '-', 
                        color = 'tab:gray')
                        
            try:
                fig.supxlabel('Correlation time [s]')
                fig.supylabel('G(\u03C4)')
            except:
                # Apparently this can fail???
                pass
            
            ax.set_xlim(data_FCS_tau_s[0], data_FCS_tau_s[-1])
            plot_y_min_max = (np.percentile(data_FCS_G, 3), np.percentile(data_FCS_G, 97))
            ax.set_ylim(0. if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                        plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)
    
            # Save and show figure
            plt.savefig(os.path.join(save_path, 
                                      out_name + '_FCS.png'), 
                        dpi=300)
            plt.show()
            
            
            # Write spreadsheet
            out_table = pd.DataFrame(data = {'Lagtime[s]':data_FCS_tau_s, 
                                              'Correlation': data_FCS_G,
                                              'Uncertainty_SD': data_FCS_sigma,
                                              'Fit': model_FCS})
            out_table.to_csv(os.path.join(save_path, 
                                          out_name + '_FCS.csv'),
                              index = False, 
                              header = True)
    
    
        if use_PCH:
    
            if not labelling_correction:
                model_PCH = fitter.get_pch_full_labelling(fit_params,
                                                          t_bin = data_PCH_bin_times[0],
                                                          spectrum_type = par_spectrum_type,
                                                          time_resolved_PCH = time_resolved_PCH,
                                                          crop_output = True,
                                                          numeric_precision = np.min(numeric_precision),
                                                          mp_pool = None
                                                          )
            else: 
                model_PCH = fitter.get_pch_partial_labelling(fit_params,
                                                             t_bin = data_PCH_bin_times[0],
                                                             time_resolved_PCH = time_resolved_PCH,
                                                             crop_output = True,
                                                             numeric_precision = np.min(numeric_precision),
                                                             mp_pool = None)
            
            # Plot PCH fit
            # Cycle through colors
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = cycle(prop_cycle.by_key()['color'])
            iter_color = next(colors)
    
            
            fig2, ax2 = plt.subplots(1, 1)
            ax2.semilogy(np.arange(0, data_PCH_hist.shape[0]),
                          data_PCH_hist[:,0],
                          marker = '.',
                          linestyle = 'none',
                          color = iter_color)
            
            ax2.semilogy(np.arange(0, model_PCH.shape[0]),
                          model_PCH * data_PCH_hist[:,0].sum(),
                          marker = '',
                          linestyle = '-',
                          color = iter_color)
            
            max_x = np.nonzero(data_PCH_hist[:,0])[0][-1]
            max_y = np.max(data_PCH_hist[:,0])
            
            if time_resolved_PCH:
                # PCMH
                for i_bin_time in range(1, data_PCH_bin_times.shape[0]):
                    iter_color = next(colors)
                    if not labelling_correction:
                        model_PCH = fitter.get_pch_full_labelling(fit_params,
                                                                  t_bin = data_PCH_bin_times[i_bin_time],
                                                                  spectrum_type = par_spectrum_type,
                                                                  time_resolved_PCH = time_resolved_PCH,
                                                                  crop_output = True,
                                                                  numeric_precision = np.min(numeric_precision),
                                                                  mp_pool = None
                                                                  )
                    else: 
                        model_PCH = fitter.get_pch_partial_labelling(fit_params,
                                                                      t_bin = data_PCH_bin_times[i_bin_time],
                                                                      time_resolved_PCH = time_resolved_PCH,
                                                                      crop_output = True,
                                                                      numeric_precision = np.min(numeric_precision),
                                                                      mp_pool = None)
                    
                    
                    ax2.semilogy(np.arange(0, data_PCH_hist.shape[0]),
                                  data_PCH_hist[:,i_bin_time],
                                  marker = '.',
                                  linestyle = 'none',
                                  color = iter_color)
                    
                    ax2.semilogy(np.arange(0, model_PCH.shape[0]),
                                  model_PCH * data_PCH_hist[:,i_bin_time].sum(),
                                  marker = '',
                                  linestyle = '-',
                                  color = iter_color)
            
                    max_x = np.max([max_x, np.nonzero(data_PCH_hist[:,i_bin_time])[0][-1]])
                    max_y = np.max([max_y, np.max(data_PCH_hist[:,i_bin_time])])
                    
                    
            ax2.set_xlim(-0.49, max_x + 1.49)
            ax2.set_ylim(0.3, max_y * 1.7)
            ax2.set_title('PCH fit')
            try:
                fig2.supxlabel('Photons in bin')
                fig2.supylabel('Counts')
            except:
                pass
            # Save and show figure
            plt.savefig(os.path.join(save_path, 
                                      out_name + '_PC'+ ('M' if time_resolved_PCH else '') +'H.png'), 
                        dpi=300)
            plt.show()

            # Write spreadsheet
            out_dict = {'Photons': np.arange(0, data_PCH_hist.shape[0]), 
                        str(data_PCH_bin_times[0]): data_PCH_hist[:,0]}
            if time_resolved_PCH:
                for i_bin_time in range(1, data_PCH_bin_times.shape[0]):
                    out_dict[str(data_PCH_bin_times[i_bin_time])] = data_PCH_hist[:,i_bin_time]
            out_table = pd.DataFrame(data = out_dict)
            out_table.to_csv(os.path.join(save_path, out_name + '_PC'+ ('M' if time_resolved_PCH else '') +'H.csv'),
                              index = False, 
                              header = True)
                
                
        else: # No valid fit result
            # Command line message
            message = f'[{job_prefix}] Failed to fit '
            message += in_file_name_FCS if use_FCS else ''
            message += ' and ' if (use_FCS and use_PCH) else ''
            message += in_file_name_PCH if use_PCH else ''
            print(message)
    except:
        traceback.print_exc()
        
    return None


#%% Preparations done- run fits

if mp_processes > 1:
    try:
        mp_pool = multiprocessing.Pool(processes = mp_processes)
        
        _ = [mp_pool.starmap(fitting_parfunc, list_of_parameter_tuples)]
    except:
        traceback.print_exception()
    finally:
        mp_pool.close()
        
else:
    # Single process analysis
    for i_fit in range(len(list_of_parameter_tuples)):
        _ = fitting_parfunc(*list_of_parameter_tuples[i_fit])