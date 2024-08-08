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

# glob_in_dir = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora'
# glob_in_dir = '/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Data/D044_MT200_Naora/'
glob_in_dir = '/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Analysis/20240731_FCS_Testdata_export'








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
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# # Repeat for FCS
# in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
#                                                                                       file_name_pattern_FCS, 
#                                                                                       _alpha_label,
#                                                                                       '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                    in_file_names_FCS_tmp,
#                                                                                                    alpha_label_FCS_tmp,
#                                                                                                    in_dir_names_PCH_tmp,
#                                                                                                    in_file_names_PCH_tmp,
#                                                                                                    alpha_label_PCH_tmp)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]




# #%% 20240521 dataset 1 - A488-labelled SUVs
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, '20240521_Test_data/20240521.sptw')
# [_in_dir_names.extend([os.path.join(local_dir, f'SUVs1_{x}_labelling_1')]) for x in ['1e-4', '2e-5', '5e-4']]
# [_alpha_label.append(x) for x in [1e-4, 2e-5, 5e-4]]

# # Naming pattern for detecting correct files within subdirs of each in_dir
# file_name_pattern_PCH = '*08_PCMH_ch0*' # Dual-channel PCH
# file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*' # CCF


# # Detect PCH files
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# # Repeat for FCS
# in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
#                                                                                       file_name_pattern_FCS, 
#                                                                                       _alpha_label,
#                                                                                       '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                    in_file_names_FCS_tmp,
#                                                                                                    alpha_label_FCS_tmp,
#                                                                                                    in_dir_names_PCH_tmp,
#                                                                                                    in_file_names_PCH_tmp,
#                                                                                                    alpha_label_PCH_tmp)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]


#%% 20240604 - A488-labelled dsDNA
_in_dir_names = []
_alpha_label = []

local_dir = os.path.join(glob_in_dir, r'20240604_Test_data/20240604_data.sptw/')
[_in_dir_names.extend([os.path.join(local_dir, f'Sample{x+1}_DNA{x}_only_1')]) for x in range(1, 8)]
[_alpha_label.append(0.025) for x in range(1, 8)]
_in_dir_names.extend([os.path.join(local_dir, 'Sample9_DNAs17_1')])
_alpha_label.append(0.025)
_in_dir_names.extend([os.path.join(local_dir, 'Sample10_DNAs157_1')])
_alpha_label.append(0.025)
_in_dir_names.extend([os.path.join(local_dir, 'Sample11_DNAs1234567_1')])
_alpha_label.append(0.025)

# Naming pattern for detecting correct files within subdirs of each in_dir
# Wohland SD
file_name_pattern_FCS = '*_ACF_ch0_bg*' 

# We do not use PCH right now
# # Detect PCH files
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# Repeat for FCS
in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
                                                                                      file_name_pattern_FCS, 
                                                                                      _alpha_label,
                                                                                      '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                     in_file_names_FCS_tmp,
#                                                                                                     alpha_label_FCS_tmp,
#                                                                                                     in_dir_names_PCH_tmp,
#                                                                                                     in_file_names_PCH_tmp,
#                                                                                                     alpha_label_PCH_tmp)

[in_dir_names.append(in_dir_name) for in_dir_name in in_dir_names_FCS_tmp]
[in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in in_file_names_FCS_tmp]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
[alpha_label.append(single_alpha_label) for single_alpha_label in alpha_label_FCS_tmp]



#%% 20240604 - A488-labelled ssDNA
_in_dir_names = []
_alpha_label = []

local_dir = os.path.join(glob_in_dir, '20240627_ssDNA_ssRNA_samples/TestData.sptw')
[_in_dir_names.extend([os.path.join(local_dir, f'ssDNA{x}_1in50_1')]) for x in range(1, 8)]
[_alpha_label.append(0.025) for x in range(1, 8)]
[_in_dir_names.extend([os.path.join(local_dir, f'ssDNAmix_125_{4//x}-4-{4*x}_1')]) for x in [1, 2, 4]]
[_alpha_label.append(0.025) for x in [1, 2, 4]]

# Naming pattern for detecting correct files within subdirs of each in_dir
# Wohland SD
file_name_pattern_FCS = '*_ACF_ch0_bg*' 


# We do not use PCH right now
# # Detect PCH files
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# Repeat for FCS
in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
                                                                                      file_name_pattern_FCS, 
                                                                                      _alpha_label,
                                                                                      '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                     in_file_names_FCS_tmp,
#                                                                                                     alpha_label_FCS_tmp,
#                                                                                                     in_dir_names_PCH_tmp,
#                                                                                                     in_file_names_PCH_tmp,
#                                                                                                     alpha_label_PCH_tmp)

[in_dir_names.append(in_dir_name) for in_dir_name in in_dir_names_FCS_tmp]
[in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in in_file_names_FCS_tmp]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
[alpha_label.append(single_alpha_label) for single_alpha_label in alpha_label_FCS_tmp]


#%% 20240604 - A488-labelled ssRNA
_in_dir_names = []
_alpha_label = []

local_dir = os.path.join(glob_in_dir, '20240627_ssDNA_ssRNA_samples/TestData.sptw')
[_in_dir_names.extend([os.path.join(local_dir, f'0628_ssRNA{x}_1in50_1')]) for x in range(1, 9)]
[_alpha_label.append(0.025) for x in range(1, 9)]

# Naming pattern for detecting correct files within subdirs of each in_dir
# Wohland SD
file_name_pattern_FCS = '*_ACF_ch0_bg*' 


# We do not use PCH right now
# # Detect PCH files
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# Repeat for FCS
in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
                                                                                      file_name_pattern_FCS, 
                                                                                      _alpha_label,
                                                                                      '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                     in_file_names_FCS_tmp,
#                                                                                                     alpha_label_FCS_tmp,
#                                                                                                     in_dir_names_PCH_tmp,
#                                                                                                     in_file_names_PCH_tmp,
#                                                                                                     alpha_label_PCH_tmp)

[in_dir_names.append(in_dir_name) for in_dir_name in in_dir_names_FCS_tmp]
[in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in in_file_names_FCS_tmp]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
[alpha_label.append(single_alpha_label) for single_alpha_label in alpha_label_FCS_tmp]



#%% 20240730 - A488-labelled ssRNA LADDERS - day 1
_in_dir_names = []
_alpha_label = []

local_dir = os.path.join(glob_in_dir, '20240730_JHK_ssRNA_ladder_run1/20240730_data.sptw')
[_in_dir_names.extend([os.path.join(local_dir, f'ssRNA_ladder_lim_{x}uM_1')]) for x in [20, 50]]
[_in_dir_names.extend([os.path.join(local_dir, f'ssRNA_ladder_RCT_{x}uM_1')]) for x in [20, 50]]
[_alpha_label.append(0.025) for x in range(4)]

# Naming pattern for detecting correct files within subdirs of each in_dir
# Wohland SD
file_name_pattern_FCS = '*_ACF_ch0_bg*' 


# We do not use PCH right now
# # Detect PCH files
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# Repeat for FCS
in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
                                                                                      file_name_pattern_FCS, 
                                                                                      _alpha_label,
                                                                                      '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                     in_file_names_FCS_tmp,
#                                                                                                     alpha_label_FCS_tmp,
#                                                                                                     in_dir_names_PCH_tmp,
#                                                                                                     in_file_names_PCH_tmp,
#                                                                                                     alpha_label_PCH_tmp)

[in_dir_names.append(in_dir_name) for in_dir_name in in_dir_names_FCS_tmp]
[in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in in_file_names_FCS_tmp]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
[alpha_label.append(single_alpha_label) for single_alpha_label in alpha_label_FCS_tmp]


#%% 20240801 - A488-labelled ssRNA LADDERS - day 2
_in_dir_names = []
_alpha_label = []

local_dir = os.path.join(glob_in_dir, '20240801_JHK_ssRNA_ladder_run2/20240801_data.sptw')
[_in_dir_names.extend([os.path.join(local_dir, f'ssRNA_ladder_lim_{x}uM_1')]) for x in [100, 200]]
[_alpha_label.append(0.025) for x in [100, 200]]

# Naming pattern for detecting correct files within subdirs of each in_dir
# Wohland SD
file_name_pattern_FCS = '*_ACF_ch0_bg*' 


# We do not use PCH right now
# # Detect PCH files
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# Repeat for FCS
in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
                                                                                      file_name_pattern_FCS, 
                                                                                      _alpha_label,
                                                                                      '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                     in_file_names_FCS_tmp,
#                                                                                                     alpha_label_FCS_tmp,
#                                                                                                     in_dir_names_PCH_tmp,
#                                                                                                     in_file_names_PCH_tmp,
#                                                                                                     alpha_label_PCH_tmp)

[in_dir_names.append(in_dir_name) for in_dir_name in in_dir_names_FCS_tmp]
[in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in in_file_names_FCS_tmp]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
[alpha_label.append(single_alpha_label) for single_alpha_label in alpha_label_FCS_tmp]




# #%% 20240801 - Series of AF488 measurements exported with varying duration
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, '20240801_JHK_ssRNA_ladder_run2/20240801_data.sptw')
# _in_dir_names.extend(os.path.join(local_dir, f'AF488_long_1'))
# _alpha_label.append(1)

# # Naming pattern for detecting correct files within subdirs of each in_dir
# # Wohland SD
# file_name_pattern_FCS = '*_ACF_ch0_bg*' 


# # We do not use PCH right now
# # # Detect PCH files
# # in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
# #                                                                                       file_name_pattern_PCH, 
# #                                                                                       _alpha_label, 
# #                                                                                       '')

# # Repeat for FCS
# in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
#                                                                                       file_name_pattern_FCS, 
#                                                                                       _alpha_label,
#                                                                                       '')

# # _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
# #                                                                                                     in_file_names_FCS_tmp,
# #                                                                                                     alpha_label_FCS_tmp,
# #                                                                                                     in_dir_names_PCH_tmp,
# #                                                                                                     in_file_names_PCH_tmp,
# #                                                                                                     alpha_label_PCH_tmp)

# [in_dir_names.append(in_dir_name) for in_dir_name in in_dir_names_FCS_tmp]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in in_file_names_FCS_tmp]
# # [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in alpha_label_FCS_tmp]





# #%% 20240604 - A488-labelled ParM
# SHould be good for incomplete-sampling stuff
# ''' Labelled protein fraction'''
# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, r'20240416_JHK_NK_New_ParM_data/20240416_data.sptw')
# _in_dir_names.extend([os.path.join(local_dir, '10uM_ParM_noATP_1')])
# _alpha_label.append(0.005) 
# [ _in_dir_names.extend([os.path.join(local_dir, f'5uM_ParM_{x}')]) for x in [1, 2, 3]]
# [_alpha_label.append(x) for x in [1e-2, 1e-2, 1e-5]]
# [ _in_dir_names.extend([os.path.join(local_dir, f'10uM_ParM_{x}')]) for x in [1, 2, 3, 4, 5]]
# [_alpha_label.append(x) for x in [5e-3, 1e-3, 2.5e-2, 5e-3, 5e-3]]

# # Naming pattern for detecting correct files within subdirs of each in_dir
# # Wohland SD, no burst removal
# # file_name_pattern_PCH = '*08_PCMH_ch0*' 
# # file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*'
# # # # Bootstrap SD, no burst removal
# # file_name_pattern_PCH = '*07_PCMH_ch0*' 
# # file_name_pattern_FCS = '*06_ACF_ch0_dt_bg*'
# # # Bootstrap SD, with burst removal
# # file_name_pattern_PCH = '*09_PCMH_ch0_br*' 
# # file_name_pattern_FCS = '*08_ACF_ch0_br_dt_bg*'
# # # Bootstrap SD, no burst removal, long lag times
# file_name_pattern_PCH = '*05_PCMH_ch0*' 
# file_name_pattern_FCS = '*04_ACF_ch0_bg*'
# # # # Bootstrap SD, with burst removal, long lag times
# # file_name_pattern_PCH = '*07_PCMH_ch0_br*' 
# # file_name_pattern_FCS = '*06_ACF_ch0_br_bg*'


# # Detect PCH files
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# # Repeat for FCS
# in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
#                                                                                       file_name_pattern_FCS, 
#                                                                                       _alpha_label,
#                                                                                       '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                     in_file_names_FCS_tmp,
#                                                                                                     alpha_label_FCS_tmp,
#                                                                                                     in_dir_names_PCH_tmp,
#                                                                                                     in_file_names_PCH_tmp,
#                                                                                                     alpha_label_PCH_tmp)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]


# _in_dir_names = []
# _alpha_label = []

# local_dir = os.path.join(glob_in_dir, r'20240416_JHK_NK_New_ParM_data/20240416_data.sptw')
# _in_dir_names.extend([os.path.join(local_dir, '10uM_ParM_noATP_1')])
# _alpha_label.append(0.005) 
# [ _in_dir_names.extend([os.path.join(local_dir, f'5uM_ParM_{x}')]) for x in [1, 2, 3]]
# [_alpha_label.append(x) for x in [1e-2, 1e-2, 1e-5]]
# [ _in_dir_names.extend([os.path.join(local_dir, f'10uM_ParM_{x}')]) for x in [1, 2, 3, 4, 5]]
# [_alpha_label.append(x) for x in [5e-3, 1e-3, 2.5e-2, 5e-3, 5e-3]]

# # Naming pattern for detecting correct files within subdirs of each in_dir
# # Wohland SD, no burst removal
# # file_name_pattern_PCH = '*08_PCMH_ch0*' 
# # file_name_pattern_FCS = '*07_ACF_ch0_dt_bg*'
# # # # Bootstrap SD, no burst removal
# # file_name_pattern_PCH = '*07_PCMH_ch0*' 
# # file_name_pattern_FCS = '*06_ACF_ch0_dt_bg*'
# # # Bootstrap SD, with burst removal
# # file_name_pattern_PCH = '*09_PCMH_ch0_br*' 
# # file_name_pattern_FCS = '*08_ACF_ch0_br_dt_bg*'
# # # # Bootstrap SD, no burst removal, long lag times
# # file_name_pattern_PCH = '*05_PCMH_ch0*' 
# # file_name_pattern_FCS = '*04_ACF_ch0_bg*'
# # # Bootstrap SD, with burst removal, long lag times
# file_name_pattern_PCH = '*07_PCMH_ch0_br*' 
# file_name_pattern_FCS = '*06_ACF_ch0_br_bg*'


# # Detect PCH files
# in_dir_names_PCH_tmp, in_file_names_PCH_tmp, alpha_label_PCH_tmp = utils.detect_files(_in_dir_names,
#                                                                                       file_name_pattern_PCH, 
#                                                                                       _alpha_label, 
#                                                                                       '')

# # Repeat for FCS
# in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
#                                                                                       file_name_pattern_FCS, 
#                                                                                       _alpha_label,
#                                                                                       '')

# _in_dir_names, _in_file_names_FCS, _in_file_names_PCH, _alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS_tmp,
#                                                                                                     in_file_names_FCS_tmp,
#                                                                                                     alpha_label_FCS_tmp,
#                                                                                                     in_dir_names_PCH_tmp,
#                                                                                                     in_file_names_PCH_tmp,
#                                                                                                     alpha_label_PCH_tmp)
# [in_dir_names.append(in_dir_name) for in_dir_name in _in_dir_names]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in _in_file_names_FCS]
# [in_file_names_PCH.append(in_file_name_PCH) for in_file_name_PCH in _in_file_names_PCH]
# [alpha_label.append(single_alpha_label) for single_alpha_label in _alpha_label]





# #%% Fit settings

# ############ Config for fitting protein filaments
# # Output dir for result file writing
# glob_out_dir = '/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Analysis/20240702_ssXNA_discrete_fits/ParM_strExp_fits'

# ### General model settings

# labelling_correction_list = [True]
# incomplete_sampling_correction_list = [False, True]
# labelling_efficiency_incomp_sampling_list = [False] 


# n_species_list = [80]
# spectrum_type_list = ['par_StrExp'] # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
# spectrum_parameter_list = ['N_monomers'] # 'Amplitude', 'N_monomers', 'N_oligomers',
# oligomer_type_list = ['single_filament'] # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'

# use_blinking_list = [False]

# # Shortest and longest diffusion time to fit (parameter bounds)
# tau_diff_min_list = [2.1E-4]
# tau_diff_max_list = [1E1]


# ### FCS settings
# use_FCS_list = [True]

# # Shortest and longest lag time to consider in fit (time axis clipping)
# FCS_min_lag_time_list = [1E-5] # Use 0. to use full range of data in .csv file
# FCS_max_lag_time_list = [3E0]  # Use np.inf to use full range of data in .csv file


# ### PCH settings
# use_PCH_list = [False]
# time_resolved_PCH_list = [False]

# # Shortest and longest bin times to consider
# PCH_min_bin_time_list = [0.] # Use 0. to use full range of data in .csv file
# PCH_max_bin_time_list = [5E-4] # Use np.inf to use full range of data in .csv file

# # Calculation settings
# NLL_funcs_accurate_list = [True] # Accurate MLE or faster least-squares approximation (affects some, not all, likelihood terms)?
# numeric_precision_list = [np.array([1E-3, 1E-4, 1E-5])] # PCH requires numerical precision cutoff, which is set here



############ Config for Amplitude fitting
# Output dir for result file writing
glob_out_dir = '/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Analysis/20240702_ssXNA_discrete_fits/XNA_par_amp_fits'
# glob_out_dir = '/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Analysis/20240702_ssXNA_discrete_fits/AF488_par_amp_fits'

### General model settings

labelling_correction_list = [False]
incomplete_sampling_correction_list = [False]
labelling_efficiency_incomp_sampling_list = [False] 

n_species_list = [80]
spectrum_type_list = ['par_Gauss', 'par_LogNorm', 'par_Gamma'] # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
spectrum_parameter_list = ['Amplitude'] # 'Amplitude', 'N_monomers', 'N_oligomers',
oligomer_type_list = ['naive'] # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'

use_blinking_list = [False]

# Shortest and longest diffusion time to fit (parameter bounds)
tau_diff_min_list = [1E-6] 
tau_diff_max_list = [1E0]

### FCS settings
use_FCS_list = [True]

# Shortest and longest lag time to consider in fit (time axis clipping)
FCS_min_lag_time_list = [1E-6] # Use 0. to use full range of data in .csv file
FCS_max_lag_time_list = [np.inf]  # Use np.inf to use full range of data in .csv file


### PCH settings
use_PCH_list = [False]
time_resolved_PCH_list = [False]

# Shortest and longest bin times to consider
PCH_min_bin_time_list = [0.] # Use 0. to use full range of data in .csv file
PCH_max_bin_time_list = [5E-4] # Use np.inf to use full range of data in .csv file

# Calculation settings
NLL_funcs_accurate_list = [False] # Accurate MLE or faster least-squares approximation (affects some, not all, likelihood terms)?
numeric_precision_list = [np.array([1E-3, 1E-4, 1E-5])] # PCH requires numerical precision cutoff, which is set here


#%% Metadata/calibration data/settings that are global for all measurements
verbosity = 1 # How much do you want the software to talk?
FCS_psf_width_nm = 210. # Roughly
FCS_psf_aspect_ratio = 6. # Roughly

acquisition_time_s = 90.

PCH_Q = 10. # More calculation parameter than metadata, but whatever

# How many parallel processes?
# If mp_processes <= 1, we use multiprocessing WITHIN the fit which allows acceleration of multi-species PCH
# If mp_processes > 1, we run multiple fits simultaneously, each in single-thread calculation
mp_processes = os.cpu_count()
# mp_processes = 1  # no multiprocessing

#%% Wrap all permutations for different fit settings and all files...Long list!


# Iterate over all settings and files
list_of_parameter_tuples = []
fit_counter = 1
print(f'Sanity check all: Found {len(in_file_names_FCS)} FCS files, {len(in_file_names_PCH)} PCH files, {len(in_dir_names)} dir names.')
for use_FCS in use_FCS_list:
    for FCS_min_lag_time in FCS_min_lag_time_list:
        for FCS_max_lag_time in FCS_max_lag_time_list:
            for use_PCH in use_PCH_list:
                for PCH_min_bin_time in PCH_min_bin_time_list:
                    for PCH_max_bin_time in PCH_max_bin_time_list:
                        for time_resolved_PCH in time_resolved_PCH_list:
                            for NLL_funcs_accurate in NLL_funcs_accurate_list:
                                for n_species in n_species_list:
                                    for tau_diff_min in tau_diff_min_list:
                                        for tau_diff_max in tau_diff_max_list:
                                            for use_blinking in use_blinking_list:
                                                for spectrum_type in spectrum_type_list:
                                                    for spectrum_parameter in spectrum_parameter_list:
                                                        for oligomer_type in oligomer_type_list:
                                                            for labelling_correction in labelling_correction_list:
                                                                for incomplete_sampling_correction in incomplete_sampling_correction_list:
                                                                    for labelling_efficiency_incomp_sampling in labelling_efficiency_incomp_sampling_list:
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
                                                                                utils.isint(n_species) and n_species > 0 and
                                                                                tau_diff_min > 0 and
                                                                                tau_diff_max >= tau_diff_min and
                                                                                type(use_blinking) == bool and
                                                                                (n_species < 5 and spectrum_type == 'discrete' or
                                                                                  n_species > 10 and spectrum_type in ['reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']) and
                                                                                spectrum_parameter in ['Amplitude', 'N_monomers', 'N_oligomers'] and
                                                                                oligomer_type in ['naive', 'spherical_shell', 'sherical_dense', 'single_filament', 'double_filament'] and
                                                                                type(labelling_correction) == bool and
                                                                                type(incomplete_sampling_correction) == bool and
                                                                                type(labelling_efficiency_incomp_sampling) == bool and
                                                                                ((not labelling_efficiency_incomp_sampling) or (labelling_efficiency_incomp_sampling and incomplete_sampling_correction and labelling_correction)) and
                                                                                (incomplete_sampling_correction == False and spectrum_type == 'discrete' or
                                                                                  spectrum_type in ['reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']) and
                                                                                ((utils.isfloat(numeric_precision) and 
                                                                                      numeric_precision < 1. and 
                                                                                      numeric_precision > 0.) or
                                                                                  (utils.isiterable(numeric_precision) and 
                                                                                      np.all(numeric_precision < 1.) and 
                                                                                      np.all(numeric_precision > 0.)))
                                                                                ):
                                                                            
                                                                                fit_settings_str = f'{spectrum_type}_{n_species}spec'
                                                                                fit_settings_str += f'_{oligomer_type}_{spectrum_parameter}' if spectrum_type != 'discrete' else ''
                                                                                fit_settings_str += '_blink' if use_blinking else ''
                                                                                fit_settings_str += '_lblcr' if labelling_correction else ''
                                                                                fit_settings_str += '_smplcr' if incomplete_sampling_correction else ''
                                                                                fit_settings_str += '_lblsmplcr' if labelling_efficiency_incomp_sampling else ''
                                                                                fit_settings_str += '_FCS' if use_FCS else ''
                                                                                fit_settings_str += ('_PCMH' if time_resolved_PCH else '_PCH') if use_PCH else ''
                                                                                fit_settings_str += ('_MLE' if NLL_funcs_accurate else '_WLSQ') if (use_PCH or incomplete_sampling_correction) else ''
                                                                                
                                                                                for i_file, dir_name in enumerate(in_dir_names):
                                                                                    job_prefix = in_file_names_FCS[i_file] + '_' + fit_settings_str
                                                                                    save_path = os.path.join(glob_out_dir, fit_settings_str)
                                                                                    if not os.path.exists(save_path):
                                                                                        os.makedirs(save_path)
                                                                                    
                                                                                    fit_res_table_path = os.path.join(save_path, 'Fit_params_' + fit_settings_str)
                                                                                    
                                                                                    parameter_tuple = (fit_counter,
                                                                                                       i_file,
                                                                                                       job_prefix,
                                                                                                        save_path,
                                                                                                        fit_res_table_path,
                                                                                                        dir_name,
                                                                                                        in_file_names_FCS[i_file],
                                                                                                        in_file_names_PCH[i_file] if len(in_file_names_PCH) > 0 else '',
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
                                                                                                        tau_diff_min,
                                                                                                        tau_diff_max,
                                                                                                        use_blinking,
                                                                                                        spectrum_type,
                                                                                                        spectrum_parameter,
                                                                                                        oligomer_type,
                                                                                                        alpha_label[i_file],
                                                                                                        labelling_correction,
                                                                                                        incomplete_sampling_correction,
                                                                                                        labelling_efficiency_incomp_sampling,
                                                                                                        numeric_precision,
                                                                                                        verbosity)
                                                                                    list_of_parameter_tuples.extend((parameter_tuple,))
                                                                                    fit_counter += 1
                                                                            
                                                                            
                                                                            
#%% Parallel processing function definition


def fitting_parfunc(fit_number,
                    i_file,
                    job_prefix,
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
                    tau_diff_min,
                    tau_diff_max,
                    use_blinking,
                    spectrum_type,
                    spectrum_parameter,
                    oligomer_type,
                    labelling_efficiency,
                    labelling_correction,
                    incomplete_sampling_correction,
                    labelling_efficiency_incomp_sampling,
                    numeric_precision,
                    verbosity,
                    ):
    
    
    # Command line message
    time_tag = datetime.datetime.now()
    message = f'[{job_prefix}] [{fit_number}] Fitting '
    message += in_file_name_FCS if use_FCS else ''
    message += ' and ' if (use_FCS and use_PCH) else ''
    message += in_file_name_PCH if use_PCH else ''
    message += ' globally:' if (use_FCS and use_PCH) else ''
    print('\n' + time_tag.strftime("%Y-%m-%d %H:%M:%S") + '\n' + message)
    
    try:
        try:
            data_FCS_tau_s, data_FCS_G, avg_count_rate, data_FCS_sigma, acquisition_time_s = utils.read_Kristine_FCS(dir_name, 
                                                                                                                      in_file_name_FCS,
                                                                                                                      FCS_min_lag_time,
                                                                                                                      FCS_max_lag_time)
            if use_PCH:
                data_PCH_bin_times, data_PCH_hist = utils.read_PCMH(dir_name,
                                                                    in_file_name_PCH,
                                                                    PCH_min_bin_time,
                                                                    PCH_max_bin_time)
        except:
            message = f'[{job_prefix}] [{fit_number}] Error in data loading for '
            message += in_file_name_FCS if use_FCS else ''
            message += ' and ' if (use_FCS and use_PCH) else ''
            message += in_file_name_PCH if use_PCH else ''
            message += ' - aborting.'
            print('\n' + time_tag.strftime("%Y-%m-%d %H:%M:%S") + '\n' + message)
            return None # Dummy return to terminate function call
            
        fitter = fitting.FCS_spectrum(FCS_psf_width_nm = FCS_psf_width_nm,
                                      FCS_psf_aspect_ratio = FCS_psf_aspect_ratio,
                                      PCH_Q = PCH_Q,
                                      acquisition_time_s = acquisition_time_s, 
                                      data_FCS_tau_s = data_FCS_tau_s if use_FCS else None,
                                      data_FCS_G = data_FCS_G if use_FCS else None,
                                      data_FCS_sigma = data_FCS_sigma if use_FCS else None,
                                      data_PCH_bin_times = data_PCH_bin_times if use_PCH else None,
                                      data_PCH_hist = data_PCH_hist if use_PCH else None,
                                      labelling_efficiency = labelling_efficiency,
                                      numeric_precision = numeric_precision,
                                      NLL_funcs_accurate = NLL_funcs_accurate,
                                      verbosity = verbosity,
                                      job_prefix = job_prefix,
                                      labelling_efficiency_incomp_sampling = labelling_efficiency_incomp_sampling
                                      )
        
        if spectrum_type in ['discrete', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            fit_result = fitter.run_fit(use_FCS = use_FCS, # bool
                                        use_PCH = use_PCH, # bool
                                        time_resolved_PCH = time_resolved_PCH, # bool
                                        spectrum_type = spectrum_type, # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
                                        spectrum_parameter = spectrum_parameter, # 'Amplitude', 'N_monomers', 'N_oligomers',
                                        oligomer_type = oligomer_type, # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
                                        labelling_correction = labelling_correction, # bool
                                        incomplete_sampling_correction = incomplete_sampling_correction, # bool
                                        n_species = n_species, # int
                                        tau_diff_min = tau_diff_min, # float
                                        tau_diff_max = tau_diff_max, # float
                                        use_blinking = use_blinking, # bool
                                        two_step_fit = True, # bool
                                        use_parallel = mp_processes <= 1 # Bool
                                        # use_parallel = False # Bool
                                        )
            
        else: # spectrum_type in ['reg_MEM', 'reg_CONTIN']
            # Here we get more complex output
            fit_result, N_pop_array, lagrange_mul = fitter.run_fit(use_FCS = use_FCS, # bool
                                                                    use_PCH = use_PCH, # bool
                                                                    time_resolved_PCH = time_resolved_PCH, # bool
                                                                    spectrum_type = spectrum_type, # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
                                                                    spectrum_parameter = spectrum_parameter, # 'Amplitude', 'N_monomers', 'N_oligomers',
                                                                    oligomer_type = oligomer_type, #  'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
                                                                    labelling_correction = labelling_correction, # bool
                                                                    incomplete_sampling_correction = incomplete_sampling_correction, # bool
                                                                    n_species = n_species, # int
                                                                    tau_diff_min = tau_diff_min, # float
                                                                    tau_diff_max = tau_diff_max, # float
                                                                    use_blinking = use_blinking, # bool
                                                                    use_parallel = mp_processes <= 1 # Bool
                                                                    # use_parallel = False # Bool
                                                                    )
            
        
        if not fit_result == None:
            if hasattr(fit_result, 'params') and hasattr(fit_result, 'covar'): 
                # dirty workaround for essentially testing "if type(fit_result) == lmfit.MinimizerResult", as MinimizerResult class cannot be explicitly referenced
                # Unpack fit result
                fit_params = fit_result.params
                covar = fit_result.covar
                # Recalculate number of species, it is possible we lose some in between
                n_species = fitter.get_n_species(fit_params)
                
            elif type(fit_result) == lmfit.Parameters:
                # Different output that can come from regularized fitting
                # Unpack fit result
                fit_params = fit_result
                covar = None
                
                n_species = N_pop_array.shape[0]
                
            else:
                raise Exception(f'Got a fit_result output, but with unsupported type. Expected lmfit.MinimizerResult or lmfit.Parameters, got {type(fit_result)}')




            out_name = os.path.join(save_path,
                                    f'{fit_number}_{i_file}_' + time_tag.strftime("%m%d-%H%M%S") + f'_{in_file_name_FCS if use_FCS else in_file_name_PCH}_fit_{spectrum_type}_{n_species}spec')
            
            # Command line preview of fit results
            print(f' [{job_prefix}]   Fitted parameters:')
            [print(f'[{job_prefix}] {key}: {fit_params[key].value}') for key in fit_params.keys() if fit_params[key].vary]
            
            
            # Dict in which we collect various mean and SD values
            avg_values_dict = {}
            
                
                
            # Small spreadsheets for species-wise parameters in case we have multiple species
            # all of these are technically redundant with the big one written later...
            # But reading these out of the main results spreadsheet is just
            # too much trouble if you have more than 1 or 2 species
            if n_species > 1:
                
                # Unpack species parameters into arrays
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    N_oligo_pop_array = N_pop_array
                elif spectrum_type == 'discrete':
                    # Historical reasons why here N_obs has to be used...
                    N_oligo_pop_array =  np.array([fit_params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])
                else:
                    N_oligo_pop_array = np.array([fit_params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species)])
                
                stoichiometry_array = np.array([fit_params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
                stoichiometry_bw_array = np.array([fit_params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species)])
                tau_diff_array = np.array([fit_params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])


                # Stoichiometry
                fit_res_tmp_path = fit_res_table_path + '_stoi.csv' 
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number                 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                for i_spec in range(n_species):
                    fit_result_dict[f'stoichiometry_{i_spec}'] = stoichiometry_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                         header = True, 
                                         index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                         mode = 'a', 
                                         header = False, 
                                         index = False)
                    
                # Stoichiometry binwidth
                fit_res_tmp_path = fit_res_table_path + '_stoi_bw.csv' 
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number                 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                for i_spec in range(n_species):
                    fit_result_dict[f'stoi_bw_{i_spec}'] = stoichiometry_bw_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                         header = True, 
                                         index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                         mode = 'a', 
                                         header = False, 
                                         index = False)


                # Diffusion time
                fit_res_tmp_path = fit_res_table_path + '_tau_diff.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                for i_spec in range(n_species):
                    fit_result_dict[f'tau_diff_{i_spec}'] = tau_diff_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                         header = True, 
                                         index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                         mode = 'a', 
                                         header = False, 
                                         index = False)


                # Amplitudes - population level - density
                fit_res_tmp_path = fit_res_table_path + '_amp_density.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                amp_density_array = N_oligo_pop_array * stoichiometry_array**2
                if labelling_correction:
                    amp_density_array *= 1 - (1 - labelling_efficiency) / (labelling_efficiency * stoichiometry_array)
                amp_density_array /= amp_density_array.max()
                
                for i_spec in range(n_species):
                    fit_result_dict[f'amp_density_{i_spec}'] = amp_density_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)
                

                # Amplitudes - population level - histogram
                fit_res_tmp_path = fit_res_table_path + '_amp_histogram.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                amp_histogram_array = N_oligo_pop_array * stoichiometry_array**2 * stoichiometry_bw_array
                if labelling_correction:
                    amp_histogram_array *= 1 - (1 - labelling_efficiency) / (labelling_efficiency * stoichiometry_array)
                amp_histogram_array /= amp_histogram_array.sum()
                
                for i_spec in range(n_species):
                    fit_result_dict[f'amp_histogram_{i_spec}'] = amp_histogram_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)
                
                # Use amp_histogram_array to get amplitude-weighted mean values
                avg_values_dict['avg_stoi_amp'] = np.sum(stoichiometry_array * amp_histogram_array)
                avg_values_dict['sd_stoi_amp'] = np.sqrt(np.sum(((stoichiometry_array - avg_values_dict['avg_stoi_amp']) * amp_histogram_array)**2) / np.sum(amp_histogram_array**2))
                avg_values_dict['avg_tau_diff_amp'] = np.sum(tau_diff_array * amp_histogram_array)
                avg_values_dict['sd_tau_diff_amp'] = np.sqrt(np.sum(((tau_diff_array - avg_values_dict['avg_tau_diff_amp']) * amp_histogram_array)**2) / np.sum(amp_histogram_array**2))

                
                # N_oligomers - population level - absolute
                fit_res_tmp_path = fit_res_table_path + '_N_pop_oligo_abs.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                N_pop_oligo_abs_array = N_oligo_pop_array * stoichiometry_bw_array
                
                for i_spec in range(n_species):
                    fit_result_dict[f'N_pop_oligo_abs_{i_spec}'] = N_pop_oligo_abs_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)

                
                # N_oligomers - population level - density
                fit_res_tmp_path = fit_res_table_path + '_N_pop_oligo_density.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                N_pop_oligo_density_array = N_oligo_pop_array / N_oligo_pop_array.max()
                
                for i_spec in range(n_species):
                    fit_result_dict[f'N_pop_oligo_density_{i_spec}'] = N_pop_oligo_density_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)


                # N_oligomers - population level - histogram
                fit_res_tmp_path = fit_res_table_path + '_N_pop_oligo_histogram.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                N_pop_oligo_histogram_array = N_oligo_pop_array * stoichiometry_bw_array
                N_pop_oligo_histogram_array /= N_pop_oligo_histogram_array.sum()
                
                for i_spec in range(n_species):
                    fit_result_dict[f'N_pop_oligo_histogram_{i_spec}'] = N_pop_oligo_histogram_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)

                # Use N_pop_oligo_histogram_array to get oligomer_N_pop-weighted mean values
                avg_values_dict['avg_stoi_N_oligo_pop'] = np.sum(stoichiometry_array * N_pop_oligo_histogram_array)
                avg_values_dict['sd_stoi_N_oligo_pop'] = np.sqrt(np.sum(((stoichiometry_array - avg_values_dict['avg_stoi_amp']) * N_pop_oligo_histogram_array)**2) / np.sum(N_pop_oligo_histogram_array**2))
                avg_values_dict['avg_tau_diff_N_oligo_pop'] = np.sum(tau_diff_array * N_pop_oligo_histogram_array)
                avg_values_dict['sd_tau_diff_N_oligo_pop'] = np.sqrt(np.sum(((tau_diff_array - avg_values_dict['avg_tau_diff_amp']) * N_pop_oligo_histogram_array)**2) / np.sum(N_pop_oligo_histogram_array**2))

                # N_monomers - population level - absolute
                fit_res_tmp_path = fit_res_table_path + '_N_pop_mono_abs.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                N_pop_mono_abs_array = N_oligo_pop_array * stoichiometry_bw_array * stoichiometry_array
                
                for i_spec in range(n_species):
                    fit_result_dict[f'N_pop_mono_abs_{i_spec}'] = N_pop_mono_abs_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)

                
                # N_monomers - population level - density
                fit_res_tmp_path = fit_res_table_path + '_N_pop_mono_density.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                N_pop_mono_density_array = N_oligo_pop_array * stoichiometry_array
                N_pop_mono_density_array /= N_pop_mono_density_array.max()
                
                for i_spec in range(n_species):
                    fit_result_dict[f'N_pop_mono_density_{i_spec}'] = N_pop_mono_density_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)


                # N_monomers - population level - histogram
                fit_res_tmp_path = fit_res_table_path + '_N_pop_mono_histogram.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                N_pop_mono_histogram_array = N_oligo_pop_array * stoichiometry_bw_array * stoichiometry_array
                N_pop_mono_histogram_array /= N_pop_mono_histogram_array.sum()
                
                for i_spec in range(n_species):
                    fit_result_dict[f'N_pop_mono_oligo_histogram_{i_spec}'] = N_pop_mono_histogram_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)

                # Use N_pop_mono_histogram_array to get monomer_N_pop-weighted mean values
                avg_values_dict['avg_stoi_N_mono_pop'] = np.sum(stoichiometry_array * N_pop_mono_histogram_array)
                avg_values_dict['sd_stoi_N_mono_pop'] = np.sqrt(np.sum(((stoichiometry_array - avg_values_dict['avg_stoi_amp']) * N_pop_mono_histogram_array)**2) / np.sum(N_pop_mono_histogram_array**2))
                avg_values_dict['avg_tau_diff_N_mono_pop'] = np.sum(tau_diff_array * N_pop_mono_histogram_array)
                avg_values_dict['sd_tau_diff_N_mono_pop'] = np.sqrt(np.sum(((tau_diff_array - avg_values_dict['avg_tau_diff_amp']) * N_pop_mono_histogram_array)**2) / np.sum(N_pop_mono_histogram_array**2))


                # The remaining ones only if we have incomplete sampling correction
                if incomplete_sampling_correction:
                    N_oligo_obs_array = np.array([fit_params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])

                    # N_oligomers - observation level - absolute
                    fit_res_tmp_path = fit_res_table_path + '_N_obs_oligo_abs.csv'
                    fit_result_dict = {}
                    fit_result_dict['fit_number'] = fit_number 
                    fit_result_dict['file_number'] = i_file
                    fit_result_dict['folder'] = dir_name 
                    fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                    fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                    if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                        fit_result_dict['lagrange_mul'] = lagrange_mul
                        
                    N_obs_oligo_abs_array = N_oligo_obs_array * stoichiometry_bw_array
                    
                    for i_spec in range(n_species):
                        fit_result_dict[f'N_obs_oligo_abs_{i_spec}'] = N_obs_oligo_abs_array[i_spec]
                    fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                    if not os.path.isfile(fit_res_tmp_path):
                        # Does not yet exist - create with header
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                header = True, 
                                                index = False)
                    else:
                        # Exists - append
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                mode = 'a', 
                                                header = False, 
                                                index = False)

                    
                    # N_oligomers - observation level - density
                    fit_res_tmp_path = fit_res_table_path + '_N_obs_oligo_density.csv'
                    fit_result_dict = {}
                    fit_result_dict['fit_number'] = fit_number 
                    fit_result_dict['file_number'] = i_file
                    fit_result_dict['folder'] = dir_name 
                    fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                    fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                    if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                        fit_result_dict['lagrange_mul'] = lagrange_mul
                    N_obs_oligo_density_array = N_oligo_obs_array / N_oligo_obs_array.max()
                    
                    for i_spec in range(n_species):
                        fit_result_dict[f'N_obs_oligo_density_{i_spec}'] = N_obs_oligo_density_array[i_spec]
                    fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                    if not os.path.isfile(fit_res_tmp_path):
                        # Does not yet exist - create with header
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                header = True, 
                                                index = False)
                    else:
                        # Exists - append
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                mode = 'a', 
                                                header = False, 
                                                index = False)


                    # N_oligomers - observation level - histogram
                    fit_res_tmp_path = fit_res_table_path + '_N_obs_oligo_histogram.csv'
                    fit_result_dict = {}
                    fit_result_dict['fit_number'] = fit_number 
                    fit_result_dict['file_number'] = i_file
                    fit_result_dict['folder'] = dir_name 
                    fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                    fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                    if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                        fit_result_dict['lagrange_mul'] = lagrange_mul
                        
                    N_obs_oligo_histogram_array = N_oligo_obs_array * stoichiometry_bw_array
                    N_obs_oligo_histogram_array /= N_obs_oligo_histogram_array.sum()
                    
                    for i_spec in range(n_species):
                        fit_result_dict[f'N_obs_oligo_histogram_{i_spec}'] = N_obs_oligo_histogram_array[i_spec]
                    fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                    if not os.path.isfile(fit_res_tmp_path):
                        # Does not yet exist - create with header
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                header = True, 
                                                index = False)
                    else:
                        # Exists - append
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                mode = 'a', 
                                                header = False, 
                                                index = False)



                    # N_monomers - observation level - absolute
                    fit_res_tmp_path = fit_res_table_path + '_N_obs_mono_abs.csv'
                    fit_result_dict = {}
                    fit_result_dict['fit_number'] = fit_number 
                    fit_result_dict['file_number'] = i_file
                    fit_result_dict['folder'] = dir_name 
                    fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                    fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                    if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                        fit_result_dict['lagrange_mul'] = lagrange_mul
                        
                    N_obs_mono_abs_array = N_oligo_obs_array * stoichiometry_bw_array * stoichiometry_array
                    
                    for i_spec in range(n_species):
                        fit_result_dict[f'N_obs_mono_abs_{i_spec}'] = N_obs_mono_abs_array[i_spec]
                    fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                    if not os.path.isfile(fit_res_tmp_path):
                        # Does not yet exist - create with header
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                header = True, 
                                                index = False)
                    else:
                        # Exists - append
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                mode = 'a', 
                                                header = False, 
                                                index = False)

                    
                    # N_monomers - observation level - density
                    fit_res_tmp_path = fit_res_table_path + '_N_obs_mono_density.csv'
                    fit_result_dict = {}
                    fit_result_dict['fit_number'] = fit_number 
                    fit_result_dict['file_number'] = i_file
                    fit_result_dict['folder'] = dir_name 
                    fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                    fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                    if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                        fit_result_dict['lagrange_mul'] = lagrange_mul
                    N_obs_mono_density_array = N_oligo_obs_array * stoichiometry_array
                    N_obs_mono_density_array /= N_obs_mono_density_array.max()
                    
                    for i_spec in range(n_species):
                        fit_result_dict[f'N_obs_mono_density_{i_spec}'] = N_obs_mono_density_array[i_spec]
                    fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                    if not os.path.isfile(fit_res_tmp_path):
                        # Does not yet exist - create with header
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                header = True, 
                                                index = False)
                    else:
                        # Exists - append
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                mode = 'a', 
                                                header = False, 
                                                index = False)


                    # N_monomers - observation level - histogram
                    fit_res_tmp_path = fit_res_table_path + '_N_obs_mono_histogram.csv'
                    fit_result_dict = {}
                    fit_result_dict['fit_number'] = fit_number 
                    fit_result_dict['file_number'] = i_file
                    fit_result_dict['folder'] = dir_name 
                    fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                    fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                    if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                        fit_result_dict['lagrange_mul'] = lagrange_mul
                        
                    N_obs_mono_histogram_array = N_oligo_obs_array * stoichiometry_bw_array * stoichiometry_array
                    N_obs_mono_histogram_array /= N_obs_mono_histogram_array.sum()
                    
                    for i_spec in range(n_species):
                        fit_result_dict[f'N_obs_mono_oligo_histogram_{i_spec}'] = N_obs_mono_histogram_array[i_spec]
                    fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                    if not os.path.isfile(fit_res_tmp_path):
                        # Does not yet exist - create with header
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                header = True, 
                                                index = False)
                    else:
                        # Exists - append
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                mode = 'a', 
                                                header = False, 
                                                index = False)

                                     
                    if labelling_efficiency_incomp_sampling:
                        # If and only if we have incomplete-sampling-of-incomplete-labelling correction, we also write that separately
                        fit_res_tmp_path = fit_res_table_path + '_label_eff_obs.csv'
                        fit_result_dict = {}
                        fit_result_dict['fit_number'] = fit_number 
                        fit_result_dict['file_number'] = i_file
                        fit_result_dict['folder'] = dir_name 
                        fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                        fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                        if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                            fit_result_dict['lagrange_mul'] = lagrange_mul
                                                    
                        for i_spec in range(n_species):
                            fit_result_dict[f'_label_eff_obs{i_spec}'] = fit_params[f'Label_efficiency_obs_{i_spec}'].value
                        fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                        if not os.path.isfile(fit_res_tmp_path):
                            # Does not yet exist - create with header
                            fit_result_df.to_csv(fit_res_tmp_path, 
                                                    header = True, 
                                                    index = False)
                        else:
                            # Exists - append
                            fit_result_df.to_csv(fit_res_tmp_path, 
                                                    mode = 'a', 
                                                    header = False, 
                                                    index = False)


            # Write large spreadsheet with ALL fit results
            fit_result_dict = {}            
            
            # Metadata
            fit_result_dict['fit_number'] = fit_number            
            fit_result_dict['file_number'] = i_file
            fit_result_dict['folder'] = dir_name
            fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
            fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
            
            
            # Insert average and standard deviation values into results dict
            for key in avg_values_dict.keys():
                fit_result_dict[key] = avg_values_dict[key]
            
            
            # Fit parameters, with uncertainty where available
            has_covar = not covar == None
            if has_covar:
                uncertainty_array = np.sqrt(np.diag(covar))
                covar_pointer = 0
            
            for key in fit_params.keys():
                fit_result_dict[key + '_val'] = fit_params[key].value
                fit_result_dict[key + '_vary'] = 'Vary' if fit_params[key].vary else 'Fix_Dep'
                if not fit_params[key].stderr == None:
                    fit_result_dict[key + '_err'] = fit_params[key].stderr
                elif has_covar and fit_params[key].vary:
                    fit_result_dict[key + '_err'] = uncertainty_array[covar_pointer]
                    covar_pointer += 1
                    
            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                # Special stuff for regularized fitting
                fit_result_dict['lagrange_mul'] = lagrange_mul
                for i_spec in range(n_species):
                    fit_result_dict[f'N_pop_{i_spec}'] = N_pop_array[i_spec]
                    
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



    
            # Show and write fits themselves
            if use_FCS:
                if not labelling_correction:
                    if spectrum_type == 'discrete':
                        model_FCS = fitter.get_acf_full_labelling(fit_params)
                    elif  spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
                        model_FCS = fitter.get_acf_full_labelling_par(fit_params)
                    else: #  spectrum_type in ['reg_MEM', 'reg_CONTIN']
                        model_FCS = fitter.get_acf_full_labelling_reg(fit_params)
    
                else:
                    if spectrum_type == 'discrete':
                        model_FCS = fitter.get_acf_partial_labelling(fit_params)
                    else: # spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp', 'reg_MEM', 'reg_CONTIN']:
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
                
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    tau_diff_array = np.array([fit_params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])
                    stoichiometry_array = np.array([fit_params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
                    stoichiometry_binwidth_array = np.array([fit_params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species)])
                    label_efficiency_array = np.array([fit_params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species)])

                    scaled_N_pop_array = N_pop_array / N_pop_array.max() * model_FCS.max()
                    
                    ax.semilogx(tau_diff_array,
                                scaled_N_pop_array, 
                                linestyle = '',
                                marker = 'x',
                                color = 'b')
                    
                    scaled_amp_array = N_pop_array * stoichiometry_binwidth_array * stoichiometry_array**2 
                    if labelling_correction:
                        scaled_amp_array  *= (1 + (1-label_efficiency_array) / label_efficiency_array / stoichiometry_array)
                    scaled_amp_array = scaled_amp_array / scaled_amp_array.max() * model_FCS.max()
                    ax.semilogx(tau_diff_array,
                                scaled_amp_array, 
                                linestyle = '',
                                marker = 'x',
                                color = 'r')

                
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
                                                              spectrum_type = spectrum_type,
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
                                                                      spectrum_type = spectrum_type,
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