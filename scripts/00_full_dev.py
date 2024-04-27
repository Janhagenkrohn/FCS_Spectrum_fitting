# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""
# I/O modules
import os
import sys

# Data processing modules
import numpy as np
import lmfit

# Plotting
import matplotlib.pyplot as plt



# Custom module
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils
from functions import fitting


#%% Define input files
# Input directory and the labelled protein fraction in each of them
in_dir_names= []
alpha_label = []
# glob_dir = '/fs/pool/pool-schwille-spt/Experiment_analysis/20231117_JHK_NKaletta_ParM_oligomerization'
glob_dir = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\simFCS2_simulations\3a\csv_exports'


''' Labelled protein fraction'''
in_dir_names.extend([os.path.join(glob_dir)])
alpha_label.append(1.) # 50 nM in 20 µM

# Naming pattern for detecting correct files within subdirs of each in_dir
file_name_pattern_PCH = '*batch3a_cond5*_PCMH_ch0*' # Dual-channel PCH
file_name_pattern_FCS = '*batch3a_cond5*_ACF_ch0*' # CCF

#%% Fit settings

### General model settings

labelling_correction = False
labelling_efficiency = 1.
incomplete_sampling_correction = False

n_species = 1
spectrum_type = 'discrete' # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
spectrum_parameter = 'Amplitude' # 'Amplitude', 'N_monomers', 'N_oligomers',
oligomer_type = 'sherical_dense' #  'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'

use_blinking = True



### FCS settings
use_FCS = True

# Shortest and longest diffusion time to fit (parameter bounds)
tau_diff_min = 1E-5
tau_diff_max = 1E-3

# Shortest and longest lag time to consider in fit (time axis clipping)
FCS_min_lag_time = 0.
FCS_max_lag_time = np.inf 


### PCH settings
use_PCH = True
time_resolved_PCH = True

# Shortest and longest bin times to consider
PCH_min_bin_time = 2E-6
PCH_max_bin_time = 5E-4

#ää Calculation settings
use_parallel = False

numeric_precision = np.array([1E-3, 1E-4, 1E-5])



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
    
    

#%% Start processing
for i_file, dir_name in enumerate(in_dir_names):
    
    data_FCS_tau_s, data_FCS_G, avg_count_rate, data_FCS_sigma, acquisition_time_s = utils.read_Kristine_FCS(dir_name, 
                                                                                                             in_file_names_FCS[i_file],
                                                                                                             FCS_min_lag_time,
                                                                                                             FCS_max_lag_time)
    
    data_PCH_bin_times, data_PCH_hist = utils.read_PCMH(dir_name,
                                                        in_file_names_PCH[i_file],
                                                        PCH_min_bin_time,
                                                        PCH_max_bin_time)
    
    fitter = fitting.FCS_spectrum(FCS_psf_width_nm = FCS_psf_width_nm,
                                  FCS_psf_aspect_ratio = FCS_psf_aspect_ratio,
                                  PCH_Q = PCH_Q,
                                  acquisition_time_s = acquisition_time_s if acquisition_time_s > 0 else 90., # Dummy for debugging with old-format data
                                  data_FCS_tau_s = data_FCS_tau_s,
                                  data_FCS_G = data_FCS_G,
                                  data_FCS_sigma = data_FCS_sigma,
                                  data_PCH_bin_times = data_PCH_bin_times,
                                  data_PCH_hist = data_PCH_hist,
                                  labelling_efficiency = labelling_efficiency,
                                  numeric_precision = numeric_precision
                                  )
    
    
    fit_result = fitter.run_fit(use_FCS = use_FCS, # bool
                                use_PCH = use_PCH, # bool
                                time_resolved_PCH = time_resolved_PCH, # bool
                                spectrum_type = spectrum_type, # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
                                spectrum_parameter = spectrum_parameter, # 'Amplitude', 'N_monomers', 'N_oligomers',
                                oligomer_type = oligomer_type, #  'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
                                labelling_correction = labelling_correction, # bool
                                incomplete_sampling_correction = incomplete_sampling_correction, # bool
                                n_species = n_species, # int
                                tau_diff_min = tau_diff_min, # float
                                tau_diff_max = tau_diff_max, # float
                                use_blinking = use_blinking, # bool
                                use_parallel = use_parallel # Bool
                                )
    
    fit_params = fit_result.params
    print('\n Fitted parameters:')
    [print(f'{key}: {fit_params[key].value} (varied: {fit_params[key].vary})') for key in fit_params.keys()]

    if use_FCS:
        if not labelling_correction:
            model_FCS = fitter.get_acf_full_labelling(fit_params)
        else:
            model_FCS = fitter.get_acf_partial_labelling(fit_params)
        
        # Create plot of fit and save
        fig, ax = plt.subplots(nrows=1, ncols=1)
        # Data
        ax.semilogx(data_FCS_tau_s, data_FCS_G, 
                    'dk')
        ax.semilogx(data_FCS_tau_s, data_FCS_G + data_FCS_sigma,
                    '-k', alpha = 0.7)
        ax.semilogx(data_FCS_tau_s, data_FCS_G - data_FCS_sigma,
                    '-k', alpha = 0.7)
        ax.semilogx(data_FCS_tau_s, model_FCS, 
                    marker = '', linestyle = '-', color = 'tab:gray')  

        fig.supxlabel('Correlation time [s]')
        fig.supylabel('G(\u03C4)')
        ax.set_xlim(data_FCS_tau_s[0], data_FCS_tau_s[-1])
        plot_y_min_max = (np.percentile(data_FCS_G, 3), np.percentile(data_FCS_G, 97))
        ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                    plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)

        plt.show()
        
    if use_PCH:
        
        fig, ax = plt.subplots(1, 1)

        if not labelling_correction:
            model_PCH = fitter.get_pch_full_labelling(fit_params,
                                                      t_bin = data_PCH_bin_times[0],
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
        
        
        ax.semilogy(np.arange(0, data_PCH_hist.shape[0]),
                     data_PCH_hist[:,0],
                     marker = '.',
                     linestyle = 'none')
        
        ax.semilogy(np.arange(0, model_PCH.shape[0]),
                     model_PCH * data_PCH_hist[:,0].sum(),
                     marker = '',
                     linestyle = '-')
        
        max_x = np.nonzero(data_PCH_hist[:,0])[0][-1]
        max_y = np.max(data_PCH_hist[:,0])
        
        if time_resolved_PCH:
            for i_bin_time in range(1, data_PCH_bin_times.shape[0]):
                if not labelling_correction:
                    model_PCH = fitter.get_pch_full_labelling(fit_params,
                                                              t_bin = data_PCH_bin_times[i_bin_time],
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
                
                
                ax.semilogy(np.arange(0, data_PCH_hist.shape[0]),
                             data_PCH_hist[:,i_bin_time],
                             marker = '.',
                             linestyle = 'none')
                
                ax.semilogy(np.arange(0, model_PCH.shape[0]),
                             model_PCH * data_PCH_hist[:,i_bin_time].sum(),
                             marker = '',
                             linestyle = '-')
        
                
                max_x = np.max([max_x, np.nonzero(data_PCH_hist[:,i_bin_time])[0][-1]])
                max_y = np.max([max_y, np.max(data_PCH_hist[:,i_bin_time])])
                
                
        ax.set_xlim(-0.49, max_x + 1.49)
        ax.set_ylim(0.3, max_y * 1.7)
        ax.set_title('PCH fit')
        fig.supxlabel('Photons in bin')
        fig.supylabel('Counts')

        plt.show()

        
        
        
