# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:01:58 2024

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

# Custom module
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils
from functions import fitting

#%% Fit settings

### General model settings

labelling_correction = False
labelling_efficiency = 1.
incomplete_sampling_correction = False

spectrum_type = 'discrete' # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
spectrum_parameter = 'N_oligomers' # 'Amplitude', 'N_monomers', 'N_oligomers',
oligomer_type = 'naive' # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'

use_blinking = False


### FCS settings
use_FCS = True

# Shortest and longest diffusion time to fit (parameter bounds)
tau_diff_min = 1E-4
tau_diff_max = 1E-0

# Shortest and longest lag time to consider in fit (time axis clipping)
FCS_min_lag_time = 0. # Use 0. to use full range of data in .csv file
FCS_max_lag_time = np.inf  # Use np.inf to use full range of data in .csv file


### PCH settings
use_PCH = True
time_resolved_PCH = True
PCH_fitting_accurate = True


# Shortest and longest bin times to consider
PCH_min_bin_time = 0. # Use 0. to use full range of data in .csv file
PCH_max_bin_time = 5E-4 # Use np.inf to use full range of data in .csv file

# Calculation settings
use_parallel = True # Mostly for multi-species PCH
numeric_precision = np.array([1E-3, 1E-4, 1E-5]) # PCH requires numerical precision cutoff, which is set here

command_line_mode = True # if true, suppresses figure display

#%% Metadata/calibration data
FCS_psf_width_nm = 210. # Roughly
FCS_psf_aspect_ratio = 6. # Roughly

acquisition_time_s = 90.

PCH_Q = 8. # More calculation parameter than metadata, but whatever

verbosity = 2

job_prefix = 'solve_model'

#%% Evaluation parameters
n_species = 2

N_avg_obs_0 = 1. #* 0.23204777405164193
cpms_0 = 1E4
tau_diff_0 = 1E-2

N_avg_obs_1 = 1. #* 2.3204777405164193
cpms_1 = 1E4# / np.sqrt(10)
tau_diff_1 = 1E-4

F = 0.4

#%% Dummy file
glob_out_dir = r'C:\Users\Krohn\Desktop\20240611_tempfits'

dir_name = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240423_Test_data\Test_data.sptw\AF488_90nM_power1350\AF488_90nM_power1350_3_T124s_1_20240610_1745'
in_file_name_FCS = '07_ACF_ch0_dt_bg'
in_file_name_PCH = '08_PCMH_ch0'



data_FCS_tau_s, data_FCS_G, avg_count_rate, data_FCS_sigma, acquisition_time_s = utils.read_Kristine_FCS(dir_name, 
                                                                                                          in_file_name_FCS,
                                                                                                          FCS_min_lag_time,
                                                                                                          FCS_max_lag_time)
if use_PCH:
    data_PCH_bin_times, data_PCH_hist = utils.read_PCMH(dir_name,
                                                        in_file_name_PCH,
                                                        PCH_min_bin_time,
                                                        PCH_max_bin_time)
    
fitter = fitting.FCS_spectrum(FCS_psf_width_nm = FCS_psf_width_nm,
                              FCS_psf_aspect_ratio = FCS_psf_aspect_ratio,
                              PCH_Q = PCH_Q,
                              acquisition_time_s = acquisition_time_s if acquisition_time_s > 0 else 90., # Dummy for debugging with old-format data
                              data_FCS_tau_s = data_FCS_tau_s if use_FCS else None,
                              data_FCS_G = data_FCS_G if use_FCS else None,
                              data_FCS_sigma = data_FCS_sigma if use_FCS else None,
                              data_PCH_bin_times = data_PCH_bin_times if use_PCH else None,
                              data_PCH_hist = data_PCH_hist if use_PCH else None,
                              labelling_efficiency = labelling_efficiency,
                              numeric_precision = numeric_precision,
                              PCH_fitting_accurate = PCH_fitting_accurate,
                              verbosity = verbosity,
                              job_prefix = job_prefix
                              )


# Parameter setup
if spectrum_type == 'discrete':
    initial_params = fitter.set_up_params_discrete(use_FCS, 
                                                    use_PCH,
                                                    time_resolved_PCH,
                                                    labelling_correction,
                                                     n_species, 
                                                     tau_diff_min, 
                                                     tau_diff_max, 
                                                     use_blinking
                                                     )
    
elif spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
    initial_params = fitter.set_up_params_par(use_FCS, 
                                               use_PCH, 
                                               time_resolved_PCH,
                                               ('par_Gauss' if spectrum_type in ['reg_MEM', 'reg_CONTIN'] else spectrum_type),
                                               spectrum_parameter, 
                                               oligomer_type, 
                                               incomplete_sampling_correction, 
                                               labelling_correction, 
                                               n_species, 
                                               tau_diff_min, 
                                               tau_diff_max, 
                                               use_blinking
                                               ) 
    
else: # spectrum_type in ['reg_MEM', 'reg_CONTIN']:
    initial_params = fitter.set_up_params_reg(use_FCS,
                                               use_PCH,
                                               time_resolved_PCH,
                                               spectrum_type,
                                               oligomer_type,
                                               incomplete_sampling_correction,
                                               labelling_correction,
                                               n_species,
                                               tau_diff_min,
                                               tau_diff_max,
                                               use_blinking
                                               )
            
initial_params['N_avg_obs_0'].value = N_avg_obs_0
initial_params['cpms_0'].value = cpms_0
initial_params['tau_diff_0'].value = tau_diff_0
initial_params['N_avg_obs_1'].value = N_avg_obs_1
initial_params['cpms_1'].value = cpms_1
initial_params['tau_diff_1'].value = tau_diff_1
initial_params['F'].value = F





time_tag = datetime.datetime.now()

out_name = time_tag.strftime("%Y%m%d-%H%M%S") + f'{in_file_name_FCS if use_FCS else in_file_name_PCH}_fit_{spectrum_type}_{n_species}spec'



if use_FCS:
    if not labelling_correction:
        if spectrum_type == 'discrete':
            model_FCS = fitter.get_acf_full_labelling(initial_params)
        elif  spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            model_FCS = fitter.get_acf_full_labelling_par(initial_params)
        else: #  spectrum_type in ['reg_MEM', 'reg_CONTIN']
            model_FCS = fitter.get_acf_full_labelling_reg(initial_params)

    else:
        if spectrum_type == 'discrete':
            model_FCS = fitter.get_acf_partial_labelling(initial_params)
        else: # spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp', 'reg_MEM', 'reg_CONTIN']:
            model_FCS = fitter.get_acf_partial_labelling_par_reg(initial_params)
    
    # Plot FCS fit
    fig, ax = plt.subplots(nrows=1, ncols=1)
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
    ax.set_ylim(0 , np.max(model_FCS) * 1.2)

    # Save and show figure
    plt.savefig(os.path.join(glob_out_dir, 
                              out_name + '_FCS.png'), 
                dpi=300)
    plt.show()
 

    # Write spreadsheet
    out_table = pd.DataFrame(data = {'Lagtime[s]':data_FCS_tau_s, 
                                      'Correlation': data_FCS_G,
                                      'Uncertainty_SD': data_FCS_sigma,
                                      'Fit': model_FCS})
    out_table.to_csv(os.path.join(glob_out_dir, 
                                  out_name + '_FCS.csv'),
                      index = False, 
                      header = True)


if use_PCH:

    if not labelling_correction:
        model_PCH = fitter.get_pch_full_labelling(initial_params,
                                                  t_bin = data_PCH_bin_times[0],
                                                  spectrum_type = spectrum_type,
                                                  time_resolved_PCH = time_resolved_PCH,
                                                  crop_output = True,
                                                  numeric_precision = np.min(numeric_precision),
                                                  mp_pool = None
                                                  )
    else: 
        model_PCH = fitter.get_pch_partial_labelling(initial_params,
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
    
    ax2.semilogy(np.arange(0, model_PCH.shape[0]),
                  model_PCH,
                  marker = '.',
                  linestyle = '-',
                  color = iter_color)
    
    max_x = model_PCH.shape[0]
    max_y = np.max(model_PCH)
    
    out_dict = {'Photons': np.arange(0, data_PCH_hist.shape[0]), 
                str(data_PCH_bin_times[0]): np.append(model_PCH, np.zeros(data_PCH_hist.shape[0] - model_PCH.shape[0]))}

    if time_resolved_PCH:
        # PCMH
        for i_bin_time in range(1, data_PCH_bin_times.shape[0]):
            iter_color = next(colors)
            if not labelling_correction:
                model_PCH = fitter.get_pch_full_labelling(initial_params,
                                                          t_bin = data_PCH_bin_times[i_bin_time],
                                                          spectrum_type = spectrum_type,
                                                          time_resolved_PCH = time_resolved_PCH,
                                                          crop_output = True,
                                                          numeric_precision = np.min(numeric_precision),
                                                          mp_pool = None
                                                          )
            else: 
                model_PCH = fitter.get_pch_partial_labelling(initial_params,
                                                              t_bin = data_PCH_bin_times[i_bin_time],
                                                              time_resolved_PCH = time_resolved_PCH,
                                                              crop_output = True,
                                                              numeric_precision = np.min(numeric_precision),
                                                              mp_pool = None)
            
            
            
            ax2.semilogy(np.arange(0, model_PCH.shape[0]),
                          model_PCH,
                          marker = '.',
                          linestyle = '-',
                          color = iter_color)
    
            max_x = np.max([max_x, model_PCH.shape[0]])
            max_y = np.max([max_y, np.max(model_PCH)])
            out_dict[str(data_PCH_bin_times[i_bin_time])] =  np.append(model_PCH, np.zeros(data_PCH_hist.shape[0] - model_PCH.shape[0]))

            
            
    ax2.set_xlim(-0.49, max_x - 0.51)
    ax2.set_ylim(1E-6, 1.2)
    ax2.set_title('PCH fit')
    try:
        fig2.supxlabel('Photons in bin')
        fig2.supylabel('Counts')
    except:
        pass
    # Save and show figure
    plt.savefig(os.path.join(glob_out_dir, 
                              out_name + '_PC'+ ('M' if time_resolved_PCH else '') +'H.png'), 
                dpi=300)
    plt.show()

    # Write spreadsheet
    out_table = pd.DataFrame(data = out_dict)
    out_table.to_csv(os.path.join(glob_out_dir, out_name + '_PC'+ ('M' if time_resolved_PCH else '') +'H.csv'),
                      index = False, 
                      header = True)
