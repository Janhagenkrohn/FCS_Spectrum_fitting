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

# Custom module
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils
from functions import fitting



#%% Define input files and output dir
# Input directory and the labelled protein fraction in each of them
in_dir_names= []
alpha_label = []
# glob_dir = '/fs/pool/pool-schwille-spt/Experiment_analysis/20231117_JHK_NKaletta_ParM_oligomerization'
glob_in_dir = '/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Data/'



''' Labelled protein fraction'''
in_dir_names.extend([os.path.join(glob_in_dir, 'simFCS2_simulations/3a/Exports')])
alpha_label.append(1.) 
in_dir_names.extend([os.path.join(glob_in_dir, 'simFCS2_simulations/3b/Exports')])
alpha_label.append(1.) 
in_dir_names.extend([os.path.join(glob_in_dir, 'simFCS2_simulations/3c/Exports')])
alpha_label.append(1.) 
in_dir_names.extend([os.path.join(glob_in_dir, 'simFCS2_simulations/3d/Exports')])
alpha_label.append(1.) 


# Naming pattern for detecting correct files within subdirs of each in_dir
file_name_pattern_PCH = '*batch*_PCMH_ch0*' # Dual-channel PCH
file_name_pattern_FCS = '*batch*_ACF_ch0*' # CCF

# Output dir for result file writing
glob_out_dir ='/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Analysis/20240608_Fits_for_figures/'


#%% Fit settings

### General model settings

labelling_correction_list = [False]
incomplete_sampling_correction_list = [False]

n_species_list = [1, 2]
spectrum_type_list = ['discrete'] # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
spectrum_parameter_list = ['Amplitude'] # 'Amplitude', 'N_monomers', 'N_oligomers',
oligomer_type_list = ['naive'] # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'

use_blinking_list = [False]


### FCS settings
use_FCS_list = [True, False]

# Shortest and longest diffusion time to fit (parameter bounds)
tau_diff_min_list = [1E-5]
tau_diff_max_list = [1E-0]

# Shortest and longest lag time to consider in fit (time axis clipping)
FCS_min_lag_time_list = [0.] # Use 0. to use full range of data in .csv file
FCS_max_lag_time_list = [np.inf]  # Use np.inf to use full range of data in .csv file


### PCH settings
use_PCH_list = [True]
time_resolved_PCH_list = [True, False]
PCH_fitting_accurate_list = [True, False] # Accurate MLE or least-squares approximation?

# Shortest and longest bin times to consider
PCH_min_bin_time_list = [0.] # Use 0. to use full range of data in .csv file
PCH_max_bin_time_list = [5E-4] # Use np.inf to use full range of data in .csv file

# Calculation settings
numeric_precision_list = [np.array([1E-3, 1E-4, 1E-5])] # PCH requires numerical precision cutoff, which is set here


#%% Metadata/calibration data/settings that are global for all measurements
verbosity = 0 # How much do you want the software to talk?
FCS_psf_width_nm = 350. # Roughly
FCS_psf_aspect_ratio = 5. # Roughly

acquisition_time_s = 90.

PCH_Q = 10. # More calculation parameter than metadata, but whatever

mp_processes = 10 # How many parallel processes?


#%% Wrap all permutations for different fit settings and all files...Long list!

# Detect PCH files
in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(in_dir_names,
                                                                          file_name_pattern_PCH, 
                                                                          alpha_label, 
                                                                          '')

# Repeat for FCS
in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(in_dir_names, 
                                                                          file_name_pattern_FCS, 
                                                                          alpha_label,
                                                                          '')
in_dir_names, in_file_names_FCS, in_file_names_PCH, alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
                                                                                               in_file_names_FCS,
                                                                                               alpha_label_FCS,
                                                                                               in_dir_names_PCH,
                                                                                               in_file_names_PCH,
                                                                                               alpha_label_PCH)



# Iterate over all settings and files
list_of_parameter_tuples = []

for use_FCS in use_FCS_list:
    for FCS_min_lag_time in FCS_min_lag_time_list:
        for FCS_max_lag_time in FCS_max_lag_time_list:
            for use_PCH in use_PCH_list:
                for PCH_min_bin_time in PCH_min_bin_time_list:
                    for PCH_max_bin_time in PCH_max_bin_time_list:
                        for time_resolved_PCH in time_resolved_PCH_list:
                            for PCH_fitting_accurate in PCH_fitting_accurate_list:
                                for n_species in n_species_list:
                                    for tau_diff_min in tau_diff_min_list:
                                        for tau_diff_max in tau_diff_max_list:
                                            for use_blinking in use_blinking_list:
                                                for spectrum_type in spectrum_type_list:
                                                    for spectrum_parameter in spectrum_parameter_list:
                                                        for oligomer_type in oligomer_type_list:
                                                            for labelling_correction in labelling_correction_list:
                                                                for incomplete_sampling_correction in incomplete_sampling_correction_list:
                                                                    for numeric_precision in numeric_precision_list:
                                                                        
                                                                        # a number of sanity-checks to make sure we skip fits that make no sense:
                                                                            # (does not capture EVERY kind of nonsense setting though) 
                                                                        if (type(use_FCS) == bool and
                                                                            FCS_min_lag_time >= 0 and
                                                                            FCS_max_lag_time > FCS_min_lag_time and
                                                                            type(use_PCH) == bool and
                                                                            PCH_min_bin_time >= 0 and
                                                                            PCH_max_bin_time >= PCH_min_bin_time and
                                                                            type(time_resolved_PCH) == bool and
                                                                            type(PCH_fitting_accurate) == bool and
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
                                                                            fit_settings_str += '_FCS' if use_FCS else ''
                                                                            fit_settings_str += ('_PCMH' if time_resolved_PCH else '_PCH') if use_PCH else ''
                                                                            fit_settings_str += ('_MLE' if PCH_fitting_accurate else '_WLSQ') if use_PCH else ''
                                                                            
                                                                            for i_file, dir_name in enumerate(in_dir_names):
                                                                                job_prefix = in_file_names_FCS[i_file] + ' -- ' + fit_settings_str
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
                                                                                                    PCH_fitting_accurate,
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
                    tau_diff_min,
                    tau_diff_max,
                    use_blinking,
                    spectrum_type,
                    spectrum_parameter,
                    oligomer_type,
                    labelling_efficiency,
                    labelling_correction,
                    incomplete_sampling_correction,
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
        if use_FCS:
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
                                        use_parallel = False # Bool
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
                                                                   use_parallel = False # Bool
                                                                   )
            
        
        if not fit_result == None:
            fit_params = fit_result.params
            
            # Recalculate number of species, it is possible we lose some in between
            n_species = fitter.get_n_species(fit_params)
            
            out_name = os.path.join(save_path,
                                    time_tag.strftime("%Y%m%d-%H%M%S") + f'{in_file_name_FCS if use_FCS else in_file_name_PCH}_fit_{spectrum_type}_{n_species}spec')
            
            # Command line preview of fit results
            print(f' [{job_prefix}]   Fitted parameters:')
            [print(f'[{job_prefix}] {key}: {fit_params[key].value}') for key in fit_params.keys() if fit_params[key].vary]
            
            # Write fit results
            fit_result_dict = {}            
            fit_result_dict['file'] = in_file_name_FCS if use_FCS else in_file_name_PCH 
            
            for key in fit_params.keys():
                # Fit parameters
                fit_result_dict[key + '_val'] = fit_params[key].value
                fit_result_dict[key + '_vary'] = 'Vary' if fit_params[key].vary else 'Fix_Dep'
                if not fit_params[key].stderr == None:
                    fit_result_dict[key + '_err'] = fit_params[key].stderr
                    
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
                
            # Additional spreadsheets for N and diffusion time spectra
            if n_species > 1:
                fit_res_N_path = fit_res_table_path + '_N.csv' # N-only spreadsheet for convenience
                fit_res_tau_diff_path = fit_res_table_path + '_tau_diff.csv' # tau_diff-only spreadsheet for convenience
                N_result_dict = {}
                tau_diff_max_result_dict = {}
                
                # Metadata
                N_result_dict['file'] = in_file_name_FCS if use_FCS else in_file_name_PCH 
                tau_diff_max_result_dict['file'] = in_file_name_FCS if use_FCS else in_file_name_PCH 
                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    N_result_dict['lagrange_mul'] = lagrange_mul
                    tau_diff_max_result_dict['lagrange_mul'] = lagrange_mul
                    
                # Species parameters
                for i_spec in range(n_species):
                    if spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
                        N_result_dict[f'N_avg_pop_{i_spec}'] = fit_params[f'N_avg_pop_{i_spec}'].value
                    elif spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                        N_result_dict[f'N_avg_pop_{i_spec}'] = N_pop_array[i_spec]
                    tau_diff_max_result_dict[f'tau_diff_{i_spec}'] = fit_params[f'tau_diff_{i_spec}'].value
                    
                if incomplete_sampling_correction or spectrum_type == 'discrete':
                    # Additional difference between sample-level and observation-level N
                    # Also, discrete only has N_avg_obs, more for historical reasons. Too lazy to fix...
                    for i_spec in range(n_species):
                        N_result_dict[f'N_avg_obs_{i_spec}'] = fit_params[f'N_avg_obs_{i_spec}'].value
                
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
                
                try:
                    fig.supxlabel('Correlation time [s]')
                    fig.supylabel('G(\u03C4)')
                except:
                    # Apparently this can fail???
                    pass
                ax.set_xlim(data_FCS_tau_s[0], data_FCS_tau_s[-1])
                plot_y_min_max = (np.percentile(data_FCS_G, 3), np.percentile(data_FCS_G, 97))
                ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                            plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)
        
                # Save and show figure
                plt.savefig(os.path.join(save_path, 
                                         out_name + '_FCS.png'), 
                            dpi=300)
                plt.close()
                
                
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
                plt.close()
    
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
        else:
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

try:
    mp_pool = multiprocessing.Pool(processes = mp_processes)
    
    _ = [mp_pool.starmap(fitting_parfunc, list_of_parameter_tuples)]
except:
    traceback.print_exception()
finally:
    mp_pool.close()