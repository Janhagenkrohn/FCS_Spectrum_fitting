# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

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
glob_dir = '/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Data/simFCS2_simulations'


''' Labelled protein fraction'''
in_dir_names.extend([os.path.join(glob_dir, 'MEM_testbatch')])

alpha_label.append(1.) 


# Naming pattern for detecting correct files within subdirs of each in_dir
file_name_pattern_PCH = '*batch*_PCMH_ch0*' # Dual-channel PCH
file_name_pattern_FCS = '*batch*_ACF_ch0*' # CCF

# Output dir for result file writing
save_path = os.path.join(glob_dir, 'Testfit/MEM_test_new')



#%% Fit settings

### General model settings

labelling_correction = False
labelling_efficiency = 1.
incomplete_sampling_correction = False

n_species = 70
spectrum_type = 'reg_MEM' # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
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
use_PCH = False
time_resolved_PCH = True

# Shortest and longest bin times to consider
PCH_min_bin_time = 0. # Use 0. to use full range of data in .csv file
PCH_max_bin_time = 5E-4 # Use np.inf to use full range of data in .csv file

# Calculation settings
use_parallel = True # Mostly for multi-species PCMH
numeric_precision = np.array([1E-3, 1E-4, 1E-5]) # PCH requires numerical precision cutoff, which is set here

command_line_mode = True # if true, suppresses figure display



#%% Metadata/calibration data
FCS_psf_width_nm = 210. # Roughly
FCS_psf_aspect_ratio = 6. # Roughly

acquisition_time_s = 90.

PCH_Q = 8. # More calculation parameter than metadata, but whatever




#%% Input interpretation

# Prepare output
if not os.path.exists(save_path):
    os.makedirs(save_path)



# Automatic input file detection

if use_PCH:
    # Detect PCH files
    in_dir_names_PCH, in_file_names_PCH, alpha_label_PCH = utils.detect_files(in_dir_names,
                                                                          file_name_pattern_PCH, 
                                                                          alpha_label, 
                                                                          save_path)

if use_FCS:
    # Repeat for FCS
    in_dir_names_FCS, in_file_names_FCS, alpha_label_FCS = utils.detect_files(in_dir_names, 
                                                                file_name_pattern_FCS, 
                                                                alpha_label,
                                                                save_path)
if use_PCH and use_FCS:
    in_dir_names, in_file_names_FCS, in_file_names_PCH, alpha_label = utils.link_FCS_and_PCH_files(in_dir_names_FCS,
                                                                                                   in_file_names_FCS,
                                                                                                   alpha_label_FCS,
                                                                                                   in_dir_names_PCH,
                                                                                                   in_file_names_PCH,
                                                                                                   alpha_label_PCH)
elif use_PCH and not use_FCS:
    in_dir_names = in_dir_names_PCH
else: # use_FCS and not use_PCH:
    in_dir_names = in_dir_names_FCS
    
    
# Build names for results spreadsheets
fit_res_path = os.path.join(save_path, 
                            datetime.datetime.now().strftime("%Y%m%d"))
fit_res_path += f'_Fit_params_{spectrum_type}' 
fit_res_path += f'_{spectrum_parameter}_{oligomer_type}' if (not spectrum_type == 'discrete') else ''
fit_res_path += f'_{n_species}spec'
fit_res_path += '_FCS' if use_FCS else '' 
fit_res_path += '_PCHM' if (use_PCH and time_resolved_PCH) else ('_PCH' if use_PCH else '')
fit_res_full_path = fit_res_path + '.csv' # General spreadsheet with all of them

if n_species > 1:
    fit_res_N_path = fit_res_path + '_N.csv' # N-only spreadsheet for convenience
    fit_res_tau_diff_path = fit_res_path + '_tau_diff.csv' # tau_diff-only spreadsheet for convenience
        

#%% Start processing
for i_file, dir_name in enumerate(in_dir_names):
    
    # Command line message
    time_tag = datetime.datetime.now()
    message = 'Fitting '
    message += in_file_names_FCS[i_file] if use_FCS else ''
    message += ' and ' if (use_FCS and use_PCH) else ''
    message += in_file_names_PCH[i_file] if use_PCH else ''
    message += ' globally:' if (use_FCS and use_PCH) else ''
    print('\n' + time_tag.strftime("%Y-%m-%d %H:%M:%S") + '\n' + message)
    

    try:
        if use_FCS:
            data_FCS_tau_s, data_FCS_G, avg_count_rate, data_FCS_sigma, acquisition_time_s = utils.read_Kristine_FCS(dir_name, 
                                                                                                                     in_file_names_FCS[i_file],
                                                                                                                     FCS_min_lag_time,
                                                                                                                     FCS_max_lag_time)
        if use_PCH:
            data_PCH_bin_times, data_PCH_hist = utils.read_PCMH(dir_name,
                                                                in_file_names_PCH[i_file],
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
                                      verbosity = 2
                                      )
        
        if spectrum_type in ['discrete', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
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
            
        else: # spectrum_type in ['reg_MEM', 'reg_CONTIN']
            # Here we get more complex output
            fit_result, N_pop_array, lagrange_mul = fitter.run_fit(use_FCS = use_FCS, # bool
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
        
        if not fit_result == None:
            fit_params = fit_result.params
            out_name = os.path.join(save_path,
                                    time_tag.strftime("%Y%m%d-%H%M%S") + f'{in_file_names_FCS[i_file] if use_FCS else in_file_names_PCH[i_file] }_fit_{spectrum_type}_{n_species}spec')
            
            # Command line preview of fit results
            print('   Fitted parameters:')
            [print(f'{key}: {fit_params[key].value}') for key in fit_params.keys() if fit_params[key].vary]
            
            # Write fit results
            fit_result_dict = {}            
            fit_result_dict['file'] = in_file_names_FCS[i_file] if use_FCS else in_file_names_PCH[i_file] 
            
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
            
            if not os.path.isfile(fit_res_full_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_full_path, 
                                     header = True, 
                                     index = False)
            else:
                # Exists - append
                fit_result_df.to_csv(fit_res_full_path, 
                                     mode = 'a', 
                                     header = False, 
                                     index = False)
                
            # Additional spreadsheets for N and diffusion time spectra
            if n_species > 1:
                N_result_dict = {}
                tau_diff_max_result_dict = {}
                
                # Metadata
                N_result_dict['file'] = in_file_names_FCS[i_file] if use_FCS else in_file_names_PCH[i_file] 
                tau_diff_max_result_dict['file'] = in_file_names_FCS[i_file] if use_FCS else in_file_names_PCH[i_file] 
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
        
                fig.supxlabel('Correlation time [s]')
                fig.supylabel('G(\u03C4)')
                ax.set_xlim(data_FCS_tau_s[0], data_FCS_tau_s[-1])
                plot_y_min_max = (np.percentile(data_FCS_G, 3), np.percentile(data_FCS_G, 97))
                ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                            plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)
        
                # Save and show figure
                plt.savefig(os.path.join(save_path, 
                                         out_name + '_FCS.png'), 
                            dpi=300)
                if not command_line_mode:
                    # Show figure
                    plt.show()
                else:
                    # Auto-close figure to avoid pileup
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
                fig2.supxlabel('Photons in bin')
                fig2.supylabel('Counts')
        
                # Save and show figure
                plt.savefig(os.path.join(save_path, 
                                         out_name + '_PC'+ ('M' if time_resolved_PCH else '') +'H.png'), 
                            dpi=300)
                if not command_line_mode:
                    # Show figure
                    plt.show()
                else:
                    # Auto-close figure to avoid pileup
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
            message = 'Failed to fit '
            message += in_file_names_FCS[i_file] if use_FCS else ''
            message += ' and ' if (use_FCS and use_PCH) else ''
            message += in_file_names_PCH[i_file] if use_PCH else ''
            print(message)
    except:
        traceback.print_exc()
        
        
