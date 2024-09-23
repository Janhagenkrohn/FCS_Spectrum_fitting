# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:56:42 2024

@author: Krohn
"""

# I/O modules
import os
import sys

# Data processing modules
import numpy as np


# Misc
import datetime # Time tags in file names
import traceback # Crash handling
import multiprocessing # Parallel processing

# Custom module
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils
from functions import fitting


'''
About the model and analysis here:
We have data from lipid nanoparticles loaded with fluorescently labelled RNA.
The key questions are:
    1. How much freely-diffusing RNA do we get outside LNPs?
    2. How many RNA molecules are loaded into an LNP on average?

Model structure:
    - Dense-sphere model, as we are observing the content of LNP lumen
    - Try different parameterized oligomer distribution shapes, but probably 
      lognormal or Gamma make most sense.
    - Oligomer size is defined as counting "unit cells" of RNA molecules that 
      could occupy inside an LNP
    - Labelling probability is product of actual labelling probability of an 
      RNA molecule times occupation probability of the "unit cell"

The challenge in this model is that the effective labelling probability is an 
unknown and must be fitted. ALSO, this LNP data, being LNP data, is chaotic and
probably requires incomplete sampling correction.


Required information:
METADATA
RNA labelling probability -> 0.025
PSF width [nm] -> 290
PSF aspect ratio -> 7
FROM MONOMER MEASUREMENT
RNA monomer diffusion time [s] -> 2.28453E-4
RNA monomer brightness [Hz] -> 5897.36792




'''


#%% Define input files and output dir
# Input directory and the labelled protein fraction in each of them
in_dir_names = []
in_file_names_FCS = []
in_file_names_PCH = []
alpha_label = []

glob_in_dir = r'/fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Analysis/20240808_IGialdini_data_export/Exports'



# #%% Fit 1: free RNA monomers to calibrate their diffusion time and brightness 

# _in_dir_names = []
# _alpha_label = []
# local_dir = os.path.join(glob_in_dir, r'siRNA_in_buffer/01_siRNA-Atto565_1to12_25Mhz_1uW__ch1_20240827_1503')
# _in_dir_names.extend([os.path.join(local_dir)])
# _alpha_label.append(0.025)


# # Naming pattern for detecting correct files within subdirs of each in_dir
# # # Bootstrap SD, no burst removal
# file_name_pattern_FCS = '*04_ACF_ch0_bg*'



# # Repeat for FCS
# in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
#                                                                                       file_name_pattern_FCS, 
#                                                                                       _alpha_label,
#                                                                                       '')

# # Workaround as we skip PCH
# [in_dir_names.append(in_dir_name) for in_dir_name in in_dir_names_FCS_tmp]
# [in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in in_file_names_FCS_tmp]
# [alpha_label.append(single_alpha_label) for single_alpha_label in alpha_label_FCS_tmp]

# # Output dir for result file writing
# glob_out_dir = os.path.join(glob_in_dir, 'Fits_discrete')




# # Fit settings
# # All "..._list" settings are handled such that the software will iterate over 
# # the elements of all lists and runs fits with attempt fits with all parameter
# # combinations. NOT ALL COMBINATIONS WORK! Some combinations are hard-coded 
# # excluded and will be skipped, others may crash. It should be easy to 
# # see that there are simply too many combinations for me to debug every one of 
# # them systematically.

# ###### Settings relating to model itself

# labelling_correction_list = [True] 
#     # Whether to consider finite fraction of labelled vs. unlabelled particles in fitting
    
# incomplete_sampling_correction_list = [False] 
#     # Whether to fit deviations between "population-level" and "observation-level"
#     # dynamics, i.e., explicit treatment of an additional layer of noise
    
# labelling_efficiency_incomp_sampling_list = [False] 
#     # Addition to combined incomplete sampling correction and labelling correction 
#     # that also considers noise in observed vs. population-level labelled fractions 
#     # for each oligomer species. CAVE: Computationally very expensive!
        
# use_blinking_list = [False]
#     # Whether to consider blinking in the particle dynamics

# n_species_list = [1]
#     # How many species to evaluate within the range of [tau_diff_min; tau_diff_max]
    
# tau_diff_min_list = [4E-5]
#     # Shortest diffusion time to fit (parameter bounds)
#     # For spectrum models, tau_diff_min is also considered the monomer diffusion time!
    
# tau_diff_max_list = [1E-2]
#     # Longest diffusion time to fit (parameter bounds)
    
# spectrum_type_list = ['discrete'] 
#     # Options: 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
#     # 'discrete' is traditional FCS mixutre model fitting, using few constraints. 
#     #   -> Not recommended for more than 1-2 species.
#     # 'reg' variants are statistically regularized fits with "CONTIN" or maximum entropy constraints.
#     #   -> The fit will attempt to automatically optimize the regularization strength.
#     # 'par' models use simple model functions to parameterize the oligomer concentration spectrum shape
    
# spectrum_parameter_list = ['Amplitude'] 
#     # On which parameter to define regularized or parameterized models
#     # Options: 'Amplitude', 'N_monomers', 'N_oligomers'
    
# oligomer_type_list = ['naive'] 
#     # Choice of oligomer type (basically which polymer-physics-based approximation to use in calculation)
#     # Options: 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
#     # use 'naive' for discrete-species fitting, and can also be used for Amplitude spectra
#     # For monomer N or oligomer N spectrum fitting, you should use a meaningful
#     # physics model to fix a relation between diffusion time and stoichiometry

# discrete_species_list = [
#     [{
#       }
#     ],
#     # [
#     #     {'N_avg_obs': 1., # default 1.
#     #      'vary_N_avg_obs': True, # default True
#     #      'tau_diff': 1E-5,  # default 1E-3
#     #      'vary_tau_diff': False, # default False
#     #      'cpms': 1000.,  # default 1.
#     #      'vary_cpms': True, # default False
#     #      'link_brightness_to_spectrum_monomer': True, # default True - see docstring for details, this one is important!!!
#     #      'stoichiometry': 1.,# default 1.
#     #      'vary_stoichiometry': False, # default (RECOMMENDED) False
#     #      'stoichiometry_binwidth': 1., # default 1.
#     #      'vary_stoichiometry_binwidth': False, # default (RECOMMENDED) False
#     #      'labelling_efficiency': 1., # default 1.
#     #      'vary_labelling_efficiency': False,  # default (RECOMMENDED) False
#     #     }
#     # ]
#     ]
#     # Definition of discrete species to add to the model in addtion to spectrum 
#     # models. Careful about the format: This is a LIST OF LISTS OF DICTS.
#     # Each dict defines the parameters for one species according to the keyword
#     # arguments to fitting.FCS_Spectrum.add_discrete_species(). Note that the
#     # available kwargs give a a lot of freedom to fix or fit whatever parameter 
#     # you want - in fact, perhaps more than is healthy. Some defaults are better 
#     # left untouched unless you understand the model well and are sure that 
#     # changing them is the right thing to do. You can leave out keywords, 
#     # these will be replaced by defaults.
#     # Each list of dicts is a set of discrete species to include in parallel 
#     # in the same fit.
#     # The list of lists of dicts finally is equivalent to the other lists here,
#     # an iteration over different configuration to try in fitting.


# ###### Settings relating to evaluation

# use_FCS_list = [True]
#     # Whether to use correlation function data at all

# use_PCH_list = [False]
#     # Whether to use photon counting histogram data
#     # CAVE: Get computationally prohibitively expensive for many species
#     # or with labelling correction 
    
# time_resolved_PCH_list = [False]
#     # Whether to use photon counting MULTIPLE histograms
#     # If false, by default only the PCH with the shortest bin time contained 
#     # in the data will be used. FCS_Spectrum.run_fit() has a handle to specify 
#     # the index of the desired PCH in case you want to use one single specific 
#     # one: i_bin_time (not used in this script currently)
    
# use_avg_count_rate_list = [True]
#     # Use average count rate to constrain fit? Allows more meaningful estimation 
#     # of molecular brightness. Also helps constrain mixture models of e.g. an
#     # oligomer spectrum and a free-dye species

# fit_label_efficiency_list = [False] 
#     # If you consider finite labelling fraction, here you can also decide to 
#     # make that a fit parameter, although that may be numerically extremely instable


# ###### Metadata/calibration data/settings (global for all fits in batch)

# FCS_min_lag_time = 1E-6
#     # Shortest lag time to consider in fit (time axis clipping)
#     # Specify 0. to use full range of data in .csv file
    
# FCS_max_lag_time = 1E0
#     # Longest lag time to consider in fit (time axis clipping)
#     # Specify np.inf to use full range of data in .csv file

# PCH_min_bin_time = 0. 
#     # Shortest PCMH bin times to consider
#     # Specify 0. to use full range of data in .csv file
    
# PCH_max_bin_time = 5E-4
#     # Longest PCMH bin times to consider
#     # Specify np.inf to use full range of data in .csv file

# NLL_funcs_accurate = True
#     # Accurate maximum-likelihood evaluation, or use faster least-squares 
#     # approximation? Affects most likelihood terms except the chi-square 
#     # minimization on the ACF correlation function
    
# numeric_precision = np.array([1E-3, 1E-4, 1E-5])
#     # PCH requires a numerical precision cutoff, which is set here. The lower 
#     # the number, the more accurate but computationally expensive the 
#     # evaluation. You can specify a single number, then it's trivial, or an 
#     # array, in which case the model is first evaluated with low accuracy and 
#     # precision is then incrementally increased according to the steps you specified.
    
# two_step_fit = False
#     # For some model configuration, you can first run a simpler, more robust, 
#     # version of the fit with some parameters fixed, and then re-fit with the 
#     # "full" model complexity

# verbosity = 0
#     # How much do you want the software to talk?

# FCS_psf_width_nm = np.mean([290])
#     # FCS calibration of PSF width in xy (w_0: 1/e^2 radius), although it is 
#     # actually not used for anything meaningful currently

# FCS_psf_aspect_ratio = np.mean([7])
#     # FCS calibration of PSF aspect ratio (w_z/w_0), also used for PCMH

# PCH_Q = 10. 
#     # Evaluation parameter for PCH

# mp_processes = os.cpu_count() // 2
#     # How many parallel processes?
#     # If mp_processes <= 1, we use multiprocessing WITHIN the fit which allows acceleration of multi-species PCH
#     # If mp_processes > 1, we run multiple fits simultaneously, each in single-thread calculation
#     # mp_processes = os.cpu_count() // 2 to use half of available logical cores 
#     # (-> on many machines all physical cores without hyperthreading)

# suppress_mp = False
#     # For debugging purposes: Forces the software to run entirely without 
#     # multiprocessing, which yields more interpretable error messages
    
# suppress_figs = True
#     # For batch processing: Suppress figure display, jsut write figures to file
    
    
#%% Fit 2: free RNA monomers to calibrate their diffusion time and brightness 

_in_dir_names = []
_alpha_label = []
_in_dir_names.extend([os.path.join(glob_in_dir, 'NP_in_buffer_time_0/01_NP-naked_batch1_HBG_time0_FCS_ch1_20240827_1503')])
_alpha_label.append(0.025)
_in_dir_names.extend([os.path.join(glob_in_dir, 'NP_in_buffer_time_2h/07_NP-naked_batch1_HBG_time2h_FCS_ch1_20240827_1503')])
_alpha_label.append(0.025)


# Naming pattern for detecting correct files within subdirs of each in_dir
# # Bootstrap SD, no burst removal
file_name_pattern_FCS = '*04_ACF_ch0_bg*'



# Repeat for FCS
in_dir_names_FCS_tmp, in_file_names_FCS_tmp, alpha_label_FCS_tmp = utils.detect_files(_in_dir_names, 
                                                                                      file_name_pattern_FCS, 
                                                                                      _alpha_label,
                                                                                      '')

# Workaround as we skip PCH
[in_dir_names.append(in_dir_name) for in_dir_name in in_dir_names_FCS_tmp]
[in_file_names_FCS.append(in_file_name_FCS) for in_file_name_FCS in in_file_names_FCS_tmp]
[alpha_label.append(single_alpha_label) for single_alpha_label in alpha_label_FCS_tmp]

# Output dir for result file writing
glob_out_dir = os.path.join(glob_in_dir, 'Fits_LNPs')




# Fit settings
# All "..._list" settings are handled such that the software will iterate over 
# the elements of all lists and runs fits with attempt fits with all parameter
# combinations. NOT ALL COMBINATIONS WORK! Some combinations are hard-coded 
# excluded and will be skipped, others may crash. It should be easy to 
# see that there are simply too many combinations for me to debug every one of 
# them systematically.

###### Settings relating to model itself

labelling_correction_list = [True] 
    # Whether to consider finite fraction of labelled vs. unlabelled particles in fitting
    
incomplete_sampling_correction_list = [True, False] 
    # Whether to fit deviations between "population-level" and "observation-level"
    # dynamics, i.e., explicit treatment of an additional layer of noise
    
labelling_efficiency_incomp_sampling_list = [False] 
    # Addition to combined incomplete sampling correction and labelling correction 
    # that also considers noise in observed vs. population-level labelled fractions 
    # for each oligomer species. CAVE: Computationally very expensive!
        
use_blinking_list = [False]
    # Whether to consider blinking in the particle dynamics

n_species_list = [50]
    # How many species to evaluate within the range of [tau_diff_min; tau_diff_max]
    
tau_diff_min_list = [2.28453E-4]
    # Shortest diffusion time to fit (parameter bounds)
    # For spectrum models, tau_diff_min is also considered the monomer diffusion time!
    
tau_diff_max_list = [1E-0]
    # Longest diffusion time to fit (parameter bounds)
    
spectrum_type_list = [ 'par_LogNorm', 'par_Gamma'] 
    # Options: 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
    # 'discrete' is traditional FCS mixutre model fitting, using few constraints. 
    #   -> Not recommended for more than 1-2 species.
    # 'reg' variants are statistically regularized fits with "CONTIN" or maximum entropy constraints.
    #   -> The fit will attempt to automatically optimize the regularization strength.
    # 'par' models use simple model functions to parameterize the oligomer concentration spectrum shape
    
spectrum_parameter_list = ['Amplitude', 'N_monomers', 'N_oligomers'] 
    # On which parameter to define regularized or parameterized models
    # Options: 'Amplitude', 'N_monomers', 'N_oligomers'
    
oligomer_type_list = ['sherical_dense'] 
    # Choice of oligomer type (basically which polymer-physics-based approximation to use in calculation)
    # Options: 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
    # use 'naive' for discrete-species fitting, and can also be used for Amplitude spectra
    # For monomer N or oligomer N spectrum fitting, you should use a meaningful
    # physics model to fix a relation between diffusion time and stoichiometry

discrete_species_list = [
    [{
      }
    ],
    [
        {
            'N_avg_obs': 0.001, # default 1.
            'vary_N_avg_obs': True, # default True
            'tau_diff': 2.28453E-4,  # default 1E-3
            'vary_tau_diff': False, # default False
            'cpms': 5897.36792,  # default 1.
            'vary_cpms': False, # default False
            'link_brightness_to_spectrum_monomer': True, # default True - see docstring for details, this one is important!!!
            'stoichiometry': 1.,# default 1.
            'vary_stoichiometry': False, # default (RECOMMENDED) False
            'stoichiometry_binwidth': 1., # default (RECOMMENDED) 1.
            'vary_stoichiometry_binwidth': False, # default (RECOMMENDED) False
            'labelling_efficiency': 0.025, # default 1.
            'vary_labelling_efficiency': False,  # default (RECOMMENDED) False
        }
    ]
    ]
    # Definition of discrete species to add to the model in addtion to spectrum 
    # models. Careful about the format: This is a LIST OF LISTS OF DICTS.
    # Each dict defines the parameters for one species according to the keyword
    # arguments to fitting.FCS_Spectrum.add_discrete_species(). Note that the
    # available kwargs give a a lot of freedom to fix or fit whatever parameter 
    # you want - in fact, perhaps more than is healthy. Some defaults are better 
    # left untouched unless you understand the model well and are sure that 
    # changing them is the right thing to do. You can leave out keywords, 
    # these will be replaced by defaults.
    # Each list of dicts is a set of discrete species to include in parallel 
    # in the same fit (e.g., one for non-conjugated free dye, one for free 
    # protein monomers).
    # The list of lists of dicts finally is equivalent to the other lists here,
    # an iteration over different configuration to try in fitting.


###### Settings relating to evaluation

use_FCS_list = [True]
    # Whether to use correlation function data at all

use_PCH_list = [False]
    # Whether to use photon counting histogram data
    # CAVE: Get computationally prohibitively expensive for many species
    # or with labelling correction 
    
time_resolved_PCH_list = [False]
    # Whether to use photon counting MULTIPLE histograms
    # If false, by default only the PCH with the shortest bin time contained 
    # in the data will be used. FCS_Spectrum.run_fit() has a handle to specify 
    # the index of the desired PCH in case you want to use one single specific 
    # one: i_bin_time (not used in this script currently)
    
use_avg_count_rate_list = [True]
    # Use average count rate to constrain fit? Allows more meaningful estimation 
    # of molecular brightness. Also helps constrain mixture models of e.g. an
    # oligomer spectrum and a free-dye species

fit_label_efficiency_list = [True, False] 
    # If you consider finite labelling fraction, here you can also decide to 
    # make that a fit parameter, although that may be numerically extremely instable


###### Metadata/calibration data/settings (global for all fits in batch)

FCS_min_lag_time = 1E-6
    # Shortest lag time to consider in fit (time axis clipping)
    # Specify 0. to use full range of data in .csv file
    
FCS_max_lag_time = 1E0
    # Longest lag time to consider in fit (time axis clipping)
    # Specify np.inf to use full range of data in .csv file

PCH_min_bin_time = 0. 
    # Shortest PCMH bin times to consider
    # Specify 0. to use full range of data in .csv file
    
PCH_max_bin_time = 5E-4
    # Longest PCMH bin times to consider
    # Specify np.inf to use full range of data in .csv file

NLL_funcs_accurate = False
    # Accurate maximum-likelihood evaluation, or use faster least-squares 
    # approximation? Affects most likelihood terms except the chi-square 
    # minimization on the ACF correlation function
    
numeric_precision = np.array([1E-3, 1E-4, 1E-5])
    # PCH requires a numerical precision cutoff, which is set here. The lower 
    # the number, the more accurate but computationally expensive the 
    # evaluation. You can specify a single number, then it's trivial, or an 
    # array, in which case the model is first evaluated with low accuracy and 
    # precision is then incrementally increased according to the steps you specified.
    
two_step_fit = False
    # For some model configuration, you can first run a simpler, more robust, 
    # version of the fit with some parameters fixed, and then re-fit with the 
    # "full" model complexity

verbosity = 0
    # How much do you want the software to talk?

FCS_psf_width_nm = np.mean([290])
    # FCS calibration of PSF width in xy (w_0: 1/e^2 radius), although it is 
    # actually not used for anything meaningful currently

FCS_psf_aspect_ratio = np.mean([7])
    # FCS calibration of PSF aspect ratio (w_z/w_0), also used for PCMH

PCH_Q = 10. 
    # Evaluation parameter for PCH

mp_processes = 24
    # How many parallel processes?
    # If mp_processes <= 1, we use multiprocessing WITHIN the fit which allows acceleration of multi-species PCH
    # If mp_processes > 1, we run multiple fits simultaneously, each in single-thread calculation
    # mp_processes = os.cpu_count() // 2 to use half of available logical cores 
    # (-> on many machines all physical cores without hyperthreading)

suppress_mp = False
    # For debugging purposes: Forces the software to run entirely without 
    # multiprocessing, which yields more interpretable error messages
    
suppress_figs = True
    # For batch processing: Suppress figure display, jsut write figures to file

#%% Wrap all permutations for different fit settings and all files...Long list!
# Iterate over all settings and files
list_of_parameter_tuples = []
fit_counter = 1
print(f'Sanity check all: Found {len(in_file_names_FCS)} FCS files, {len(in_file_names_PCH)} PCH files, {len(in_dir_names)} dir names.')
for use_FCS in use_FCS_list:
    for use_PCH in use_PCH_list:
        for time_resolved_PCH in time_resolved_PCH_list:
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
                                                    for use_avg_count_rate in use_avg_count_rate_list:
                                                        for fit_label_efficiency in fit_label_efficiency_list:
                                                            for discrete_species in discrete_species_list:
                                                    
                                                                # a number of sanity-checks to make sure we do not waste time planning fits that cannot work:
                                                                if (type(use_FCS) == bool and
                                                                    type(use_PCH) == bool and
                                                                    type(time_resolved_PCH) == bool and
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
                                                                    type(use_avg_count_rate) == bool and
                                                                    type(fit_label_efficiency) == bool and
                                                                    type(discrete_species) == list and
                                                                    np.all([type(element) == dict for element in discrete_species])
                                                                    ):
                                                                
                                                                    fit_settings_str = f'{spectrum_type}_{n_species}spec'
                                                                    fit_settings_str += f'_{oligomer_type}_{spectrum_parameter}' if spectrum_type != 'discrete' else ''
                                                                    fit_settings_str += '_blink' if use_blinking else ''
                                                                    fit_settings_str += '_lblcr' if labelling_correction else ''
                                                                    fit_settings_str += '_smplcr' if incomplete_sampling_correction else ''
                                                                    fit_settings_str += '_lblsmplcr' if labelling_efficiency_incomp_sampling else ''
                                                                    fit_settings_str += f'_{len(discrete_species)}discr' if discrete_species != [{}] else ''
                                                                    fit_settings_str += '_FCS' if use_FCS else ''
                                                                    fit_settings_str += ('_PCMH' if time_resolved_PCH else '_PCH') if use_PCH else ''
                                                                    fit_settings_str += ('_MLE' if NLL_funcs_accurate else '_WLSQ') if (use_PCH or incomplete_sampling_correction) else ''
                                                                    
                                                                    for i_file, dir_name in enumerate(in_dir_names):
                                                                        job_prefix = in_file_names_FCS[i_file] + '_' + fit_settings_str
                                                                        save_path = os.path.join(glob_out_dir, fit_settings_str)
                                                                        if not os.path.exists(save_path):
                                                                            os.makedirs(save_path)
                                                                        
                                                                        fit_res_table_path = os.path.join(save_path, 'Fit_params_' + fit_settings_str)
                                                                        
                                                                        parameter_tuple = (fit_res_table_path,
                                                                                           fit_counter,
                                                                                           i_file,
                                                                                           job_prefix,
                                                                                           save_path,
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
                                                                                           NLL_funcs_accurate,
                                                                                           use_avg_count_rate,
                                                                                           fit_label_efficiency,
                                                                                           numeric_precision,
                                                                                           two_step_fit,
                                                                                           discrete_species,
                                                                                           verbosity,
                                                                                           mp_processes <= 1 and not suppress_mp,
                                                                                           suppress_figs)
                                                                        list_of_parameter_tuples.extend((parameter_tuple,))
                                                                        fit_counter += 1
                                                                            
                                                                            
                                                                            
#%% Parallel execution function...
def par_func(fit_res_table_path,
             fit_number,
             i_file,
             job_prefix,
             save_path,
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
             time_resolved_PCH,
             n_species_target,
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
             NLL_funcs_accurate,
             use_avg_count_rate,
             fit_label_efficiency,
             numeric_precision,
             two_step_fit,
             discrete_species,
             verbosity,
             use_parallel,
             suppress_figs):
    
    # Command line message
    if verbosity >= 1:
        time_tag = datetime.datetime.now()
        message = f'[{job_prefix}] [{fit_number}] Fitting '
        message += in_file_name_FCS if use_FCS else ''
        message += ' and ' if (use_FCS and use_PCH) else ''
        message += in_file_name_PCH if use_PCH else ''
        message += ' globally:' if (use_FCS and use_PCH) else ''
        print('\n' + time_tag.strftime("%Y-%m-%d %H:%M:%S") + '\n' + message)
    
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
        if verbosity >= 1:
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
                                  data_avg_count_rate = avg_count_rate,
                                  labelling_efficiency = labelling_efficiency,
                                  numeric_precision = numeric_precision,
                                  NLL_funcs_accurate = NLL_funcs_accurate,
                                  verbosity = verbosity,
                                  job_prefix = job_prefix,
                                  labelling_efficiency_incomp_sampling = labelling_efficiency_incomp_sampling
                                  )
    try:
        if spectrum_type in ['discrete', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            fit_result = fitter.run_fit(use_FCS = use_FCS, # bool
                                        use_PCH = use_PCH, # bool
                                        time_resolved_PCH = time_resolved_PCH, # bool
                                        spectrum_type = spectrum_type, # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
                                        spectrum_parameter = spectrum_parameter, # 'Amplitude', 'N_monomers', 'N_oligomers',
                                        oligomer_type = oligomer_type, # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
                                        labelling_correction = labelling_correction, # bool
                                        incomplete_sampling_correction = incomplete_sampling_correction, # bool
                                        n_species = n_species_target, # int
                                        tau_diff_min = tau_diff_min, # float
                                        tau_diff_max = tau_diff_max, # float
                                        use_blinking = use_blinking, # bool
                                        use_avg_count_rate = use_avg_count_rate, # Bool
                                        fit_label_efficiency = fit_label_efficiency, # Bool
                                        two_step_fit = two_step_fit, # bool
                                        discrete_species = discrete_species, # list of dicts
                                        use_parallel = use_parallel, # Bool
                                        )
            
                        
            # Dummies
            N_pop_array = np.array([])
            lagrange_mul = np.inf # Is defined such that high lagrange_mul means weak/no regularization
            
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
                                                                    n_species = n_species_target, # int
                                                                    tau_diff_min = tau_diff_min, # float
                                                                    tau_diff_max = tau_diff_max, # float
                                                                    use_blinking = use_blinking, # bool
                                                                    use_avg_count_rate = use_avg_count_rate, # Bool
                                                                    fit_label_efficiency = fit_label_efficiency, # Bool
                                                                    two_step_fit = two_step_fit, # bool
                                                                    discrete_species = discrete_species, # list of dicts
                                                                    use_parallel = use_parallel # Bool
                                                                    )
            
        if not fit_result == None:    
            # [print(key) for key in fit_result.params.keys()]
            _ = utils.write_fit_results(fit_result,
                                        fitter,
                                        save_path,
                                        spectrum_type,
                                        labelling_correction,
                                        incomplete_sampling_correction,
                                        use_FCS,
                                        use_PCH,
                                        time_resolved_PCH,
                                        job_prefix,
                                        fit_res_table_path,
                                        N_pop_array,
                                        lagrange_mul,
                                        fit_number,
                                        i_file,
                                        in_file_name_FCS,
                                        in_file_name_PCH,
                                        dir_name,
                                        suppress_figs = suppress_figs
                                        )
            
            return fit_result, fitter, N_pop_array, lagrange_mul

        else: # No valid fit result
            if verbosity >= 1:
                # Command line message
                message = f'[{job_prefix}] Failed to fit '
                message += in_file_name_FCS if use_FCS else ''
                message += ' and ' if (use_FCS and use_PCH) else ''
                message += in_file_name_PCH if use_PCH else ''
                print(message)
            return None, fitter, None, None
        
        
    except:
        traceback.print_exc()
        return None, fitter, None, None


#%% Preparations done - run fits

if mp_processes > 1:
    try:
        mp_pool = multiprocessing.Pool(processes = mp_processes)
        
        _ = [mp_pool.starmap(par_func, list_of_parameter_tuples)]
    except:
        traceback.print_exception()
    finally:
        mp_pool.close()
        
else:
    # Single process analysis
    for i_fit in range(len(list_of_parameter_tuples)):
        _ = par_func(*list_of_parameter_tuples[i_fit])
        
print('Job done.')