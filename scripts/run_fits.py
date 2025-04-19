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



#%% Define input files and output dir
# Input directory and the labelled protein fraction in each of them
in_dir_names = []
in_file_names_FCS = []
in_file_names_PCH = []
alpha_label = []

glob_in_dir = r'D:\temp\FCS_Spectrum_debug\__Data'



#%% 20240604 - A488-labelled ParM 

####### NO BURST REMOVAL #########
# SHould be good for incomplete-sampling stuff
''' Labelled protein fraction'''
_in_dir_names = []
_alpha_label = []
local_dir = os.path.join(glob_in_dir, r'ssRNA_ladder_RCT_50uM_1_T0s_1_20240826_1203')
_in_dir_names.extend([os.path.join(local_dir)])
_alpha_label.append(0.025)
# local_dir = os.path.join(glob_in_dir, r'_4s_0_AF488_long_1_T0s_120240826_1234')
# _in_dir_names.extend([os.path.join(local_dir)])
# _alpha_label.append(1.)

# [ _in_dir_names.extend([os.path.join(local_dir, f'20240808_more_ssRNA/20240808_data.sptw/ssRNA_IVT{x}_1')]) for x in [*range(1,9), 'mix']]
# [_alpha_label.append(5E-3) for x in [*range(1,9), 'mix']]

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
glob_out_dir = r'D:\temp\FCS_Spectrum_debug\5'




#%% Fit settings
# All "..._list" settings are handled such that the software will iterate over 
# the elements of all lists and runs fits with attempt fits with all parameter
# combinations. NOT ALL COMBINATIONS WORK! Some combinations are hard-coded 
# excluded and will be skipped, others may crash. It should be easy to 
# see that there are simply too many combinations for me to debug every one of 
# them systematically.

###### Settings relating to model itself

labelling_correction_list = [False, True] 
    # Whether to consider finite fraction of labelled vs. unlabelled particles in fitting
    
incomplete_sampling_correction_list = [True] 
    # Whether to fit deviations between "population-level" and "observation-level"
    # dynamics, i.e., explicit treatment of an additional layer of noise
    # If True, very strongy recommended to use with settings:
        # two_step_fit == True and NLL_funcs_accurate == True
    
labelling_efficiency_incomp_sampling_list = [False] 
    # Addition to combined incomplete sampling correction and labelling correction 
    # that also considers noise in observed vs. population-level labelled fractions 
    # for each oligomer species. CAVE: Computationally very expensive!
        
use_blinking_list = [False]
    # Whether to consider blinking in the particle dynamics
    # Careful: In the implementation here, tau_diff_min serves as an UPPER 
    # bound on the blinking time! So when using blinking, make sure not to set 
    # tau_diff_min too short, otherwise the blinking term will just to some 
    # weird nonsense.
        
n_species_list = [70]
    # How many species to evaluate within the range of [tau_diff_min; tau_diff_max]
    
tau_diff_min_list = [1.56E-5]
# tau_diff_min_list = [1e-6]
    # Shortest diffusion time to fit (parameter bounds)
    # For spectrum models, tau_diff_min is also considered the monomer diffusion time!
    
tau_diff_max_list = [1E0]
    # Longest diffusion time to fit (parameter bounds)
    
spectrum_type_list = ['par_LogNorm'] 
    # Options: 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
    # 'discrete' is traditional FCS mixutre model fitting, using few constraints. 
    #   -> Not recommended for more than 1-2 species.
    # 'reg' variants are statistically regularized fits with "CONTIN" or maximum entropy constraints.
    #   -> The fit will attempt to automatically optimize the regularization strength.
    # 'par' models use simple model functions to parameterize the oligomer concentration spectrum shape
    
spectrum_parameter_list = ['N_monomers'] 
    # On which parameter to define regularized or parameterized models
    # Options: 'Amplitude', 'N_monomers', 'N_oligomers'
    
oligomer_type_list = ['spherical_shell'] 
    # Choice of oligomer type (basically which polymer-physics-based approximation to use in calculation)
    # Options: 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
    # use 'naive' for discrete-species fitting, and can also be used for Amplitude spectra
    # For monomer N or oligomer N spectrum fitting, you should use a meaningful
    # physics model to fix a relation between diffusion time and stoichiometry

fixed_spectrum_params_list = [
    {
      'N_dist_amp': np.nan,
      'N_dist_a': np.nan,
      'N_dist_b': np.nan,
      'j_avg_N_oligo' : 1E4
      }, # Average oligomer size fixed
    # {
    #   'N_dist_amp': np.nan,
    #   'N_dist_a': np.nan,
    #   'N_dist_b': np.nan,
    #   'j_avg_N_oligo' : np.nan
    #   } # Nothing fixed
    ]
    # If you use a parameterized spectrum, you can fix some parameters of the 
    # expression. There are four parameters, independent of the model chosen :
    # (Gauss, LogNorm, Gamma, StrExp): 
        # 'N_dist_amp' -> Amplitude
        # 'N_dist_a' -> Position parameter (mean for Gaussian)
        # 'N_dist_b' -> Shape parameter (Variance for Gaussian)
        # 'j_avg_N_oligo' -> Average stoichiometry weighted by N_oligo (particle number). Fixing this overwrites N_dist_a and N_dist_b!
    # Note that is you set the value for one of the parameters to something 
    # that is not a single number > 0, it will be considered invalid input, 
    # and the parameter will NOT be fixed.

discrete_species_list = [
    [{
      }
    ],
    # [
    #     {'N_avg_obs': 1., # default 1.
    #      'vary_N_avg_obs': True, # default True
    #      'tau_diff': 1E-5,  # default 1E-3
    #      'vary_tau_diff': False, # default False
    #      'cpms': 1000.,  # default 1.
    #      'vary_cpms': True, # default False
    #      'link_brightness_to_spectrum_monomer': True, # default True - see docstring for details, this one is important!!!
    #      'stoichiometry': 1.,# default 1.
    #      'vary_stoichiometry': False, # default (RECOMMENDED) False
    #      'stoichiometry_binwidth': 1., # default 1.
    #      'vary_stoichiometry_binwidth': False, # default (RECOMMENDED) False
    #      'labelling_efficiency': 1., # default 1.
    #      'vary_labelling_efficiency': False,  # default (RECOMMENDED) False
    #     }
    # ]
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
    # in the same fit.
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

fit_label_efficiency_list = [True] 
    # If you consider finite labelling fraction, here you can also decide to 
    # make that a fit parameter, although that may be numerically extremely instable



###### Metadata/calibration data/settings (global for all fits in batch)

FCS_min_lag_time = 1E-6
    # Shortest lag time to consider in fit (time axis clipping)
    # Specify 0. to use full range of data in .csv file
    
FCS_max_lag_time = 3E-1
    # Longest lag time to consider in fit (time axis clipping)
    # Specify np.inf to use full range of data in .csv file

PCH_min_bin_time = 0. 
    # Shortest PCMH bin times to consider
    # Specify 0. to use full range of data in .csv file
    
PCH_max_bin_time = 5E-4
    # Longest PCMH bin times to consider
    # Specify np.inf to use full range of data in .csv file

NLL_funcs_accurate = True
    # Accurate maximum-likelihood evaluation, or use faster least-squares 
    # approximation? Affects most likelihood terms except the chi-square 
    # minimization on the ACF correlation function. True is highly recommended for 
    # incomplete sampling fit as least-squares is prone to crashing due to 
    # zeros!
    
numeric_precision = np.array([1E-3, 1E-4, 1E-5])
    # PCH requires a numerical precision cutoff, which is set here. The lower 
    # the number, the more accurate but computationally expensive the 
    # evaluation. You can specify a single number, then it's trivial, or an 
    # array, in which case the model is first evaluated with low accuracy and 
    # precision is then incrementally increased according to the steps you specified.
    
two_step_fit = True
    # For some model configuration, you can first run a simpler, more robust, 
    # version of the fit with some parameters fixed, and then re-fit with the 
    # "full" model complexity. Highly recommended for incomplete sampling fit
    # for numeric stability!

verbosity = 2
    # How much do you want the software to talk?

FCS_psf_width_nm = np.mean([210])
    # FCS calibration of PSF width in xy (w_0: 1/e^2 radius), although it is 
    # actually not used for anything meaningful currently

FCS_psf_aspect_ratio = np.mean([6])
    # FCS calibration of PSF aspect ratio (w_z/w_0), also used for PCMH

PCH_Q = 10. 
    # Evaluation parameter for PCH

mp_processes = 1 
    # How many parallel processes?
    # If mp_processes <= 1, we use multiprocessing WITHIN the fit which allows acceleration of multi-species PCH
    # If mp_processes > 1, we run multiple fits simultaneously, each in single-thread calculation
    # mp_processes = os.cpu_count() // 2 to use half of available logical cores 
    # (-> on many machines all physical cores without hyperthreading)

suppress_mp = True
    # For debugging purposes: Forces the software to run entirely without 
    # multiprocessing, which yields more interpretable error messages
    
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
                                                            for fixed_spectrum_params in fixed_spectrum_params_list:
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
                                                                                               fixed_spectrum_params,
                                                                                               numeric_precision,
                                                                                               two_step_fit,
                                                                                               discrete_species,
                                                                                               verbosity,
                                                                                               mp_processes <= 1 and not suppress_mp)
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
             fixed_spectrum_params,
             numeric_precision,
             two_step_fit,
             discrete_species,
             verbosity,
             use_parallel):
    
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
                                        fixed_spectrum_params = fixed_spectrum_params, # Dict
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
                                        dir_name
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

if len(list_of_parameter_tuples) == 0:
    raise Exception('No valid parameter configurations found in input! Sanity-check input parameters, invalid values do not raise specific error messages at this point.')

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