# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""

# File I/O and path manipulation
import os
import glob
# import sys
import stat

# Data handling
import numpy as np
import pandas as pd

# misc
import lmfit # only used for type recognition
import traceback
import datetime
import matplotlib.pyplot as plt
from itertools import cycle # used only in plotting

# Custom module
# For localizing module
# repo_dir = os.path.abspath('..')
# sys.path.insert(0, repo_dir)
# from functions import fitting


def detect_files(in_dir_names, 
                 file_name_pattern, 
                 other_info, 
                 exclude_path,
                 file_type_suffix = '.csv'):
    # Automatic input file detection
    
    # Containers for storing results
    _in_file_names=[]
    _in_dir_names = [] 
    _other_info = []
    
    # Iteration over directories
    for i_dir, directory in enumerate(in_dir_names):
        
        # Iteration over files: Find all that match pattern, and list        
        search_pattern = os.path.join(directory, '**' + os.path.sep + '*' + file_name_pattern)
    
        for name in glob.glob(search_pattern, 
                              recursive = True):
    
            dir_name, file_name_full = os.path.split(name)
            file_name, file_suffix = os.path.splitext(file_name_full)
            
            # Register if the found file is a .csv file and is not inside the exclude_path
            if file_suffix == file_type_suffix and dir_name != exclude_path and not (path_is_parent(exclude_path, dir_name)):
                _in_file_names.extend([file_name])
                _in_dir_names.extend([dir_name])
                _other_info.append(other_info[i_dir])
                
    if len(_in_dir_names) == 0:
        raise Exception(f'Searched {i_dir+1} directories, but could not detect any files (base dirs: {in_dir_names})')
        
    return _in_dir_names, _in_file_names, _other_info


def path_is_parent(parent_path, child_path):
    # Copy-paste with addition of drive check from 
    # https://stackoverflow.com/questions/3812849/how-to-check-whether-a-directory-is-a-sub-directory-of-another-directory
    # Smooth out relative path names, note: if you are concerned about symbolic links, you should use os.path.realpath too
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)
        
    if os.stat(parent_path).st_dev == os.stat(child_path).st_dev:
        # Files are on same drive, check further
        
        # Compare the common path of the parent and child path with the common path of just the parent path. 
        # Using the commonpath method on just the parent path will regularise the path name in the same way 
        # as the comparison that deals with both paths, removing any trailing path separator
        return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])
    else:
        # Different drives anyway, nevermind
        return False
    

def read_Kristine_FCS(dir_name,
                      file_name,
                      FCS_min_lag_time = 0.,
                      FCS_max_lag_time = np.inf):
    try:
        # Load Kristine format FCS data
        in_path = os.path.join(dir_name, file_name)
        data = pd.read_csv(in_path + '.csv', header = None)
            
        # Col 0: Lag times - Use for cropping, if needed
        lag_times = data.iloc[:,0].to_numpy()
        lag_time_mask = np.logical_and(lag_times > FCS_min_lag_time, 
                                       lag_times < FCS_max_lag_time)
        data_FCS_tau_s = lag_times[lag_time_mask]
        
        # Col 1: Correlation
        G = data.iloc[:, 1].to_numpy()
        data_FCS_G = G[lag_time_mask]
        
        # Col 2: Count rate, and in my modification also acquisition time
        avg_count_rate = data.iloc[0:2,2].mean() 
            # In case of cross-correlation, we average count rates of both channels for simplicity - not entirely accurate, but good enough I guess
        acquisition_time_s = data.iloc[2,2] 
    
        # Col 3: Uncertainty
        sigma_G = data.iloc[:, 3].to_numpy()
        data_FCS_sigma = sigma_G[lag_time_mask]
    except:
        raise Exception(f'Error in reading {in_path} \n {traceback.print_exc()}')
    return data_FCS_tau_s, data_FCS_G, avg_count_rate, data_FCS_sigma, acquisition_time_s


def read_PCMH(dir_name,
              file_name,
              PCH_min_bin_time,
              PCH_max_bin_time):

    # Load PCH data
    in_path_PCH = os.path.join(dir_name, file_name)
    data_PCH = pd.read_csv(in_path_PCH + '.csv', header = 0)
    
    # Row 0 = Column names = bin times
    bin_times = np.array([float(bin_time) for bin_time in data_PCH.keys()])
    bin_time_mask = np.logical_and(bin_times > PCH_min_bin_time,
                                   bin_times < PCH_max_bin_time)
    data_PCH_bin_times = bin_times[bin_time_mask]
    
    # Rows 1...end: PC(M)H data
    PCH_hist = data_PCH.to_numpy()
    data_PCH_hist = PCH_hist[:,bin_time_mask]
    
    return data_PCH_bin_times, data_PCH_hist


def link_FCS_and_PCH_files(in_dir_names_FCS,
                           in_file_names_FCS,
                           other_info_FCS,
                           in_dir_names_PCH,
                           in_file_names_PCH,
                           other_info_PCH):
    
    if len(in_dir_names_PCH) <= len(in_dir_names_FCS):
        # One PCH file per FCS file or more FCS files than PCH files
        
        # We use FCS as reference to sort PCH
        in_file_names_FCS_argsort = []
        in_file_names_PCH_argsort = []
        
        for i_fcs, in_dir_name_FCS in enumerate(in_dir_names_FCS):
        # Find the index of the corresponding PCH file
            
            found = False
            for i_pch, in_dir_name_PCH in enumerate(in_dir_names_PCH):
                
                if in_dir_name_PCH == in_dir_name_FCS and not found:
                    # This looks like the right one - Store index for sorting PCH information
                    in_file_names_FCS_argsort.append(i_fcs)
                    in_file_names_PCH_argsort.append(i_pch)
                    found = True
                    
                elif in_dir_name_PCH == in_dir_name_FCS and found:
                    # We already found the correct PCH, but now we find another? That does not work!
                    raise Exception(f'Assignment of FCS data and PCH data is ambiguous in {in_dir_name_PCH}! More advanced user-defined logic may be needed.')
            
        in_dir_names = [in_dir_names_FCS[i_fcs] for i_fcs in in_file_names_FCS_argsort]
        in_file_names_FCS = [in_file_names_FCS[i_fcs] for i_fcs in in_file_names_FCS_argsort]
        in_file_names_PCH = [in_file_names_PCH[i_pch] for i_pch in in_file_names_PCH_argsort]
        other_info = [other_info_FCS[i_fcs] for i_fcs in in_file_names_FCS_argsort]
        
        
    # elif len(in_dir_names_PCH) < len(in_dir_names_FCS):
            
    #     # We use FCS as reference to sort PCH
    #     in_file_names_FCS_argsort = []
    #     in_file_names_PCH_argsort = []
        
    #     for i_fcs, in_dir_name_FCS in enumerate(in_dir_names_FCS):
    #     # Find the index of the corresponding PCH file
            
    #         found = False
    #         for i_pch, in_dir_name_PCH in enumerate(in_dir_names_PCH):
                
    #             if in_dir_name_PCH == in_dir_name_FCS and not found:
    #                 # This looks like the right one - Store index for sorting PCH information
    #                 in_file_names_FCS_argsort.append(i_fcs)
    #                 in_file_names_PCH_argsort.append(i_pch)
    #                 found = True
                    
    #             elif in_dir_name_PCH == in_dir_name_FCS and found:
    #                 # We already found the correct PCH, but now we find another? That does not work!
    #                 raise Exception('Assignment of FCS data and PCH data is ambiguous! More advanced user-defined assignment code may be needed.')
                    
    #     in_dir_names = [in_dir_names_FCS[i_fcs] for i_fcs in in_file_names_FCS_argsort]
    #     in_file_names_FCS = [in_file_names_FCS[i_fcs] for i_fcs in in_file_names_FCS_argsort]
    #     in_file_names_PCH = [in_file_names_PCH[i_pch] for i_pch in in_file_names_PCH_argsort]
    #     other_info = other_info_FCS


    else: # len(in_dir_names_PCH) > len(in_dir_names_FCS)
        # We have more PCH files than FCS files - use PCH as reference
        
        in_file_names_FCS_argsort = []
        in_file_names_PCH_argsort = []

        for i_pch, in_dir_name_PCH in enumerate(in_dir_names_PCH):
        # Find the index of the corresponding PCH file
            
            found = False
            for i_fcs, in_dir_name_FCS in enumerate(in_dir_names_FCS):
                
                if in_dir_name_PCH == in_dir_name_FCS and not found:
                    # This looks like the right one - Store index for sorting PCH information
                    in_file_names_FCS_argsort.append(i_fcs)
                    in_file_names_PCH_argsort.append(i_pch)
                    found = True

                elif in_dir_name_PCH == in_dir_name_FCS and found:
                    # We already found the correct FCS dataset, but now we find another? That does not work!
                    raise Exception('Assignment of FCS data and PCH data is ambiguous! More advanced user-defined assignment code may be needed.')
            
        in_dir_names = [in_dir_names_PCH[i_pch] for i_pch in in_file_names_PCH_argsort]
        in_file_names_FCS = [in_file_names_FCS[i_fcs] for i_fcs in in_file_names_FCS_argsort]
        in_file_names_PCH = [in_file_names_PCH[i_pch] for i_pch in in_file_names_PCH_argsort]
        other_info = [other_info_PCH[i_pch] for i_pch in in_file_names_PCH_argsort]

        
    return in_dir_names, in_file_names_FCS, in_file_names_PCH, other_info

        
def isint(object_to_check):
    '''
    Just a shorthand to check if the object in question is one out of many 
    int types, which we use in other functions.

    Parameters
    ----------
    object_to_check : 
        Some object whose type we want to check.

    Returns
    -------
    Bool
        True if the object is an int type, else False.

    '''
    return type(object_to_check) in [int, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]


def isempty(object_to_check):
    '''
    Just a shorthand to check if the object in question is one out of a few 
    representations of an empty value, which we use in other functions.

    Parameters
    ----------
    object_to_check : 
        Some object whose type we want to check.

    Returns
    -------
    Bool
        True if the object is an empty type, else False.

    '''
    if type(object_to_check) == np.ndarray:
        return object_to_check.shape[0] == 0
    else:
        return object_to_check in [None, np.nan, [], ()]


def isiterable(object_to_check):
    '''
    Just a shorthand to check if the object in question is one out of a few 
    iterable types and with nonzero length, which we use in other functions.

    Parameters
    ----------
    object_to_check : 
        Some object whose type we want to check.

    Returns
    -------
    Bool
        True if the object is an allowed iterable type, else False.

    '''
    check_1 = type(object_to_check) in [list, tuple, np.ndarray]
    if check_1:
        if len(object_to_check) > 0 :
            return True
        else: 
            return False
    else:
        return False


def isfloat(object_to_check):
    '''
    Just a shorthand to check if the object in question is one out of a few 
    float types, which we use in other functions.

    Parameters
    ----------
    object_to_check : 
        Some object whose type we want to check.

    Returns
    -------
    Bool
        True if the object is an allowed float type, else False.

    '''
    return type(object_to_check) in [float, np.float16, np.float32, np.float64]


def dense_to_sparse(dense_trace, 
                    dtype = None, 
                    overflow_marker = None,
                    overflow_policy = 'wrap'):
    '''
    Convert sparse representation of photon data to dense time binned trace 
    representation (except without any binning applied yet).

    Parameters
    ----------
    dense_trace : 
        1D np.array with dense time trace of photons
    dtype :
        Data type for sparse_trace output. Must be a numpy unsigned integer type
        or None. Optional with default None, in which case it will be heuristically 
        chosen to minimize data size.
    overflow_marker :
        Marker for wrap-around of time tags in sparse_trace to allow effective 
        time tags beyond the dynamic range of the sparse_trace data type. 
        Optional with default None, which assumes that the overflow_marker is 
        the highest number in the dynamic range dtype. MUST be None if dtype is None.
    overflow_policy :
        How to handle time tags of photons that end up beyond overflow_marker, 
        i.e., too long for naive storage in the designated dynamic range:
            'wrap' - values exceeding overflow_marker are wrapped around and a overflow_marker placed in the data
            'ignore' (not recommended) - do nothing, accepting nonsense values from numeric overflow
            'raise' - Break calculation and raise an exception when an overflow is encountered
    Returns
    -------
    sparse_trace :
        1D np.array with unsigned integers that denote photon time tags
    '''
    
    # A bit of input check
    if not (dense_trace.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]):
        raise ValueError("dense_trace must be of an unsigned integer numpy type!")
        
    if not (dtype in [np.uint8, np.uint16, np.uint32, np.uint64, None]):
        raise ValueError("Invalid dtype: Can only output unsigned integer numpy types or None for auto-selection!")

    if (dtype is None) and not (overflow_marker is None):
        raise ValueError("If you specify overflow_marker, you also need to specify dtype!")
        
    if dtype is None:
        # Roughly predict number of bits for uint8, uint16, uint32 and uint64, and pick most efficient one
        # Calculation: minimize(bits_per_element * (number_of_photons + number_of_overflow_markers))
        dtype_autopick = np.argmin(np.array([8, 16, 32, 64]) * (np.sum(dense_trace) + np.floor(dense_trace.shape[0] / (2.0**(np.array([8, 16, 32, 64], dtype=np.float64))-1))))
        dtype = [np.uint8, np.uint16, np.uint32, np.uint64][dtype_autopick]
        
    if overflow_marker is None:
        overflow_marker = dtype(np.iinfo(dtype).max)
        
    if not (overflow_policy in ['wrap', 'ignore', 'raise']):
        raise ValueError("Invalid overflow_policy: Can only be 'wrap', 'ignore' (not recommended), or 'raise'!")
       
    if (overflow_policy == 'wrap') and (\
                                        (overflow_marker > np.iinfo(dtype).max) or \
                                        (overflow_marker < 0) or \
                                        (not ((type(overflow_marker) in [np.uint8, np.uint16, np.uint32, np.uint64]) or (overflow_marker is None))) \
                                        ):
        raise ValueError("chosen overflow_marker does not work for chosen dtype!")
    
    encounters_overflow = dense_trace.shape[0] >= overflow_marker
    if encounters_overflow and (overflow_policy == 'raise'):
        raise ValueError("dense_trace is too long for the chosen output data type without overflow wrapping!")
    
    # Do conversion
    max_count = np.max(dense_trace)
    if (overflow_policy == 'ignore') or \
        (overflow_policy in ['wrap', 'raise'] and not encounters_overflow): 
        if max_count <= 1: # Only 0s and 1s
            sparse_trace = np.nonzero(dense_trace)[0].astype(dtype)
        else: # multiple photons
            sparse_trace = np.nonzero(dense_trace)[0].astype(dtype)
            for n_photons in range(1, max_count):
                sparse_trace = np.append(sparse_trace, np.nonzero(dense_trace > n_photons)[0].astype(dtype))
            sparse_trace = np.sort(sparse_trace)
    elif overflow_policy == 'raise' and encounters_overflow:
        raise ValueError("Encountered overflow in dense_trace relative to dynamic range allowed by the chosen dtype!")
    else: # overflow_policy == 'wrap' and encounters_overflow
        # Convert to sparse representation, for now as np.uint64 for practically unlimited dynamic range (>10^19)
        sparse_trace = np.nonzero(dense_trace)[0].astype(np.uint64)
        if max_count > 1: # multiple photons
            for n_photons in range(1, max_count):
                sparse_trace = np.append(sparse_trace, np.nonzero(dense_trace > n_photons)[0].astype(np.uint64))
            sparse_trace = np.sort(sparse_trace)

        # Find overflows and wrap
        overflows = []
        overflow = np.nonzero(sparse_trace >= overflow_marker)[0] # Look for overflows
        while overflow.shape[0] > 0:
            overflows.append(overflow[0]) 
            sparse_trace[overflow[0]:] -= (overflow_marker -1)
            overflow = np.nonzero(sparse_trace >= overflow_marker)[0] # Look for next remaining overflow

        # Convert to target dtype and insert overflows
        sparse_trace = sparse_trace.astype(dtype)
        sparse_trace = np.insert(sparse_trace, overflows, overflow_marker) 
            
    return sparse_trace


def sparse_to_dense(sparse_trace, 
                    dtype = np.uint8, 
                    overflow_marker = None, 
                    clipping_policy = 'clip'):
    '''
    Convert sparse representation of photon data to dense time binned trace 
    representation (except without any binning applied yet).

    Parameters
    ----------
    sparse_trace : 
        1D np.array with unsigned integers that denote photon time tags
    dtype :
        Data type for dense_trace output. Must be a numpy unsigned integer type. 
        Optional with default np.uint8
    overflow_marker :
        Marker for wrap-around of time tags in sparse_trace to allow effective 
        time tags beyond the dynamic range of the sparse_trace data type. 
        Optional with default None, which assumes that the overflow_marker is 
        the highest number in the dynamic range of sparse_trace data type.
    clipping_policy :
        How to handle cases in which photons came so bunched that the data 
        type of dense_trace overflows. Optional with default 'clip'. Allowed:
            'clip' - values exceeding allowed range are clipped to the maximum allowed value
            'ignore' (not recommended) - do nothing, accepting nonsense values from numeric overflow
            'raise' - Break calculation and raise an exception when a clipped value is found
    Returns
    -------
    dense_trace :
        1D np.array with dense time trace of photons
    '''
    
    dtype_in = sparse_trace.dtype
    
    # A bit of input check
    if not (dtype_in in [np.uint8, np.uint16, np.uint32, np.uint64]):
        raise ValueError("sparse_trace must be of an unsigned integer numpy type!")
        
    if not (dtype in [np.uint8, np.uint16, np.uint32, np.uint64]):
        raise ValueError("Invalid dtype: Can only output unsigned integer numpy types!")

    if overflow_marker is None:
        overflow_marker = np.iinfo(dtype_in).max
    elif ((overflow_marker > np.iinfo(dtype_in).max) or (overflow_marker < 0)) or \
        (type(overflow_marker) != dtype_in):
        raise ValueError("overflow_marker must have the same datatype as sparse_trace!")
    elif np.any(sparse_trace > overflow_marker):
        raise ValueError("sparse_trace contains values not allowed by chosen overflow_marker!")
        
    if not clipping_policy in ['clip', 'ignore', 'raise']:
        raise ValueError("Invalid overflow_policy: Can only be 'clip', 'ignore' (not recommended), or 'raise'!")

    # Unwrap overflows, if needed
    if np.any(sparse_trace == overflow_marker):
        sparse_trace = sparse_trace.astype(np.uint64)
        overflows = np.nonzero(sparse_trace == overflow_marker)[0]
        for overflow in overflows:
            sparse_trace[overflow+1:] += (overflow_marker -1) # Apply unwrap
        sparse_trace = np.delete(sparse_trace, overflows) # Remove overflow_marker

    # Reconstruct dense_trace
    if np.all(np.diff(sparse_trace) > 0): # Only zeros and ones in dense_trace - easy
        dense_trace = np.zeros(np.int64(sparse_trace[-1]+1), dtype = dtype)
        dense_trace[sparse_trace[:]] = 1
    else: # Multi-photon - more tricky...
        dense_trace, _ = np.histogram(sparse_trace, bins = np.arange(sparse_trace[-1]+2))
        if np.any(dense_trace > np.iinfo(dtype).max): # Clipping handling
            if clipping_policy == 'clip':
                dense_trace[dense_trace > np.iinfo(dtype).max] = np.iinfo(dtype).max
            elif clipping_policy == 'raise':
                raise ValueError("Cannot convert sparse_trace to chosen data type for dense_trace due to too-high photon count!")
        dense_trace = dense_trace.astype(dtype)

    return dense_trace

        
def write_fit_results(fit_result,
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
                      N_pop_array = np.array([]),
                      lagrange_mul = np.inf,
                      fit_number = -1,
                      i_file = -1,
                      in_file_name_FCS = '',
                      in_file_name_PCH = '',
                      dir_name = '',
                      verbosity = 0,
                      suppress_figs = False
                      ):
    
    # Unpack simple stuff
    if not fit_result == None:
        if hasattr(fit_result, 'params') and hasattr(fit_result, 'covar'): 
            # dirty workaround for essentially testing "if type(fit_result) == lmfit.MinimizerResult", as MinimizerResult class cannot be explicitly referenced
            # Unpack fit result
            fit_params = fit_result.params
            covar = fit_result.covar
            # Recalculate number of species, it is possible we lose some in between
            n_species_spec, n_species_disc = fitter.get_n_species(fit_params)
            
        elif type(fit_result) == lmfit.Parameters:
            # Different output that can come from regularized fitting
            # Unpack fit result
            fit_params = fit_result
            covar = None
            
            n_species_spec = N_pop_array.shape[0]
            _, n_species_disc = fitter.get_n_species(fit_params) 
        else:
            raise Exception(f'Got a fit_result output, but with unsupported type. Expected lmfit.MinimizerResult or lmfit.Parameters, got {type(fit_result)}')

        n_species_tot = n_species_spec + n_species_disc
        
        
        # File name construction
        time_tag = datetime.datetime.now()
        out_suffix = time_tag.strftime("%m%d-%H%M%S")
        if i_file >= 0:
            out_suffix = f'{i_file}_' + out_suffix
        if fit_number >= 0:
            out_suffix = f'{fit_number}_' + out_suffix
        out_suffix += f'_{in_file_name_FCS if use_FCS else in_file_name_PCH}_fit_{spectrum_type}_{n_species_tot}spec'
        out_name = os.path.join(save_path, out_suffix)
        
        # Command line preview of fit results
        if verbosity > 0:
            print(f' [{job_prefix}]   Fitted parameters:')
            [print(f'[{job_prefix}] {key}: {fit_params[key].value}') for key in fit_params.keys() if fit_params[key].vary]
        
        # Show and write fits themselves
        if use_FCS:
            data_FCS_tau_s = fitter.data_FCS_tau_s
            data_FCS_G = fitter.data_FCS_G
            data_FCS_sigma = fitter.data_FCS_sigma

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
                tau_diff_array = np.array([fit_params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
                stoichiometry_array = np.array([fit_params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species_spec)])
                stoichiometry_binwidth_array = np.array([fit_params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
                label_efficiency_array = np.array([fit_params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])

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
            if suppress_figs:
                plt.close()
            else:
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
            # Set rights
            os.chmod(os.path.join(save_path, 
                                  out_name + '_FCS.csv'), stat.S_IRWXU)

            # Get chi-square
            FCS_chi_square = np.mean((data_FCS_G - model_FCS / data_FCS_sigma)**2)

    
    
        if use_PCH:
            data_PCH_hist = fitter.data_PCH_hist
            data_PCH_bin_times = fitter.data_PCH_bin_times
            numeric_precision = fitter.numeric_precision
            if not labelling_correction:
                # Get model
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
            if suppress_figs:
                plt.close()
            else:
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
            # Set rights
            os.chmod(os.path.join(save_path, out_name + '_PC'+ ('M' if time_resolved_PCH else '') +'H.csv'), stat.S_IRWXU)

            # Get chi-square
            if not time_resolved_PCH and not labelling_correction:
                PCH_chi_square = fitter.negloglik_pch_single_full_labelling(fit_params,
                                                                            spectrum_type = spectrum_type,
                                                                            i_bin_time = 0,
                                                                            numeric_precision = np.min(numeric_precision),
                                                                            mp_pool = None)
            elif not time_resolved_PCH and labelling_correction:
                PCH_chi_square = fitter.negloglik_pch_single_partial_labelling(fit_params,
                                                                               i_bin_time = 0,
                                                                               numeric_precision = np.min(numeric_precision),
                                                                               mp_pool = None)
            elif time_resolved_PCH and not labelling_correction:
                PCH_chi_square = fitter.negloglik_pcmh_full_labelling(fit_params,
                                                                      spectrum_type = spectrum_type,
                                                                      numeric_precision = np.min(numeric_precision),
                                                                      mp_pool = None)
            else: # time_resolved_PCH and labelling_correction
                PCH_chi_square = fitter.negloglik_pcmh_partial_labelling(fit_params,
                                                                         numeric_precision = np.min(numeric_precision),
                                                                         mp_pool = None)
                
                
        # Compile all parameters for all discrete species in a single smaller table
        if n_species_disc > 0:
            fit_res_tmp_path = fit_res_table_path + '_discrete_species.csv' 
            disc_spec_params_dict = {}
            disc_spec_params_dict['fit_number'] = fit_number                 
            disc_spec_params_dict['file_number'] = i_file
            disc_spec_params_dict['folder'] = dir_name 
            disc_spec_params_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
            disc_spec_params_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
            disc_spec_params_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            disc_spec_params_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf
            
            query_keys = ['tau_diff_d_', 'N_avg_obs_d_', 'cpms_d_', 'stoichiometry_d_', 'stoichiometry_binwidth_d_', 'Label_efficiency_obs_d_']
            for i_spec in range(n_species_disc):
                for query_key in query_keys:
                    if query_key + str(i_spec) in fit_params.keys():
                        disc_spec_params_dict[query_key + str(i_spec)] = fit_params[query_key + str(i_spec)].value
                        
            fit_result_df = pd.DataFrame(disc_spec_params_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                     header = True, 
                                     index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
            else:
                # Exists - append
                fit_result_df.to_csv(fit_res_tmp_path, 
                                     mode = 'a', 
                                     header = False, 
                                     index = False)

                        
        # Smaller spreadsheets for species-wise parameters in case we have multiple species
        # all of these are technically redundant with the big one written later...
        # But reading these out of the main results spreadsheet is just
        # too much trouble if you have more than 1 or 2 species
        
        # Dict in which we collect various mean and SD values
        # We create this even if we do not create the result spreadsheets - a dummy whose existence avoids errors
        avg_values_dict = {}   
             
        if n_species_spec > 1:

            
            # Unpack species parameters into arrays
            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                N_oligo_pop_array = N_pop_array
            elif spectrum_type == 'discrete':
                # Historical reasons why here N_obs has to be used...
                N_oligo_pop_array =  np.array([fit_params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_spec)])
            else:
                N_oligo_pop_array = np.array([fit_params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species_spec)])
            
            stoichiometry_array = np.array([fit_params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species_spec)])
            stoichiometry_bw_array = np.array([fit_params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
            if 'tau_diff_0' in fit_params.keys():
                tau_diff_array = np.array([fit_params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
            else:
                # There were no tau_diff values in fit_params - can happen, it should be in class attribute then
                tau_diff_array = fitter.tau__tau_diff_array[0,:]

            
            tau_diff_array = np.array([fit_params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
            labelling_efficiency_array = np.ones_like(tau_diff_array) * fit_params['Label_efficiency'].value


            # Stoichiometry
            fit_res_tmp_path = fit_res_table_path + '_stoi.csv' 
            fit_result_dict = {}
            fit_result_dict['fit_number'] = fit_number                 
            fit_result_dict['file_number'] = i_file
            fit_result_dict['folder'] = dir_name 
            fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
            fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
            for i_spec in range(n_species_spec):
                fit_result_dict[f'stoichiometry_{i_spec}'] = stoichiometry_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                     header = True, 
                                     index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
            for i_spec in range(n_species_spec):
                fit_result_dict[f'stoi_bw_{i_spec}'] = stoichiometry_bw_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                     header = True, 
                                     index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
            for i_spec in range(n_species_spec):
                fit_result_dict[f'tau_diff_{i_spec}'] = tau_diff_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                     header = True, 
                                     index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
                
            amp_density_array = N_oligo_pop_array * stoichiometry_array**2
            if labelling_correction:
                amp_density_array *= 1 - (1 - labelling_efficiency_array) / (labelling_efficiency_array * stoichiometry_array)
            amp_density_array /= amp_density_array.max()
            
            for i_spec in range(n_species_spec):
                fit_result_dict[f'amp_density_{i_spec}'] = amp_density_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                        header = True, 
                                        index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
                
            amp_histogram_array = N_oligo_pop_array * stoichiometry_array**2 * stoichiometry_bw_array
            if labelling_correction:
                amp_histogram_array *= 1 - (1 - labelling_efficiency_array) / (labelling_efficiency_array * stoichiometry_array)
            amp_histogram_array /= amp_histogram_array.sum()
            
            for i_spec in range(n_species_spec):
                fit_result_dict[f'amp_histogram_{i_spec}'] = amp_histogram_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                        header = True, 
                                        index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
                
            N_pop_oligo_abs_array = N_oligo_pop_array * stoichiometry_bw_array
            
            for i_spec in range(n_species_spec):
                fit_result_dict[f'N_pop_oligo_abs_{i_spec}'] = N_pop_oligo_abs_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                        header = True, 
                                        index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
            N_pop_oligo_density_array = N_oligo_pop_array / N_oligo_pop_array.max()
            
            for i_spec in range(n_species_spec):
                fit_result_dict[f'N_pop_oligo_density_{i_spec}'] = N_pop_oligo_density_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                        header = True, 
                                        index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
                
            N_pop_oligo_histogram_array = N_oligo_pop_array * stoichiometry_bw_array
            N_pop_oligo_histogram_array /= N_pop_oligo_histogram_array.sum()
            
            for i_spec in range(n_species_spec):
                fit_result_dict[f'N_pop_oligo_histogram_{i_spec}'] = N_pop_oligo_histogram_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                        header = True, 
                                        index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
                
            N_pop_mono_abs_array = N_oligo_pop_array * stoichiometry_bw_array * stoichiometry_array
            
            for i_spec in range(n_species_spec):
                fit_result_dict[f'N_pop_mono_abs_{i_spec}'] = N_pop_mono_abs_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                        header = True, 
                                        index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
            N_pop_mono_density_array = N_oligo_pop_array * stoichiometry_array
            N_pop_mono_density_array /= N_pop_mono_density_array.max()
            
            for i_spec in range(n_species_spec):
                fit_result_dict[f'N_pop_mono_density_{i_spec}'] = N_pop_mono_density_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                        header = True, 
                                        index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
            fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
            fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

            if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                fit_result_dict['lagrange_mul'] = lagrange_mul
                
            N_pop_mono_histogram_array = N_oligo_pop_array * stoichiometry_bw_array * stoichiometry_array
            N_pop_mono_histogram_array /= N_pop_mono_histogram_array.sum()
            
            for i_spec in range(n_species_spec):
                fit_result_dict[f'N_pop_mono_oligo_histogram_{i_spec}'] = N_pop_mono_histogram_array[i_spec]
            fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
            if not os.path.isfile(fit_res_tmp_path):
                # Does not yet exist - create with header
                fit_result_df.to_csv(fit_res_tmp_path, 
                                        header = True, 
                                        index = False)
                # Set rights
                os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
                N_oligo_obs_array = np.array([fit_params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])

                # N_oligomers - observation level - absolute
                fit_res_tmp_path = fit_res_table_path + '_N_obs_oligo_abs.csv'
                fit_result_dict = {}
                fit_result_dict['fit_number'] = fit_number 
                fit_result_dict['file_number'] = i_file
                fit_result_dict['folder'] = dir_name 
                fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
                fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                N_obs_oligo_abs_array = N_oligo_obs_array * stoichiometry_bw_array
                
                for i_spec in range(n_species_spec):
                    fit_result_dict[f'N_obs_oligo_abs_{i_spec}'] = N_obs_oligo_abs_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                    # Set rights
                    os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
                fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
                fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                N_obs_oligo_density_array = N_oligo_obs_array / N_oligo_obs_array.max()
                
                for i_spec in range(n_species_spec):
                    fit_result_dict[f'N_obs_oligo_density_{i_spec}'] = N_obs_oligo_density_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                    # Set rights
                    os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
                fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
                fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                N_obs_oligo_histogram_array = N_oligo_obs_array * stoichiometry_bw_array
                N_obs_oligo_histogram_array /= N_obs_oligo_histogram_array.sum()
                
                for i_spec in range(n_species_spec):
                    fit_result_dict[f'N_obs_oligo_histogram_{i_spec}'] = N_obs_oligo_histogram_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                    # Set rights
                    os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
                fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
                fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                N_obs_mono_abs_array = N_oligo_obs_array * stoichiometry_bw_array * stoichiometry_array
                
                for i_spec in range(n_species_spec):
                    fit_result_dict[f'N_obs_mono_abs_{i_spec}'] = N_obs_mono_abs_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                    # Set rights
                    os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
                fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
                fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                N_obs_mono_density_array = N_oligo_obs_array * stoichiometry_array
                N_obs_mono_density_array /= N_obs_mono_density_array.max()
                
                for i_spec in range(n_species_spec):
                    fit_result_dict[f'N_obs_mono_density_{i_spec}'] = N_obs_mono_density_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                         header = True, 
                                         index = False)
                    # Set rights
                    os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
                fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
                fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

                if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                    fit_result_dict['lagrange_mul'] = lagrange_mul
                    
                N_obs_mono_histogram_array = N_oligo_obs_array * stoichiometry_bw_array * stoichiometry_array
                N_obs_mono_histogram_array /= N_obs_mono_histogram_array.sum()
                
                for i_spec in range(n_species_spec):
                    fit_result_dict[f'N_obs_mono_oligo_histogram_{i_spec}'] = N_obs_mono_histogram_array[i_spec]
                fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                if not os.path.isfile(fit_res_tmp_path):
                    # Does not yet exist - create with header
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            header = True, 
                                            index = False)
                    # Set rights
                    os.chmod(fit_res_tmp_path, stat.S_IRWXU)
                else:
                    # Exists - append
                    fit_result_df.to_csv(fit_res_tmp_path, 
                                            mode = 'a', 
                                            header = False, 
                                            index = False)

                                 
                if fitter.labelling_efficiency_incomp_sampling:
                    # If and only if we have incomplete-sampling-of-incomplete-labelling correction, we also write that separately
                    fit_res_tmp_path = fit_res_table_path + '_label_eff_obs.csv'
                    fit_result_dict = {}
                    fit_result_dict['fit_number'] = fit_number 
                    fit_result_dict['file_number'] = i_file
                    fit_result_dict['folder'] = dir_name 
                    fit_result_dict['file_FCS'] = in_file_name_FCS if use_FCS else 'unused' 
                    fit_result_dict['file_PCH'] = in_file_name_PCH if use_PCH else 'unused' 
                    fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
                    fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

                    if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
                        fit_result_dict['lagrange_mul'] = lagrange_mul
                                                
                    for i_spec in range(n_species_spec):
                        fit_result_dict[f'_label_eff_obs{i_spec}'] = fit_params[f'Label_efficiency_obs_{i_spec}'].value
                    fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
                    if not os.path.isfile(fit_res_tmp_path):
                        # Does not yet exist - create with header
                        fit_result_df.to_csv(fit_res_tmp_path, 
                                                header = True, 
                                                index = False)
                        # Set rights
                        os.chmod(fit_res_tmp_path, stat.S_IRWXU)
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
        fit_result_dict['FCS_chisq'] = FCS_chi_square if use_FCS else np.inf
        fit_result_dict['PCH_chisq'] = PCH_chi_square if use_PCH else np.inf

        
        
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
            for i_spec in range(n_species_spec):
                fit_result_dict[f'N_pop_{i_spec}'] = N_pop_array[i_spec]
                
        fit_result_df = pd.DataFrame(fit_result_dict, index = [1]) 
        
        fit_res_table_path_full = fit_res_table_path + '.csv'
        if not os.path.isfile(fit_res_table_path_full):
            # Does not yet exist - create with header
            fit_result_df.to_csv(fit_res_table_path_full, 
                                  header = True, 
                                  index = False)
            # Set rights
            os.chmod(fit_res_tmp_path, stat.S_IRWXU)
        else:
            # Exists - append
            fit_result_df.to_csv(fit_res_table_path_full, 
                                  mode = 'a', 
                                  header = False, 
                                  index = False)


                
                
    return None


