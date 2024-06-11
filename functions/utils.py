# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""

# File I/O and path manipulation
import os
import glob

# Data handling
import numpy as np
import pandas as pd


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
        raise Exception(f'Searched {i_dir+1} directories, but could not detect any files.')
        
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
                    raise Exception('Assignment of FCS data and PCH data is ambiguous! More advanced user-defined logic may be needed.')
            
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

