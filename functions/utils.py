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


def detect_files(in_dir_names, file_name_pattern, other_info, exclude_path):
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
            if file_suffix == '.csv' and dir_name != exclude_path:
                _in_file_names.extend([file_name])
                _in_dir_names.extend([dir_name])
                _other_info.append(other_info[i_dir])
                
    if len(_in_dir_names) == 0:
        raise Exception(f'Searched {i_dir+1} directories, but could not detect any files.')
        
    return _in_dir_names, _in_file_names, _other_info


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
              file_name):

    # Load PCH data
    in_path_PCH = os.path.join(dir_name, file_name)
    data_PCH = pd.read_csv(in_path_PCH + '.csv', header = 0)
    
    # Row 0 = Column names = bin times
    data_PCH_bin_times = np.array([float(bin_time) for bin_time in data_PCH.keys()])
    
    # Rows 1...end: PC(M)H data
    data_PCH_hist = data_PCH.to_numpy()
    
    return data_PCH_bin_times, data_PCH_hist


def link_FCS_and_PCH_files(in_dir_names_FCS,
                           in_file_names_FCS,
                           other_info_FCS,
                           in_dir_names_PCH,
                           in_file_names_PCH,
                           other_info_PCH):
    if len(in_dir_names_PCH) == len(in_dir_names_FCS):
        # One PCH file per FCS file
        
        # We use FCS as reference to sort PCH
        in_file_names_PCH_argsort = []
        
        for in_dir_name_FCS in in_dir_names_FCS:
        # Find the index of the corresponding PCH file
            
            found = False
            for i_pch, in_dir_name_PCH in enumerate(in_dir_names_PCH):
                
                if in_dir_name_PCH == in_dir_name_FCS and not found:
                    # This looks like the right one - Store index for sorting PCH information
                    in_file_names_PCH_argsort.append(i_pch)
                    
                elif in_dir_name_PCH == in_dir_name_FCS and found:
                    # We already found the correct PCH, but now we find another? That does not work!
                    raise Exception('Assignment of FCS data and PCH data is ambiguous! More advanced user-defined logic may be needed.')
            
        in_dir_names = in_dir_names_FCS
        # in_file_names_FCS unchanged
        in_file_names_PCH = [in_file_names_PCH[i_pch] for i_pch in in_file_names_PCH_argsort]
        other_info = other_info_FCS
        
        
    elif len(in_dir_names_PCH) < len(in_dir_names_FCS):
        # We have more FCS files than PCH files
            
        # We use FCS as reference to sort PCH
        in_file_names_PCH_argsort = []
        
        for in_dir_name in in_dir_names_FCS:
        # Find the index of the corresponding PCH file
            
            found = False
            for i_pch, in_dir_name_PCH in enumerate(in_dir_names_PCH):
                
                if in_dir_name_PCH == in_dir_name_FCS and not found:
                    # This looks like the right one - Store index for sorting PCH information
                    in_file_names_PCH_argsort.append(i_pch)
                    
                elif in_dir_name_PCH == in_dir_name_FCS and found:
                    # We already found the correct PCH, but now we find another? That does not work!
                    raise Exception('Assignment of FCS data and PCH data is ambiguous! More advanced user-defined assignment code may be needed.')
                    
        in_dir_names = in_dir_names_FCS
        # in_file_names_FCS unchanged
        in_file_names_PCH = [in_file_names_PCH[i_pch] for i_pch in in_file_names_PCH_argsort]
        other_info = other_info_FCS


    else: # len(in_dir_names_PCH) > len(in_dir_names_FCS)
        # We have more PCH files than FCS files - use PCH as reference
        
        in_file_names_FCS_argsort = []
        
        for in_dir_name in in_dir_names_PCH:
        # Find the index of the corresponding PCH file
            
            found = False
            for i_fcs, in_dir_name_FCS in enumerate(in_dir_names_FCS):
                
                if in_dir_name_PCH == in_dir_name_FCS and not found:
                    # This looks like the right one - Store index for sorting PCH information
                    in_file_names_FCS_argsort.append(i_fcs)
                    
                elif in_dir_name_PCH == in_dir_name_FCS and found:
                    # We already found the correct FCS dataset, but now we find another? That does not work!
                    raise Exception('Assignment of FCS data and PCH data is ambiguous! More advanced user-defined assignment code may be needed.')
            
        in_dir_names = in_dir_names_PCH
        in_file_names_FCS = [in_file_names_FCS[i_fcs] for i_fcs in in_file_names_FCS_argsort]
        # in_file_names_PCH unchanged
        other_info = other_info_PCH
        
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
