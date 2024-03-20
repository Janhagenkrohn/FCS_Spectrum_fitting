# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""

# File I/O and path manipulation
import os
import glob

# Data andling/types
import numpy as np


def detect_files(in_dir_names, file_name_pattern, other_info, exclude_path):
    # Automatic input file detection
    
    # Containers for storing results
    _in_file_names=[]
    _in_dir_names = [] 
    _other_info = []
    
    # Iteration over directories
    for i_dir, directory in enumerate(in_dir_names):
        
        # Iteration over files: Find all that match pattern, and list        
        search_pattern = os.path.join(directory, '**/*', file_name_pattern)
                
        for name in glob.glob(search_pattern, 
                              recursive = True):
            
            head, tail = os.path.split(name)
            tail = tail.strip('.csv')
            
            if head != exclude_path:
                _in_file_names.extend([tail])
                _in_dir_names.extend([head])
                _other_info.append(other_info[i_dir])
    
    if len(_in_dir_names) == 0:
        raise Exception('Could not detect any files.')

    return _in_dir_names, _in_file_names, _other_info




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
