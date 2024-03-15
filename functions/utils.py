# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""

import glob


def detect_files(in_dir_names, file_name_pattern, other_info, exclude_path):
    # Automatic input file detection
    
    # Containers for storing results
    _in_file_names=[]
    _in_dir_names = [] 
    _other_info = []
    
    # Iteration over directories
    for i_dir, directory in enumerate(in_dir_names):
    
        # Iteration over files: Find all that match pattern, and list
        for name in glob.glob(directory + '/**/*' + file_name_pattern, recursive = True):
            
            head, tail = os.path.split(name)
            tail = tail.strip('.csv')
            
            if head != exclude_path
                _in_file_names.extend([tail])
                _in_dir_names.extend([head])
                _other_info.append(other_info[i_dir])
    
    if len(in_dir_names) == 0:
        raise ValueError('Could not detect any files.')

    return _in_dir_names, _in_file_names, _other_info