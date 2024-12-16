# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:16:39 2024

@author: Krohn
"""

import numpy as np
import pandas as pd
import os
import re
import stat

# Short, crude script for sorting the fits from a number of datasets analyzed 
# on a single batch, into files that are easier to work with in Origin which I
# use for follow-up statistics

# Note that this is based on naming patterns introduced by PicoQuant's 
# SymPhoTime software and my personal habits of naming files. Adapting this to 
# other people's data may require a significant re-write of the string 
# comparisons.



folder = r'D:\temp\20242116\par_StrExp_70spec_spherical_shell_N_oligomers_lblcr_FCS'
# spectrum_parameter = 'N_pop_mono_histogram'
spectrum_parameter = 'N_pop_oligo_histogram'
# spectrum_parameter = 'N_obs_oligo_histogram'


condition_name_patterns = ['dummy']
sample_type_name_patterns = ['dummy']



#%% Functions
def filepart(col, indx, trim = 0):
    new_col = []
    indx = int(indx)
    for element in col:
    
        for iter in range(indx):
            if iter == indx - 1:
                if trim > 0:
                    new_col.append(os.path.split(element)[1][:-1 * trim])
                else:
                    new_col.append(os.path.split(element)[1])
            else:
                element = os.path.split(element)[0]
    return new_col


def search_for_patterns(col,
                        list_of_patterns):
    new_col = []
    for element in col:
        found = False
        for pattern in list_of_patterns:
            if pattern.lower() in element.lower():
                new_col.append(pattern)
                found = True
                break
        if not found:
            new_col.append('none')
    return new_col


def extract_time(col):
    new_col = []
    for element in col:
        match = re.search(r'_T(\d+)s_1_', element)
        if match:
            new_col.append(float(match.group(1)))
        else:
            new_col.append(np.nan)
    return new_col

#%% Run

file_spectra = 'Fit_params_' + os.path.split(folder)[-1] + '_' + spectrum_parameter + '.csv'
file_stoi = 'Fit_params_' + os.path.split(folder)[-1] + '_stoi.csv'
file_stoi_bw = 'Fit_params_' + os.path.split(folder)[-1] + '_stoi_bw.csv'
file_tau =  'Fit_params_' + os.path.split(folder)[-1] + '_tau_diff.csv'



path_spectra = os.path.join(folder, file_spectra)
path_stoi = os.path.join(folder, file_stoi)
path_stoi_bw = os.path.join(folder, file_stoi_bw)
path_tau = os.path.join(folder, file_tau)

data_spectra = pd.read_csv(path_spectra)
data_stoi = pd.read_csv(path_stoi)
data_stoi_bw = pd.read_csv(path_stoi_bw)
data_tau = pd.read_csv(path_tau)



data_spectra['folder_short'] = filepart(data_spectra['folder'],
                                        1)

data_spectra['condition'] = search_for_patterns(data_spectra['folder_short'],
                                                condition_name_patterns)

data_spectra['sample_type'] = search_for_patterns(data_spectra['folder_short'],
                                                  sample_type_name_patterns)

data_spectra['time_tag'] = extract_time(data_spectra['folder_short'])



for condition in data_spectra['condition'].unique():
    for sample_type in data_spectra['sample_type'].unique():
        keep_indices = []
        time_tags = []
        
        # Identify datasets that belong to this group
        for i_dataset, condition_test in enumerate(data_spectra['condition']):
            sample_type_test = data_spectra.loc[i_dataset, 'sample_type']
            
            if condition_test == condition and sample_type_test == sample_type:
                # Correct dataset
                keep_indices.append(i_dataset)
                time_tags.append(data_spectra.loc[i_dataset, 'time_tag'])
                
        if len(keep_indices) == 0:
            # Skip if there was no data matching this combination of condition and sample_type
            continue
            
        
        # Sort by time point
        sort_order = np.argsort(time_tags)
        keep_indices = np.array(keep_indices)
        keep_indices = keep_indices[sort_order]
        
        # New dataframe
        out_df = pd.DataFrame()
        
        
        # Add diffusion time column
        pointer = 3
        out_df.loc[0, 'tau_diff'] = np.nan # dummy rows
        out_df.loc[1, 'tau_diff'] = np.nan # dummy rows
        out_df.loc[2, 'tau_diff'] = np.nan # dummy rows
        for key in data_tau.keys():
            if re.search('tau_diff_', key):
                out_df.loc[pointer, 'tau_diff'] = data_tau.loc[keep_indices[0], key]
                pointer += 1

        # Add stoichiometry column
        pointer = 3
        out_df.loc[0, 'stoichiometry'] = np.nan # dummy rows
        out_df.loc[1, 'stoichiometry'] = np.nan # dummy rows
        out_df.loc[2, 'stoichiometry'] = np.nan # dummy rows
        for key in data_stoi.keys():
            if re.search('stoichiometry_', key):
                out_df.loc[pointer, 'stoichiometry'] = data_stoi.loc[keep_indices[0], key]
                pointer += 1

        # Add stoichiometry binwidth column
        pointer = 3
        out_df.loc[0, 'stoi_bw'] = np.nan # dummy rows
        out_df.loc[1, 'stoi_bw'] = np.nan # dummy rows
        out_df.loc[2, 'stoi_bw'] = np.nan # dummy rows
        for key in data_stoi_bw.keys():
            if re.search('stoi_bw_', key):
                out_df.loc[pointer, 'stoi_bw'] = data_stoi_bw.loc[keep_indices[0], key]
                pointer += 1


        # Add data, while effectively flipping and sorting spreadsheet axes
        for i_dataset_old in keep_indices:
            dataset_name = data_spectra.loc[i_dataset_old, 'folder_short']
            
            # Here comes why we needed the dummy rows: To annotate time tags and quality of fit
            out_df.loc[0, dataset_name] = data_spectra.loc[i_dataset_old, 'time_tag']  # Time
            out_df.loc[1, dataset_name] = data_spectra.loc[i_dataset_old, 'FCS_chisq'] # goodness of fit FCS
            out_df.loc[2, dataset_name] = data_spectra.loc[i_dataset_old, 'PCH_chisq'] # goodness of fit PCH
            
            pointer = 3     
            for key in data_spectra.keys():
                if re.search(spectrum_parameter, key):
                    out_df.loc[pointer, dataset_name] = data_spectra.loc[i_dataset_old, key]
                    pointer += 1        

        # Done, write this collection of datasets into its own spreadsheet
        out_name = os.path.join(folder,
                                f'{sample_type}_{condition}_{spectrum_parameter}_sorted.csv')
        out_df.to_csv(out_name)
        # Set rights
        os.chmod(out_name, stat.S_IRWXU)

        
print('Job done.')
