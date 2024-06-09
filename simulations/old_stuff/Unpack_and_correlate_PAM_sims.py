# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:34:31 2024

@author: Krohn
"""

import h5py
import os



filenames = []
foldernames = []


filenames.extend(['batch3e_1_label1e-1_simData.mat'])
# foldernames.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3e'])
foldernames.extend([r'C:\Users\Krohn\Desktop\dilutedY_diluted_X_partitioning'])



for i_file in range(len(filenames)):
    filename = filenames[i_file]
    foldername = foldernames[i_file]
    path = os.path.join(filename, foldername)
    
    
    with h5py.File(path, 'r') as h5file:
        [print(key) for key in h5file.keys()]
        
        macro_times = h5file['Sim_Photons']   
        header = h5file['Header']
        