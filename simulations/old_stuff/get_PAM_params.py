# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:46:36 2024

@author: Krohn
"""

import numpy as np
import scipy.stats as sstats
import pandas as pd
import os

'''
Simple script for generating an array of FCS simulation parameters to run a
simulation of a distribution of particle sizes

'''


oligomer_type = 'spherical_shell' # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
label_efficiency = 8e-4
n_species = 10

# Total number of particles to consider
N_total = 30000


# Relative species abundances - will be renormalized and scaled with N_total
distribution = np.array([20, 30, 50, 70, 50, 30, 20])

monomer_brightness = 100000
monomer_tau_diff = 2E-4 # 200 us, typical protein

FCS_psf_width_um = 0.350 # To match SimFCS settings 


# e.g. for stoichiometry_scaling_base = 2, species with stoichiometries 1, 2, 4, 8, ... will be simulated
stoichiometry_scaling_base = 2.

save_folder = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3e'
save_name = 'batch3e_1_label8e-4_simParams'



#%% Processing

distribution = distribution / distribution.sum() * N_total
n_species = distribution.shape[0]


stoichiometry_array = stoichiometry_scaling_base ** np.arange(0, n_species)
n_effective_species = int(stoichiometry_array.sum())
effective_species_tau_diff = np.zeros(n_effective_species)
effective_species_N = np.zeros(n_effective_species)
effective_species_cpms = np.zeros(n_effective_species)
effective_species_stoichiometry = np.zeros(n_effective_species)

pointer = 0

for i_stoichiometry, stoichiometry in enumerate(stoichiometry_array):
    
    if oligomer_type == 'spherical_shell':
        # Parameterize Binomial dist object
        binomial_dist = sstats.binom(stoichiometry, 
                                     label_efficiency)
        
        for i_labelling in range(int(stoichiometry)):
            effective_species_tau_diff[pointer] = stoichiometry ** (1/3)
            effective_species_stoichiometry[pointer] = stoichiometry
            effective_species_cpms[pointer] = i_labelling + 1
            effective_species_N[pointer] = distribution[i_stoichiometry] * binomial_dist.pmf(i_labelling + 1)
            pointer += 1
            
effective_species_cpms *= monomer_brightness
effective_species_tau_diff *= monomer_tau_diff
effective_species_weights = effective_species_cpms**2 * effective_species_N 
effective_species_weights /= effective_species_weights.sum()
effective_species_D = FCS_psf_width_um**2 / 4 / effective_species_tau_diff

# Write stuff
out_table = pd.DataFrame(data = {'stoichiometry':effective_species_stoichiometry,
                                 'N': effective_species_N,
                                 'cpms':effective_species_cpms,
                                 'D': effective_species_D,
                                 'tau_diff':effective_species_tau_diff, 
                                 'rel_weights': effective_species_weights})

out_table.to_csv(os.path.join(save_folder, 
                              save_name + '.csv'),
                 index = False, 
                 header = True)
