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
label_efficiency = 1e-1
n_species = 10

# Total number of particles to consider
N_total = 30000


# Relative species abundances - will be renormalized and scaled with N_total
n_species = 20
gauss_params = np.array([
    [1, 10, 2]
    [0.5, 50, 30],
    [1, 100, 10]]) # relative AUC, mean, sigma for each population

monomer_brightness = 100000
monomer_tau_diff = 2E-4 # 200 us, typical protein

FCS_psf_width_um = 0.350 # To match SimFCS settings 


# e.g. for stoichiometry_scaling_base = 2, species with stoichiometries 1, 2, 4, 8, ... will be simulated
stoichiometry_scaling_base = 2.

save_folder = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3e'
save_name = 'batch3e_2_label1e-1_simParams'



#%% Processing

distribution_x = np.logspace(start = 0, 
                             stop = np.log10(gauss_params[-1, 1] + 2 * gauss_params[-1, 2]), 
                             num = n_species)
log_distribution_x = np.log(distribution_x)
log_binwidth = np.mean(np.diff(log_distribution_x))
binwidth = (np.exp(log_distribution_x + log_binwidth/2) - np.exp(log_distribution_x - log_binwidth/2))


distribution_y = np.zeros(n_species)
for i_gauss in gauss_params.shape[0]:
    auc = gauss_params[i_gauss, 0]
    logmu = np.log(gauss_params[i_gauss, 1])
    logsigma = np.log(gauss_params[i_gauss, 2])
    distribution_y += auc / np.sqrt(2 * np.pi) / distribution_x / logsigma * np.exp(-0.5 * ((log_distribution_x - logmu) / logsigma)**2) * binwidth
distribution_y = distribution_y / distribution_y.sum() * N_total

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
            effective_species_N[pointer] = distribution_y[i_stoichiometry] * binomial_dist.pmf(i_labelling + 1)
            pointer += 1
            
effective_species_cpms *= monomer_brightness
effective_species_tau_diff *= monomer_tau_diff
effective_species_weights = effective_species_cpms**2 * effective_species_N 
effective_species_weights /= effective_species_weights.sum()
effective_species_D = FCS_psf_width_um**2 / 4 / effective_species_tau_diff

# Write stuff sorted from species with largest weight to species with smallest weight:
# In an older version intended for use with SimFCS, there was an additional 
# sorting step here as SimFCS can only simulate 50 effective species, so we 
# stick to the most significant amplitude terms - or that was the idea, turns
#  out SimFCS did not handle that either.
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

