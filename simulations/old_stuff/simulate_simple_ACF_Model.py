# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:46:36 2024

@author: Krohn
"""

import numpy as np
import scipy.stats as sstats
import pandas as pd
import os
import matplotlib.pyplot as plt

'''
Simple script for generating an array of FCS simulation parameters to run a
simulation of a distribution of particle sizes

'''


oligomer_type = 'spherical_shell' # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
label_efficiency = 5e-1

# Total number of particles to consider
N_total = 10


# Relative species abundances - will be renormalized and scaled with N_total
distribution = np.array([20, 30, 50, 70, 50, 30, 20])

monomer_brightness = 10000
monomer_tau_diff = 2E-4 # 200 us, typical protein

min_lag_time = 1E-6
max_lag_time = 1.
n_acf_data_points = 200

psf_aspect_ratio= 5

noise_scaling_factor = 1e-2

# e.g. for stoichiometry_scaling_base = 2, species with stoichiometries 1, 2, 4, 8, ... will be simulated
stoichiometry_scaling_base = 2.

save_folder = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\ACF_simulations_direct\3f'
save_name = 'batch3f_1_label5e-1'



#%% Processing

distribution = distribution / distribution.sum() * N_total
n_species = distribution.shape[0]

tau = np.logspace(start = np.log10(min_lag_time),
                  stop = np.log10(max_lag_time),
                  num = n_acf_data_points)
acf_num = np.zeros_like(tau)
acf_den = np.zeros_like(tau)
acf_noise_var_num = np.zeros_like(tau)
acf_noise_var_den = np.zeros_like(tau)

stoichiometry_array = stoichiometry_scaling_base ** np.arange(0, n_species)
stoichiometry_array = np.unique(np.round(stoichiometry_array))
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
            # Get species parameters
            effective_species_tau_diff[pointer] = stoichiometry ** (1/3) * monomer_tau_diff
            effective_species_stoichiometry[pointer] = stoichiometry
            effective_species_cpms[pointer] = (i_labelling + 1) * monomer_brightness 
            effective_species_N[pointer] = distribution[i_stoichiometry] * binomial_dist.pmf(i_labelling + 1)
            
            # Simulate ACF for this species
            g_norm = 1 / (1 + tau/effective_species_tau_diff[pointer]) / np.sqrt(1 + tau / (np.square(psf_aspect_ratio) * effective_species_tau_diff[pointer]))
            if np.any(np.isnan(g_norm)): raise Exception('NaN in g_norm!')
            acf_num += g_norm * effective_species_N[pointer] * effective_species_cpms[pointer]**2
            if np.any(np.isnan(acf_num)): raise Exception('NaN in acf_num!')
            acf_den += effective_species_N[pointer] * effective_species_cpms[pointer]
            
            # Get a noise curve for this species
            # Model that we work with (not accurate by any means, but sort of captures the basic trends): 
                # Gaussian noise
                # SD per data point is linear with relation of molecular brightness and tau
                # Variance per data point is proportional to 1/g_norm
                # Variance overall is proportional to 1/(1-(probability to have 0 particles in PSF of this species at given time))
                # SD has an arbitrary user-supplied SD scaling factor
                # Variance adds up species-wise with weights given by ACF amplitude weights
            
            acf_noise_var_spec = ((tau * effective_species_cpms[pointer])**-2 / # Term for photon shot noise
                                  g_norm / # term for survival function for a particle to still be in PFS
                                  (1 - sstats.poisson(effective_species_N[pointer]).pmf(0)) * # Term for dead time from zero particles in observation volume
                                  noise_scaling_factor**2) # Scaling factor
            if np.any(np.isnan(acf_noise_var_spec)): raise Exception('NaN in acf_noise_var_spec!')

            # Add to total variance 
            acf_noise_var_num += acf_noise_var_spec * effective_species_N[pointer] * effective_species_cpms[pointer]**2
            acf_noise_var_den += effective_species_N[pointer] * effective_species_cpms[pointer]
            
            pointer += 1

# Get total ACF            
acf = acf_num / acf_den**2

# Get uncertainty, normalizing by ACF weights
sd_acf = np.sqrt(acf_noise_var_num / acf_noise_var_den**2 / acf_num[0] * acf_den[0]**2)

# Get noise term
acf += np.random.standard_normal(size = n_acf_data_points) * sd_acf


#%% Write parameters        
effective_species_weights = effective_species_cpms**2 * effective_species_N 
effective_species_weights /= effective_species_weights.sum()

# Write stuff sorted from species with largest weight to species with smallest weight:
# In an older version intended for use with SimFCS, there was an additional 
# sorting step here as SimFCS can only simulate 50 effective species, so we 
# stick to the most significant amplitude terms - or that was the idea, turns
#  out SimFCS did not handle that either.
out_table = pd.DataFrame(data = {'stoichiometry':effective_species_stoichiometry,
                                 'N': effective_species_N,
                                 'cpms':effective_species_cpms,
                                 'tau_diff':effective_species_tau_diff, 
                                 'rel_weights': effective_species_weights})

out_table.to_csv(os.path.join(save_folder, 
                              save_name + '_params.csv'),
                 index = False, 
                 header = True)


#%% FCS Figure
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.semilogx(tau, acf, 'dk')
ax.semilogx(tau, acf + sd_acf, '-k', alpha = 0.7)
ax.semilogx(tau, acf - sd_acf, '-k', alpha = 0.7)
plot_y_min_max = (np.percentile(acf, 15), np.percentile(acf, 85))
ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
            plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)
plt.savefig(os.path.join(save_folder, 
                         save_name + '_ACF_ch0.png'), 
            dpi=300)
plt.show()


#%% Export Kristine csv
acr_col = np.zeros_like(tau)
average_count_rate = np.sum(effective_species_cpms * effective_species_N)
acr_col[:3] = np.array([average_count_rate, average_count_rate, 1E3])
out_table = pd.DataFrame(data = {'Lagtime[s]':tau, 
                                 'Correlation': acf,
                                 'ACR[Hz]': acr_col,
                                 'Uncertainty_SD': sd_acf})
out_table.to_csv(os.path.join(save_folder, 
                              save_name + '_ACF_ch0.csv'),
                 index = False, 
                 header = False)

