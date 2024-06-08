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
import multiprocessing
import traceback
'''
Simple script for generating an array of FCS simulation parameters to run a
simulation of a distribution of particle sizes

'''


oligomer_types = ['spherical_shell', 'spherical_dense', 'single_filament'] # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'
# oligomer_types = ['spherical_dense', 'single_filament'] # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'


label_efficiencies = [1E-4, 1e-2, 1e-1, 1e-0]
save_names = []
[save_names.append(f'3peaks_mu5-20-50_sigma2-30-10_label1e-{log_le}') for log_le in [4, 2, 1, 0]]

# Total number of particles to consider
N_total = 10


# Relative species abundances - will be renormalized and scaled with N_total
# Many species are needed as this is actually not Gaussian but lognormal 
n_species = 100
gauss_params = np.array([
    [1, 5, 2],
    [1E-1, 10, 3],
    [1e-3, 20, 5]]) # relative AUC, mean, sigma for each population

# gauss_params = np.array([
#     [1E-3, 10, 5]
#     ]) # relative AUC, mean, sigma for each population


monomer_brightness = 10000
monomer_tau_diff = 2E-4 # 200 us, typical protein

min_lag_time = 1E-6
max_lag_time = 1.
n_acf_data_points = 200

psf_aspect_ratio= 5


add_noise = False
noise_scaling_factor = 1e-4

save_folder_glob = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\ACF_simulations_direct\3f'


#%% Processing

n_sims = len(label_efficiencies) * len(oligomer_types)


# Parallel processing function
def run_single_sim(i_simulation,
                   stoichiometry_array,
                   distribution_y,
                   tau,
                   oligomer_type,
                   monomer_tau_diff,
                   monomer_brightness,
                   label_efficiency,
                   save_path):
    print(f'Starting simulation number {i_simulation} with lablling efficiency {label_efficiency}...')

    acf_num = np.zeros_like(tau)
    acf_den = np.zeros_like(tau)
    acf_noise_var_num = np.zeros_like(tau)
    acf_noise_var_den = np.zeros_like(tau)
    
    stoichiometry_array = np.round(stoichiometry_array)
    tau_diff_array = np.zeros_like(stoichiometry_array)
    species_weights = np.zeros_like(stoichiometry_array)
    species_signal = np.zeros_like(stoichiometry_array)

    n_effective_species = int(stoichiometry_array.sum())
    effective_species_tau_diff = np.zeros(n_effective_species)
    effective_species_N = np.zeros(n_effective_species)
    effective_species_cpms = np.zeros(n_effective_species)
    effective_species_stoichiometry = np.zeros(n_effective_species)
    
    pointer = 0
    
    for i_stoichiometry, stoichiometry in enumerate(stoichiometry_array):

        # Parameterize Binomial dist object
        binomial_dist = sstats.binom(stoichiometry, 
                                     label_efficiency)
        
        if oligomer_type == 'spherical_dense':
            tau_diff_array[i_stoichiometry] = stoichiometry ** (1/3) * monomer_tau_diff
    
        elif oligomer_type == 'spherical_shell':
            tau_diff_array[i_stoichiometry] = stoichiometry ** (1/2) * monomer_tau_diff
            
        elif oligomer_type == 'single_filament':
            tau_diff_array[i_stoichiometry] = monomer_tau_diff * 2. * stoichiometry / (2. * np.log(stoichiometry) + 0.632 + 1.165 * stoichiometry ** (-1.) + 0.1 * stoichiometry ** (-2.))

        elif oligomer_type == 'double_filament':
            axial_ratio = stoichiometry / 2.
            tau_diff_array[i_stoichiometry] = monomer_tau_diff * 2. * axial_ratio / (2. * np.log(axial_ratio) + 0.632 + 1.165 * axial_ratio ** (-1.) + 0.1 * axial_ratio ** (-2.))
        else:
            raise Exception(f"Invalid oligomer_type. Must be 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament', got {oligomer_type}")
            
        # Normalized correlation 
        g_norm_spec = 1 / (1 + tau/tau_diff_array[i_stoichiometry]) / np.sqrt(1 + tau / (np.square(psf_aspect_ratio) * tau_diff_array[i_stoichiometry]))

        for i_labelling in range(int(stoichiometry)):
            if i_labelling % 100000 == 0 and i_labelling > 1:
                print(f'Label efficiency {i_labelling} ({np.round(i_labelling / stoichiometry * 100, 3)} %)...')
                
            # Get species parameters
            effective_species_tau_diff[pointer] = tau_diff_array[i_stoichiometry]
            effective_species_stoichiometry[pointer] = stoichiometry
            effective_species_cpms[pointer] = (i_labelling + 1) * monomer_brightness 
            effective_species_N[pointer] = distribution_y[i_stoichiometry] * binomial_dist.pmf(i_labelling + 1)
                            
            species_weights[i_stoichiometry] += effective_species_N[pointer] * effective_species_cpms[pointer]**2
            species_signal[i_stoichiometry] += effective_species_N[pointer] * effective_species_cpms[pointer]
            
            if add_noise:
                # Get a noise curve for this species
                # Model that we work with (not accurate by any means, but sort of captures the basic trends): 
                    # Gaussian noise
                    # SD per data point is linear with relation of molecular brightness and tau
                    # Variance per data point is proportional to 1/g_norm
                    # Variance overall is proportional to 1/(1-(probability to have 0 particles in PSF of this species at given time))
                    # SD has an arbitrary user-supplied SD scaling factor
                    # Variance adds up species-wise with weights given by ACF amplitude weights
                
                acf_noise_var_spec = ((tau * effective_species_cpms[pointer])**-2 / # Term for photon shot noise
                                      g_norm_spec * # term for survival function for a particle to still be in PFS
                                      (sstats.poisson(effective_species_N[pointer]).pmf(0) + 1e-6) * # Term for dead time from zero particles in observation volume
                                      noise_scaling_factor**2) # Scaling factor
                if np.any(np.isnan(acf_noise_var_spec)): raise Exception('NaN in acf_noise_var_spec!')
                if np.any(np.isinf(acf_noise_var_spec)): raise Exception('Inf in acf_noise_var_spec!')
        
                # Add to total variance 
                acf_noise_var_num += acf_noise_var_spec * effective_species_N[pointer] * effective_species_cpms[pointer]**2
                acf_noise_var_den += effective_species_N[pointer] * effective_species_cpms[pointer]
            
            pointer += 1
            
        # Get correlation fucntion weights for this species

        acf_num += g_norm_spec * species_weights[i_stoichiometry]
        acf_den += species_signal[i_stoichiometry]

    
    # Get total ACF            
    acf = acf_num / acf_den**2
    
    # Get uncertainty, normalizing by ACF weights
    if add_noise:
        sd_acf = np.sqrt(acf_noise_var_num / acf_noise_var_den**2 / acf_num[0] * acf_den[0]**2)
        # Get noise term
        acf_noise = acf + np.random.standard_normal(size = n_acf_data_points) * sd_acf
    else:
        sd_acf = np.ones_like(acf)
        acf_noise = acf
    
    ################### Write parameters        
    
    # Effective species (resolved by labelling statistics)
    effective_species_weights = effective_species_cpms**2 * effective_species_N 
    effective_species_weights /= effective_species_weights.sum()
    
    out_table = pd.DataFrame(data = {'stoichiometry':effective_species_stoichiometry,
                                     'N': effective_species_N,
                                     'cpms':effective_species_cpms,
                                     'tau_diff':effective_species_tau_diff, 
                                     'rel_weights': effective_species_weights})
    
    out_table.to_csv(save_path + '_sim_params.csv',
                     index = False, 
                     header = True)
    
    # Diffusion species
    species_weights /= species_weights.sum()
    species_signal /= species_signal.sum()
    
    out_table = pd.DataFrame(data = {'stoichiometry':stoichiometry_array,
                                     'N': distribution_y,
                                     'tau_diff':tau_diff_array, 
                                     'rel_weights': species_weights,
                                     'rel_fluorescence': species_signal})
    
    out_table.to_csv(save_path + '_species_params.csv',
                     index = False, 
                     header = True)


    ############# FCS Figure
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    ax.semilogx(tau, acf_noise, 'dk')
    ax.semilogx(tau, acf_noise + sd_acf, '-k', alpha = 0.7)
    ax.semilogx(tau, acf_noise - sd_acf, '-k', alpha = 0.7)
    plot_y_min_max = (np.percentile(acf_noise, 15), np.percentile(acf_noise, 85))
    ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)
    plt.savefig(save_path + '_ACF_ch0.png', 
                dpi=300)
    # plt.show()
    
    ########## Spectrum figure
    fig, ax = plt.subplots(nrows=3, ncols=1)
    
    ax[0].semilogx(tau_diff_array, distribution_y, 'dk', label = '<N> over tau_diff')
    ax[1].semilogx(tau_diff_array, species_signal, 'dk', label = 'Rel. fluorescence signal over tau_diff')
    ax[2].semilogx(tau_diff_array, species_weights, 'dk', label = 'Rel. ACF weights over tau_diff')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.savefig(save_path + '_stoichiometry.png', 
                dpi=300)
    # plt.show()
    
    
    ######## Export Kristine csv
    acr_col = np.zeros_like(tau)
    average_count_rate = np.sum(effective_species_cpms * effective_species_N)
    acr_col[:3] = np.array([average_count_rate, average_count_rate, 1E3])
    out_table = pd.DataFrame(data = {'Lagtime[s]':tau, 
                                     'Correlation': acf_noise,
                                     'ACR[Hz]': acr_col,
                                     'Uncertainty_SD': sd_acf})
    out_table.to_csv(save_path + '_ACF_ch0.csv',
                     index = False, 
                     header = False)
    
    

    
    
#%% Wrap parameters

i_simulation_list = []
stoichiometry_array_list = []
distribution_y_list = []
tau_list = []
oligomer_type_list = []
monomer_tau_diff_list = []
monomer_brightness_list = []
label_efficiency_list = []
save_path_list = []

list_of_param_tuples = []



pointer = 0
for oligomer_type in oligomer_types:
    save_folder = os.path.join(save_folder_glob, oligomer_type)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for i_simulation, label_efficiency in enumerate(label_efficiencies):

        save_name = save_names[i_simulation]
    

        
        stoichiometry_array = np.logspace(start = 0, 
                                          stop = np.log10(gauss_params[-1, 1] * (gauss_params[-1, 2]**8 if gauss_params[-1, 2]**8 > 2. else 2.)), 
                                          num = n_species)
        stoichiometry_array = np.unique(np.round(stoichiometry_array))
        log_distribution_x = np.log(stoichiometry_array)
        log_binwidth = np.mean(np.diff(log_distribution_x))
        binwidth = (np.exp(log_distribution_x + log_binwidth/2) - np.exp(log_distribution_x - log_binwidth/2))
        
        
        distribution_y = np.zeros_like(stoichiometry_array)
        for i_gauss in range(gauss_params.shape[0]):
            auc = gauss_params[i_gauss, 0]
            logmu = np.log(gauss_params[i_gauss, 1])
            logsigma = np.log(gauss_params[i_gauss, 2])
            distribution_y += auc / np.sqrt(2 * np.pi) / stoichiometry_array / logsigma * np.exp(-0.5 * ((log_distribution_x - logmu) / logsigma)**2) * binwidth
        distribution_y = distribution_y / distribution_y.sum() * N_total
        
        tau = np.logspace(start = np.log10(min_lag_time),
                          stop = np.log10(max_lag_time),
                          num = n_acf_data_points)
        
        
        list_of_param_tuples.append((pointer,
                                     stoichiometry_array,
                                     distribution_y,
                                     tau,
                                     oligomer_type,
                                     monomer_tau_diff,
                                     monomer_brightness,
                                     label_efficiency,
                                     os.path.join(save_folder, save_name)))
        
        pointer += 1
        
        
#%% Run parallel
try:
    mp_pool = multiprocessing.Pool(processes = (os.cpu_count() - 1 if os.cpu_count() - 1 < pointer else pointer))
    
    _ = [mp_pool.starmap(run_single_sim, list_of_param_tuples)]
except:
    traceback.print_exception()
finally:
    mp_pool.close()
    
    
"""
run_single_sim(i_simulation,
                   stoichiometry_array,
                   distribution_y,
                   tau,
                   oligomer_type,
                   monomer_tau_diff,
                   monomer_brightness,
                   label_efficiency,
                   save_path)"""
