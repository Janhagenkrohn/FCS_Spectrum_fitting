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
import imageio
'''
Simple script for generating an array of FCS simulation parameters to run a
simulation of a distribution of particle sizes

'''


oligomer_types = ['single_filament'] # 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'


label_efficiencies = [1.]*20
save_names = []
[save_names.append(f'Batch1_run_{x}') for x in range(len(label_efficiencies))]

# Write the huge CSV file with all the species (This thing tends to be GB scale, and take a few min to write)?
write_big_spreadsheet = False


# Total number of particles to consider
N_total = 10

# Random fluctautions of particle numbers observed around expectation values?
particle_number_fluctuations = True

# Add Gaussian noise to ACF (set to <= 0 to not use noise)?
gaussian_noise = 0.05 # Noise coefficient of variation, Float

# Only has an effect if particle_number_fluctuations == True
acquisition_time = 60 # seconds

# Relative species abundances - will be renormalized and scaled with N_total
# Many species are needed as this is actually not Gaussian but lognormal 
n_species = 100


gauss_params = np.array([
    [1E-3, 8, 8]
    ]) # relative AUC, mean, sigma for each population

monomer_brightness = 10000
monomer_tau_diff = 2E-4 # 200 us, rather typical protein

min_lag_time = 1E-6
max_lag_time = 1.
n_acf_data_points = 200

psf_width_xy = 0.2 # micrometers
psf_aspect_ratio= 5

RICS_scan_speeds = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4]) # seconds / px
RICS_pixel_size = 0.04 # micrometers
RICS_line_length = 128 # pixels per line, sets time scales assuming no dead time (also assumes a square FOV); must be even integer!
pCF_distances = np.array([0.1, 0.25, 0.5, 1., 2.]) # micrometers


# save_folder_glob = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\ACF_simulations_direct\3f'
save_folder_glob = 'C:\TEMP'


#%% Processing
n_sims = len(label_efficiencies) * len(oligomer_types)

# We actually need this one to enumerate over axis 1
pCF_distances = np.reshape(pCF_distances,
                           newshape = [1, pCF_distances.shape[0]])

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
    print(f'Starting simulation number {i_simulation} -- labelling efficiency {label_efficiency}')
    
    # Normalization factor is the same for all correlation functions
    acf_den = 0.
    
    # Single-spot FCS
    single_spot_acf_num = np.zeros_like(tau)
    
    # RICS, also defining some geometry-related parameters
    RICS_map_num = np.zeros(shape = [RICS_line_length, RICS_line_length, RICS_scan_speeds.shape[0]])
    RICS_mesh_enumeration_fast = np.arange(-RICS_line_length // 2, RICS_line_length // 2)
    RICS_meshgrid_fast_axis = np.repeat(np.reshape(RICS_mesh_enumeration_fast,
                                                   newshape = [RICS_mesh_enumeration_fast.shape[0], 1]), 
                                        repeats = RICS_line_length,
                                        axis = 1)
    RICS_meshgrid_slow_axis = np.transpose(RICS_meshgrid_fast_axis)
    RICS_meshgrid_fast_axis = np.reshape(RICS_meshgrid_fast_axis,
                                         newshape = [RICS_meshgrid_fast_axis.shape[0], RICS_meshgrid_fast_axis.shape[1], 1])
    RICS_meshgrid_slow_axis = np.reshape(RICS_meshgrid_slow_axis,
                                         newshape = [RICS_meshgrid_slow_axis.shape[0], RICS_meshgrid_slow_axis.shape[1], 1])
    RICS_pixel_times = np.reshape(RICS_scan_speeds,
                                  newshape = [1, 1, RICS_scan_speeds.shape[0]])
    RICS_line_times = RICS_pixel_times * RICS_line_length
    RICS_abs_lag_map = np.abs(RICS_pixel_times * RICS_meshgrid_fast_axis + RICS_line_times * RICS_meshgrid_slow_axis)

    # pCF
    pCF_num = np.zeros(shape = [tau.shape[0], pCF_distances.shape[1]])
    
    stoichiometry_array = np.round(stoichiometry_array)
    n_stoichiometries = stoichiometry_array.shape[0]
    tau_diff_array = np.zeros_like(stoichiometry_array)
    diff_coeff_array = np.zeros_like(stoichiometry_array)
    species_weights = np.zeros_like(stoichiometry_array)
    species_signal = np.zeros_like(stoichiometry_array)

    n_effective_species = int(stoichiometry_array.sum())
    effective_species_tau_diff = np.zeros(n_effective_species)
    effective_species_N = np.zeros(n_effective_species)
    if particle_number_fluctuations:
        effective_species_N_obs = np.zeros(n_effective_species)
    effective_species_cpms = np.zeros(n_effective_species)
    effective_species_stoichiometry = np.zeros(n_effective_species)
    effective_species_diff_coeff = np.zeros(n_effective_species)
    
    pointer = 0
    
    for i_stoichiometry, stoichiometry in enumerate(stoichiometry_array):
        print(f'Simulation {i_simulation} -- stoichiometry {stoichiometry} ({i_stoichiometry+1} / {n_stoichiometries})')

        # Parameterize Binomial dist object
        binomial_dist = sstats.binom(stoichiometry, 
                                     label_efficiency)
        
        # Get diffusion times
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
        
        # Recalculate diffusion coefficient
        diff_coeff_array[i_stoichiometry] = psf_width_xy**2 / 4 / tau_diff_array[i_stoichiometry]
        
        # Normalized correlation single-spot
        single_spot_g_norm_spec = 1 / (1 + tau/tau_diff_array[i_stoichiometry]) / np.sqrt(1 + tau / (np.square(psf_aspect_ratio) * tau_diff_array[i_stoichiometry]))
        
        # Normalized correlation RICS
        RICS_g_norm_spec = 1 / (1 + 4 * diff_coeff_array[i_stoichiometry] * RICS_abs_lag_map  / psf_width_xy**2) / \
            np.sqrt(1 + 4 * diff_coeff_array[i_stoichiometry] * RICS_abs_lag_map  / psf_aspect_ratio**2 / psf_width_xy**2) * \
            np.exp(- (RICS_pixel_size**2 * (RICS_meshgrid_fast_axis**2 + RICS_meshgrid_slow_axis**2)) / (psf_width_xy**2 + 4 * diff_coeff_array[i_stoichiometry] * RICS_abs_lag_map))
        
        # normalized correlations pCF
        pCF_g_norm_spec = np.repeat(np.reshape(single_spot_g_norm_spec,
                                               newshape = [single_spot_g_norm_spec.shape[0], 1]),
                                    repeats = pCF_distances.shape[1],
                                    axis = 1)
        pCF_g_norm_spec *= np.exp(- pCF_distances**2 / (4 * diff_coeff_array[i_stoichiometry] * tau.reshape([tau.shape[0],1]) + psf_width_xy**2))
        
        
        # For going from oligomer size to label count
        int_stoichiometry = int(stoichiometry)
        labelling_efficiencies = np.arange(1, int_stoichiometry + 1)
        
        # Get species parameters
        effective_species_tau_diff[pointer:pointer+int_stoichiometry] = tau_diff_array[i_stoichiometry]
        effective_species_stoichiometry[pointer:pointer+int_stoichiometry] = stoichiometry
        effective_species_cpms[pointer:pointer+int_stoichiometry] = labelling_efficiencies * monomer_brightness 
        effective_species_N[pointer:pointer+int_stoichiometry] = distribution_y[i_stoichiometry] * binomial_dist.pmf(labelling_efficiencies)
        if particle_number_fluctuations:
            effective_species_N_obs[pointer:pointer+int_stoichiometry] = np.random.poisson(effective_species_N[pointer:pointer+int_stoichiometry] * acquisition_time / tau_diff_array[i_stoichiometry]) / acquisition_time * tau_diff_array[i_stoichiometry]
        effective_species_diff_coeff[pointer:pointer+int_stoichiometry] = diff_coeff_array[i_stoichiometry]
        
        # Weights for correlation functions
        if particle_number_fluctuations:
            # Including particle number fluctuations
            species_weights[i_stoichiometry] = np.sum(effective_species_N_obs[pointer:pointer+int_stoichiometry] * effective_species_cpms[pointer:pointer+int_stoichiometry]**2)
            species_signal[i_stoichiometry] = np.sum(effective_species_N_obs[pointer:pointer+int_stoichiometry] * effective_species_cpms[pointer:pointer+int_stoichiometry])

        else:
            # The normal way, assuming full sampling
            species_weights[i_stoichiometry] = np.sum(effective_species_N[pointer:pointer+int_stoichiometry] * effective_species_cpms[pointer:pointer+int_stoichiometry]**2)
            species_signal[i_stoichiometry] = np.sum(effective_species_N[pointer:pointer+int_stoichiometry] * effective_species_cpms[pointer:pointer+int_stoichiometry])
        
        # Update pointer for next iteration
        pointer += int_stoichiometry
            
        # Finalize correlation fucntion weights for this species
        single_spot_acf_num += single_spot_g_norm_spec * species_weights[i_stoichiometry]
        RICS_map_num += RICS_g_norm_spec * species_weights[i_stoichiometry]
        pCF_num += pCF_g_norm_spec * species_weights[i_stoichiometry]
        acf_den += species_signal[i_stoichiometry]

    print(f'Simulation {i_simulation} -- Wrapping up and saving spreadsheets')

    ################### Write parameters        
    
    if write_big_spreadsheet:
        # Effective species (resolved by labelling statistics)
        effective_species_weights = effective_species_cpms**2 * effective_species_N 
        effective_species_weights /= effective_species_weights.sum()
        out_table = pd.DataFrame(data = {'stoichiometry':effective_species_stoichiometry,
                                         'N': effective_species_N,
                                         'cpms':effective_species_cpms,
                                         'tau_diff':effective_species_tau_diff, 
                                         'rel_weights': effective_species_weights,
                                         'diff_coeff': effective_species_diff_coeff})
        if particle_number_fluctuations:
            out_table['N_obs'] = effective_species_N_obs
    
        out_table.to_csv(save_path + '_sim_params.csv',
                         index = False, 
                         header = True)

    # Normalize and write ACF            
    single_spot_acf = single_spot_acf_num / acf_den**2
    if gaussian_noise > 0:
        single_spot_acf += np.random.normal(loc = 0,
                                            scale = np.abs(single_spot_acf) * gaussian_noise)

    acr_col = np.zeros_like(tau)
    average_count_rate = np.sum(effective_species_cpms * effective_species_N)
    acr_col[:3] = np.array([average_count_rate, average_count_rate, acquisition_time])
    out_table = pd.DataFrame(data = {'Lagtime[s]':tau, 
                                     'Correlation': single_spot_acf,
                                     'ACR[Hz]': acr_col,
                                     'Uncertainty_SD': np.ones_like(single_spot_acf)})
    print('Saving ' + save_path + '_ACF_ch0.csv' )
    out_table.to_csv(save_path + '_ACF_ch0.csv',
                     index = False, 
                     header = False)
    
        
    
    # Normalize and write RICS data
    # As we save 16 bit images, we normalize differently
    RICS_maps = RICS_map_num.copy()
    if gaussian_noise > 0:
        RICS_maps += np.random.normal(loc = 0,
                                      scale = np.abs(RICS_maps) * gaussian_noise)
        RICS_maps[RICS_maps < 0] = 0

    for i_map in range(RICS_maps.shape[2]):
        RICS_maps[:,:, i_map] = np.round(RICS_maps[:,:, i_map] / RICS_maps[:,:, i_map].max() * 65535)
    RICS_maps = np.uint16(RICS_maps)
    print('Saving ' + save_path + '_RICS_maps.tiff' )
    imageio.mimwrite(save_path + '_RICS_maps.tiff',
                     np.moveaxis(np.moveaxis(RICS_maps,
                                             -1,
                                             0),
                                 -1,
                                 1),
                     ) 
    # Double moveaxis to compensate between ImageJ and Python 
    # Could probably be done easier, but whatever, it works and is not performance-critical
    
    # Normalize and write pCF data
    pCF_ccs = pCF_num / acf_den**2
    if gaussian_noise > 0:
        pCF_ccs += np.random.normal(loc = 0,
                                    scale = np.abs(pCF_ccs) * gaussian_noise)

    acr_col = np.zeros_like(tau)
    average_count_rate = np.sum(effective_species_cpms * effective_species_N)
    acr_col[:3] = np.array([average_count_rate, average_count_rate, acquisition_time])
    out_table = pd.DataFrame(data = {'Lagtime[s]':tau, 
                                     'ACF': single_spot_acf})
    for i_dist, dist in enumerate(pCF_distances[0,:]):
        out_table[f'pCF_{int(dist*1E3)}nm'] = pCF_ccs[:, i_dist]
    print('Saving ' + save_path + '_RICS_maps.tiff' )
    out_table.to_csv(save_path + '_pCF.csv',
                     index = False, 
                     header = True)

    
    print(f'Simulation {i_simulation} -- Saving figures')
    ############# FCS Figure
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    ax.semilogx(tau, single_spot_acf, 'dk')
    ax.set_ylim(single_spot_acf.min() / 1.2 if single_spot_acf.min() > 0 else single_spot_acf.min() * 1.2,
                single_spot_acf.max() * 1.2 if single_spot_acf.max() > 0 else single_spot_acf.max() / 1.2)
    plt.savefig(save_path + '_single_spot_ACF_ch0.png', 
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
    
    

    
    

    
    
#%% Wrap parameters

print('Planning simulation runs...')
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
        
        save_path = os.path.join(save_folder, save_name)
        
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
                                     save_path))
        
        pointer += 1
        
print('Done planning.')

#%% Run parallel
try:
    # print('Starting parallel pool...')

    # mp_pool = multiprocessing.Pool(processes = (os.cpu_count() - 1 if os.cpu_count() - 1 < len(list_of_param_tuples) else len(list_of_param_tuples)))
    print('Starting simulations...')

    # _ = [mp_pool.starmap(run_single_sim, list_of_param_tuples)]
    
    _ = [run_single_sim(*params_tuple) for params_tuple in list_of_param_tuples]
except:
    traceback.print_exception()
# finally:
    # mp_pool.close()

print('Job done.')
    
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