# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:17:18 2024

@author: Krohn
"""

# Data manipulation
import numpy as np
from scipy.optimize import minimize_scalar as sminimize_scalar

def g3Ddiff_MEMFCS_fit(FCS_data_tau, 
                       FCS_data_G, 
                       FCS_data_sigma_G, 
                       PSF_aspect_ratio, 
                       tau_D_min = 1e-6, 
                       tau_D_max = 1e-1, 
                       n_tau_D = 100,
                       tau_D_array = np.array([]),
                       initial_amp_array = np.array([]),
                       convergence_criterion_inner = 'parallel_gradients', # 'parallel_gradients' or 'chi-square'
                       max_iter_inner = 2.5E4,
                       max_iter_outer = 30,
                       reg_method = 'MEM', # 'MEM' or 'CONTIN'
                       max_lagrange_mul = 100
                       ):
    
    # Input check/interpretation
    
    # Are tau, G, sd(G) datasets consistent in size?
    if not (FCS_data_tau.shape[0] == FCS_data_G.shape[0] and
            FCS_data_tau.shape[0] == FCS_data_sigma_G.shape[0]):
        raise Exception(f'Inconsistent sizes of FCS_data_tau (size {FCS_data_tau.shape[0]}), FCS_data_G (size {FCS_data_G.shape[0]}), and FCS_data_sigma_G (size {FCS_data_sigma_G.shape[0]})!')
        
    # Number of data points
    n_tau = FCS_data_tau.shape[0]
    
    # Do we already have a pre-iniitalized amplitude array?
    if initial_amp_array.shape[0] > 0:
        n_tau_D = initial_amp_array.shape[0]
        
    # Pre-defined array of tau_D values?
    if tau_D_array.shape[0] > 0:
        # Yes, we have that. Does it fit to initial_amp_array (if we have that)?
        if initial_amp_array.shape[0] > 0 and tau_D_array.shape[0] != initial_amp_array.shape[0]:
            raise Exception(f'tau_D_array (size {tau_D_array.shape[0]}) input does not match initial_amp_array (size {initial_amp_array.shape[0]})!')
        
        elif initial_amp_array.shape[0] == 0:
            # No initial_amp_array, then tau_D_array overwrites n_tau_D
            n_tau_D = tau_D_array.shape[0] 
            
    else: # tau_D_array.shape[0] == 0:
        # No user-supplied tau_D_array, so we auto-generate it
        tau_D_array = np.logspace(start = np.log10(tau_D_min),
                                  stop = np.log10(tau_D_max),
                                  num = n_tau_D) # evenly spaced tau_D in the log scale base = 10

    # Declare amplitude array for fitting
    if initial_amp_array.shape[0] == 0:
        # No user-supplied amp_array
        amp_array = np.ones(n_tau_D) / n_tau_D # setting up the flat distribution
    else:
        # We have a user-supplied amp array, use that
        amp_array = initial_amp_array
        
    # Sanity-check convergence_criterion_inner: Convergence criterion for inner iteration
    if not convergence_criterion_inner in ['chi-square',  'parallel_gradients']:
        raise Exception(f'Invalid convergence_criterion_inner: Must be "chi-square" or "parallel_gradients", got {convergence_criterion_inner}')

    # Ensure integer iteration count limits
    max_iter_inner = int(max_iter_inner)
    max_iter_outer = int(max_iter_outer)

    # Declare various variables...
    
    # Now, each row will correspond to a different tau_D value in the evenly 
    # distributed log space, and the columns are different values of lag time 
    tauD_tau_array = np.zeros((n_tau_D, n_tau))
    for i_tau in range(n_tau_D):
        tauD_tau_array[i_tau,:] = 1/((1 + FCS_data_tau/tau_D_array[i_tau]) * np.sqrt(1 + FCS_data_tau/(tau_D_array[i_tau]*PSF_aspect_ratio**2)))
        
    # But we actually also need the transpose of that array for some matrix 
    # manipulation, which we also pre-calculate to avoid shuffling stuff around too much
    tau_tauD_array = np.transpose(tauD_tau_array)
    
    # iteration settings
        
    # Gradient scaling factor for (inner) optimization
    gradient_scaling = 2e-4
    convergence_threshold_chi_sq = 5e-6
    convergence_threshold_par_grad = 1e-1
        
    # Initial value of Lagrange multiplier
    lagrange_mul = 20
    lagrange_mul_array = np.zeros(max_iter_outer)
    chi_sq_array_outer = np.zeros(max_iter_outer)
    lagrange_mul_del_old = 1
    iterator_outer = 1
    while True:
        print(f'Iteration {iterator_outer}: lagrange_mul = {lagrange_mul}')

        chi_sq_array_inner = np.zeros(max_iter_inner)
        
        iterator_inner = 0
        while True:
            
            # Get kinetic model
            G_fit_normalized = np.dot(tau_tauD_array, 
                                      amp_array) # ndarray wrt lag time (sum over all diffusion time components)
            
            # Amplitude is locally optimized
            G0 = sminimize_scalar(lambda G0: np.sum(((G_fit_normalized * G0 - FCS_data_G) / FCS_data_sigma_G)**2)).x
            G_fit = G_fit_normalized * G0 
            
            # Goodness-of-fit statistics
            weighted_residual = (G_fit - FCS_data_G) / FCS_data_sigma_G
            chi_sq_array_inner[iterator_inner] = np.mean(weighted_residual**2)
            
            # Calculation of regularization terms
            # These actually used only implicitly, so we only show them for reference
            # if _reg_method:
            #     # Entropy (to be maximized)
            #     S = -np.sum(amp_array * np.log(amp_array))
            # else:
            #     # CONTIN -> Magnitude of second derivative (to be minimized)
            #     # One caveat here: CONTIN regularization, differnet from MEM, 
            #     # requires consideration of local neighborhood around each 
            #     # data point. To ensure that this is defined for data points 
            #     # at the edges, we implicitly assume a periodicity in amp_array.
            #     S = np.sum((2 * amp_array - np.roll(amp_array, -1) - np.roll(amp_array, +1))**2)
               
            # Gradients
            # First order derivative of least-squares
            chi_sq_gradient = np.mean(2 * weighted_residual * tauD_tau_array / FCS_data_sigma_G, 
                                      axis = 1)
            chi_sq_length = np.sqrt(np.sum(chi_sq_gradient**2))
            
            # first order derivative of entropy/CONTIN derivative
            # Sanity-check reg_method and redefine to quicker-evaluated bool: 
            if reg_method == 'MEM':
                S_gradient = - 1 - np.log(amp_array)
            elif reg_method == 'CONTIN':
                S_gradient = - (12 * amp_array 
                               - 8 * (np.roll(amp_array, -1) + np.roll(amp_array, +1))
                               + 2 * (np.roll(amp_array, -2) - np.roll(amp_array, +2))
                               )
            elif reg_method == 'hybrid':
                # Average of MEM and CONTIN, I want to see what happens...
                S_gradient = - (6 * amp_array 
                               - 4 * (np.roll(amp_array, -1) + np.roll(amp_array, +1))
                               + 4 * (np.roll(amp_array, -2) - np.roll(amp_array, +2))
                               ) - 0.5 - 0.5 * np.log(amp_array)


            S_length = np.sqrt(np.sum(S_gradient**2))

            # Once in a while, check for convergence 
            if (iterator_inner + 1) % 1000 == 0 and iterator_inner > 1:
                
                if (iterator_inner + 1) >= max_iter_inner:
                    # Iteration limit hit
                    print(f'Stopping inner loop after {iterator_inner + 1} iterations (iteration limit), current chi-square: {chi_sq_array_inner[iterator_inner]}')
                    break
                
                elif convergence_criterion_inner == 'chi-square':
                    # Check for levelling-off of chi-square changes
                    
                    old_chi_sq = np.sum(chi_sq_array_inner[iterator_inner-200:iterator_inner-100])
                    recent_chi_sq = np.sum(chi_sq_array_inner[iterator_inner-100:iterator_inner])
        
                    if np.abs((old_chi_sq - recent_chi_sq)/recent_chi_sq) < convergence_threshold_chi_sq:
                        # Negligible chi-square change - stop inner loop
                        print(f'Stopping inner loop after {iterator_inner + 1} iterations (converged), current chi-square: {chi_sq_array_inner[iterator_inner]}')
                        break
                    
                else: # convergence_criterion_inner == 'parallel_gradients'
                    # Check if gradients for S and chi-square are parallel, which they
                    # should be at the optimal point for a given Lagrange multiplier
                    
                    S_direction = S_gradient / S_length                        
                    chi_sq_direction = chi_sq_gradient / chi_sq_length
                    test_stat = 0.5 * np.sum((S_direction - chi_sq_direction)**2)

                    if test_stat < convergence_threshold_par_grad: 
                        # gradients approximately parallel - stop inner loop
                        print(f'Stopping inner loop after {iterator_inner + 1} iterations (converged), current chi-square: {chi_sq_array_inner[iterator_inner]}')
                        break

            # We continue - In that case update amp_array
            
            # Scaling factor from Euclician norms of gradients
            alpha_f = chi_sq_length / S_length / lagrange_mul
            
            # search direction construct
            e_G = alpha_f * S_gradient - chi_sq_gradient / 2 # del Q
    
            # update amp_array
            amp_array += amp_array * e_G * gradient_scaling
            
            # Enforce positive-nonzero values, and renormalize
            amp_array_nonzeros = amp_array > 0
            n_amp_array_nonzeros = np.sum(amp_array_nonzeros)
            amp_array = np.where(amp_array_nonzeros,
                                 amp_array,
                                 amp_array[amp_array_nonzeros].min() if n_amp_array_nonzeros > 0 else 1E-6)
            amp_array /= amp_array.sum()
            
            # Iterate
            iterator_inner +=1 
            
        # Inner iteration stopped - check for globally optimal solution
        # We check if chi-square is not within tolerance region

        if iterator_outer >= max_iter_outer:
            # Iteration limit hit 
            print(f'Stopping outer loop after {iterator_outer} iterations (iteration limit)')
            break
        
        chi_sq_array_outer[iterator_outer] = chi_sq_array_inner[iterator_inner]
        if chi_sq_array_outer[iterator_outer] < 1 or chi_sq_array_outer[iterator_outer] > 1.3 or iterator_outer < 3: 
            # Over- or underfit: Update Lagrange multiplier!
            # The way it is defined, high lagrange_mul leads to high weight for 
            # chi-square gradients. 
                # So, we reduce lagrange_mul if we had on overfit (too-low chi-sq),
                # and we increase lagrange_mul if we had an underfit
            # For this, we use the ratio S_length / chi_sq_length:
                # High S_length/chi_sq_length implies that we had too much weight 
                # on chi-square (too-high lagrange_mul), and vice versa, so
                # we change lagrange_mul based on that ratio (nonlinearly, and 
                # witha bit of a gradient boost)
                
            # Try to equalize gradients - converges for MEM, but not necessarily on good chi-sq
            # lagrange_mul_del_new = np.sqrt(chi_sq_length/S_length)
            
            # Combined update that tries to move chi-square towards 1.3 and equalize the gradient lengths
            # lagrange_mul_del_new = np.sqrt(chi_sq_array_outer[iterator_outer] / 1.3) * (chi_sq_length/S_length)**(-3)
            
            # 
            lagrange_mul_del_new = np.sqrt(chi_sq_array_outer[iterator_outer] / 1.3) * (chi_sq_length/S_length)**(1/3)
            lagrange_mul *= lagrange_mul_del_new * lagrange_mul_del_old**(1/3)
            
            # Bounds for lagrange_mul
            if lagrange_mul > max_lagrange_mul:
                lagrange_mul = max_lagrange_mul
            elif lagrange_mul < 1/max_lagrange_mul:
                lagrange_mul = 1/max_lagrange_mul
                
            lagrange_mul_array[iterator_outer] = lagrange_mul
            
            # Another stop criterion comes in here: If neither lagrange_mul nor
            # chi-square have really changed for three iterations
            if iterator_outer >= 3:
                
                lagrange_mul_recent = lagrange_mul_array[iterator_outer-2:iterator_outer+1]
                lagrange_mul_recent_rel_span = (np.max(lagrange_mul_recent) - np.min(lagrange_mul_recent)) / np.mean(lagrange_mul_recent)

                chi_sq_recent = chi_sq_array_outer[iterator_outer-2:iterator_outer+1]
                chi_sq_recent_rel_span = (np.max(chi_sq_recent) - np.min(chi_sq_recent)) / np.mean(chi_sq_recent)
                
                if lagrange_mul_recent_rel_span < 0.01 and chi_sq_recent_rel_span < 0.001:
                    print(f'Stopping outer loop after {iterator_outer} iterations (no longer changing)')
                    break
            
            lagrange_mul_del_old = np.copy(lagrange_mul_del_new)
            iterator_outer += 1
        
        else:
            # Convergence criterion hit - stop outer loop
            print(f'Stopping outer loop after {iterator_outer} iterations (converged)')
            break
    
    return G_fit, amp_array, tau_D_array, G0

def n_total_from_N_avg(N_avg, 
                       tau_diff,
                       acquisition_time_s):
    '''
    For converting the particle number parameter N_avg in FCS/PCH
    to the estiamted total number of observed particles during acquisition
    '''
    
    n_total = N_avg * acquisition_time_s / tau_diff
    
    return n_total


def get_tau_diff_array(tau_diff_min,
                       tau_diff_max,
                       n_species
                       ):
            
    return np.logspace(start = np.log(tau_diff_min),
                       stop = np.log(tau_diff_max),
                       num = n_species,
                       base = np.e)

def single_filament_tau_diff_fold_change(j):
    return 2. * j / (2. * np.log(j) + 0.632 + 1.165 * j ** (-1.) + 0.1 * j ** (-2.))


def double_filament_tau_diff_fold_change(j):
    axial_ratio = j / 2.
    return 2. * axial_ratio / (2. * np.log(axial_ratio) + 0.632 + 1.165 * axial_ratio ** (-1.) + 0.1 * axial_ratio ** (-2.))


def single_filament_tau_diff_fold_change_deviation(log_j,
                                                   tau_diff_fold_change):
    return (single_filament_tau_diff_fold_change(np.exp(log_j)) - tau_diff_fold_change) ** 2


def double_filament_tau_diff_fold_change_deviation(log_j,
                                                   tau_diff_fold_change):
    return (double_filament_tau_diff_fold_change(np.exp(log_j)) - tau_diff_fold_change) ** 2



def stoichiometry_from_tau_diff_array(tau_diff_array,
                                      oligomer_type
                                      ):
    
    # The monomer by definition has stoichiometry 1, so we start with the second element
    fold_changes = tau_diff_array[1:] / tau_diff_array[0]
    stoichiometry = np.ones_like(tau_diff_array)
    
    if oligomer_type in ['sherical_dense']:
        # tau_diff proportional hydrodyn. radius
        # Stoichiometry proportional volume
        stoichiometry[1:] = fold_changes ** 3
    
    elif oligomer_type == 'spherical_shell':
        # tau_diff proportional hydrodyn. radius
        # Stoichiometry proportional surface area
        stoichiometry[1:] = fold_changes ** 2

    elif oligomer_type == 'single_filament':
        # For the filament models, we have more complicated expressions based
        # on Seils & Pecora 1995. We numerically solve the expression,
        # which cannot be decently inverted. 
        
        for i_spec, tau_diff_fold_change in enumerate(fold_changes[:-1]):
            res = sminimize_scalar(fun = single_filament_tau_diff_fold_change_deviation, 
                                   args = (tau_diff_fold_change,))
            stoichiometry[i_spec + 1] = np.exp(res.x)
    
    elif oligomer_type == 'double_filament':
    
        stoichiometry = np.zeros_like(fold_changes)
        
        for i_spec, tau_diff_fold_change in enumerate(fold_changes):

            res = sminimize_scalar(fun = double_filament_tau_diff_fold_change_deviation, 
                                   args = (tau_diff_fold_change,))
            stoichiometry[i_spec + 1] = np.exp(res.x)
    else: # oligomer_type == naive
        # Dummy ones
        stoichiometry = np.arange(1, tau_diff_array.shape[0] + 1)
        stoichiometry_binwidth = 1. / stoichiometry
    
    if not oligomer_type == 'naive':
        stoichiometry, indices = np.unique(np.round(stoichiometry),
                                            return_index = True)
        tau_diff_array = tau_diff_array[indices]
    
        # Also get approximate bin widths of log-spaced distribution
        log_stoichiometry = np.log(stoichiometry)
        log_binwidth = np.mean(np.diff(log_stoichiometry))
        stoichiometry_binwidth = (np.exp(log_stoichiometry + log_binwidth/2) - np.exp(log_stoichiometry - log_binwidth/2))
        
    return stoichiometry, tau_diff_array, stoichiometry_binwidth



#%% Define data

# I/O
import os
import sys

# Custom
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils

# Plotting
import matplotlib.pyplot as plt



dir_names = []
in_file_names_FCS = []

# Low labelling batch without burst removal
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k1_T0s_1_20240613_1016'])
dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k2_T92s_1_20240613_1010'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k3_T184s_1_20240613_1025'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k4_T275s_1_20240613_1025'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k5_T367s_1_20240613_1010'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k6_T459s_1_20240613_1017'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k7_T551s_1_20240613_1010'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k8_T642s_1_20240613_1016'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k9_T734s_1_20240613_1016'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k10_T826s_1_20240613_1010'])
in_file_names_FCS.extend(['06_ACF_ch0_dt_bg'])
# in_file_names_FCS.extend(['06_ACF_ch0_dt_bg']*10)


# # Low labelling batch with burst removal
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k1_T0s_1_20240613_1700'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k2_T92s_1_20240613_1654'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k3_T184s_1_20240613_1708'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k4_T275s_1_20240613_1707'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k5_T367s_1_20240613_1654'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k6_T459s_1_20240613_1701'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k7_T551s_1_20240613_1654'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k8_T642s_1_20240613_1700'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k9_T734s_1_20240613_1700'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_2\ParM_10uM_1in10k10_T826s_1_20240613_1654'])
# in_file_names_FCS.extend(['08_ACF_ch0_br_dt_bg']*10)

# # High labelling batch without burst removal
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in401_T0s_1_20240613_1435'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in402_T92s_1_20240613_1116'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in403_T184s_1_20240613_1324'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in404_T276s_1_20240613_1031'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in405_T367s_1_20240613_1032'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in406_T459s_1_20240613_1233'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in407_T551s_1_20240613_1112'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in408_T643s_1_20240613_1236'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in409_T735s_1_20240613_1318'])
# dir_names.extend([r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\D044_MT200_Naora\20240416_JHK_NK_New_ParM_data\20240416_data.sptw\10uM_ParM_3\ParM_10uM_1in4010_T826s_1_20240613_1513'])
# in_file_names_FCS.extend(['06_ACF_ch0_dt_bg']*10)


# # High labelling batch with burst removal not ready yet

acquisition_time= 90.

n_fits = len(dir_names)

PSF_aspect_ratio = 5
tau_D_min = 1E-5 
tau_D_max = 1e1
n_species = 120
tau_D_array = np.array([])
initial_amp_array = np.array([])
convergence_criterion_inner = 'parallel_gradients' # 'parallel_gradients', 
max_iter_inner = 2.5E4
max_iter_outer = 10
reg_method = 'CONTIN' #'MEM', 'CONTIN', 'hybrid'
max_lagrange_mul = 1e6


for i_fit, dir_name in enumerate(dir_names):
    
    data_FCS_tau_s, data_FCS_G, avg_count_rate, data_FCS_sigma, _ = utils.read_Kristine_FCS(dir_name, 
                                                                                            in_file_names_FCS[i_fit],
                                                                                            1e-6,
                                                                                            1.)   
    
    
    G_fit, amp_array, tau_D_array, G0 = g3Ddiff_MEMFCS_fit(data_FCS_tau_s, 
                                                           data_FCS_G, 
                                                           data_FCS_sigma, 
                                                           PSF_aspect_ratio = PSF_aspect_ratio, 
                                                           tau_D_min = tau_D_min, 
                                                           tau_D_max = tau_D_max , 
                                                           n_tau_D = n_species,
                                                           tau_D_array = tau_D_array ,
                                                           initial_amp_array = initial_amp_array,
                                                           convergence_criterion_inner = convergence_criterion_inner, # 'parallel_gradients', 
                                                           max_iter_inner = max_iter_inner,
                                                           max_iter_outer = max_iter_outer,
                                                           reg_method = reg_method,# 'MEM', 'CONTIN'
                                                           max_lagrange_mul = max_lagrange_mul
                                                           )
    
    
    
    if i_fit == 0:
        all_data_tau = np.zeros((data_FCS_tau_s.shape[0], n_fits))
        all_data_G = np.zeros((data_FCS_tau_s.shape[0], n_fits))
        all_data_G_fit = np.zeros((data_FCS_tau_s.shape[0], n_fits))
        all_data_amp_array = np.zeros((amp_array.shape[0], n_fits))
        all_data_tau_D_array = np.zeros((tau_D_array.shape[0], n_fits))
        all_data_avg_count_rate = np.zeros((n_fits))

    all_data_tau[:,i_fit] = data_FCS_sigma
    all_data_G[:,i_fit] = data_FCS_G
    all_data_G_fit[:,i_fit] = G_fit
    all_data_amp_array[:,i_fit] = amp_array
    all_data_tau_D_array[:,i_fit] = tau_D_array
    all_data_avg_count_rate[i_fit] = avg_count_rate





    fig, ax = plt.subplots(nrows=2, 
                           ncols=1,
                           sharex=True,
                           sharey=False)
    ax[0].semilogx(data_FCS_tau_s,
                   data_FCS_G, 
                   'dk')
    ax[0].semilogx(data_FCS_tau_s, 
                   data_FCS_G + data_FCS_sigma,
                   '-k', 
                   alpha = 0.7)
    ax[0].semilogx(data_FCS_tau_s, 
                   data_FCS_G - data_FCS_sigma,
                   '-k', 
                   alpha = 0.7)
    ax[0].semilogx(data_FCS_tau_s,
                   G_fit, 
                   marker = '',
                   linestyle = '-', 
                   color = 'tab:gray')  
    ax[0].set_xlim(np.min([data_FCS_tau_s[0], tau_D_array[0]]), np.max([data_FCS_tau_s[-1], tau_D_array[-1]]))
    plot_y_min_max = (np.percentile(data_FCS_G, 3), np.percentile(data_FCS_G, 97))
    ax[0].set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)

    ax[1].semilogx(tau_D_array,
                   amp_array, 
                   '-dk')
    ax[1].set_ylim(0.,
                   np.max(amp_array) * 1.2 if np.max(amp_array) < 0.8 else 1)
    fig.supxlabel('Correlation time [s]')
    fig.supylabel('G(\u03C4)')

    # Show figure
    plt.show()
    
    #TODO: Save spreadsheet
    
