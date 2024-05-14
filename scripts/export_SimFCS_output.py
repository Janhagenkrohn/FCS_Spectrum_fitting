# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:38:58 2024

@author: Krohn
"""

# I/O
import os
import sys
import pandas as pd

# Data processing and plotting
import numpy as np
from multipletau import autocorrelate
import matplotlib.pyplot as plt
from itertools import cycle # used only in plotting

# Custom module
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils



in_dir_names= []
glob_dir = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\simFCS2_simulations'
sim_time_resolution = [] # Simulation time step in seconds - differs between sims sometimes, so we expand it

''' Which batch(es)'''
# in_dir_names.extend([os.path.join(glob_dir, r'3a\raw_data')])
# sim_time_resolution = 1E-6 # 1 us 

# in_dir_names.extend([os.path.join(glob_dir, r'3b\slow')])
# sim_time_resolution = 1E-5 # 10 us 

# in_dir_names.extend([os.path.join(glob_dir, r'3b\fast')])
# sim_time_resolution = 1E-6 # 1 us 

# in_dir_names.extend([os.path.join(glob_dir, r'3c')])
# sim_time_resolution = 1E-6 # 1 us 

in_dir_names.extend([os.path.join(glob_dir, '3d')])
sim_time_resolution = 1E-6 # 1 us 


# Naming pattern for detecting correct files within subdirs of each in_dir
file_name_pattern = '*batch*'



sd_method = 'bootstrap'


FCS_tau_min = 1E-6
FCS_tau_max = 1.



#%% Input file search

# Detect PCH files
in_dir_names, in_file_names, _ = utils.detect_files(in_dir_names,
                                                    file_name_pattern, 
                                                    ['' for in_dir in in_dir_names], 
                                                    '',
                                                    file_type_suffix = '.bin')

for i_file, in_dir_name in enumerate(in_dir_names):
    
    # Unique for today: Skip some files to continue where the code broke
    if i_file < 103: continue
    
    #%% Load data
    in_file_name = in_file_names[i_file] + '.bin'
    path = os.path.join(in_dir_name, in_file_name)
    
    data = np.uint16(np.fromfile(path,
                                 np.int16))
    n_photons = data.sum()
    n_bins = data.shape[0]
    sim_duration = n_bins * sim_time_resolution
    average_count_rate = n_photons / sim_duration
    
    # Prepare output dirs as needed
    # Here, in_file_name without its file type extension becomes a subdir
    save_path = os.path.join(in_dir_name, in_file_names[i_file])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    #%% FCS Export
    # Get autocorrelation
    res = autocorrelate(data, 
                        m = 8, 
                        normalize = True,
                        dtype=np.float_)
    tau = res[:,0]
    tau *= sim_time_resolution
    acf = res[:,1]
    
    tau_mask = np.logical_and(tau > FCS_tau_min,
                              tau < FCS_tau_max)
    
    FCS_lag_times = tau[tau_mask]
    FCS_G = acf[tau_mask]
    
    # Get uncertainty
    
    if sd_method == 'wohland':
        segment_length = np.floor(n_bins / 10)
        
        for i_segment in range(10):
            # Crop data
            data_crop = data[i_segment*segment_length : (i_segment+1)*segment_length]
    
            # Get segment correlation, store
            res = autocorrelate(data_crop, 
                                m = 8, 
                                normalize = True,
                                dtype=np.float_)        
            segment_tau = res[:,0]
            segment_acf = res[:,1]
                    
            if i_segment == 0:
                segment_acfs = np.zeros((segment_acf.shape[0], 10))
            
            segment_acfs[:, i_segment] = segment_acf
        
        # Get uncertainty
        tau_mask = np.logical_and(segment_tau > FCS_tau_min / sim_time_resolution,
                                  segment_tau < FCS_tau_max / sim_time_resolution)
        FCS_sigma = np.sqrt(np.var(segment_acfs[tau_mask, :], 
                                   axis = 1) / 9)
        
        
    elif sd_method == 'bootstrap':
        # Prepare data for bootstrapping
        sparse_data = utils.dense_to_sparse(data,
                                            dtype = np.uint64)
        rng = np.random.default_rng()
    
        for i_bs_rep in range(10):
            # Resample photon data
            resample_indxs = rng.choice(n_photons, 
                                        size = n_photons)
            resample_indxs_sort = np.sort(resample_indxs) 
            
            # Reconstruct bootstrapped trace
            data_resample = utils.sparse_to_dense(sparse_data[resample_indxs_sort])
            
            # Get bootstrap correlation, store
            res = autocorrelate(data_resample, 
                                m = 8, 
                                normalize = True,
                                dtype=np.float_)        
            bs_tau = res[:,0]
            bs_acf = res[:,1]
                    
            if i_bs_rep == 0:
                bs_acfs = np.zeros((bs_acf.shape[0], 10))
            
            bs_acfs[:, i_bs_rep] = bs_acf
    
        
        # Get uncertainty
        tau_mask = np.logical_and(bs_tau > FCS_tau_min / sim_time_resolution,
                                  bs_tau < FCS_tau_max / sim_time_resolution)
        FCS_sigma = np.std(bs_acfs[tau_mask, :], 
                           axis = 1)
        
    else: # sd_method not 'wohland' or 'bootstrap'
        raise Exception('Invalid SD calculation method')
        
        
    # Export Kristine csv
    acr_col = np.zeros_like(FCS_lag_times)
    acr_col[:3] = np.array([average_count_rate, average_count_rate, sim_duration])
    out_table = pd.DataFrame(data = {'Lagtime[s]':FCS_lag_times, 
                                     'Correlation': FCS_G,
                                     'ACR[Hz]': acr_col,
                                     'Uncertainty_SD': FCS_sigma})
    out_table.to_csv(os.path.join(save_path, in_file_name + '_ACF_ch0.csv'),
                     index = False, 
                     header = False)
    
    
    #%% FCS Figure
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    ax.semilogx(FCS_lag_times, FCS_G, 'dk')
    ax.semilogx(FCS_lag_times, FCS_G + FCS_sigma, '-k', alpha = 0.7)
    ax.semilogx(FCS_lag_times, FCS_G - FCS_sigma, '-k', alpha = 0.7)
    plot_y_min_max = (np.percentile(FCS_G, 3), np.percentile(FCS_G, 97))
    ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)
    plt.savefig(os.path.join(save_path, in_file_name + '_ACF_ch0.png'), 
                dpi=300)
    plt.close()
    
    
    #%% Calculate PCMH
    # We use all bin widths starting at 2 sim time bins, going up in factors of 
    # "spacing" until the last one where we get 10000 bins out of the data
    max_bin_time = sim_duration / 1E4
    spacing = np.sqrt(2.)
    
    bin_times = [sim_time_resolution * 2]
    next_bin_time = sim_time_resolution * 2 * spacing
    while next_bin_time < max_bin_time:
        bin_times.append(next_bin_time)
        next_bin_time *= spacing
    bin_times = np.array(bin_times)
    
    sparse_data = utils.dense_to_sparse(data,
                                        dtype = np.uint64)
    
    # Rebin time trace at lowest resolution
    # This gives us the dimensions of the required PCMH array
    time_trace_bins = np.arange(0, 
                                n_bins + 1, 
                                bin_times[-1] / sim_time_resolution, 
                                dtype = float)
    time_trace = np.histogram(sparse_data,
                              bins = time_trace_bins,
                              density = False)[0]
    photon_count_maximum = np.max(time_trace)
    
    # Get actual PCH
    pch = np.histogram(time_trace,
                       bins = np.arange(0, photon_count_maximum + 1),
                       density = False)[0]
    
    photon_count_variance = np.var(time_trace)
    photon_count_mean = np.mean(time_trace)
    Mandel_Q = photon_count_variance / photon_count_mean - 1
    
    # Initialize PCMH arrays and write
    pcmh = np.zeros((pch.shape[0], len(bin_times)))
    pcmh[:,-1] = pch
    Mandel_Q_series = np.zeros(len(bin_times))
    Mandel_Q_series[-1] = Mandel_Q
    
    # Other bin times
    for i_bin_time, bin_time in enumerate(bin_times[:-1]):
        # Rebin time trace at desired resolution
        time_trace_bins = np.arange(0, 
                                    n_bins + 1, 
                                    bin_time / sim_time_resolution, 
                                    dtype = float)
        time_trace = np.histogram(sparse_data,
                                  bins = time_trace_bins,
                                  density = False)[0]
        photon_count_maximum = np.max(time_trace)
        
        # Get actual PCH
        pch = np.histogram(time_trace,
                           bins = np.arange(0, photon_count_maximum + 1),
                           density = False)[0]
        
        photon_count_variance = np.var(time_trace)
        photon_count_mean = np.mean(time_trace)
        Mandel_Q = photon_count_variance / photon_count_mean - 1
    
        if pch.shape[0] > pcmh.shape[0]:
            # The super-rare exception where a smaller bin width reaches the 
            # highest number of photons in a bin due to frame shifts in
            # binning. It can happen, did crash my code once...
            pcmh = np.append(pcmh, 
                             np.zeros((pch.shape[0]-pcmh.shape[0], len(bin_times))),
                             axis = 0)
        pcmh[0:pch.shape[0], i_bin_time] = pch

        Mandel_Q_series[i_bin_time] = Mandel_Q
    
    # Create and write spreadsheets
    out_table = pd.DataFrame(data = {str(bin_time): pcmh[:,i_bin_time] for i_bin_time, bin_time in enumerate(bin_times)})
    
    
    out_table.to_csv(os.path.join(save_path, in_file_name + '_PCMH_ch0.csv'), 
                     index = False, 
                     header = True)
    
    out_table = pd.DataFrame(data = {'Bin Times [s]': bin_times,
                                     'Mandel Q': Mandel_Q_series})
    out_table.to_csv(os.path.join(save_path, in_file_name + '_Mandel_Q_ch0.csv'), 
                     index = False, 
                     header = True)
    
    #%% Create and write PCMH figure
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex = False)
    pch_x = np.arange(0, pcmh.shape[0])
    
    # Left panel: PCMH
    # Cycle through colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    
    for i_pch in range(pcmh.shape[1]):
        iter_color = next(colors)
        # Plot is nicer/easier to compare with normalized PCMH
        pch_norm = pcmh[:,i_pch] / np.sum(pcmh[:,i_pch])
        ax[0].semilogy(pch_x, 
                       pch_norm,
                       marker = '', 
                       linestyle = '-', 
                       alpha = 0.7,
                       color = iter_color)
    ax[0].set_title('Norm. PCH over bin time')
    ax[0].set_ylim(np.min(pch_norm[pch_norm>0]), 1.1)
    
    # Right panel: Mandel's Q
    ax[1].plot(bin_times,
                Mandel_Q_series, 
                marker = 'o',
                linestyle = '-',
                color = 'k')
    ax[1].set_title("Mandel's Q")
    
    
    plt.savefig(os.path.join(save_path, in_file_name + '_PCMH_ch0.png'), 
                dpi=300)
    plt.close()
