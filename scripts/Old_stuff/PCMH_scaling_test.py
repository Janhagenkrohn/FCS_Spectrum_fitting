# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:07:20 2024

@author: Krohn
"""

import numpy as np
import scipy.special as sspecial
import matplotlib.pyplot as plt
    
def pch_bin_time_correction(t_bin,
                            cpms,
                            N_avg,
                            tau_diff,
                            tau_blink,
                            beta_blink,
                            F_blink,
                            FCS_psf_aspect_ratio):
    '''
    Key function for generalizing PCH to time-resolved PCMH. Essentially 
    uses diffusion time and blinking parameters from FCS math to correct 
    apparent brightness and particle number parameters for the "blurring"
    induced by particle dynamics during integration time

    Parameters
    ----------
    t_bin : 
        Float. Bin time in seconds.
    cpms : 
        Float or array of float. Molecular brightness in counts per molecule and second.
    N_avg : 
        Float or array of float. <N> as defined in FCS math.
    tau_diff : 
        Float or array of float. Diffusion time in seconds as defined in FCS math.
    tau_blink : 
        Float or array of float. Blinking relaxation time in seconds as defined in FCS math.
    beta_blink : 
        Float or array of float. Stretched-exponential scale parameter for blinking.
    F_blink : 
        Float or array of float. Blinking off-state fraction as defined in FCS math.

    Returns
    -------
    cpms_corr : 
        Float or array of float. Integration-time corrected apparent cpms.
    N_avg_corr : 
        Float or array of float. Integration-time corrected apparent N_avg.

    '''
    
    if F_blink > 0.:
        # Blinking correction is to be done
        tau_blink_avg = tau_blink / beta_blink * sspecial.gamma(1 / beta_blink)
        
        blink_corr = 1 + 2 * F_blink * tau_blink_avg / (t_bin * (1 - F_blink)) * \
            (1 - tau_blink_avg / t_bin * (1 - np.exp(- t_bin / tau_blink_avg)))
            
    else:
        blink_corr = 1.

    temp1 = np.sqrt(FCS_psf_aspect_ratio**2 - 1)
    temp2 = np.sqrt(FCS_psf_aspect_ratio**2 + t_bin / tau_diff)
    temp3 = np.arctanh(temp1 * (temp2 - FCS_psf_aspect_ratio) / (1 + FCS_psf_aspect_ratio * temp2 - FCS_psf_aspect_ratio**2))
    
    diff_corr = 4 * FCS_psf_aspect_ratio * tau_diff / t_bin**2 / temp1 * \
        ((tau_diff + t_bin) * temp3 + tau_diff * temp1 * (FCS_psf_aspect_ratio - temp2)) 
    
    cpms_corr = cpms * blink_corr * diff_corr
    N_avg_corr = N_avg / blink_corr / diff_corr
    
    return cpms_corr, N_avg_corr



bin_times = np.logspace(start = np.log(1e-6),
                   stop = np.log(1e-2),
                   num = 100,
                   base = np.e)

cpms_true = 188700
N_true = 2.255
tau_diff = 5.54e-5
tau_blink = 3e-6
f_blink = 0.0
beta_blink = 1.
FCS_psf_aspect_ratio = 6.2

cpms_app = np.zeros_like(bin_times)
N_app = np.zeros_like(bin_times)


for i_bin_time, bin_time in enumerate(bin_times):
    cpms_app[i_bin_time], N_app[i_bin_time] = pch_bin_time_correction(bin_time,
                                                                        cpms_true,
                                                                        N_true,
                                                                        tau_diff,
                                                                        tau_blink,
                                                                        beta_blink,
                                                                        f_blink,
                                                                        FCS_psf_aspect_ratio)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].semilogx(bin_times,
            cpms_app, 
            marker = '',
            linestyle = '-', 
            color = 'g')
ax[1].semilogx(bin_times,
            N_app, 
            marker = '',
            linestyle = '-', 
            color = 'm')  

        
