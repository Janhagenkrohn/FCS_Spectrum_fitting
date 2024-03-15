# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""

import numpy as np
from scipy.special import factorial

class FCS_spectrum():

    def __init__(self,
                 data_tau_s,
                 data_correlation,
                 data_sigma,
                 psf_width_nm,
                 psf_aspect_ratio,
                 acquisition_time_s):
         
        # Acquisition metadata
        self.psf_width_nm = psf_width_nm
        self.psf_aspect_ratio = psf_aspect_ratio
        self.acquisition_time_s = acquisition_time_s
        
        # Data to be fitted
        self.data_tau_s = data_tau_s
        self.data_correlation = data_correlation
        self.data_sigma = data_sigma

    def g_3d_diff_single_species(self, tau_diff):
        # Normalized 3D diffusion autocorrelation for a single species
        return self.g_2d_diff_single_species(tau_diff) / np.sqrt(1 + self.tau / (np.square(self.psf_aspect_ratio) * self.data_tau_s))

    def g_2d_diff_single_species(self, tau_diff):
        # Normalized 2D diffusion autocorrelation for a single species
        return 1 / (1 + self.data_tau_s/tau_diff)
    
    @staticmethod
    def negloglik_poisson(rates, observations):
        # Poisson penalty in fitting. Implemented to allow "observations" 
        # themselves to vary in a hierarchical model with a hidden Poisson process
        # (if "observations" were fixed, the factorial term could be omitted)
        return np.sum(np.log(rates ** observations) - rates - np.log(factorial(observations)))
        
    def n_observed_from_N(self, N, tau_diff):
        # For converting the particle number parameter in FCS
        # to the estiamted total number of observed particles during acquisition
        return N * self.acquisition_time_s / tau_diff
        
    def N_from_n_observed(self, n_observed, tau_diff):
        # Inverse of FCS_spectrum.n_observed_from_N
        return n_observed * tau_diff / self.acquisition_time_s 
        
        
    