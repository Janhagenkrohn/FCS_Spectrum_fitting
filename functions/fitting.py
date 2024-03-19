# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""

import numpy as np
import scipy.special as sspecial
import scipy.integrate as sintegrate

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


    def g_3d_diff_single_species(self, 
                                 tau_diff):
        '''
        Normalized 3D diffusion autocorrelation for a single species
        '''
        return self.g_2d_diff_single_species(tau_diff) / np.sqrt(1 + self.tau / (np.square(self.psf_aspect_ratio) * self.data_tau_s))


    def g_2d_diff_single_species(self, 
                                 tau_diff):
        '''
        Normalized 2D diffusion autocorrelation for a single species
        '''
        return 1 / (1 + self.data_tau_s/tau_diff)
    
    
    @staticmethod
    def negloglik_poisson_full(rates, 
                               observations):
        '''
        Poisson penalty in fitting. Full model to allow "observations" 
        themselves to vary in a hierarchical model with a hidden Poisson process
        (if "observations" were fixed, the factorial term could be omitted)
        '''
        return np.sum(np.log(rates ** observations) - rates - np.log(sspecial.factorial(observations)))


    @staticmethod
    def negloglik_poisson_simple(rates, 
                                 observations):
        '''
        Poisson penalty in fitting. Simplified version for fixed observations.
        '''
        return np.sum(np.log(rates ** observations) - rates)

        
    def n_observed_from_N(self, 
                          N, 
                          tau_diff):
        '''
        For converting the particle number parameter in FCS
        to the estiamted total number of observed particles during acquisition
        '''
        return N * self.acquisition_time_s / tau_diff
        
    def N_from_n_observed(self, 
                          n_observed, 
                          tau_diff):
        '''
        Inverse of FCS_spectrum.n_observed_from_N
        '''
        return n_observed * tau_diff / self.acquisition_time_s 
        
    
    @staticmethod
    def pch_3dgauss_1part_int(x,
                              k,
                              cpms_eff):
        '''
        Helper function for implementing the numeric integral in pch_3dgauss_1part
        '''
        return sspecial.gammainc(k, cpms_eff * np.exp(-2*x**2))
    
    
    def pch_3dgauss_1part(self,
                          n_photons_max,
                          Q,
                          t_bin,
                          cpms):
        '''
        Calculates the single-particle compound PCH to be used in subsequent 
        calculation of the "real" multi-particle PCH.

        Parameters
        ----------
        n_photons_max : 
            Int. Highest photon count to consider in prediction.
        Q : 
            Float. Excess volume factor used in defining the PCH reference volume
            relative to standard FCS volume.
        t_bin :
            Float. Bin time in seconds.
        cpms : 
            Float. Molecular brightness in counts per molecule and second.

        Returns
        -------
        pch : 
            np.array with (normalized) PCH for given parameters. 
            Cave: This is a truncated PCH that starts at the bin for 1 photon!

        '''
        # Array with all photon counts from 1-maximum
        photon_counts_array = np.arange(1, n_photons_max+1)
        
        # Simple prefactor...
        prefactor = 1 / Q / np.sqrt(2.) / sspecial.factorial(photon_counts_array)
        
        # The more annoying one: Numeric integration over incomplete gamma 
        # function, for each photon count
        integ_res = np.zeros(photon_counts_array.shape)
        for k in photon_counts_array:
            integ_res[k-1] = sintegrate.quad(lambda x: self.pch_3dgauss_1part_int(x,
                                                                                  k,
                                                                                  t_bin*cpms),
                                             0, np.inf)
            
        # Scipy.special actually implements regularized lower incompl. gamma func,
        # so we need to scale it
        integ_res *= sspecial.gamma(photon_counts_array)
        
        pch = prefactor * integ_res
        return pch
    
    
    def pch_3dgauss_nonideal_1part(self,
                                   F,
                                   n_photons_max,
                                   Q,
                                   t_bin,
                                   cpms):
        '''
        Calculates the single-particle compound PCH to be used in subsequent 
        calculation of the "real" multi-particle PCH. See:
        Huang, Perroud, Zare ChemPhysChem 2004 DOI: 10.1002/cphc.200400176

        Parameters
        ----------
        F :
            Float. Weight for non-Gaussian "out-of-focus" light contribution 
            correction.
        n_photons_max : 
            Int. Highest photon count to consider in prediction.
        Q : 
            Float. Excess volume factor used in defining the PCH reference volume
            relative to standard FCS volume.
        t_bin :
            Float. Bin time in seconds.
        cpms : 
            Float. Molecular brightness in counts per molecule and second.

        Returns
        -------
        pch : 
            np.array with (normalized) PCH for given parameters. 
            Cave: This is a truncated PCH that starts at the bin for 1 photon!

        '''

        pch = self.pch_3dgauss_1part(n_photons_max, Q, t_bin, cpms)
        
        pch /= (1 + F)
        pch[0] += 2**(-3/2) / Q * t_bin * cpms * F
        
        
    @staticmethod
    def negloglik_binomial_simple(n_trials,
                                  k_successes, 
                                  probabilities):
        '''
        Likelihood model for fitting a histogram described by k_successes from
        n_trials observations (where sum(k_successes)==n_trails when summing 
        over all histogram bins = fit data points) with a probability model
        described by probabilities.
        '''
        term1 = -k_successes * np.log(probabilities)
        term2 = -(n_trials - k_successes) * np.log(1 - probabilities)
        return np.sum(term1 + term2)
        
    
    
    