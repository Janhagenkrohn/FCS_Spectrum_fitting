# -*- coding: utf-8 -*-
"""
Created on Fri 15 March 2024

@author: Krohn
"""

# File I/O and path manipulation
import sys
import os

# Data manipulation
import numpy as np
import scipy.special as sspecial
import scipy.integrate as sintegrate
import scipy.stats as sstats
import lmfit

# Plotting
from matplotlib import pyplot as plt

# Custom module
# For localizing module
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils


class FCS_spectrum():

    def __init__(self,
                 FCS_psf_width_nm = None,
                 FCS_psf_aspect_ratio = None,
                 PCH_Q = None,
                 acquisition_time_s = None,
                 data_FCS_tau_s = None,
                 data_FCS_G = None,
                 data_FCS_sigma = None,
                 data_PCH_bin_times = None,
                 data_PCH_hist = None,
                 ):
         
        # Acquisition metadata
        
        if utils.isfloat(FCS_psf_width_nm):
            if FCS_psf_width_nm > 0:
                self.FCS_psf_width_nm = FCS_psf_width_nm
                self.FCS_possible = True
            else:
                raise ValueError('FCS_psf_width_nm must be float > 0')
        elif utils.isempty(FCS_psf_width_nm):
            self.psf_width = None
            self.FCS_possible = False
        else:
            raise ValueError('FCS_psf_width_nm must be float > 0')


        if utils.isfloat(FCS_psf_aspect_ratio):
            if FCS_psf_aspect_ratio > 0:
                self.FCS_psf_aspect_ratio = FCS_psf_aspect_ratio
                self.FCS_possible = True
            else:
                raise ValueError('FCS_psf_aspect_ratio must be float > 0')
        elif utils.isempty(FCS_psf_aspect_ratio):
            self.FCS_psf_aspect_ratio = None
            self.FCS_possible = False
        else:
            raise ValueError('FCS_psf_aspect_ratio must be float > 0')


        if utils.isfloat(PCH_Q):
            if PCH_Q > 0:
                self.PCH_Q = PCH_Q
                self.PCH_possible = True
            else:
                raise ValueError('FCS_psf_aspect_ratio must be float > 0')
        elif utils.isempty(PCH_Q):
            self.PCH_Q = None
            self.PCH_possible = False
        else:
            raise ValueError('FCS_psf_aspect_ratio must be float > 0')


        if utils.isfloat(acquisition_time_s):
            if acquisition_time_s > 0:
                self.acquisition_time_s = acquisition_time_s
                self.incomplete_sampling_possible = True
            else: 
                raise ValueError('acquisition_time_s must be empty value or float > 0')
        elif utils.isempty(acquisition_time_s):
            self.acquisition_time_s = None
            self.incomplete_sampling_possible = False
        else:
            raise ValueError('acquisition_time_s must be empty value or float > 0')
    
    
        # FCS input data to be fitted
        if utils.isempty(data_FCS_tau_s) or not self.FCS_possible:
            self.data_FCS_tau_s = None  
            self.FCS_possible = False
        elif utils.isiterable(data_FCS_tau_s):
            self.data_FCS_tau_s = np.array(data_FCS_tau_s)
            self.FCS_possible = True
        else:
            raise ValueError('data_FCS_tau_s must be array (or can be left empty for PCH only)')
            
            
        if utils.isempty(data_FCS_G) or not self.FCS_possible:
            self.data_FCS_G = None
            self.FCS_possible = False
        elif utils.isiterable(data_FCS_G):
            data_FCS_G = np.array(data_FCS_G)
            if data_FCS_G.shape[0] == data_FCS_tau_s.shape[0]:
                self.data_FCS_G = data_FCS_G
            else:
                raise ValueError('data_FCS_G must be array of same length as data_FCS_tau_s (or can be left empty for PCH only)')
        else:
            raise ValueError('data_FCS_G must be array of same length as data_FCS_tau_s (or can be left empty for PCH only)')


        if utils.isempty(data_FCS_sigma) or not self.FCS_possible:
            self.data_FCS_sigma = None
            self.FCS_possible = False
        if utils.isiterable(data_FCS_sigma):
            data_FCS_sigma = np.array(data_FCS_sigma)
            if data_FCS_sigma.shape[0] == data_FCS_tau_s.shape[0]:
                self.data_FCS_sigma = data_FCS_sigma
            else:
                raise ValueError('data_FCS_sigma must be array of same length as data_FCS_tau_s (or can be left empty for PCH only)')
        else:
            raise ValueError('data_FCS_sigma must be array of same length as data_FCS_tau_s (or can be left empty for PCH only)')


        # PC(M)H input data to be fitted
        if utils.isempty(data_PCH_bin_times) or not self.PCH_possible:
            self.data_PCH_bin_times = None  
            self.PCH_possible = False
        elif utils.isiterable(data_PCH_bin_times) or utils.isfloat(data_PCH_bin_times):
            self.data_PCH_bin_times = np.array(data_PCH_bin_times)
            self.PCH_possible = True
        else:
            raise ValueError('data_PCH_bin_times must be float or array (or can be left empty for FCS only)')


        if utils.isempty(data_PCH_hist) or not self.PCH_possible:
            self.data_PCH_hist = None
            self.PCH_possible = False
        elif utils.isiterable(data_PCH_hist):
            data_PCH_hist = np.array(data_PCH_hist)
            if data_PCH_hist.shape[1] == data_PCH_bin_times.shape[0]:
                self.data_PCH_hist = data_PCH_hist
            else:
                raise ValueError('data_PCH_hist must be array with axis 1 same length as same length as data_PCH_bin_times (or can be left empty for FCS only)')
        else:
            raise ValueError('data_PCH_hist must be array with axis 1 same length as same length as data_PCH_bin_times (or can be left empty for FCS only)')


    def g_3d_diff_single_species(self, 
                                 tau_diff):
        '''
        Normalized 3D diffusion autocorrelation for a single species
        '''
        return self.g_2d_diff_single_species(tau_diff) / np.sqrt(1 + self.tau / (np.square(self.FCS_psf_aspect_ratio) * self.data_tau_s))


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
                          cpms_eff,
                          precision = 1E-6):
        '''
        Calculates the single-particle compound PCH to be used in subsequent 
        calculation of the "real" multi-particle PCH.

        Parameters
        ----------
        cpms_eff : 
            Float. Molecular brightness in counts per molecule and bin.
        float :
            OPTIONAL Float with default 1E-6. Numeric precision parameter that
            determines at what photon count the iteration will terminate.
        Returns
        -------
        pch : 
            np.array with (normalized) PCH for given parameters. 
            Cave: This is a truncated PCH that starts at the bin for 1 photon!

        '''
        # Array with all photon counts from 1 to 1000 * average
        n_photons_max = np.floor(cpms_eff * 1E3)
        photon_counts_array = np.arange(1, n_photons_max+1)
        
        
        # Iterate over photon counts until we have essentially all the probability mass covered
        pch = []
        pch_cumsum = 0.
        k = 1
        while True:
            # Numeric integration over incomplete gamma function
            new_value = sintegrate.quad(lambda x: self.pch_3dgauss_1part_int(x,
                                                                             k,
                                                                             cpms_eff),
                                        0, np.inf)
            
            # Scipy.special actually implements regularized lower incompl. gamma func, so we need to scale it
            new_value *= sspecial.gamma(photon_counts_array)
            
            # A simpler prefactor
            new_value *=  1 / self.PCH_Q / np.sqrt(2.) / sspecial.factorial(photon_counts_array)
            
            # Write
            pch.append(new_value)
            
            # Converged?
            if k > 1:
                # From k=2 on, check for convergence within numeric precision
                if pch[-2] / pch_cumsum < precision:
                    # Tiny increment in probability mass, so we consider this converged
                    break
            
            # Iterate
            pch_cumsum += new_value
            k += 1
            
        return np.array(pch)
    
    
    def pch_3dgauss_nonideal_1part(self,
                                   F,
                                   cpms_eff):
        '''
        Calculates the single-particle compound PCH to be used in subsequent 
        calculation of the "real" multi-particle PCH. See:
        Huang, Perroud, Zare ChemPhysChem 2004 DOI: 10.1002/cphc.200400176

        Parameters
        ----------
        F :
            Float. Weight for non-Gaussian "out-of-focus" light contribution 
            correction.
        cpms_eff : 
            Float. Molecular brightness in counts per molecule and bin.

        Returns
        -------
        pch : 
            np.array with (normalized) PCH for given parameters. 
            Cave: This is a truncated PCH that starts at the bin for 1 photon!

        '''
        
        # Get ideal-Gauss PCH
        pch = self.pch_3dgauss_1part(cpms_eff,
                                     precision = 1E-6)
        
        # Apply correction
        pch /= (1 + F)
        pch[0] += 2**(-3/2) / self.PCH_Q * cpms_eff * F
        
        
    def pch_get_N_spectrum(self,
                           N_avg,
                           precision = 1E-4):
        '''
        Get the weighting function p(N) for different N in the box defined by Q
        for PCH. 

        Parameters
        ----------
        N_avg : 
            Float. <N> as defined in FCS math.
        precision : 
            OPTIONAL float with default 1E-4. Precision parameter defining at 
            which value for p(N) to truncate the spectrum

        Returns
        -------
        p_of_N : 
            np.array (1D). probabilities for N_box = 0, 1, 2, ...

        '''
        # Get scaled <N> as mean for Poisson dist object
        N_avg_box = self.PCH_Q * N_avg
        poisson_dist = sstats.poisson(N_avg_box)
        
        # x axis array
        N_box_array = np.arange(0, np.round(N_avg_box * 1E3))
        
        # Clip N_box to useful significant values within precision
        poisson_sf = poisson_dist.sf(N_box_array)
        N_box_array_clip = N_box_array[poisson_sf > precision]
        
        # Get probability mass function
        p_of_N = poisson_dist.pmf(N_box_array_clip)
        
        return p_of_N
    
    
    def pch_N_part(self,
                    pch_single_particle,
                    p_of_N):
        
        # Initialize a long array of zeros to contain the full PCH
        pch_full = np.zeros((pch_single_particle.shape[0] + 1 ) * (p_of_N.shape[0] + 1))
        
        # Append the probability mass for 0 photons to pch_single_particle
        pch_1 = np.concatenate((np.array([pch_single_particle.sum()]), pch_single_particle))
        
        # Write probability for 0 particles -> 0 photons into full PCH
        pch_full[0] += p_of_N[0]
        
        # Write single-particle weighted PCH into full PCH
        pch_full[0:pch_1.shape[0]] += p_of_N[1] * pch_1
        
        # Get N-particle PCHs and write them into the full thing through 
        # iterative convolutions
        pch_N = pch_1.copy()
        for pN in p_of_N[2:]:
            pch_N = np.convolve(pch_N, 
                                pch_1, 
                                mode='full')
            pch_full[0:pch_N.shape[0]] += pN * pch_N
        
        return pch_full[pch_full>0]
        
    
    def get_pch(self,
                F,
                t_bin,
                cpms,
                N_avg):
        '''
        Wrapper for get_pch using a syntax that works with lmfit.minimize()

        Parameters
        ----------
        F :
            Float. Weight for non-Gaussian "out-of-focus" light contribution 
            correction.
        t_bin : 
            Float. Bin time in seconds.
        cpms : 
            Float. Molecular brightness in counts per molecule and second.
        N_avg : 
            Float. <N> as defined in FCS math.
        Returns
        -------
        pch
            np.array (1D) with normalized pch model.
        '''

        # "Fundamental" single-particle PCH
        pch_single_particle = self.pch_3dgauss_nonideal_1part(F, 
                                                              t_bin * cpms)
        
        # Weights for observation of 1, 2, 3, 4, ... particles
        p_of_N = self.pch_get_N_spectrum(N_avg,
                                         precision = 1E-4)
        
        # Put weights and fundamental PCH together for full PCH
        pch = self.pch_N_part(pch_single_particle,
                              p_of_N)

        return pch


    def get_pch_lmfit(self,
                      params,
                      t_bin
                      ):
        '''
        Wrapper for get_pch using a syntax that works with lmfit.minimize()

        Parameters
        ----------
        params : 
            lmfit.parameters object

        Returns
        -------
        pch
            np.array (1D) with normalized pch model.

        '''
        
        return self.get_pch(params['F'].value,
                            params['t_bin'].value,
                            params['cpms'].value,
                            params['N_avg'].value)
        
    
    @staticmethod
    def negloglik_binomial_simple(n_trials,
                                  k_successes, 
                                  probabilities):
        '''
        Likelihood model for fitting a histogram described by k_successes from
        n_trials observations (where sum(k_successes)==n_trials when summing 
        over all histogram bins = fit data points) with a probability model
        described by probabilities.
        '''
        term1 = -k_successes * np.log(probabilities)
        term2 = -(n_trials - k_successes) * np.log(1 - probabilities)
        return np.sum(term1 + term2)
        
    
    
    def simple_pch_penalty(self,
                           params,
                           pch_data,
                           ):
        return self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                              k_successes = pch_data,
                                              probabilities = self.get_pch_lmfit(params))
    
    
    def run_simple_PCH_fit(self,
                           i_bin_time = None):
        
        if not self.PCH_possible:
            raise Exception('Cannot run PCH fit - not all required attributes set in class')
            
        if self.data_PCH_bin_times.shape[0] == 1 or utils.isempty(i_bin_time):
            bin_time = self.data_PCH_bin_times[0]
            pch = self.data_PCH_hist[:,0]

        elif utils.isint(i_bin_time) and i_bin_time >= 0 and i_bin_time < self.data_PCH_bin_times.shape[0]:
            bin_time = self.data_PCH_bin_times[i_bin_time]
            pch = self.data_PCH_hist[:,i_bin_time]
            
        else:
            raise Exception('Cannor run PCH fit. It seems you specified an invalid value for i_bin_time.')
        
        init_params = lmfit.Parameters()
        init_params.add('F', 
                        value = 0.4, 
                        min = 0, 
                        max = 1.,
                        vary=True)
        
        init_params.add('t_bin', 
                        value = bin_time, 
                        vary = False)
        
        init_params.add('cpms', 
                        value = 1E3, 
                        min = 0, 
                        vary = True)

        init_params.add('N_avg', 
                        value = 1., 
                        min = 0, 
                        vary = True)

        fit_result = lmfit.minimize(self.simple_pch_penalty, 
                                    init_params, 
                                    args=(pch),
                                    method='nelder') 
        
        print(lmfit.fit_report(fit_result), "\n")

        prediction = self.get_pch_lmfit(fit_result.params) * np.sum(pch)
        
        x_for_plot = np.arange(0, np.max([pch.shape[0], prediction.shape[0]]))
        plt.plot(x = x_for_plot,
                 y = pch,
                 marker = '.',
                 linestyle = 'none')
        plt.plot(x = x_for_plot,
                 y = prediction,
                 marker = '',
                 linestyle = '-')

        return fit_result
        
        
        
        
    
    