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
import scipy.optimize.minimize as sminimize
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
                 labelling_efficiency = 1.
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
            self.PCH_n_photons_max = 0
        elif utils.isiterable(data_PCH_hist):
            data_PCH_hist = np.array(data_PCH_hist)
            if data_PCH_hist.shape[1] == data_PCH_bin_times.shape[0]:
                self.data_PCH_hist = data_PCH_hist
                self.PCH_n_photons_max = data_PCH_hist.shape[0] + 1
            else:
                raise ValueError('data_PCH_hist must be array with axis 1 same length as same length as data_PCH_bin_times (or can be left empty for FCS only)')
        else:
            raise ValueError('data_PCH_hist must be array with axis 1 same length as same length as data_PCH_bin_times (or can be left empty for FCS only)')


        if utils.isfloat(labelling_efficiency) and labelling_efficiency > 0. and labelling_efficiency <= 1.:
            self.labelling_efficiency = labelling_efficiency
        else:
            raise ValueError('labelling_efficiency must be float with 0 < labelling_efficiency <= 1.')

    #%% Collection of "simple" expressions for fit models, cost terms, etc.
    def fcs_3d_diff_single_species(self, 
                                   tau_diff):
        '''
        Normalized 3D diffusion autocorrelation for a single species
        '''
        return 1 / (1 + self.data_tau_s/tau_diff) / np.sqrt(1 + self.tau / (np.square(self.FCS_psf_aspect_ratio) * self.data_tau_s))


    def fcs_2d_diff_single_species(self, 
                                   tau_diff):
        '''
        Normalized 2D diffusion autocorrelation for a single species
        '''
        return 1 / (1 + self.data_tau_s/tau_diff)
    
    def fcs_blink_stretched_exp(self,
                                tau_blink,
                                beta_blink):
        return np.exp(-(self.data_FCS_tau_s / tau_blink)**beta_blink)
        

    @staticmethod
    def single_filament_tau_diff_fold_change(j):
        return 2. * j / (2. * np.log(j) + 0.632 + 1.165 * j ** (-1.) + 0.1 * j ** (-2.))


    @staticmethod
    def double_filament_tau_diff_fold_change(j):
        axial_ratio = j / 2.
        return 2. * axial_ratio / (2. * np.log(axial_ratio) + 0.632 + 1.165 * axial_ratio ** (-1.) + 0.1 * axial_ratio ** (-2.))

    
    @staticmethod
    def negloglik_poisson_full(rates, 
                               observations):
        '''
        Poisson neg log lik in fitting. Full model to allow "observations" 
        themselves to vary in a hierarchical model with a hidden Poisson process
        (if "observations" were fixed, the factorial term could be omitted)
        '''
        return np.sum(np.log(rates ** observations) - rates - np.log(sspecial.factorial(observations)))


    @staticmethod
    def negloglik_poisson_simple(rates, 
                                 observations):
        '''
        Poisson neg log lik in fitting. Simplified version for fixed observations.
        '''
        return np.sum(np.log(rates ** observations) - rates)

        
    @staticmethod
    def negloglik_binomial_simple(n_trials,
                                  k_successes, 
                                  probabilities):
        '''
        Neg log likelihood function for fitting a histogram described by k_successes from
        n_trials observations (where for PCH sum(k_successes)==n_trials when summing 
        over all histogram bins = fit data points) with a probability model
        described by probabilities.
        '''
        term1 = -k_successes * np.log(probabilities)
        term2 = -(n_trials - k_successes) * np.log(1 - probabilities)
        return np.sum(term1 + term2)


    #%% Other small stuff
    @staticmethod
    def get_n_species(params):
        '''
        Brute-force method to figure out number of species in a multi-species 
        lmfit.Parameters object as used in this class. Not an efficient way to
        do it I guess, but used to avoid explicitly passing the number of 
        species through the code stack, making things more robust in coding. 
        Or that's the idea at least.
        '''
        n_species = 0
        while True:
            if f'cpms_{n_species}' in params.keys():
                n_species += 1
            else:
                break
        return n_species


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


    #%% PCH model code stack
    
    @staticmethod
    def pch_3dgauss_1part_int(x,
                              k,
                              cpm_eff):
        '''
        Helper function for implementing the numeric integral in pch_3dgauss_1part
        '''
        return sspecial.gammainc(k, cpm_eff * np.exp(-2*x**2))
    
    
    def pch_3dgauss_1part(self,
                          cpm_eff):
        '''
        Calculates the single-particle compound PCH to be used in subsequent 
        calculation of the "real" multi-particle PCH.

        Parameters
        ----------
        cpm_eff : 
            Float. Molecular brightness in counts per molecule and bin.
        n_photons_max :
            Int that will determine the highest photon count to consider.
        Returns
        -------
        pch : 
            np.array with (normalized) PCH for given parameters. 
            Cave: This is a truncated PCH that starts at the bin for 1 photon!

        '''
        # Array with all photon counts > 0 to sample
        photon_counts_array = np.arange(1, self.PCH_n_photons_max+1)
        
        # Results container
        pch = np.zeros_like(photon_counts_array, 
                            dtype = np.float64)
        
        # Iterate over photon counts 
        for k in photon_counts_array:
            # Numeric integration over incomplete gamma function
            pch[k-1], _ = sintegrate.quad(lambda x: self.pch_3dgauss_1part_int(x,
                                                                                k,
                                                                                cpm_eff),
                                          0, np.inf)
            
        # Scipy.special actually implements regularized lower incompl. gamma func, so we need to scale it
        pch *= sspecial.gamma(photon_counts_array)
        
        # A simpler prefactor
        pch *=  1 / self.PCH_Q / np.sqrt(2.) / sspecial.factorial(photon_counts_array)
        
        return pch
    
    
    def pch_3dgauss_nonideal_1part(self,
                                   F,
                                   cpm_eff):
        '''
        Calculates the single-particle compound PCH to be used in subsequent 
        calculation of the "real" multi-particle PCH. See:
        Huang, Perroud, Zare ChemPhysChem 2004 DOI: 10.1002/cphc.200400176

        Parameters
        ----------
        F :
            Float. Weight for non-Gaussian "out-of-focus" light contribution 
            correction.
        cpm_eff : 
            Float. Molecular brightness in counts per molecule and bin.

        Returns
        -------
        pch : 
            np.array with (normalized) PCH for given parameters. 
            Cave: This is a truncated PCH that starts at the bin for 1 photon!

        '''
        
        # Get ideal-Gauss PCH
        pch = self.pch_3dgauss_1part(cpm_eff,
                                     self.PCH_n_photons_max)

        # Apply correction
        pch /= (1 + F)
        pch[0] += 2**(-3/2) / self.PCH_Q * cpm_eff * F
        
        return pch
        
    
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
        
        # Clip N_box to useful significant values within precision (on righthand side)
        poisson_sf = poisson_dist.sf(N_box_array)
        N_box_array_clip = N_box_array[:np.nonzero(poisson_sf > precision)[0][-1] + 1]
        
        # Get probability mass function
        p_of_N = poisson_dist.pmf(N_box_array_clip)
        
        return p_of_N


    def pch_get_stoichiometry_spectrum(self,
                                       max_label,
                                       precision = 1E-4):
        '''
        Get the weighting function p(n_labels) for different labelling 
        stoichiometries of the oligomer based on binomial stats

        Parameters
        ----------
        max_label : 
            Scalar. The maximum number of labels possible on the particle.
        precision : 
            OPTIONAL float with default 1E-4. Precision parameter defining at 
            which value for p(n_labels) to truncate the spectrum

        Returns
        -------
        p_of_n_labels : 
            np.array (1D). probabilities for n_labels = 0, 1, 2, ...

        '''
        
        # Parameterize Binomial dist object
        binomial_dist = sstats.binom(max_label, self.labelling_efficiency)

        # x axis array
        n_labels_array = np.arange(0, max_label + 1)
        
        # Clip n_labels_array to useful significant values within precision
        binomial_sf = binomial_dist.sf(n_labels_array)
        n_labels_array_clip = n_labels_array[binomial_sf > precision]
        
        # Get probability mass function
        p_of_n_labels = binomial_dist.pmf(n_labels_array_clip)
        
        return n_labels_array_clip, p_of_n_labels
    
    @staticmethod
    def pch_N_part(pch_single_particle,
                   p_of_N):
        '''
        Take a single-particle "fundamental" PCH and an array of probabilities
        for N = 0,1,2,... particles to calculate an N-particle PCH

        Parameters
        ----------
        pch_single_particle : 
            np.array with (normalized) PCH for given parameters. 
            Cave: This is a truncated PCH that starts at the bin for 1 photon!
        p_of_N : 
            np.array (1D). probabilities for N_box = 0, 1, 2, ...

        Returns
        -------
        pch : 
            np.array with (normalized) N-particle PCH for given parameters. 
            Cave: This is a truncated PCH that starts at the bin for 1 photon!

        '''
        # Initialize a long array of zeros to contain the full PCH
        pch_full = np.zeros((pch_single_particle.shape[0] + 1 ) * (p_of_N.shape[0] + 1))
        
        # Append the probability mass for 0 photons to pch_single_particle
        pch_1 = np.concatenate((np.array([1 - pch_single_particle.sum()]), pch_single_particle))
        
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
                N_avg,
                crop_output = True):
        '''
        Calculate a single-species PCH without diffusion or blinking information

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
        crop_output :
            OPTIONAL Bool with default True. Whether or not to cut the output 
            of the PCH to the length defined by self.PCH_n_photons_max.
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

        if crop_output:
            pch = pch[:self.PCH_n_photons_max + 1]
            
        return pch
    
    
    def pch_bin_time_correction(self,
                                t_bin,
                                cpms,
                                N_avg,
                                tau_diff,
                                tau_blink,
                                beta_blink,
                                F_blink
                                ):
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
            Float or array of float. . <N> as defined in FCS math.
        tau_diff : 
            Float or array of float. . Diffusion time in seconds as defined in FCS math.
        tau_blink : 
            Float or array of float. . Blinking relaxation time in seconds as defined in FCS math.
        beta_blink : 
            Float or array of float. . Stretched-exponential scale parameter for blinking.
        F_blink : 
            Float or array of float. . Blinking off-state fraction as defined in FCS math.

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

        temp1 = np.sqrt(self.FCS_psf_aspect_ratio**2 - 1)
        temp2 = np.sqrt(self.FCS_psf_aspect_ratio**2 - t_bin / tau_diff)
        temp3 = np.arctanh(temp1 * (temp2 - self.FCS_psf_aspect_ratio) / 1 + self.FCS_psf_aspect_ratio * temp2 - self.FCS_psf_aspect_ratio)
        
        diff_corr = 4 * self.FCS_psf_aspect_ratio * tau_diff / t_bin**2 * temp1 * \
            ((tau_diff + t_bin) * temp3 + tau_diff * temp1 * (self.FCS_psf_aspect_ratio * temp2)) 
        
        cpms_corr = cpms * blink_corr * diff_corr
        N_avg_corr = N_avg / blink_corr / diff_corr
        
        return cpms_corr, N_avg_corr


    def multi_species_pch(self,
                          F,
                          t_bin,
                          cpms_array,
                          N_avg_array,
                          crop_output = True
                          ):
        
        if not utils.isfloat(F) or F <= 0.:
           raise Exception('Invalid input for F: Must be float > 0.') 

        if not utils.isfloat(t_bin) or t_bin <= 0.:
           raise Exception('Invalid input for t_bin: Must be float > 0.') 

        if not utils.isiterable(cpms_array):
           raise Exception('Invalid input for cpms_array: Must be array.') 
            
        if not utils.isiterable(N_avg_array):
           raise Exception('Invalid input for N_avg_array: Must be array.')
           
        if not cpms_array.shape[0] == N_avg_array.shape[0]:
           raise Exception('cpms_array and N_avg_array must have same length.')
           
        if not type(crop_output) == bool:
           raise Exception('Invalid input for crop_output: Must be bool.')
        

        for i_spec, cpms_spec in enumerate(cpms_array):
            N_avg_spec = N_avg_array[i_spec]
            
            if i_spec == 0:
                # Get first species PCH
                pch = self.get_pch(F,
                                   t_bin,
                                   cpms_spec,
                                   N_avg_spec,
                                   crop_output = False)        
            
            else:
                # Colvolve with further species PCH
                pch = np.convolve(pch, 
                                  self.get_pch(F,
                                               t_bin,
                                               cpms_spec,
                                               N_avg_spec,
                                               crop_output = False),
                                  mode = 'full')      
        
        if crop_output:
            pch = pch[:self.PCH_n_photons_max + 1]
            
        return pch

    #%% More complete/complex FCS models
                
    
    def get_acf_full_labelling(self, 
                               params):
                
        acf_num = np.zeros_like(self.data_FCS_G)
        acf_den = np.zeros_like(self.data_FCS_G)
        
        n_species = self.get_n_species(params)
            
        # Extract parameters
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])

        for i_spec in range(n_species):
            g_norm = self.fcs_3d_diff_single_species(tau_diff_array[i_spec])
            acf_num += g_norm * N_avg_array[i_spec] * cpms_array[i_spec]**2
            acf_den += N_avg_array[i_spec] * cpms_array[i_spec]
        
        acf = acf_num / acf_den**2

        if params['F_blink'].value > 0:
            acf *= 1 + params['F_blink'].value / (1 - params['F_blink'].value) * self.fcs_blink_stretched_exp(params['tau_blink'].value,
                                                                                                                params['beta_blink'].value)
        
        acf *= 2**(-2/3) # Amplitude correction
        acf += params['acf_offset'].value # Offset

        return acf


    def fcs_chisq(self, 
                  params):
        
        # Count variable parameters in fit
        n_vary = np.sum([1. for key in params.keys() if params[key].vary])
                
        # Get model
        acf_model = self.get_acf_full_labelling(params)
        
        # Calc weighted residual sum of squares
        wrss = np.sum(((acf_model - self.data_FCS_G) / self.data_FCS_sigma) ** 2)
        
        # Return reduced chi-square
        return wrss / (self.data_FCS_G.shape[0] - n_vary)


    #%% More complete/complex PCH models
    def get_simple_pch_lmfit(self,
                             params,
                             ):
        '''
        Wrapper for get_pch using a syntax that works easily with lmfit.minimize()

        Parameters
        ----------
        params : 
            lmfit.parameters object containing all arguments of self.get_pch()
            as parameters

        Returns
        -------
        pch
            np.array (1D) with normalized pch model.

        '''
        
        return self.get_pch(params['F'].value,
                            params['t_bin'].value,
                            params['cpms'].value,
                            params['N_avg'].value)
        
    
    def simple_pch_penalty(self,
                           params,
                           pch
                           ):
        pch_model = self.get_simple_pch_lmfit(params)
        return self.negloglik_binomial_simple(n_trials = np.sum(pch),
                                              k_successes = pch,
                                              probabilities = pch_model)
    
    
    def get_pch_full_labelling(self,
                               params,
                               n_species,
                               t_bin,
                               time_resolved = False,
                               crop_output = False
                               ):
                
        # Extract parameters
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])

        tau_blink = params['tau_blink'].value, 
        beta_blink = params['beta_blink'].value, 
        F_blink = params['F_blink'].value

        if time_resolved:
            # Bin time correction
            cpms_array, N_avg_array = self.pch_bin_time_correction(t_bin = t_bin, 
                                                                 cpms = cpms_array, 
                                                                 N_avg = N_avg_array,
                                                                 tau_diff = tau_diff_array, 
                                                                 tau_blink = tau_blink, 
                                                                 beta_blink = beta_blink, 
                                                                 F_blink = F_blink)
        
        pch = self.multi_species_pch(F = params['F'].value,
                                     t_bin = t_bin,
                                     cpms_array = cpms_array,
                                     N_avg_array = N_avg_array,
                                     crop_output = True
                                     )
        if crop_output:
            pch = pch[:self.PCH_n_photons_max + 1]

        return pch


    def get_pch_partial_labelling(self,
                                  params,
                                  t_bin,
                                  time_resolved = False,
                                  crop_output = False):
                
        cpms_0 = params['cpms_0'].value
        
        tau_blink = params['tau_blink'].value, 
        beta_blink = params['beta_blink'].value, 
        F_blink = params['F_blink'].value
        
        # Iterate over species in spectrum
        n_species = self.get_n_species(params)
        for i_spec in range(n_species):
            
            # Get parameters
            max_labels = params[f'stoichiometry_{i_spec}'].value
            N_avg = params[f'N_avg_obs_{i_spec}'].value
            
            # Get probabilities for different label stoichiometries, 
            # clipping away extremely unlikely ones for computational feasibility
            n_labels_array, p_of_n_labels = self.pch_get_stoichiometry_spectrum(max_labels, 
                                                                                precision = 1E-4)
            
            # Use parameters to get full array of PCH parameters
            cpms_array_spec = n_labels_array * cpms_0
            N_array_spec = N_avg * p_of_n_labels
            
            if time_resolved:
                # Bin time correction
                cpms_array_spec, N_array_spec = self.pch_bin_time_correction(t_bin = t_bin, 
                                                                     cpms = cpms_array_spec, 
                                                                     N_avg = N_array_spec ,
                                                                     tau_diff = params[f'tau_diff_{i_spec}'].value, 
                                                                     tau_blink = tau_blink, 
                                                                     beta_blink = beta_blink, 
                                                                     F_blink = F_blink)
                
                
            if i_spec == 0:
                # Get first species PCH
                pch = self.multi_species_pch(F = params['F'].value,
                                             t_bin = t_bin,
                                             cpms_array = cpms_array_spec,
                                             N_avg_array = N_array_spec,
                                             crop_output = True
                                             )
            else:
                # Colvolve with further species PCH
                pch = np.convolve(pch,
                                 self.multi_species_pch(F = params['F'].value,
                                                              t_bin = t_bin,
                                                              cpms_array = cpms_array_spec,
                                                              N_avg_array = N_array_spec,
                                                              crop_output = True
                                                              ),
                                 mode = 'full')
                
            if crop_output:
                pch = pch[:self.PCH_n_photons_max + 1]

        return pch

        
    #%% Stuff for fit parameter initialization
    def get_tau_diff_array(self,
                           tau_diff_min,
                           tau_diff_max,
                           n_species
                           ):
                
        return np.logspace(start = np.log(tau_diff_min),
                           stop = np.log(tau_diff_max),
                           num = n_species,
                           base = np.e)
    
    
    def single_filament_tau_diff_fold_change_deviation(self,
                                                       log_j,
                                                       tau_diff_fold_change):
        return (self.single_filament_tau_diff_fold_change(np.exp(log_j)) - tau_diff_fold_change) ** 2
    
    
    def double_filament_tau_diff_fold_change_deviation(self,
                                                       log_j,
                                                       tau_diff_fold_change):
        return (self.double_filament_tau_diff_fold_change(np.exp(log_j)) - tau_diff_fold_change) ** 2

    
    def stoichiometry_from_tau_diff_array(self,
                                          tau_diff_array,
                                          oligomer_type
                                          ):
        
        if not utils.isiterable(tau_diff_array):
            raise Exception("Invalid input for tau_diff_array - must be np.array")
            
        if not (oligomer_type in ['continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', 'discrete_double_filament']):
            raise Exception("Invalid input for oligomer_type - oligomer_type must be one out of 'continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', or 'discrete_double_filament'")

        # The monomer by definition has stoichiometry 1, so we start with the second element
        fold_changes = tau_diff_array[1:] / tau_diff_array[0]
        stoichiometry = np.ones_like(tau_diff_array)
        
        if oligomer_type in ['continuous_spherical', 'discrete_spherical']:
            # tau_diff proportional hydrodyn. radius
            # Stoichiometry proportional volume
            stoichiometry[1:] = fold_changes ** 3
        
        elif oligomer_type == 'continuous_shell':
            # tau_diff proportional hydrodyn. radius
            # Stoichiometry proportional surface area
            stoichiometry[1:] = fold_changes ** 2

        elif oligomer_type == 'discrete_single_filament':
            # For the filament models, we have more complicated expressions based
            # on Seils & Pecora 1995. We numerically solve the expression,
            # which cannot be decently inverted. This is a one-parameter 
            # optimization and needs to be done only once per species, so it's not a big deal
            
            for i_spec, tau_diff_fold_change in enumerate(fold_changes):
                res = sminimize(fun = self.single_filament_tau_diff_fold_change_deviation, 
                                x0 = np.array([1.]),
                                args = (tau_diff_fold_change,))
                stoichiometry[i_spec + 1] = np.exp(res.x)
        
        else: # oligomer_type == 'discrete_double_filament'
        
            stoichiometry = np.zeros_like(fold_changes)
            
            for i_spec, tau_diff_fold_change in enumerate(fold_changes):

                res = sminimize(fun = self.double_filament_tau_diff_fold_change_deviation, 
                                x0 = np.array([1.]),
                                args = (tau_diff_fold_change,))
                stoichiometry[i_spec + 1] = np.exp(res.x)
        
        return np.round(stoichiometry)
    
       
    def set_blinking_initial_params(self,
                                    initial_params,
                                    use_blinking,
                                    tau_diff_min):
        
        # Blinking parameters 
        if use_blinking:
            initial_params.add('tau_blink', 
                               value = tau_diff_min / 10., 
                               min = 0., 
                               max = tau_diff_min, 
                               vary = True)

            initial_params.add('F_blink', 
                               value = 0.1, 
                               min = 0., 
                               max = 1., 
                               vary = True)

            initial_params.add('beta_blink', 
                               value = 1., 
                               min = 0., 
                               max = 10., 
                               vary = True)
            
        else: # not use_blinking -> Dummy values
            initial_params.add('tau_blink', 
                               value = tau_diff_min / 10., 
                               vary = False)

            initial_params.add('F_blink', 
                               value = 0., 
                               vary = False)

            initial_params.add('beta_blink', 
                               value = 1., 
                               vary = False)

        return initial_params
    
    
    def set_up_params_discrete(self,
                               use_FCS,
                               use_PCH,
                               n_species,
                               tau_diff_min,
                               tau_diff_max,
                               use_blinking
                               ):
        
        if use_FCS and not self.FCS_possible:
            raise Exception('Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception('Cannot run PCH fit - not all required attributes set in class')

        if not (utils.isint(n_species) and n_species > 0):
            raise Exception("Invalid input for n_species - must be int > 0")
        
        if not (utils.isfloat(tau_diff_min) and tau_diff_min > 0):
            raise Exception("Invalid input for tau_diff_min - must be float > 0")

        if not (utils.isfloat(tau_diff_max) and tau_diff_max > 0):
            raise Exception("Invalid input for tau_diff_max - must be float > 0")

        if type(use_blinking) != bool:
            raise Exception("Invalid input for use_blinking - must be bool")


        initial_params = lmfit.Parameters()
        
        # More technical parameters
        initial_params.add('acf_offset', 
                           value = 0., 
                           vary=True)

        if use_PCH:
            initial_params.add('F', 
                               value = 0.4, 
                               min = 0, 
                               max = 1.,
                               vary=True)
                                
        
        # Actual model parameters 
        for i_spec in range(1, n_species+1):
            initial_params.add(f'N_avg_obs_{i_spec}', 
                               value = 1., 
                               min = 0., 
                               vary = True)

            initial_params.add(f'tau_diff_{i_spec}', 
                               value = 1E-3, 
                               min = tau_diff_min,
                               max = tau_diff_max,
                               vary = True)

            
            if use_PCH:
                initial_params.add(f'cpms_{i_spec}', 
                                   value = 1E3, 
                                   min = 0., 
                                   vary = True)
            else:
                # If we do not use PCH, use a dummy, as we won't be able to tell from FCS alone
                initial_params.add(f'cpms_{i_spec}', 
                                   value = 1., 
                                   min = 0., 
                                   vary = False)
                
        # Add blinking parameters - real or dummy
        initial_params = self.set_blinking_initial_params(initial_params,
                                                          use_blinking,
                                                          tau_diff_min)

        return initial_params


    def set_up_params_reg(self,
                          use_FCS,
                          use_PCH,
                          spectrum_type,
                          oligomer_type,
                          incomplete_sampling_correction,
                          labelling_correction,
                          n_species,
                          tau_diff_min,
                          tau_diff_max,
                          use_blinking
                          ):
    
        if use_FCS and not self.FCS_possible:
            raise Exception('Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception('Cannot run PCH fit - not all required attributes set in class')

        if not spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            raise Exception("Invalid input for spectrum_type for set_up_params_reg - must be one out of 'reg_MEM', 'reg_CONTIN'")

        if not (oligomer_type in ['continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', 'discrete_double_filament']):
            raise Exception("Invalid input for oligomer_type - oligomer_type must be one out of 'continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', or 'discrete_double_filament'")

        if not (utils.isint(n_species) and n_species >= 10):
            raise Exception("Invalid input for n_species - must be int >= 10 for regularized fitting")
            
        tau_diff_array = self.get_tau_diff_array(tau_diff_min, 
                                                 tau_diff_max, 
                                                 n_species)
        
        stoichiometry = self.stoichiometry_from_tau_diff_array(tau_diff_array, 
                                                               oligomer_type)
            
        initial_params = lmfit.Parameters()

        # More technical parameters
        initial_params.add('acf_offset', 
                           value = 0., 
                           vary=True)

        if use_PCH:
            initial_params.add('F', 
                               value = 0.4, 
                               min = 0, 
                               max = 1.,
                               vary=True)

        initial_params.add('Label_efficiency', 
                           value = self.labelling_efficiency if labelling_correction else 1.,
                           vary = False)

        # Species-wise parameters
        for i_spec, tau_diff_i in enumerate(tau_diff_array):
            
            initial_params.add(f'N_avg_pop_{i_spec}', 
                               value = 1., 
                               min = 0., 
                               vary = True)

            if incomplete_sampling_correction:
                # Allow fluctuations of observed apparent particle count
                initial_params.add(f'N_avg_obs_{i_spec}', 
                                   value = 1., 
                                   min = 0., 
                                   vary = True)
            else:
                # Dummy
                initial_params.add(f'N_avg_obs_{i_spec}', 
                                   expr = f'N_avg_pop_{i_spec}', 
                                   vary = False)

            initial_params.add(f'tau_diff_{i_spec}', 
                               value = tau_diff_array[i_spec], 
                               vary = False)
            
            initial_params.add(f'stoichiometry_{i_spec}', 
                               value = stoichiometry[i_spec], 
                               vary = False)

            # An additional factor for translating between "sample-level" and
            # "population-level" observed label efficiency if and only if we 
            # use both incomplete_sampling_correction and labelling_correction
            initial_params.add(f'Label_obs_factor_{i_spec}', 
                               value = 1.,
                               vary = True if incomplete_sampling_correction and labelling_correction else False)

            initial_params.add(f'Label_efficiency_obs_{i_spec}', 
                               expr = f'Label_efficiency * Label_obs_factor_{i_spec}',
                               vary = False)

            if i_spec == 0:
                
                # Monomer brightness
                if use_PCH:
                    initial_params.add('cpms_0', 
                                       value = 1E3, 
                                       min = 0, 
                                       vary = True)
                else:
                    # If we do not use PCH, use a dummy, as we won't be able to tell from FCS alone
                    initial_params.add('cpms_0', 
                                       value = 1., 
                                       min = 0, 
                                       vary = False)
            else: # i_spec >= 1
            
                # Oligomer cpms is defined by monomer and stoichiometry factor
                initial_params.add(f'cpms_{i_spec}', 
                                   expr = f'cpms_0 * stoichiometry_{i_spec}', 
                                   vary = False)
            
        # Add blinking parameters - real or dummy
        initial_params = self.set_blinking_initial_params(initial_params,
                                                          use_blinking,
                                                          tau_diff_min)

        return initial_params



    def set_up_params_par(self,
                          use_FCS,
                          use_PCH,
                          spectrum_type,
                          oligomer_type,
                          incomplete_sampling_correction,
                          labelling_correction,
                          n_species,
                          tau_diff_min,
                          tau_diff_max,
                          use_blinking
                          ):
    
        if use_FCS and not self.FCS_possible:
            raise Exception('Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception('Cannot run PCH fit - not all required attributes set in class')

        if not spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            raise Exception("Invalid input for spectrum_type for set_up_params_par - must be one out of 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'")

        if not (oligomer_type in ['continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', 'discrete_double_filament']):
            raise Exception("Invalid input for oligomer_type - oligomer_type must be one out of 'continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', or 'discrete_double_filament'")

        if not (utils.isint(n_species) and n_species >= 10):
            raise Exception("Invalid input for n_species - must be int >= 10 for parameterized spectrum fitting")
            
        tau_diff_array = self.get_tau_diff_array(tau_diff_min, 
                                                 tau_diff_max, 
                                                 n_species)
        
        stoichiometry = self.stoichiometry_from_tau_diff_array(tau_diff_array, 
                                                               oligomer_type)
            
        initial_params = lmfit.Parameters()

        # More technical parameters
        if use_PCH:
            initial_params.add('F', 
                               value = 0.4, 
                               min = 0, 
                               max = 1.,
                               vary=True)
            
        initial_params.add('Label_efficiency', 
                           value = self.labelling_efficiency if labelling_correction else 1.,
                           vary = False)
        
        # N distribution parameters
        initial_params.add('N_dist_amp', 
                           value = 10., 
                           min = 0., 
                           vary=True)

        initial_params.add('N_dist_a', 
                           value = 10., 
                           min = 0., 
                           vary=True)

        initial_params.add('N_dist_b', 
                           value = 1., 
                           min = 0.,
                           vary=True)
            
        for i_spec, tau_diff_i in enumerate(tau_diff_array):
            
            initial_params.add(f'tau_diff_{i_spec}', 
                               value = tau_diff_array[i_spec], 
                               vary = False)
            
            initial_params.add(f'stoichiometry_{i_spec}', 
                               value = stoichiometry[i_spec], 
                               vary = False)
            
            # Define 
            if spectrum_type == 'par_Gauss':
                initial_params.add(f'N_avg_pop_{i_spec}', 
                                   expr = f'N_dist_amp / sqrt(2 * pi) / N_dist_b * exp(-0.5 * ((stoichiometry_{i_spec} - N_dist_a) / N_dist_b) ** 2)', 
                                   vary = False)
                
            if spectrum_type == 'par_LogNorm':
                initial_params.add(f'N_avg_pop_{i_spec}', 
                                   expr = f'N_dist_amp / sqrt(2 * pi) / N_dist_b / stoichiometry_{i_spec} * exp(-0.5 * ((log(stoichiometry_{i_spec}) - N_dist_a) / N_dist_b) ** 2)',
                                   vary = False)
                
            if spectrum_type == 'par_Gamma':
                # We cannot use sspecial.gamma inside the "expr" attribute. We instead use the Lanczos approximation for the gamma function.
                initial_params.add(f'N_avg_pop_{i_spec}', 
                                   expr = f'N_dist_amp * N_dist_b ** N_dist_a / sqrt(2 * pi) * N_dist_a ** (0.5 - N_dist_a) * stoichiometry_{i_spec} ** (N_dist_a - 1) * exp(N_dist_a - N_dist_b * stoichiometry_{i_spec})', 
                                   vary = False)
                
            if spectrum_type == 'par_StrExp':
                # Again Lanczos approximation for gamma function.
                initial_params.add(f'N_avg_pop_{i_spec}', 
                                   expr = f'N_dist_amp * N_dist_a / sqrt(2 * pi) * (1 / N_dist_b) ** (0.5 - (1 / N_dist_b)) * exp(1 / N_dist_b - (stoichiometry_{i_spec} * N_dist_a) ** N_dist_b)', 
                                   vary = False)

            if incomplete_sampling_correction:
                # Allow fluctuations of observed apparent particle count
                initial_params.add(f'N_avg_obs_{i_spec}', 
                                   value = 1., 
                                   min = 0., 
                                   vary = True)
                
            else:
                # Dummy
                initial_params.add(f'N_avg_obs_{i_spec}', 
                                   expr = f'N_avg_pop_{i_spec}', 
                                   vary = False)

            # An additional factor for translating between "sample-level" and
            # "population-level" observed label efficiency if and only if we 
            # use both incomplete_sampling_correction and labelling_correction
            initial_params.add(f'Label_obs_factor_{i_spec}', 
                               value = 1.,
                               vary = True if incomplete_sampling_correction and labelling_correction else False)

            initial_params.add(f'Label_efficiency_obs_{i_spec}', 
                               expr = f'Label_efficiency * Label_obs_factor_{i_spec}',
                               vary = False)
                
            if i_spec == 0:
                
                # Monomer brightness
                if use_PCH:
                    initial_params.add('cpms_0', 
                                       value = 1E3, 
                                       min = 0, 
                                       vary = True)
                else:
                    # If we do not use PCH, use a dummy, as we won't be able to tell from FCS alone
                    initial_params.add('cpms_0', 
                                       value = 1., 
                                       min = 0, 
                                       vary = False)
                    
            else: # i_spec >= 1
                # Oligomer cpms is defined by monomer and stoichiometry factor
                initial_params.add(f'cpms_{i_spec}', 
                                   expr = f'cpms_0 * stoichiometry_{i_spec}', 
                                   vary = False)
            
        # Add blinking parameters - real or dummy
        initial_params = self.set_blinking_initial_params(initial_params,
                                                          use_blinking,
                                                          tau_diff_min)

        return initial_params


    #%% Complete single-call fit routines
    
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
        
        # We set the range such that we calculate the PCH until the first EMPTY 
        # bin in the actual data, as "zero photons in this bin" actually is 
        # a little bit of information for PCH fitting
        
        if self.PCH_n_photons_max > pch.shape[0] - 1:
            #The last PCH bin is nonzero: We actually need to zero-pad the PCH
            pch_fit = np.append(pch, np.array([0]))
            
        elif self.PCH_n_photons_max == pch.shape[0] - 1:
            # The PCH is just right
            pch_fit = pch.copy()
            
        else:
            # The PCH contains more zeros than we need
            pch_fit = pch[:self.PCH_n_photons_max + 1]
            

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
                                    args = (pch_fit,), 
                                    method='nelder') 
        
        print(lmfit.fit_report(fit_result), "\n")

        prediction = self.get_pch_lmfit(fit_result.params) * np.sum(pch_fit)
        
        x_for_plot = np.arange(0, np.max([pch_fit.shape[0], prediction.shape[0]]))
        
        fig, ax = plt.subplots(1, 1)
        ax.semilogy(x_for_plot,
                     pch_fit,
                     marker = '.',
                     linestyle = 'none')
        ax.semilogy(x_for_plot,
                     prediction,
                     marker = '',
                     linestyle = '-')
        ax.set_ylim(0.3, np.max(pch_fit) * 1.25)
        ax.set_title(f'PCH fit bin time {bin_time} s')
        fig.supxlabel('Photons in bin')
        fig.supylabel('Counts')

        plt.show()
        
        return fit_result


    def run_fit(self,
                use_FCS, # bool
                use_PCH, # bool
                spectrum_type, # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
                spectrum_parameter, # 'Amplitude', 'N_monomers', 'N_oligomers',
                oligomer_type, # 'continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', 'discrete_double_filament'
                labelling_correction, # bool
                incomplete_sampling_correction, # bool
                n_species, # int
                tau_diff_min, # float
                tau_diff_max, # float
                use_blinking # bool
                ):
        
        # A bunch of input and compatibility checks
        if use_FCS and not self.FCS_possible:
            raise Exception('Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception('Cannot run PCH fit - not all required attributes set in class')

        if not spectrum_type in ['discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            raise Exception("Invalid input for spectrum_type - must be one out of 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', or 'par_StrExp'")

        if not (spectrum_parameter in ['Amplitude', 'N_monomers', 'N_oligomers'] or  spectrum_type == 'discrete'):
            raise Exception("Invalid input for spectrum_parameter - unless spectrum_type is 'discrete', spectrum_parameter must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
    
        if not (oligomer_type in ['continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', 'discrete_double_filament'] or spectrum_type == 'discrete'):
            raise Exception("Invalid input for oligomer_type - unless spectrum_type is 'discrete', oligomer_type must be one out of 'continuous_spherical', 'continuous_shell', 'discrete_spherical', 'discrete_single_filament', or 'discrete_double_filament'")
        
        if type(labelling_correction) != bool:
            raise Exception("Invalid input for labelling_correction - must be bool")

        if type(incomplete_sampling_correction) != bool:
            raise Exception("Invalid input for incomplete_sampling_correction - must be bool")

        if incomplete_sampling_correction and spectrum_type == 'discrete':
            raise Exception("incomplete_sampling_correction does not work for spectrum_type 'discrete' and must be set to false")

        if not (utils.isint(n_species) and n_species > 0):
            raise Exception("Invalid input for n_species - must be int > 0")
        
        if n_species < 10 and spectrum_type != 'discrete':
            raise Exception("For any spectrum_type other than 'discrete', use n_species >= 10")

        if not (utils.isfloat(tau_diff_min) and tau_diff_min > 0):
            raise Exception("Invalid input for tau_diff_min - must be float > 0")

        if not (utils.isfloat(tau_diff_max) and tau_diff_max > 0):
            raise Exception("Invalid input for tau_diff_max - must be float > 0")

        if type(use_blinking) != bool:
            raise Exception("Invalid input for use_blinking - must be bool")






        if spectrum_type == 'discrete':
            initial_params = self.set_up_params_discrete(use_FCS, 
                                                         use_PCH, 
                                                         n_species, 
                                                         tau_diff_min, 
                                                         tau_diff_max, 
                                                         use_blinking
                                                         )
            
        elif spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            pass
        else: # spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']
            pass
            
        


        return None
        
        
        
        
    
    