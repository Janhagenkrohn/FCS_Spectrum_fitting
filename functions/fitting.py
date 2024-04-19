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
from scipy.optimize import minimize as sminimize
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
                 labelling_efficiency = 1.,
                 numeric_precision = 1E-4
                 ):
        '''
        Class for single or combined fitting of FCS and/or PCH.
        Allows fitting of:
            - Single correlation function
            - Single photon counting histogram
            - Photon counting multiple histograms
            - Global fitting of an ACF and a PC(M)H dataset
        Possible models:
            - N-species diffusion models (FCS and/or PCMH)
            - N-species particle number and molecular brightness models
              (PC(M)H, possibly combined with FCS)
            - Explicit treatment of partial labelling in both FCS and PC(M)H
            - Treatment of blinking via a stretched-exponential approximation 
              in FCS and PCMH
            - #TODO: Explicit treatment of short-acquisition time bias in FCS
            - #TODO: Fitting of spectra of particle numbers over diffusion coefficients
              in FCS and PC(M)H using statistical regularization techniques 
              (Maximum Entropy Method, second-derivative minimization) or 
              parameterization of an underlying distribution (Gaussian, 
              Lognormal, Gamma, stretched exponential)
            - #TODO: Explicit treatment of statistically insufficient sampling 
              of rare species in particle number spectra, using correlations
              from regularization or parameterization to compensate missing 
              information and simple Poisson distribution approximations to 
              allow plausible and penalize unphysical deviations.

        Parameters
        ----------
        FCS_psf_width_nm :
            Float > 0. 1/e^2 width in nm of observation volume as defined in 
            standard FCS math. Required for FCS fitting, not for PCH.
        FCS_psf_aspect_ratio : 
            Float > 0. z/xy aspect ratio of the observation volume as defined in 
            standard FCS math. Required for FCS and PCMH fitting, not for 
            standard PCH.
        PCH_Q : 
            Float > 0. Scaling parameter for the PCH reference volume. Required 
            for PCH fitting, not for FCS.
        acquisition_time_s : 
            Float > 0. Acquisition time used for correcting short-acquisition 
            time bias in FCS, and for insufficient sampling correciton in FCS
            and PC(M)H. Neither of these corrections work without this parameter,
            but if standard math with assumption of representative sampling of 
            the ensemble is to be used, this parameter is not required.
        data_FCS_tau_s : 
            Array of float. Lag time array to be used as independend variable 
            in FCS fitting. Not required for PC(M)H.
        data_FCS_G : 
            Array of float with same length as data_FCS_tau_s. Correlation 
            function to be used as dependent variable in FCS fitting. Not 
            required for PC(M)H.
        data_FCS_sigma : 
            Array of float with same length as data_FCS_tau_s. of float. 
            Point-wise uncertainty associated with data_FCS_G. Required for 
            FCS fitting. For unweighted fitting, you may of course supply a
            dummy along the lines of np.ones_like(data_FCS_G), but do not 
            expect the global fits pf FCS+PCH, or the more advanced model 
            architectures, to yield good results that way. Not required for 
            PC(M)H.
        data_PCH_bin_times : 
            Float or array of floats. Bin times required for PCH (float) or 
            PCMH (array of floats) fitting. Not required for FCS.
        data_PCH_hist : 
            Array of int or float. Non-normalized histograms to be used as 
            dependent variable PC(M)H fitting. 
                - 1D array for PCH where it is implicitly assumed that the 
                  i-th element of the array represents the frequency of 
                  "i photon counts in bin"
                - 2D for PCH fitting or PCMH fitting, where iteration over 
                  axis 0 is iteration over photon counts and axis 1 is iteration
                  over bin times
        labelling_efficiency : 
            OPTIONAL float 0<labelling_efficiency<=1 with default 1. Labelling 
            efficiency expressed as a probability for a randomly chosen monomer 
            particle in the population to be labelled. Required for partial-
            labelling correction models in both FCS and PC(M)H, otherwise 
            irrelevant.
        numeric_precision : 
            OPTIONAL float 0<numeric_precision<1, with default is 1E-4. In PCH, 
            some steps in model calculation require truncation of the model at 
            a certain numerical precision. Some other steps do not strictly 
            require it, but profit greatly in terms of performance. This 
            parameter tunes the precision at which these calculations are to be 
            truncated. The smaller the value of numeric_precision, the more 
            accurate, but also the more computationally expensive, the 
            calculation gets.


        '''
        # Acquisition metadata and technical settings
        
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
    
    
        if utils.isfloat(numeric_precision) and numeric_precision > 0. and numeric_precision < 1.:
            self.numeric_precision = numeric_precision
        else:
            raise ValueError('numeric_precision must be float with 0 < numeric_precision < 1.')


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



    #%% Collection of some miscellaneous expressions
    def fcs_3d_diff_single_species(self, 
                                   tau_diff):
        '''
        Normalized 3D diffusion autocorrelation for a single species
        '''
        return 1 / (1 + self.data_FCS_tau_s/tau_diff) / np.sqrt(1 + self.data_FCS_tau_s / (np.square(self.FCS_psf_aspect_ratio) * tau_diff))


    def fcs_2d_diff_single_species(self, 
                                   tau_diff):
        '''
        Normalized 2D diffusion autocorrelation for a single species
        '''
        return 1 / (1 + self.data_FCS_tau_s/tau_diff)
    
    
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


    def n_total_from_N_avg(self, 
                           N_avg, 
                           tau_diff):
        '''
        For converting the particle number parameter N_avg in FCS/PCH
        to the estiamted total number of observed particles during acquisition
        '''
        
        n_total = N_avg * self.acquisition_time_s / tau_diff
        
        return n_total
        
    
    def N_avg_from_n_total(self, 
                           n_total, 
                           tau_diff):
        '''
        Inverse of FCS_spectrum.n_total_from_N_avg
        '''
        N_avg = n_total * tau_diff / self.acquisition_time_s 
        return N_avg


    #%% Penalty terms in fitting
    def negloglik_incomplete_sampling_full_labelling(self,
                                                     params):
        '''
        Neg log likelihood function for deviation between population-level
        and observed N, but without treatment of labelling statistics
        '''
        n_species = self.get_n_species(params)
        
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_pop_array = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_obs_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])
            
        n_pop = self.n_total_from_N_avg(N_avg_pop_array, 
                                        tau_diff_array)
        
        n_obs = self.n_total_from_N_avg(N_avg_obs_array, 
                                        tau_diff_array)
        
        negloglik = self.negloglik_poisson_full(rates = n_pop,
                                                observations = n_obs)
        
        return negloglik
    
    
    def negloglik_incomplete_sampling_partial_labelling(self,
                                                        params):
        '''
        Neg log likelihood function for deviation between population-level
        and observed N and labelling statistics
        '''
        n_species = self.get_n_species(params)
        
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_pop_array = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_obs_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])
        labelling_efficiency_obs_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species)])
        lebelling_efficiency_pop = params['Label_efficiency'].value
        stoichiometry_array = np.array([params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
        
        # Likelihood function for particle numbers
        n_pop = self.n_total_from_N_avg(N_avg_pop_array, 
                                        tau_diff_array)
        
        n_obs = self.n_total_from_N_avg(N_avg_obs_array, 
                                        tau_diff_array)
        
        negloglik_N = self.negloglik_poisson_full(rates = n_pop,
                                                  observations = n_obs)
        
        # Here, we accumulate the negloglik iteratively over species
        negloglik_labelling = 0.
        
        # Likelihood function for labelling efficiency fluctuations
        # We iterate over species, and for each species calculate a likelihood 
        # for the observed labelling
        for i_spec in range(n_species):
            
            # "Histogram" of "observed" labelling with "observed" N spectrum
            labelling_efficiency_array_spec_obs = self.pch_get_stoichiometry_spectrum(stoichiometry_array[i_spec], 
                                                                                      labelling_efficiency_obs_array[i_spec])
            labelling_efficiency_array_spec_obs *= n_obs[i_spec]
            
            # Reference labelling statistics based on labelling efficiency metadata and "observed" N spectrum
            labelling_efficiency_array_spec_pop = self.pch_get_stoichiometry_spectrum(stoichiometry_array[i_spec], 
                                                                                      lebelling_efficiency_pop)
            
            negloglik_labelling += self.negloglik_binomial_full(n_trials = n_obs[i_spec],
                                                                k_successes = labelling_efficiency_array_spec_obs, 
                                                                probabilities = labelling_efficiency_array_spec_pop)
            
        negloglik = negloglik_N + negloglik_labelling
        
        return negloglik
    
    
    def negloglik_acf_full_labelling(self, 
                                     params):
        '''
        Tiny wrapper for fcs_chisq_full_labelling() that corrects for the fact that reduced 
        chi-square is actually 2*neg log likelihood, which is usually irrelevant,
        but for global maximum-likelihood fitting worth correcting.
        '''        
        
        negloglik = self.fcs_chisq_full_labelling(params) / 2.
        
        return negloglik
    
    
    def fcs_chisq_full_labelling(self, 
                                 params):
        '''
        Reduced chi-square for FCS fitting with full labelling.
        '''
        
        # Count variable parameters in fit
        n_vary = np.sum([1. for key in params.keys() if params[key].vary])
                
        # Get model
        acf_model = self.get_acf_full_labelling(params)
        
        # Calc weighted residual sum of squares
        wrss = np.sum(((acf_model - self.data_FCS_G) / self.data_FCS_sigma) ** 2)
        
        # Return reduced chi-square
        red_chi_sq =  wrss / (self.data_FCS_G.shape[0] - n_vary)
        
        return red_chi_sq


    def negloglik_acf_partial_labelling(self, 
                                        params):
        '''
        Tiny wrapper for fcs_chisq_partial_labelling() that corrects for the fact that reduced 
        chi-square is actually 2*neg log likelihood, which is usually irrelevant,
        but for global maximum-likelihood fitting worth correcting.
        '''        
        
        negloglik = self.fcs_chisq_partial_labelling(params) / 2.
        
        return negloglik
    
    
    def fcs_chisq_partial_labelling(self, 
                                 params):
        '''
        Reduced chi-square for FCS fitting with partial labelling.
        '''
        
        # Count variable parameters in fit
        n_vary = np.sum([1. for key in params.keys() if params[key].vary])
                
        # Get model
        acf_model = self.get_acf_partial_labelling(params)
        
        # Calc weighted residual sum of squares
        wrss = np.sum(((acf_model - self.data_FCS_G) / self.data_FCS_sigma) ** 2)
        
        # Return reduced chi-square
        red_chi_sq =  wrss / (self.data_FCS_G.shape[0] - n_vary)
        
        return red_chi_sq


    @staticmethod
    def negloglik_poisson_full(rates, 
                               observations):
        '''
        Poisson neg log lik in fitting. Full model to allow "observations" 
        themselves to vary in a hierarchical model with a hidden Poisson process.
        '''
        
        negloglik = np.sum(np.log(rates ** observations) - rates - np.log(sspecial.factorial(observations)))
        
        return negloglik


    @staticmethod
    def negloglik_poisson_simple(rates, 
                                 observations):
        '''
        Poisson neg log lik in fitting. Simplified version for fixed observations.
        '''
        
        negloglik = np.sum(np.log(rates ** observations) - rates)
        
        return negloglik 

        
    @staticmethod
    def negloglik_binomial_simple(n_trials,
                                  k_successes, 
                                  probabilities):
        '''
        Neg log likelihood function for fitting a histogram described by k_successes from
        n_trials observations (where for PCH sum(k_successes)==n_trials when summing 
        over all histogram bins = fit data points) with a probability model
        described by probabilities. Simplified version for fitting models where
        only probabilities is varied.
        '''
        # Renormalize for possible truncation artifacts
        probabilities /= probabilities.sum()
        
        successes_term = -k_successes * np.log(probabilities)
        failures_term = -(n_trials - k_successes) * np.log(1 - probabilities)
        
        negloglik = np.sum(successes_term + failures_term)
        
        return negloglik


    @staticmethod
    def negloglik_binomial_full(n_trials,
                                k_successes, 
                                probabilities):
        '''
        Neg log likelihood function for fitting a histogram described by k_successes from
        n_trials observations. Full version for fitting models where
        only all parameters can be varied, used here in incomplete labelling + 
        incomplete sampling treatment.
        '''
        # Renormalize for possible truncation artifacts
        probabilities /= probabilities.sum()

        neglog_binom_coeff = - np.log(sspecial.binom(n_trials, 
                                                     k_successes))
        successes_term = -k_successes * np.log(probabilities)
        failures_term = -(n_trials - k_successes) * np.log(1 - probabilities)
        
        negloglik = np.sum(neglog_binom_coeff + successes_term + failures_term)
        
        return negloglik


    def regularization_MEM(self,
                           params,
                           spectrum_parameter):
        
        # Unpack parameters
        n_species = self.get_n_species(params)
        if spectrum_parameter == 'Amplitude':
            reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value * params[f'cpms_{i_spec}'].value**2 for i_spec in range(n_species)])
        elif spectrum_parameter == 'N_monomers':
            reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value * params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
        elif spectrum_parameter == 'N_oligomers':
            reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species)])
        else:
            raise Exception("Invalid input for spectrum_parameter - must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
        
        # Normalize and remove zeros
        frequency_array = reg_target / reg_target.sum()
        frequency_array_nonzero = frequency_array[frequency_array > 0.] 
        
        # neg entropy regularizer
        regularizer = np.sum(frequency_array_nonzero * np.log(frequency_array_nonzero))
        return regularizer


    def regularization_CONTIN(self,
                              params,
                              spectrum_parameter):

        # Unpack parameters
        n_species = self.get_n_species(params)
        
        if spectrum_parameter == 'Amplitude':
            reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value * params[f'cpms_{i_spec}'].value**2 for i_spec in range(n_species)])
        elif spectrum_parameter == 'N_monomers':
            reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value * params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
        elif spectrum_parameter == 'N_oligomers':
            reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species)])
        else:
            raise Exception("Invalid input for spectrum_parameter - must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
            
        # Normalize
        frequency_array = reg_target / reg_target.sum()
        
        # Numerically approximate second derivative of N distribution
        second_numerical_diff = np.diff(frequency_array, 2)
        
        # Get sum of squared second derivative as single number that reports on non-smoothness of distribution
        regularizer = np.sum(second_numerical_diff**2)
        
        return regularizer
        
        
    def simple_pch_penalty(self,
                           params,
                           pch
                           ):
        
        pch_model = self.get_simple_pch_lmfit(params)
        return self.negloglik_binomial_simple(n_trials = np.sum(pch),
                                              k_successes = pch,
                                              probabilities = pch_model)


    def negloglik_pch_single_full_labelling(self,
                                            params,
                                            i_bin_time = 0):
        
        pch_model = self.get_pch_full_labelling(params,
                                                self.data_PCH_bin_times[i_bin_time],
                                                time_resolved_PCH = False,
                                                crop_output = True)
        
        # Crop data/model where needed...Can happen for data handling reasons
        pch_data = self.data_PCH_hist[:,i_bin_time]
        if pch_model.shape[0] > pch_data.shape[0]:
            pch_model = pch_model[:pch_data.shape[0]]
        elif pch_model.shape[0] < pch_data.shape[0]:
            pch_data = pch_data[:pch_model.shape[0]]

        negloglik = self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                                   k_successes = pch_data,
                                                   probabilities = pch_model)
            
        return negloglik


    def negloglik_pch_single_partial_labelling(self,
                                               params,
                                               i_bin_time = 0):
        
        pch_model = self.get_pch_partial_labelling(params,
                                                   self.data_PCH_bin_times[i_bin_time],
                                                   time_resolved_PCH = False,
                                                   crop_output = True)
        
        # Crop data/model where needed...Can happen for data handling reasons
        pch_data = self.data_PCH_hist[:,i_bin_time]
        if pch_model.shape[0] > pch_data.shape[0]:
            pch_model = pch_model[:pch_data.shape[0]]
        elif pch_model.shape[0] < pch_data.shape[0]:
            pch_data = pch_data[:pch_model.shape[0]]

        negloglik = self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                                   k_successes = pch_data,
                                                   probabilities = pch_model)
        
        return negloglik
        
    
    def negloglik_pcmh_full_labelling(self,
                                      params):
        
        negloglik = 0
                
        for i_bin_time, t_bin in enumerate(self.data_PCH_bin_times):
            
            pch_model = self.get_pch_full_labelling(params,
                                                    t_bin,
                                                    time_resolved_PCH = True,
                                                    crop_output = True)
            
            # Crop data/model where needed...Can happen for data handling reasons
            pch_data = self.data_PCH_hist[:,i_bin_time]
            if pch_model.shape[0] > pch_data.shape[0]:
                pch_model = pch_model[:pch_data.shape[0]]
            elif pch_model.shape[0] < pch_data.shape[0]:
                pch_data = pch_data[:pch_model.shape[0]]

            negloglik += self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                                        k_successes = pch_data,
                                                        probabilities = pch_model)
            
        return negloglik

    
    def negloglik_pcmh_partial_labelling(self,
                                         params):
        
        negloglik = 0
                
        for i_bin_time, t_bin in enumerate(self.data_PCH_bin_times):
            
            pch_model = self.get_pch_partial_labelling(params,
                                                       t_bin,
                                                       time_resolved_PCH = True,
                                                       crop_output = True)
            
            # Crop data/model where needed...Can happen for data handling reasons
            pch_data = self.data_PCH_hist[:,i_bin_time]
            if pch_model.shape[0] > pch_data.shape[0]:
                pch_model = pch_model[:pch_data.shape[0]]
            elif pch_model.shape[0] < pch_data.shape[0]:
                pch_data = pch_data[:pch_model.shape[0]]
                
            negloglik += self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                                        k_successes = pch_data,
                                                        probabilities = pch_model)
            
        return negloglik


    def negloglik_global_fit(self,
                             params,
                             use_FCS,
                             use_PCH,
                             time_resolved_PCH,
                             spectrum_type,
                             spectrum_parameter,
                             labelling_correction,
                             incomplete_sampling_correction):
        
        negloglik = 0
        
        if use_FCS:
            # Correlation function is being fitted
            
            if labelling_correction:
                negloglik += self.negloglik_acf_partial_labelling(params)
            else: # not labelling_correction
                negloglik += self.negloglik_acf_full_labelling(params)
                
        if use_PCH and (not time_resolved_PCH):
            # Conventional single PCH is being fitted
            
            if labelling_correction:
                negloglik += self.negloglik_pch_single_partial_labelling(params)
            else: # not labelling_correction
                negloglik += self.negloglik_pch_single_full_labelling(params)
                
        elif use_PCH and time_resolved_PCH:
            # PCMH fit
            
            if labelling_correction:
                negloglik += self.negloglik_pcmh_partial_labelling(params)
            else: # not labelling_correction
                negloglik += self.negloglik_pcmh_full_labelling(params)
                
        if spectrum_type == 'reg_CONTIN':
            negloglik += self.regularization_CONTIN(params, 
                                                    spectrum_parameter)
        
        elif spectrum_type == 'reg_MEM':
            negloglik += self.regularization_MEM(params,
                                                 spectrum_parameter)
            
        if incomplete_sampling_correction:
            # Incomplete sampling correction included in fit
            
            if labelling_correction:
                negloglik += self.negloglik_incomplete_sampling_partial_labelling(params)
            else: # not labelling_correction
                negloglik += self.negloglik_incomplete_sampling_full_labelling(params)
        
        return negloglik
        
        
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
        pch = self.pch_3dgauss_1part(cpm_eff)

        # Apply correction
        pch /= (1 + F)
        pch[0] += 2**(-3/2) / self.PCH_Q * cpm_eff * F
        
        return pch
        
    
    def pch_get_N_spectrum(self,
                           N_avg):
        '''
        Get the weighting function p(N) for different N in the box defined by Q
        for PCH. 

        Parameters
        ----------
        N_avg : 
            Float. <N> as defined in FCS math.
        Returns
        -------
        p_of_N : 
            np.array (1D). probabilities for N_box = 0, 1, 2, ...

        '''
        # Get scaled <N> as mean for Poisson dist object
        N_avg_box = self.PCH_Q * N_avg
        poisson_dist = sstats.poisson(N_avg_box)
        
        # x axis array
        N_box_array = np.arange(0, np.ceil(N_avg_box * 1E3))
        # Clip N_box to useful significant values within precision (on righthand side)
        poisson_sf = poisson_dist.sf(N_box_array)
        N_box_array_clip = N_box_array[:np.nonzero(poisson_sf > self.numeric_precision)[0][-1] + 1]
        
        # Get probability mass function
        p_of_N = poisson_dist.pmf(N_box_array_clip)
        
        return p_of_N


    def pch_get_stoichiometry_spectrum(self,
                                       max_label,
                                       Label_efficiency):
        '''
        Get the weighting function p(n_labels) for different labelling 
        stoichiometries of the oligomer based on binomial stats

        Parameters
        ----------
        max_label : 
            Scalar. The maximum number of labels possible on the particle.
        Label_efficiency :
            Float 0<Label_efficiency<=1. The probability for a single 
            labelling site to be occupied.
        Returns
        -------
        p_of_n_labels : 
            np.array (1D). probabilities for n_labels = 0, 1, 2, ...

        '''
        
        # Parameterize Binomial dist object
        binomial_dist = sstats.binom(max_label, Label_efficiency)

        # x axis array
        n_labels_array = np.arange(0, max_label + 1)
        
        # Clip n_labels_array to useful significant values within precision
        # Note that weights are given by both particle frequency and particle stoichiometry
        binomial_pmf = binomial_dist.pmf(n_labels_array)
        weights = binomial_pmf * (n_labels_array + 1) # +1 to avoid zeros, and we need not be super-accurate here
        weights /= weights.sum()
        n_labels_array_clip = n_labels_array[weights > self.numeric_precision]
        
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
        p_of_N = self.pch_get_N_spectrum(N_avg)
        
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

        temp1 = np.sqrt(self.FCS_psf_aspect_ratio**2 - 1)
        temp2 = np.sqrt(self.FCS_psf_aspect_ratio**2 + t_bin / tau_diff)
        temp3 = np.arctanh(temp1 * (temp2 - self.FCS_psf_aspect_ratio) / (1 + self.FCS_psf_aspect_ratio * temp2 - self.FCS_psf_aspect_ratio**2))
        
        diff_corr = 4 * self.FCS_psf_aspect_ratio * tau_diff / t_bin**2 / temp1 * \
            ((tau_diff + t_bin) * temp3 + tau_diff * temp1 * (self.FCS_psf_aspect_ratio - temp2)) 
        
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
        
        acf += params['acf_offset'].value # Offset

        return acf


    def get_acf_partial_labelling(self, 
                                  params):
                
        acf_num = np.zeros_like(self.data_FCS_G)
        acf_den = np.zeros_like(self.data_FCS_G)
        
        n_species = self.get_n_species(params)
            
        # Extract parameters
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species)])
        labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])

        for i_spec in range(n_species):
            g_norm = self.fcs_3d_diff_single_species(tau_diff_array[i_spec])
            acf_num += g_norm * N_avg_array[i_spec] * cpms_array[i_spec]**2 * (1 + (1 - labelling_efficiency_array[i_spec]) / cpms_array[i_spec] / labelling_efficiency_array[i_spec])
            acf_den += N_avg_array[i_spec] * cpms_array[i_spec] * labelling_efficiency_array[i_spec]
        
        acf = acf_num / acf_den**2

        if params['F_blink'].value > 0:
            acf *= 1 + params['F_blink'].value / (1 - params['F_blink'].value) * self.fcs_blink_stretched_exp(params['tau_blink'].value,
                                                                                                              params['beta_blink'].value)
        
        acf += params['acf_offset'].value # Offset

        return acf




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
        
    
   
    
    def get_pch_full_labelling(self,
                               params,
                               t_bin,
                               time_resolved_PCH = False,
                               crop_output = False
                               ):
        
        n_species = self.get_n_species(params)

        # Extract parameters
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])

        if time_resolved_PCH:
            # Bin time correction
            tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])
            cpms_array, N_avg_array = self.pch_bin_time_correction(t_bin = t_bin, 
                                                                   cpms = cpms_array, 
                                                                   N_avg = N_avg_array,
                                                                   tau_diff = tau_diff_array, 
                                                                   tau_blink = params['tau_blink'].value, 
                                                                   beta_blink = params['beta_blink'].value, 
                                                                   F_blink = params['F_blink'].value)
        
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
                                  time_resolved_PCH = False,
                                  crop_output = False):
        
        # Unpack parameters
        cpms_0 = params['cpms_0'].value
        n_species = self.get_n_species(params)
        
        stoichiometry_array = np.array([params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
        labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species)])

        if time_resolved_PCH:
            tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])
            tau_blink = params['tau_blink'].value, 
            beta_blink = params['beta_blink'].value, 
            F_blink = params['F_blink'].value
        
        # Iterate over species in spectrum
        for i_spec in range(n_species):
            
            
            # Get probabilities for different label stoichiometries, 
            # clipping away extremely unlikely ones for computational feasibility
            n_labels_array, p_of_n_labels = self.pch_get_stoichiometry_spectrum(stoichiometry_array[i_spec],
                                                                                labelling_efficiency_array[i_spec])
            
            # Use parameters to get full array of PCH parameters
            cpms_array_spec = n_labels_array * cpms_0 #  frequency of 0,1,2,3,... labels
            N_array_spec =  N_avg_array[i_spec] * p_of_n_labels # N_avg_array[i_spec] as a kind of global amplitude for the frequencies
            
            if time_resolved_PCH:
                # Bin time correction
                cpms_array_spec, N_array_spec = self.pch_bin_time_correction(t_bin = t_bin, 
                                                                             cpms = cpms_array_spec, 
                                                                             N_avg = N_array_spec ,
                                                                             tau_diff = tau_diff_array[i_spec], 
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
            
        if not (oligomer_type in ['spherical_shell', 'sherical_dense', 'single_filament', 'double_filament']):
            raise Exception("Invalid input for oligomer_type - oligomer_type must be one out of 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'")

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
            # which cannot be decently inverted. This is a one-parameter 
            # optimization and needs to be done only once per species, so it's not a big deal
            
            for i_spec, tau_diff_fold_change in enumerate(fold_changes):
                res = sminimize(fun = self.single_filament_tau_diff_fold_change_deviation, 
                                x0 = np.array([1.]),
                                args = (tau_diff_fold_change,))
                stoichiometry[i_spec + 1] = np.exp(res.x)
        
        else: # oligomer_type == 'double_filament'
        
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
                               time_resolved_PCH,
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
            
        if (use_PCH and time_resolved_PCH) and not self.FCS_possible:
            raise Exception('Cannot run PCMH fit - not all required attributes set in class')

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
        initial_params.add('Label_efficiency', 
                           value = self.labelling_efficiency if labelling_correction else 1.,
                           vary = False)
        
        if use_FCS:
            initial_params.add('acf_offset', 
                                value = 0., 
                                vary=True)

        if use_PCH:
            initial_params.add('F', 
                               value = 0.4, 
                               min = 0., 
                               max = 1.,
                               vary = True)
                                
        
        # Actual model parameters 
        for i_spec in range(0, n_species):
            initial_params.add(f'N_avg_obs_{i_spec}', 
                               value = 1., 
                               min = 0., 
                               vary = True)

            if use_FCS or (use_PCH and time_resolved_PCH):
                # Diffusion time only for FCS and PCMH
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
                
            # Additional parameter that is included only as a dummy for 
            # consistency with spectrum models
            initial_params.add(f'Label_obs_factor_{i_spec}', 
                               value = 1.,
                               vary = False)

            initial_params.add(f'Label_efficiency_obs_{i_spec}', 
                               expr = f'Label_efficiency * Label_obs_factor_{i_spec}',
                               vary = False)

                
        # Add blinking parameters for FCS and PCMH - real or dummy
        if use_FCS or (use_PCH and time_resolved_PCH):
            initial_params = self.set_blinking_initial_params(initial_params,
                                                              use_blinking,
                                                              tau_diff_min)

        return initial_params


    def set_up_params_reg(self,
                          use_FCS,
                          use_PCH,
                          time_resolved_PCH,
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
                    
        if (use_PCH and time_resolved_PCH) and not self.FCS_possible:
            raise Exception('Cannot run PCMH fit - not all required attributes set in class')
            
        if not spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            raise Exception("Invalid input for spectrum_type for set_up_params_reg - must be one out of 'reg_MEM', 'reg_CONTIN'")

        if not (oligomer_type in ['continuous_spherical', 'spherical_shell', 'sherical_dense', 'single_filament', 'double_filament']):
            raise Exception("Invalid input for oligomer_type - oligomer_type must be one out of 'continuous_spherical', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'")

        if not (utils.isint(n_species) and n_species >= 10):
            raise Exception("Invalid input for n_species - must be int >= 10 for regularized fitting")
            
        tau_diff_array = self.get_tau_diff_array(tau_diff_min, 
                                                 tau_diff_max, 
                                                 n_species)
        
        stoichiometry = self.stoichiometry_from_tau_diff_array(tau_diff_array, 
                                                               oligomer_type)
            
        initial_params = lmfit.Parameters()

        # More technical parameters
        if use_FCS:
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

            if use_FCS or (use_PCH and time_resolved_PCH):
                # Diffusion time only for FCS and PCMH
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
                               vary = True if (incomplete_sampling_correction and labelling_correction) else False)

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
            
        # Add blinking parameters for FCS and PCMH - real or dummy
        if use_FCS or (use_PCH and time_resolved_PCH):
            initial_params = self.set_blinking_initial_params(initial_params,
                                                              use_blinking,
                                                              tau_diff_min)

        return initial_params



    def set_up_params_par(self,
                          use_FCS,
                          use_PCH,
                          time_resolved_PCH,
                          spectrum_type,
                          spectrum_parameter,
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

        if (use_PCH and time_resolved_PCH) and not self.FCS_possible:
            raise Exception('Cannot run PCMH fit - not all required attributes set in class')

        if not spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            raise Exception("Invalid input for spectrum_type for set_up_params_par - must be one out of 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'")

        if not (oligomer_type in ['spherical_shell', 'sherical_dense', 'single_filament', 'double_filament']):
            raise Exception("Invalid input for oligomer_type - oligomer_type must be one out of spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'")

        if not (utils.isint(n_species) and n_species >= 10):
            raise Exception("Invalid input for n_species - must be int >= 10 for parameterized spectrum fitting")
            
        tau_diff_array = self.get_tau_diff_array(tau_diff_min, 
                                                 tau_diff_max, 
                                                 n_species)
        
        stoichiometry = self.stoichiometry_from_tau_diff_array(tau_diff_array, 
                                                               oligomer_type)
            
        initial_params = lmfit.Parameters()

        # More technical parameters
        if use_FCS:
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
            
            if use_FCS or (use_PCH and time_resolved_PCH):
                # Diffusion time only for FCS and PCMH
                initial_params.add(f'tau_diff_{i_spec}', 
                                   value = tau_diff_array[i_spec], 
                                   vary = False)
            
            initial_params.add(f'stoichiometry_{i_spec}', 
                               value = stoichiometry[i_spec], 
                               vary = False)
            
            # Weighting function that essentially decides which number the parameterization acts on
            if spectrum_parameter == 'Amplitude':
                initial_params.add(f'spectrum_weight_{i_spec}', 
                                   value = stoichiometry[i_spec]**(-2), 
                                   vary = False)
            elif spectrum_parameter == 'N_monomers':
                initial_params.add(f'spectrum_weight_{i_spec}', 
                                   value = stoichiometry[i_spec]**(-1), 
                                   vary = False)
            elif spectrum_parameter == 'N_oligomers':
                initial_params.add(f'spectrum_weight_{i_spec}', 
                                   value = 1., 
                                   vary = False)
            else:
                raise Exception("Invalid input for spectrum_parameter - must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
            
            
            # Define particle numbers via parameterized distributions
            if spectrum_type == 'par_Gauss':
                initial_params.add(f'N_avg_pop_{i_spec}', 
                                   expr = f'N_dist_amp * spectrum_weight_{i_spec} * exp(-0.5 * ((stoichiometry_{i_spec} - N_dist_a) / N_dist_b) ** 2)', 
                                   vary = False)
                
            if spectrum_type == 'par_LogNorm':
                initial_params.add(f'N_avg_pop_{i_spec}', 
                                   expr = f'N_dist_amp / spectrum_weight_{i_spec} / stoichiometry_{i_spec} * exp(-0.5 * ((log(stoichiometry_{i_spec}) - N_dist_a) / N_dist_b) ** 2)',
                                   vary = False)
                
            if spectrum_type == 'par_Gamma':
                initial_params.add(f'N_avg_pop_{i_spec}', 
                                   expr = f'N_dist_amp * spectrum_weight_{i_spec} * stoichiometry_{i_spec}**(N_dist_a - 1) * exp(N_dist_a - N_dist_b * stoichiometry_{i_spec})', 
                                   vary = False)
                
            if spectrum_type == 'par_StrExp':
                initial_params.add(f'N_avg_pop_{i_spec}', 
                                   expr = f'N_dist_amp * spectrum_weight_{i_spec} * exp(1 / N_dist_b - (stoichiometry_{i_spec} * N_dist_a) ** N_dist_b)', 
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
                               vary = True if (incomplete_sampling_correction and labelling_correction) else False)

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
            
        # Add blinking parameters for FCS and PCMH - real or dummy
        if use_FCS or (use_PCH and time_resolved_PCH):
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
                time_resolved_PCH, # bool
                spectrum_type, # 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'
                spectrum_parameter, # 'Amplitude', 'N_monomers', 'N_oligomers',
                oligomer_type, # 'continuous_spherical', 'spherical_shell', 'sherical_dense', 'single_filament', 'double_filament'
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

        if time_resolved_PCH and not self.FCS_possible:
            raise Exception('Cannot run PCMH fit - not all required attributes set in class')

        if not spectrum_type in ['discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            raise Exception("Invalid input for spectrum_type - must be one out of 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', or 'par_StrExp'")

        if not (spectrum_parameter in ['Amplitude', 'N_monomers', 'N_oligomers'] or  spectrum_type == 'discrete'):
            raise Exception("Invalid input for spectrum_parameter - unless spectrum_type is 'discrete', spectrum_parameter must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
    
        if not (oligomer_type in ['continuous_spherical', 'spherical_shell', 'sherical_dense', 'single_filament', 'double_filament'] or spectrum_type == 'discrete'):
            raise Exception("Invalid input for oligomer_type - unless spectrum_type is 'discrete', oligomer_type must be one out of 'continuous_spherical', 'spherical_shell', 'sherical_dense', 'single_filament', or 'double_filament'")
        
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
            # Parameter setup
            initial_params = self.set_up_params_discrete(use_FCS, 
                                                         use_PCH,
                                                         time_resolved_PCH,
                                                         labelling_correction,
                                                         n_species, 
                                                         tau_diff_min, 
                                                         tau_diff_max, 
                                                         use_blinking
                                                         )
            
            
        elif spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            # Parameter setup
            initial_params = self.set_up_params_reg(use_FCS,
                                                    use_PCH,    
                                                    time_resolved_PCH,
                                                    spectrum_type,
                                                    oligomer_type,
                                                    incomplete_sampling_correction,
                                                    labelling_correction,
                                                    n_species,
                                                    tau_diff_min,
                                                    tau_diff_max,
                                                    use_blinking
                                                    )
            
            
        else: # spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']
            # Parameter setup
            initial_params = self.set_up_params_par(use_FCS, 
                                                    use_PCH, 
                                                    time_resolved_PCH,
                                                    spectrum_type,
                                                    spectrum_parameter, 
                                                    oligomer_type, 
                                                    incomplete_sampling_correction, 
                                                    labelling_correction, 
                                                    n_species, 
                                                    tau_diff_min, 
                                                    tau_diff_max, 
                                                    use_blinking
                                                    )
            
        print('\n Initial parameters:')
        [print(f'{key}: {initial_params[key].value} (varied: {initial_params[key].vary})') for key in initial_params.keys()]
        
        # Define minimization target
        fitter = lmfit.Minimizer(self.negloglik_global_fit, 
                                 params = initial_params, 
                                 fcn_args = (use_FCS, 
                                             use_PCH, 
                                             time_resolved_PCH,
                                             spectrum_type, 
                                             spectrum_parameter,
                                             labelling_correction, 
                                             incomplete_sampling_correction),
                                 calc_covar = True)
            
        fit_result = fitter.minimize(method = 'nelder')

        


        return fit_result
        
        
        
        
    
    