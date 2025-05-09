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
# from scipy.optimize import minimize as sminimize
from scipy.optimize import minimize_scalar as sminimize_scalar
from scipy.optimize import minimize as sminimize
import lmfit


# Plotting
from matplotlib import pyplot as plt

# Parallel processing and related
import multiprocessing
import traceback

# To suppress unnecessary command line warnings
import warnings


# Custom module
# For localizing module
repo_dir = os.path.abspath('..')
sys.path.insert(0, repo_dir)
from functions import utils

# Some hard-coded metaparameters for regularized minimization
# Weighting factor between likelihood function and regularization term
REG_MAX_LAGRANGE_MUL = 1E3
# "Momentum" factors for iterations in reg fitting
REG_LAGRANGE_MUL_MOMENTUM = 1/3 # Exponent
REG_GRAD_MOMENTUM = 0.1 # Multiplicative
# Gradient scaling factor for (inner) iteration update
REG_GRADIENT_SCALING = 2e-4
# Inner and outer iteration convergence criteria, and maximum iteration counts
REG_MAX_ITER_INNER = 1E4
REG_MAX_ITER_OUTER = 30
REG_CONV_THRESH_INNER = 1e-1
REG_CONV_THRESH_OUTER = 1e-2


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
                 data_avg_count_rate = 0.,
                 labelling_efficiency = 1.,
                 numeric_precision = np.array([1E-3, 1E-4, 1E-5]),
                 NLL_funcs_accurate = False,
                 verbosity = 1,
                 job_prefix = '',
                 labelling_efficiency_incomp_sampling = False 
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
            - Fitting of spectra of particle numbers over diffusion coefficients
              in FCS and PC(M)H using statistical regularization techniques 
              (Maximum Entropy Method, second-derivative minimization) or 
              parameterization of an underlying distribution (Gaussian, 
              Lognormal, Gamma, stretched exponential)
            - Explicit treatment of statistically insufficient sampling 
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
            time bias in FCS, for insufficient sampling correciton in FCS
            and PC(M)H, and for including average count rate in the fit. 
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
        data_avg_count_rate :
            Optional float >=0 with default 0. If specified > 0, it can be used 
            as an additional constraint parameter in fitting that affects the 
            fit of molecular brightness.
        labelling_efficiency : 
            OPTIONAL float 0<labelling_efficiency<=1 with default 1. Labelling 
            efficiency expressed as a probability for a randomly chosen monomer 
            particle in the population to be labelled. Required for partial-
            labelling correction models in both FCS and PC(M)H, otherwise 
            irrelevant.
        numeric_precision : 
            OPTIONAL float 0<numeric_precision<1, or array of such floats.
            Default is [1E-3, 1E-4, 1E-5]. In PCH, and in the 
            incomplete-sampling + incomplete labelling likelihood function,
            some steps in model calculation require truncation of the model at 
            a certain numerical precision, or at least profit greatly in 
            terms of performance. This 
            parameter tunes the precision at which these calculations are to be 
            truncated. The smaller the value of numeric_precision, the more 
            accurate, but also the more computationally expensive, the 
            calculation gets. Using an array instead of a single value is meant
            as a means of saving time by first fitting a low-precision model, 
            and incrementally fine-tune parameters at higher calcualtion 
            precision, rather than solving the computationally expensive 
            high-precision model at every step of the fit procedure.
        NLL_funcs_accurate :
            OPTIONAL bool with default False. If False, PC(M)H fitting 
            and incomplete-samplign likelihood functions will be
            done with the typical least-squares approximation widespread in the
            literature. If True, a more accurate but computationally expensive
            binomial maximum likelihood model will be used.
        verbosity :
            OPTIONAL int with default 1. Tunes the amount of command line 
            feedback, with more feedback at higher number. Currently meaningful 
            levels are 0 ... 3.
        job_prefix :
            A user-defined string that is printed as a preamble to most command 
            line output from this instance, meant as a means of keeping track 
            for example when you have large batches running in parallel
        labelling_efficiency_incomp_sampling :
            OPTIONAL bool with default False. Whether to consider incomplete 
            sampling-deviation of labelling statistics. Without this, an 
            incomplete sampling fit with incomplete labelling only allows 
            deviations in the particle number, but not the label stoichiometry.
            If True, also the label stoichiometry is varied. CAVE: 
            Computationally extremely expensive!

        '''
        # Acquisition metadata and technical settings
        
        if type(job_prefix) == str:
            self.job_prefix = job_prefix
        else:
            raise ValueError('job_prefix must be string. Got {job_prefix}')
        
        # This parameter is actually currently unused...Whatever.
        if utils.isfloat(FCS_psf_width_nm):
            if FCS_psf_width_nm > 0:
                self.FCS_psf_width_nm = FCS_psf_width_nm
                self.FCS_possible = True
            else:
                raise ValueError(f'[{self.job_prefix}] FCS_psf_width_nm must be float > 0. Got {FCS_psf_width_nm}')
        elif utils.isempty(FCS_psf_width_nm):
            self.psf_width = None
            self.FCS_possible = False
        else:
            raise ValueError(f'[{self.job_prefix}] FCS_psf_width_nm must be float > 0. Got {FCS_psf_width_nm}')


        if utils.isfloat(FCS_psf_aspect_ratio):
            if FCS_psf_aspect_ratio > 0:
                self.FCS_psf_aspect_ratio = FCS_psf_aspect_ratio
                self.FCS_possible = True
            else:
                raise ValueError(f'[{self.job_prefix}] FCS_psf_aspect_ratio must be float > 0. Got {FCS_psf_aspect_ratio}')
        elif utils.isempty(FCS_psf_aspect_ratio):
            self.FCS_psf_aspect_ratio = None
            self.FCS_possible = False
        else:
            raise ValueError(f'[{self.job_prefix}] FCS_psf_aspect_ratio must be float > 0. Got {FCS_psf_aspect_ratio}')


        if utils.isfloat(PCH_Q):
            if PCH_Q > 0:
                self.PCH_Q = PCH_Q
                self.PCH_possible = True
            else:
                raise ValueError(f'[{self.job_prefix}] PCH_Q must be float > 0. Got {PCH_Q}')
        elif utils.isempty(PCH_Q):
            self.PCH_Q = None
            self.PCH_possible = False
        else:
            raise ValueError(f'[{self.PCH_Q}] FCS_psf_aspect_ratio must be float > 0. Got {PCH_Q}')


        if utils.isfloat(acquisition_time_s):
            if acquisition_time_s > 0:
                self.acquisition_time_s = acquisition_time_s
                self.incomplete_sampling_possible = True
            else: 
                raise ValueError(f'[{self.job_prefix}] acquisition_time_s must be empty value or float > 0. Got {acquisition_time_s}')
                
        elif utils.isempty(acquisition_time_s):
            self.acquisition_time_s = None
            self.incomplete_sampling_possible = False
            
        else:
            raise ValueError(f'[{self.job_prefix}] acquisition_time_s must be empty value or float > 0. Got {acquisition_time_s}')
    
    
        if utils.isfloat(numeric_precision) and numeric_precision > 0. and numeric_precision < 1.:
            self.numeric_precision = numeric_precision
            self.precision_incremental = False
            
        elif utils.isiterable(numeric_precision):
            # Convert to array and sort so that we start with coarsest precision and increment towards finer parameters
            numeric_precision = np.sort(np.array(numeric_precision))[::-1]
            if np.all(numeric_precision > 0.) and np.all(numeric_precision < 1.):
                self.numeric_precision = numeric_precision
                self.precision_incremental = True
            else:
                raise ValueError(f'[{self.job_prefix}] numeric_precision must be float with 0 < numeric_precision < 1., or array of such float. Got {numeric_precision}')
        else:
            raise ValueError(f'[{self.job_prefix}] numeric_precision must be float with 0 < numeric_precision < 1., or array of such float. Got {numeric_precision}')


        if utils.isint(verbosity):
            self.verbosity = verbosity
            
        else:
            raise ValueError(f'[{self.job_prefix}] verbosity must be int. Got {verbosity}')
            
            
        # FCS input data to be fitted
        if utils.isempty(data_FCS_tau_s) or not self.FCS_possible:
            self.data_FCS_tau_s = None  
            self.FCS_possible = False
        elif utils.isiterable(data_FCS_tau_s):
            self.data_FCS_tau_s = np.array(data_FCS_tau_s)
            self.FCS_possible = True
        else:
            raise ValueError(f'[{self.job_prefix}] data_FCS_tau_s must be array (or can be left empty for PCH only). Got {data_FCS_tau_s}')
            
            
        if utils.isempty(data_FCS_G) or not self.FCS_possible:
            self.data_FCS_G = None
            self.FCS_possible = False
        elif utils.isiterable(data_FCS_G):
            data_FCS_G = np.array(data_FCS_G)
            if data_FCS_G.shape[0] == data_FCS_tau_s.shape[0]:
                self.data_FCS_G = data_FCS_G
            else:
                raise ValueError(f'[{self.job_prefix}] data_FCS_G must be array of same length as data_FCS_tau_s (or can be left empty for PCH only). Got {data_FCS_G}')
        else:
            raise ValueError(f'[{self.job_prefix}] data_FCS_G must be array of same length as data_FCS_tau_s (or can be left empty for PCH only). Got {data_FCS_G}')


        if utils.isempty(data_FCS_sigma) or not self.FCS_possible:
            self.data_FCS_sigma = None
            self.FCS_possible = False
        elif utils.isiterable(data_FCS_sigma):
            data_FCS_sigma = np.array(data_FCS_sigma)
            if data_FCS_sigma.shape[0] == data_FCS_tau_s.shape[0]:
                self.data_FCS_sigma = data_FCS_sigma
            else:
                raise ValueError(f'[{self.job_prefix}] data_FCS_sigma must be array of same length as data_FCS_tau_s (or can be left empty for PCH only). Got {data_FCS_sigma}')
        else:
            raise ValueError(f'[{self.job_prefix}] data_FCS_sigma must be array of same length as data_FCS_tau_s (or can be left empty for PCH only). Got {data_FCS_sigma}')


        # PC(M)H input data to be fitted
        if utils.isempty(data_PCH_bin_times) or not self.PCH_possible:
            self.data_PCH_bin_times = None  
            self.PCH_possible = False
        elif utils.isiterable(data_PCH_bin_times) or utils.isfloat(data_PCH_bin_times):
            self.data_PCH_bin_times = np.array(data_PCH_bin_times)
            self.PCH_possible = True
        else:
            raise ValueError(f'[{self.job_prefix}] data_PCH_bin_times must be float or array (or can be left empty for FCS only). Got {data_PCH_bin_times}')


        if utils.isempty(data_PCH_hist) or not self.PCH_possible:
            self.data_PCH_hist = None
            self.PCH_possible = False
            self.PCH_n_photons_max = 0
            self.data_PCH_sigma = None
            
        elif utils.isiterable(data_PCH_hist):
            data_PCH_hist = np.array(data_PCH_hist)
            if data_PCH_hist.shape[1] == data_PCH_bin_times.shape[0]:
                self.data_PCH_hist = data_PCH_hist
                self.PCH_n_photons_max = np.array([np.nonzero(data_PCH_hist[:, i_bin_time])[0][-1] + 1 for i_bin_time in range(data_PCH_bin_times.shape[0])])
                
                # Also get uncertainty estimate for least-squares fitting - not used in high-accuracy MLE fitting mode
                data_PCH_sigma = np.zeros_like(data_PCH_hist)
                data_PCH_norm = data_PCH_hist / data_PCH_hist.sum(axis=0)
                data_PCH_max_counts = data_PCH_hist.max(axis=0)
                for i_bin_time in range(data_PCH_bin_times.shape[0]):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data_PCH_sigma[:,i_bin_time] = np.where(data_PCH_hist[:,i_bin_time] > 0,
                                                                np.sqrt(data_PCH_hist[:,i_bin_time] * (1 - data_PCH_norm[:,i_bin_time])),
                                                                data_PCH_max_counts[i_bin_time])
                self.data_PCH_sigma = data_PCH_sigma
                
            else:
                raise ValueError(f'[{self.job_prefix}] data_PCH_hist must be array with axis 1 same length as same length as data_PCH_bin_times (or can be left empty for FCS only). Got {data_PCH_hist}')
        else:
            raise ValueError(f'[{self.job_prefix}] data_PCH_hist must be array with axis 1 same length as same length as data_PCH_bin_times (or can be left empty for FCS only). Got {data_PCH_hist}')

        if utils.isfloat(data_avg_count_rate):
            if data_avg_count_rate > 0:
                self.data_avg_count_rate = np.array([data_avg_count_rate])
                self.avg_count_rate_given = True
                                
            elif data_avg_count_rate == 0:
                self.data_avg_count_rate = np.array([0.])
                self.avg_count_rate_given = False
                
            else: # data_average_count_rate < 0
                raise ValueError(f'[{self.job_prefix}] data_avg_count_rate must be float >= 0. Got {data_avg_count_rate}')
                
        else:
            raise ValueError(f'[{self.job_prefix}] data_avg_count_rate must be float >= 0. Got {data_avg_count_rate}')

        if type(NLL_funcs_accurate) == bool:
            self.NLL_funcs_accurate = NLL_funcs_accurate
        else:
            raise ValueError(f'[{self.job_prefix}] NLL_funcs_accurate must be bool (ignored if no PCH is loaded). Got {NLL_funcs_accurate}')
        
        if utils.isfloat(labelling_efficiency) and labelling_efficiency > 0. and labelling_efficiency <= 1.:
            self.labelling_efficiency = labelling_efficiency
        else:
            raise ValueError(f'[{self.job_prefix}] labelling_efficiency must be float with 0 < labelling_efficiency <= 1. Got {labelling_efficiency}')

        if type(labelling_efficiency_incomp_sampling) == bool:
            self.labelling_efficiency_incomp_sampling = labelling_efficiency_incomp_sampling
        else:
            raise ValueError(f'[{self.job_prefix}] labelling_efficiency_incomp_sampling must be bool. Got {labelling_efficiency_incomp_sampling}')

            
            
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
    def wlc_tau_diff_fold_change(j,
                                 r_mono,
                                 l_Kuhn,
                                 ):
        l_contour = j * 2 * r_mono 
        msd_fold_change = l_Kuhn * l_contour - l_Kuhn**2 / 2. * (1 - np.exp(-2* l_contour / l_Kuhn))
        return np.sqrt(msd_fold_change)

    @staticmethod
    def helical_wlc_tau_diff_fold_change(j,
                                         helix_radius,
                                         helix_pitch,
                                         r_mono,
                                         l_Kuhn,
                                         ):
        
        kappa_wlc = (4*l_Kuhn*helix_radius*np.pi**2)/(helix_pitch**2 + 4*np.pi**2*helix_radius**2)
        tau_wlc = (2*helix_pitch*l_Kuhn*np.pi)/(helix_pitch**2 + 4*np.pi**2*helix_radius**2)
        c_inf = (4 + tau_wlc**2) / (4 + kappa_wlc**2 + tau_wlc**2)
        l_contour = j * 2 * r_mono / l_Kuhn # not real contour length, but normalized to Kuhn segment length
        nu = np.sqrt(kappa_wlc**2 + tau_wlc**2)
        msd_fold_change = c_inf * l_contour - 0.5 * (tau_wlc/nu)**2 - 2 * (kappa_wlc/nu)**2 * (4 - nu**2) / (4+nu**2)**2 + np.exp(-2*l_contour) * (0.5* (tau_wlc / nu)**2 + 2 * (kappa_wlc/nu)**2 * ((4 - nu**2) * np.cos(nu*l_contour) - 4 * nu * np.sin(nu*l_contour)) / (4 + nu**2)**2)
        return np.sqrt(msd_fold_change)
    

    @staticmethod
    def get_n_species(params):
        '''
        Brute-force method to figure out number of species in a multi-species 
        lmfit.Parameters object as used in this class. Not an efficient way to
        do it I guess, but used to avoid explicitly passing the number of 
        species through the code stack, making things more robust in coding. 
        Or that's the idea at least.
        '''
        
        n_species_spec = 0
        n_species_disc = 0
        
        while f'cpms_{n_species_spec}' in params.keys():
            n_species_spec += 1
            
        while f'cpms_d_{n_species_disc}' in params.keys():
            n_species_disc += 1
            
        return n_species_spec, n_species_disc


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

    @staticmethod
    def invgamma_helper(mean, a):
        # Helper function for brute-force inversion of an awkward gamma function 
        # construct that can appear when using the stretched-exponential particle 
        # number distribution model
        return sminimize_scalar(lambda x: sspecial.loggamma(1/x) - np.log(mean * a * x),
                                bounds = (1E-6, 1E3)).x
    #%% Regularized minimization code
    
    def regularized_minimization_fit(self,
                                     params,
                                     use_FCS,
                                     use_PCH,
                                     time_resolved_PCH,
                                     spectrum_type,
                                     spectrum_parameter,
                                     labelling_correction,
                                     incomplete_sampling_correction,
                                     i_bin_time = 0,
                                     numeric_precision = 1e-4,
                                     N_pop_array = None,
                                     mp_pool = None,
                                     use_avg_count_rate = False
                                     ):
                    
        
        verbose = self.verbosity > 1
        
        # Read out iniital parameters and set up fit
        n_species_spec, _ = self.get_n_species(params)
                    
        stoichiometry_array = np.array([params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species_spec)]).astype(np.float64)
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)]).astype(np.float64)
        labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        
        # Do we need to optimize other parameters besides those tuned through MEM algorithm?
        has_other_params = np.any(np.array([params[key].vary for key in params.keys()]))
        
        # Initialize amplitudes and construct  N_population array for model functions as attribute 
        if not utils.isiterable(N_pop_array):
            if N_pop_array == None:
                # Initalize from nothing
                if incomplete_sampling_correction:
                    N_avg_obs_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
                else:
                    N_avg_obs_array = stoichiometry_array**(-2)
                    
                if verbose:
                    print(f'[{self.job_prefix}] Initializing default amp_array for {n_species_spec} species as N_pop_array is empty')
                if spectrum_parameter  == 'Amplitude':
                    amp_array = N_avg_obs_array * stoichiometry_binwidth_array * stoichiometry_array**2 * (1 + (1 - labelling_efficiency_array) / stoichiometry_array / labelling_efficiency_array)
                elif spectrum_parameter == 'N_monomers':
                    amp_array = N_avg_obs_array * stoichiometry_array * stoichiometry_binwidth_array
                else: # spectrum_parameter == 'N_oligomers'
                    amp_array = N_avg_obs_array * stoichiometry_binwidth_array

                self._N_pop_array = N_avg_obs_array

        else:
            if np.all(N_pop_array >= 0.) and np.sum(N_pop_array) > 0. :
                # We have an initial array
                if verbose:
                    print(f'[{self.job_prefix}] Initializing amp_array with {n_species_spec} species from pre-initialized N_pop_array')
                if spectrum_parameter  == 'Amplitude':
                    amp_array = N_pop_array * stoichiometry_binwidth_array * stoichiometry_array**2 * (1 + (1 - labelling_efficiency_array) / stoichiometry_array / labelling_efficiency_array)
                elif spectrum_parameter == 'N_monomers':
                    amp_array = N_pop_array * stoichiometry_array * stoichiometry_binwidth_array
                else: # spectrum_parameter == 'N_oligomers'
                    amp_array = N_pop_array * stoichiometry_binwidth_array
                self._N_pop_array = N_pop_array
            else:
                raise Exception(f'[{self.job_prefix}] Invalid N_pop_array: Must be None, or np.array with non-negative initial estimates, got {type(N_pop_array)}')
        
        # Enforce positive-nonzero values
        amp_array_nonzeros = amp_array > 0
        n_amp_array_nonzeros = np.sum(amp_array_nonzeros)
        amp_array = np.where(amp_array_nonzeros,
                             amp_array,
                             amp_array[amp_array_nonzeros].min() if n_amp_array_nonzeros > 0 else 1E-6)
        amp_array /= amp_array.sum() # Renormalize
        if verbose:
            print(f'Initial amp_array: [{amp_array}]')

        # Ensure integer iteration count limits
        _REG_MAX_ITER_INNER = int(REG_MAX_ITER_INNER)
        _REG_MAX_ITER_OUTER = int(REG_MAX_ITER_OUTER)

        # Sanity-check reg_method and redefine to quicker-evaluated bool: 
        if spectrum_type == 'reg_MEM':
            _reg_method = True
        elif spectrum_type == 'reg_CONTIN':
            _reg_method = False
        else:
            raise Exception(f'[{self.job_prefix}] Invalid spectrum_type: Must be "MEM" or "CONTIN", got {spectrum_type}')


        # Declare some variables...
        
        # But we also need the transpose of self.tau__tau_diff_array for some matrix 
        # manipulation, which we also pre-calculate to avoid shuffling stuff around too much
        tauD__tau_array = np.transpose(self.tau__tau_diff_array)
        
        # iteration settings
            
        # Initial value of Lagrange multiplier
        lagrange_mul = 20
        lagrange_mul_array = np.zeros(REG_MAX_ITER_OUTER)
        NLL_array_outer = np.zeros(REG_MAX_ITER_OUTER)
        lagrange_mul_del_old = 1
        iterator_outer = 1


        while True:
            # Outer iteration is an iterative tuning of the lagrange multiplier,
            # i.e., the regularization weight
            
            if verbose:
                print(f'[{self.job_prefix}] Outer loop iteration {iterator_outer}: lagrange_mul = {lagrange_mul}')

            NLL_array_inner = np.zeros(_REG_MAX_ITER_INNER)
            
            # lmfit minimizer for other parameters
            # We create a new instance with every outer iteration to ensure 
            # regular reset of the function evaluation count of the Minimizer 
            # object, which can otherwise apparently overflow and cause issues
            
            fitter = lmfit.Minimizer(self.negloglik_global_fit, 
                                      params, 
                                      fcn_args = (use_FCS, 
                                                  use_PCH, 
                                                  time_resolved_PCH,
                                                  spectrum_type, 
                                                  spectrum_parameter,
                                                  labelling_correction, 
                                                  incomplete_sampling_correction
                                                  ),
                                      fcn_kws = {'i_bin_time': i_bin_time,
                                                'numeric_precision': numeric_precision,
                                                'mp_pool': mp_pool,
                                                'use_avg_count_rate': use_avg_count_rate},
                                      nan_policy = 'propagate',
                                      calc_covar = False)
            
            # Initial values
            iterator_inner = 0
            N_pop_tot_del = 1. 
            old_amp_increment = 0.
            while True:
                # Each inner iteration is a cycle of three optimizations:
                # 1. Single scalar optimization (until converged) of total particle number based 
                #    on correlation function alone
                # 2. MEM-/ or CONTIN-regularized single iteration of spectrum 
                #    amplitudes, i.e. species-wise particle numbers
                # 3. lmfit/scipy-based MLE increment (1 iteration) of "other" parameters 
                #    not affected by regularization, which can involve any of
                #    the likelihood function terms implemented in this class,
                #    also PC(M)H
                
                ### Step 1: Total particle number
                # We do this on population-level particle numbers, not observation-level
                
                if incomplete_sampling_correction:
                    # Recalculate if needed
                    labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
                
                temp_n_array = self._N_pop_array / np.sum(self._N_pop_array)
                
                w_array = temp_n_array * stoichiometry_binwidth_array * stoichiometry_array**2 * (1 + (1 - labelling_efficiency_array) / stoichiometry_array / labelling_efficiency_array)* 2**(-3/2) 
                w_array /= np.sum(temp_n_array * stoichiometry_binwidth_array * stoichiometry_array)**2
                
                N_pop_tot_del = sminimize_scalar(lambda N_pop_total: np.sum(((np.dot(self.tau__tau_diff_array, 
                                                                                     w_array / N_pop_tot_del)
                                                                                + params['acf_offset'].value - self.data_FCS_G) / self.data_FCS_sigma)**2)).x
                self._N_pop_array = temp_n_array * N_pop_tot_del
                
                if not incomplete_sampling_correction:
                    # if we do not have incomplete sampling correction, we must explicitly update this thing
                    for i_spec in range(n_species_spec):
                        params[f'N_avg_obs_{i_spec}'].value = self._N_pop_array[i_spec]                
                
                ### Step 2 (which is far more code than 1 and 3): Species amplitudes
                # Recalculate explicit FCS model, and from that weighted residuals
                # on population statistics alone
                G_fit = np.dot(self.tau__tau_diff_array, 
                               w_array / N_pop_tot_del)
                
                weighted_residual = (G_fit + params['acf_offset'].value - self.data_FCS_G) / self.data_FCS_sigma
                
                # Calculation of regularization terms
                # These actually used only implicitly, so we only show them for reference
                # if _reg_method:
                #     # Entropy (to be maximized)
                #     S = -np.sum(amp_array * np.log(amp_array))
                # else:
                #     # CONTIN -> Squared second derivative (to be minimized)
                #     # One caveat here: CONTIN regularization, differnet from MEM, 
                #     # requires consideration of local neighborhood around each 
                #     # data point. To ensure that this is defined for data points 
                #     # at the edges, we implicitly assume a periodicity in amp_array.
                #     S = np.sum((2 * amp_array - np.roll(amp_array, -1) - np.roll(amp_array, +1))**2)

                # Gradients
                # First derivative of least-squares with amplitude
                chisq_gradient = np.mean(2 * weighted_residual * tauD__tau_array / self.data_FCS_sigma, 
                                         axis = 1) * 2**(-3/2)
                
                # Recalculate gradient with amplitude to gradient with 
                # N_monomers or N_oligomers, and according to labelling efficiency if needed
                # Full disclosure: Some of the math here is actually based on calculations
                # I did with pen and paper, some of it (especially the *= or /= lines) was found 
                # to be necessary by trial and error and I can't really explain them even if I want to...
                if labelling_correction:
                    label_factor = 1 + (1 - labelling_efficiency_array) / labelling_efficiency_array / stoichiometry_array
                else:
                    label_factor = 1.
                    
                if spectrum_parameter == 'Amplitude':
                    chisq_gradient *= label_factor
                
                elif spectrum_parameter == 'N_monomers':
                    x = self._N_pop_array * stoichiometry_array * stoichiometry_binwidth_array
                    weight = stoichiometry_array * label_factor
                    den = np.sum(x)
                    gradient_conversion_factor = weight / den**2 - 2 * np.sum(x*weight) / den**3
                    # gradient_conversion_factor = weight * label_factor / den**2 - 2 * np.sum(x*weight) / den**3
                    gradient_conversion_factor /= stoichiometry_binwidth_array
                    chisq_gradient *= gradient_conversion_factor
                    
                elif spectrum_parameter == 'N_oligomers':
                    x = self._N_pop_array * stoichiometry_binwidth_array
                    weight = stoichiometry_array**2 * label_factor
                    den = np.sum(x*stoichiometry_array)
                    gradient_conversion_factor = weight / den**2 - 2 * stoichiometry_array * np.sum(x*weight) / den**3
                    # gradient_conversion_factor /= stoichiometry_binwidth_array * stoichiometry_array
                    chisq_gradient *= gradient_conversion_factor
                    
                # first derivative of entropy/CONTIN 
                if _reg_method:
                    # Maximum entropy gradient
                    S_gradient = - 1 - np.log(amp_array)
                else:
                    # CONTIN gradient (inverted to match entropy gradient handling)
                    S_gradient = - (12 * amp_array 
                                   - 8 * (np.roll(amp_array, -1) + np.roll(amp_array, +1))
                                   + 2 * (np.roll(amp_array, -2) - np.roll(amp_array, +2))
                                   )

                # Scaling factor from Euclician norms of gradients
                chisq_grad_length = np.sqrt(np.sum(chisq_gradient**2))
                S_grad_length = np.sqrt(np.sum(S_gradient**2))
                alpha_f = chisq_grad_length / S_grad_length / lagrange_mul
                
                # search direction construct
                e_G = alpha_f * S_gradient - chisq_gradient / 2 # del Q
        
                # update amp_array
                amp_increment = amp_array * e_G * REG_GRADIENT_SCALING
                amp_array += amp_increment + REG_GRAD_MOMENTUM * old_amp_increment
                
                # Store amp_increment for next iteration for fitting with momentum
                old_amp_increment = np.copy(amp_increment)
                # Enforce positive-nonzero values, and renormalize
                amp_array_nonzeros = amp_array > 0
                n_amp_array_nonzeros = np.sum(amp_array_nonzeros)
                amp_array = np.where(amp_array_nonzeros,
                                     amp_array,
                                     amp_array[amp_array_nonzeros].min() if n_amp_array_nonzeros > 0 else 1E-6)
                amp_array /= amp_array.sum() # Renormalize
                
                # Update self._N_pop_array from amplitudes array
                # amp_array -> temp_n_array -> noramlize -> scale to real N
                if incomplete_sampling_correction:
                    # Recalculate if needed
                    labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
                    
                if spectrum_parameter  == 'Amplitude':
                    temp_n_array = amp_array / (stoichiometry_binwidth_array * stoichiometry_array**2 * (1 + (1 - labelling_efficiency_array) / stoichiometry_array / labelling_efficiency_array))
                elif spectrum_parameter == 'N_monomers':
                    temp_n_array = amp_array / stoichiometry_array / stoichiometry_binwidth_array
                else: # spectrum_parameter == 'N_oligomers'
                    temp_n_array = amp_array / stoichiometry_binwidth_array
                temp_n_array /= temp_n_array.sum() 
                
                
                
                # Scaling factor for absolute N needs to be recalculated
                # There might be a simpler solution to that, but I am going for the
                # brute-force approach here...
                w_array = temp_n_array * stoichiometry_binwidth_array * stoichiometry_array**2 * (1 + (1 - labelling_efficiency_array) / stoichiometry_array / labelling_efficiency_array)* 2**(-3/2) 
                w_array /= np.sum(temp_n_array * stoichiometry_binwidth_array * stoichiometry_array)**2
                
                N_pop_tot_del = sminimize_scalar(lambda N_pop_total: np.sum(((np.dot(self.tau__tau_diff_array, 
                                                                                     w_array / N_pop_tot_del)
                                                                              + params['acf_offset'].value - self.data_FCS_G) / self.data_FCS_sigma)**2)).x
                self._N_pop_array = temp_n_array * N_pop_tot_del
                
                if not incomplete_sampling_correction:
                    # if we do not have incomplete sampling correction, we must explicitly update this thing
                    for i_spec in range(n_species_spec):
                       params[f'N_avg_obs_{i_spec}'].value = self._N_pop_array[i_spec]



                
                ### Step 3: MLE iteration of other parameters 
                if has_other_params:
                    # Update other parameters
                    minimization_result = fitter.minimize(method = 'nelder',
                                                          params = params,
                                                          max_nfev = 1).params
                    # Also write into params variable for next iteration
                    params = minimization_result
                    
                    if incomplete_sampling_correction:
                        # If and only if we have incomplete labelling correction, we have to recalculate the spectrum here
                        labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
                        if spectrum_parameter  == 'Amplitude':
                            temp_n_array = amp_array / (stoichiometry_binwidth_array * stoichiometry_array**2 * (1 + (1 - labelling_efficiency_array) / stoichiometry_array / labelling_efficiency_array))
                        elif spectrum_parameter == 'N_monomers':
                            temp_n_array = amp_array / stoichiometry_binwidth_array / stoichiometry_array
                        else: # spectrum_parameter == 'N_oligomers'
                            temp_n_array = amp_array / stoichiometry_binwidth_array
                            
                        temp_n_array /= temp_n_array.sum() 
                        w_array = temp_n_array * stoichiometry_binwidth_array * stoichiometry_array**2 * (1 + (1 - labelling_efficiency_array) / stoichiometry_array / labelling_efficiency_array)* 2**(-3/2) 
                        w_array /= np.sum(temp_n_array * stoichiometry_binwidth_array * stoichiometry_array)**2
                        
                        N_pop_tot_del = sminimize_scalar(lambda N_pop_total: np.sum(((np.dot(self.tau__tau_diff_array, 
                                                                                             w_array / N_pop_tot_del)
                                                                                        + params['acf_offset'].value - self.data_FCS_G) / self.data_FCS_sigma)**2)).x
                        self._N_pop_array = temp_n_array * N_pop_tot_del

                    
                else: # not has_other_params
                    # Just get the minimization result as a dummy variable
                    minimization_result = params

                                
                # Current chi-square as goodness of fit
                NLL_array_inner[iterator_inner] = np.sum(weighted_residual**2) / (weighted_residual.shape[0] - amp_array.shape[0] - 1)
                            
                    
                # Once in a while, check for convergence 
                if (iterator_inner + 1) % 500 == 0 and iterator_inner > 1:
                    
                    if (iterator_inner + 1) >= _REG_MAX_ITER_INNER:
                        # Iteration limit hit
                        if verbose:
                            print(f'[{self.job_prefix}] Stopping inner loop after {iterator_inner + 1} iterations (iteration limit), current NLL: {NLL_array_inner[iterator_inner]}')
                        break
                    
                    # Check if gradients for S and chi-square are parallel, which they
                    # should be at the optimal point for a given Lagrange multiplier
                    S_direction = S_gradient / S_grad_length                        
                    NLL_direction = chisq_gradient / chisq_grad_length
                    test_stat = 0.5 * np.sum((S_direction - NLL_direction)**2)

                    if test_stat < REG_CONV_THRESH_INNER: 
                        # gradients approximately parallel - stop inner loop
                        if verbose:
                            print(f'[{self.job_prefix}] Stopping inner loop after {iterator_inner + 1} iterations (converged), current NLL: {NLL_array_inner[iterator_inner]}')
                        break

                ### Inner iteration done: on to next
                iterator_inner +=1 
                
            if verbose:
                # Plot FCS fit
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.semilogx(self.data_FCS_tau_s,
                            self.data_FCS_G, 
                            'dk')
                ax.semilogx(self.data_FCS_tau_s, 
                            self.data_FCS_G + self.data_FCS_sigma,
                            '-k', 
                            alpha = 0.7)
                ax.semilogx(self.data_FCS_tau_s, 
                            self.data_FCS_G - self.data_FCS_sigma,
                            '-k', 
                            alpha = 0.7)
                ax.semilogx(self.data_FCS_tau_s,
                            np.dot(self.tau__tau_diff_array, 
                                   w_array / N_pop_tot_del) + params['acf_offset'].value, 
                            marker = '',
                            linestyle = '-', 
                            color = 'tab:gray')
                plt.show()

                
            # Inner iteration stopped - check for globally optimal solution
            # We check if chi-square is not within tolerance region

            if iterator_outer >= _REG_MAX_ITER_OUTER:
                # Iteration limit hit 
                if verbose:
                    print(f'[{self.job_prefix}] Stopping outer loop after {iterator_outer} iterations (iteration limit)')
                break
            
            NLL_array_outer[iterator_outer] = NLL_array_inner[iterator_inner]
            if NLL_array_outer[iterator_outer] < 1 or NLL_array_outer[iterator_outer] > 1.3: 
                # Over- or underfit: Update Lagrange multiplier!
                # The way it is defined, high lagrange_mul leads to high weight for 
                # chi-square gradients. 
                    # So, we reduce lagrange_mul if we had on overfit (too-low chi-sq),
                    # and we increase lagrange_mul if we had an underfit
                # For this, we use the ratio S_grad_length / chisq_grad_length:
                    # High S_grad_length/chisq_grad_length implies that we had too much weight 
                    # on chi-square (too-high lagrange_mul), and vice versa, so
                    # we change lagrange_mul based on that ratio (nonlinearly, and 
                    # with a bit of a gradient boost)
                lagrange_mul_del_new = np.sqrt(chisq_grad_length/S_grad_length)
                                
                lagrange_mul *= lagrange_mul_del_new * lagrange_mul_del_old**(REG_LAGRANGE_MUL_MOMENTUM)
                
                # Bounds for lagrange_mul
                if lagrange_mul > REG_MAX_LAGRANGE_MUL:
                    lagrange_mul = REG_MAX_LAGRANGE_MUL
                elif lagrange_mul < 1/REG_MAX_LAGRANGE_MUL:
                    lagrange_mul = 1/REG_MAX_LAGRANGE_MUL
                    
                lagrange_mul_array[iterator_outer] = lagrange_mul
                
                # Another stop criterion comes in here: If neither lagrange_mul 
                # nor chi-square have really changed for three iterations
                if iterator_outer >= 3:
                    
                    lagrange_mul_recent = lagrange_mul_array[iterator_outer-2:iterator_outer+1]
                    lagrange_mul_recent_rel_span = (np.max(lagrange_mul_recent) - np.min(lagrange_mul_recent)) / np.mean(lagrange_mul_recent)

                    NLL_recent = NLL_array_outer[iterator_outer-2:iterator_outer+1]
                    NLL_recent_rel_span = (np.max(NLL_recent) - np.min(NLL_recent)) / np.mean(NLL_recent)
                    
                    if lagrange_mul_recent_rel_span < REG_CONV_THRESH_OUTER and NLL_recent_rel_span < REG_CONV_THRESH_OUTER:
                        if verbose:
                            print(f'[{self.job_prefix}] Stopping outer loop after {iterator_outer} iterations (no longer changing)')
                        break
                
                lagrange_mul_del_old = np.copy(lagrange_mul_del_new)
                iterator_outer += 1
            
            else:
                # Convergence criterion hit - stop outer loop
                if verbose:
                    print(f'[{self.job_prefix}] Stopping outer loop after {iterator_outer} iterations (converged)')
                break
        

        return minimization_result, self._N_pop_array, lagrange_mul

    
    #%% Penalty terms in fitting
    def negloglik_incomplete_sampling_full_labelling_old(self,
                                                         params,
                                                         spectrum_type):
        '''
        Neg log likelihood function for deviation between population-level
        and observed N, but without treatment of labelling statistics
        '''
        
        # Unpack parameters
        n_species_spec, _ = self.get_n_species(params)
        
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_obs_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        
        if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            N_avg_pop_array = self._N_pop_array
        else:
            N_avg_pop_array = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species_spec)])
            
        # Likelihood function for particle numbers
        n_pop = self.n_total_from_N_avg(N_avg_pop_array, 
                                        tau_diff_array)

        n_obs = self.n_total_from_N_avg(N_avg_obs_array, 
                                        tau_diff_array)
        if self.NLL_funcs_accurate:
            # Poisson likelihood
            negloglik = self.negloglik_poisson_full(rates = n_pop,
                                                    observations = n_obs,
                                                    scale = stoichiometry_binwidth_array)
        else:
            # wlsq approximation
            negloglik = 0.5 * np.sum((n_pop - n_obs)**2 / np.where(n_pop > 0,
                                                                   n_pop,
                                                                   np.min(n_pop[n_pop > 0])) * stoichiometry_binwidth_array)
            
        # Normalize by number of species as "pseudo-datapoints"
        negloglik /= n_species_spec
        return negloglik


    def negloglik_incomplete_sampling_static_labelling(self,
                                                       params,
                                                       spectrum_type,
                                                       labelling_correction = False):
        '''
        Neg log likelihood function for deviation between population-level
        and observed N, but without treatment of labelling efficiency variations
        (but WITH treatment of how a limited fixed labelled fraction changes
        likelihood function)
        '''
        
        # Unpack parameters
        n_species_spec, _ = self.get_n_species(params)
        
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_obs_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        
        if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            N_avg_pop_array = self._N_pop_array
        else:
            N_avg_pop_array = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species_spec)])
            
        # Likelihood function for particle numbers
        n_pop = self.n_total_from_N_avg(N_avg_pop_array, 
                                        tau_diff_array)

        n_obs = self.n_total_from_N_avg(N_avg_obs_array, 
                                        tau_diff_array)
        
        
        if labelling_correction:
            negloglik = 0.
            labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
            
            for i_spec in range(n_species_spec):
                label_fractions = sstats.binom.pmf(k = np.arange(1, i_spec + 1),
                                                    n = i_spec + 1, 
                                                    p = labelling_efficiency_array[i_spec])
                n_pop_spec_label = n_pop[i_spec] * label_fractions
                n_obs_spec_label = n_obs[i_spec] * label_fractions
                
                if self.NLL_funcs_accurate:
                    # Poisson likelihood
                    negloglik += self.negloglik_poisson_full(rates = n_pop_spec_label,
                                                              observations = n_obs_spec_label,
                                                              scale = stoichiometry_binwidth_array[i_spec])
                else:
                    # wlsq approximation
                    negloglik += 0.5 * np.sum((n_pop_spec_label - n_obs_spec_label)**2 / np.where(n_pop_spec_label > 1E-50,
                                                                                                  n_pop_spec_label,
                                                                                                  # np.min(n_pop_spec_label[n_pop_spec_label > 1E-50]))) * stoichiometry_binwidth_array[i_spec]
                                                                                                  np.max(n_pop_spec_label))) * stoichiometry_binwidth_array[i_spec]
                    
                    
        else:
            if self.NLL_funcs_accurate:
                # Poisson likelihood
                negloglik = self.negloglik_poisson_full(rates = n_pop,
                                                        observations = n_obs,
                                                        scale = stoichiometry_binwidth_array)
            else:
                # wlsq approximation
                negloglik = 0.5 * np.sum((n_pop - n_obs)**2 / np.where(n_pop > 1E-50,
                                                                       n_pop,
                                                                       np.max(n_pop)) * stoichiometry_binwidth_array)
            
            
        # Normalize by number of species as "pseudo-datapoints"
        negloglik /= n_species_spec
        
        return negloglik
    
    
    def negloglik_incomplete_sampling_variable_labelling(self,
                                                         params,
                                                         spectrum_type,
                                                         numeric_precision = 1e-4):
        '''
        Neg log likelihood function for deviation between population-level
        and observed N and labelling statistics
        '''
        n_species_spec, _ = self.get_n_species(params)
        
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_obs_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        labelling_efficiency_obs_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        labelling_efficiency_pop = params['Label_efficiency'].value
        stoichiometry_array = np.array([params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species_spec)]).astype(np.float64)
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)]).astype(np.float64)

        if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            N_avg_pop_array = self._N_pop_array
        else:
            N_avg_pop_array = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species_spec)])
        
        # Likelihood function for particle numbers
        n_pop = self.n_total_from_N_avg(N_avg_pop_array * stoichiometry_binwidth_array, 
                                        tau_diff_array)
        
        n_obs = self.n_total_from_N_avg(N_avg_obs_array * stoichiometry_binwidth_array, 
                                        tau_diff_array)
        
        if self.NLL_funcs_accurate:
            # Poisson likelihood
            negloglik_N = self.negloglik_poisson_full(rates = n_pop,
                                                      observations = n_obs,
                                                      scale = stoichiometry_binwidth_array)
        else:
            # wlsq approximation
            negloglik_N = 0.5 * np.sum((n_pop - n_obs)**2 / np.where(n_pop > 0,
                                                                     n_pop,
                                                                     np.min(n_pop[n_pop > 0])) * stoichiometry_binwidth_array)
        
        # Here, we accumulate the negloglik iteratively over species
        negloglik_labelling = 0.
        
        # Likelihood function for labelling efficiency fluctuations
        # We iterate over species, and for each species calculate a likelihood 
        # for the observed labelling
        for i_spec in range(n_species_spec):
            
            # "Histogram" of "observed" labelling with "observed" N spectrum
            _, labelling_efficiency_array_spec_obs = self.pch_get_stoichiometry_spectrum(stoichiometry_array[i_spec], 
                                                                                         labelling_efficiency_obs_array[i_spec],
                                                                                         numeric_precision = numeric_precision)
            labelling_efficiency_array_spec_obs *= n_obs[i_spec]
            
            # Reference labelling statistics based on labelling efficiency metadata and "population" N spectrum
            _, labelling_efficiency_array_spec_pop = self.pch_get_stoichiometry_spectrum(stoichiometry_array[i_spec], 
                                                                                         labelling_efficiency_pop,
                                                                                         numeric_precision = numeric_precision)
            
            # Match array lengths
            if labelling_efficiency_array_spec_obs.shape[0] > labelling_efficiency_array_spec_pop.shape[0]:
                labelling_efficiency_array_spec_pop = np.append(labelling_efficiency_array_spec_pop, np.zeros(labelling_efficiency_array_spec_obs.shape[0] - labelling_efficiency_array_spec_pop.shape[0]))
            elif labelling_efficiency_array_spec_obs.shape[0] < labelling_efficiency_array_spec_pop.shape[0]:
                labelling_efficiency_array_spec_obs = np.append(labelling_efficiency_array_spec_obs, np.zeros(labelling_efficiency_array_spec_pop.shape[0] - labelling_efficiency_array_spec_obs.shape[0]))
                
            # Likelihood function for current estimate of "observed" labelling statistics given "population" frequencies
            if self.NLL_funcs_accurate:
                # Binomial likelihood
                negloglik_labelling_spec = self.negloglik_binomial_full(n_trials = n_obs[i_spec],
                                                                        k_successes = labelling_efficiency_array_spec_obs, 
                                                                        probabilities = labelling_efficiency_array_spec_pop,
                                                                        scale = stoichiometry_binwidth_array[i_spec])
            
            else:
                # wslq approximation
                labelling_efficiency_sigma = np.where(labelling_efficiency_array_spec_pop > 0,
                                                      np.sqrt(n_obs[i_spec] * labelling_efficiency_array_spec_pop * (1 - labelling_efficiency_array_spec_pop)),
                                                      n_obs[i_spec] * labelling_efficiency_array_spec_pop.max() * 1E6) 
                negloglik_labelling_spec = np.sum(((labelling_efficiency_array_spec_obs - labelling_efficiency_array_spec_pop * n_obs[i_spec]) / labelling_efficiency_sigma)**2)
            
            # Normalize by number of labelling "data points" 
            negloglik_labelling_spec /= labelling_efficiency_array_spec_obs.shape[0]
            
            negloglik_labelling += negloglik_labelling_spec

        # Combine neglogliks and normalize by number of species as "pseudo-datapoints"
        negloglik = (negloglik_N + negloglik_labelling) / np.sum(stoichiometry_binwidth_array)
        
        return negloglik
    
    
    def fcs_red_chisq_full_labelling(self, 
                                     params):
        '''
        Reduced chi-square for FCS fitting with full labelling - not actually 
        used as we need to count the DOF in more complicated manner.
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


    def fcs_chisq_full_labelling(self, 
                                 params):
        '''
        Chi-square without normalization for parameter number for FCS fitting 
        with full labelling.
        '''
                        
        # Get model
        acf_model = self.get_acf_full_labelling(params)
        
        # Calc weighted residual sum of squares
        wrss = np.sum(((acf_model - self.data_FCS_G) / self.data_FCS_sigma) ** 2)
                
        return wrss


    # def fcs_chisq_full_labelling_par(self, 
    #                                  params):
    #     '''
    #     Chi-square without normalization for parameter number for FCS fitting 
    #     with full labelling. unusued.
    #     '''
                        
    #     # Get model
    #     acf_model = self.get_acf_full_labelling_par(params)
        
    #     # Calc weighted residual sum of squares
    #     wrss = np.sum(((acf_model - self.data_FCS_G) / self.data_FCS_sigma) ** 2)
        
    #     # Return reduced chi-square
    #     red_chi_sq =  wrss / self.data_FCS_G.shape[0]
        
    #     return red_chi_sq


    def fcs_chisq_full_labelling_par_reg(self, 
                                         params,
                                         spectrum_type):
        '''
        Chi-square without normalization for parameter number for FCS fitting 
        with full labelling.
        '''
        
        # Get model
        if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            acf_model = self.get_acf_full_labelling_reg(params)
        else:
            acf_model = self.get_acf_full_labelling_par(params)
        
        # Calc weighted residual sum of squares
        wrss = np.sum(((acf_model - self.data_FCS_G) / self.data_FCS_sigma) ** 2)
                
        return wrss


    
    def fcs_red_chisq_partial_labelling(self, 
                                        params):
        '''
        Reduced chi-square for FCS fitting with partial labelling - not actually 
        used as we need to count the DOF in more complicated manner.
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


    def fcs_chisq_partial_labelling(self, 
                                    params):
        '''
        Chi-square without normalization for parameter number for FCS fitting with partial labelling.
        '''
                        
        # Get model
        acf_model = self.get_acf_partial_labelling(params)
        
        # Calc weighted residual sum of squares
        wrss = np.sum(((acf_model - self.data_FCS_G) / self.data_FCS_sigma) ** 2)
                
        return wrss


    def fcs_chisq_partial_labelling_par_reg(self, 
                                            params):
        '''
        Chi-square without normalization for parameter number for FCS fitting 
        with partial labelling. Version with shortcut for using pre-calculated
        correlation functions for parameterized distributions and regularized fitting.
        '''
                        
        # Get model
        acf_model = self.get_acf_partial_labelling_par_reg(params)
        
        # Calc weighted residual sum of squares
        wrss = np.sum(((acf_model - self.data_FCS_G) / self.data_FCS_sigma) ** 2)
                
        return wrss




    @staticmethod
    def negloglik_poisson_full(rates, 
                               observations,
                               scale = 1):
        '''
        Poisson neg log lik in fitting. Full model to allow "observations" 
        themselves to vary in a hierarchical model with a hidden Poisson process.
        "scale" is a handle for additional weighting/scaling factors, global or element-wise
        '''
        
        # Error handling for zeros in observations
        if np.any(observations == 0): 
            if np.any(observations > 0):
                observations = np.where(observations > 0,
                                        observations,
                                        np.min(observations[observations > 0]))
            else:
                observations[:] = 1E-13

        # Error handling for zeros in rates
        if np.any(rates == 0):   
            if np.any(rates > 0):
                rates = np.where(rates > 0,
                                 rates,
                                 np.min(rates[rates > 0]))
            else:
                rates[:] = 1E-13

        negloglik = - np.sum(observations * np.log(rates) - rates - sspecial.loggamma(observations + 1) * scale)
        
        return negloglik


    @staticmethod
    def negloglik_poisson_simple(rates, 
                                 observations,
                                 scale = 1):
        '''
        Poisson neg log lik in fitting. Simplified version for fixed observations.
        "scale" is a handle for additional weighting/scaling factors, global or element-wise

        '''
        # Error handling for zeros in observations
        if np.any(observations == 0):            
            observations = np.where(observations > 0,
                                    observations,
                                    np.min(observations[observations > 0]))

        # Error handling for zeros in rates
        if np.any(rates == 0):            
            rates = np.where(rates > 0,
                             rates,
                             np.min(rates[rates > 0]))

        negloglik = - np.sum((observations * np.log(rates) - rates) * scale)
        
        return negloglik 

        
    @staticmethod
    def negloglik_binomial_simple(n_trials,
                                  k_successes, 
                                  probabilities,
                                  scale = 1):
        '''
        Neg log likelihood function for fitting a histogram described by k_successes from
        n_trials observations (where for PCH sum(k_successes)==n_trials when summing 
        over all histogram bins = fit data points) with a probability model
        described by probabilities. Simplified version for fitting models where
        only probabilities is varied.
        "scale" is a handle for additional weighting/scaling factors, global or element-wise

        '''
        # Error handling for zeros in probabilities
        if np.any(probabilities == 0):            
            probabilities = np.where(probabilities > 0,
                                     probabilities,
                                     np.min(probabilities[probabilities > 0]))
        
        # Renormalize for possible truncation artifacts
        probabilities /= probabilities.sum()

        successes_term = -k_successes * np.log(probabilities)
        failures_term = -(n_trials - k_successes) * np.log(1 - probabilities)
        
        negloglik = np.sum((successes_term + failures_term) * scale)
        
        return negloglik


    @staticmethod
    def negloglik_binomial_full(n_trials,
                                k_successes, 
                                probabilities,
                                scale = 1):
        '''
        Neg log likelihood function for fitting a histogram described by k_successes from
        n_trials observations. Full version for fitting models where
        only all parameters can be varied, used here in incomplete labelling + 
        incomplete sampling treatment.
        "scale" is a handle for additional weighting/scaling factors, global or element-wise

        '''
        
        # Error handling for zeros in probabilities
        mask = probabilities == 0
        if np.any(mask):            
            probabilities = np.where(np.logical_not(mask),
                                    probabilities,
                                    np.min(probabilities[np.logical_not(mask)]) * 1E-3)
        
        # Renormalize for possible truncation artifacts
        probabilities /= probabilities.sum()

        neglog_binom_coeff = - np.log(sspecial.binom(n_trials, 
                                                     k_successes))
        successes_term = -k_successes * np.log(probabilities)
        failures_term = -(n_trials - k_successes) * np.log(1 - probabilities)

        negloglik = np.sum((neglog_binom_coeff + successes_term + failures_term) * scale)
        
        return negloglik

    @staticmethod
    def PCH_chisq(pch_data,
                  pch_model,
                  pch_sigma):
        '''
        Simple sum-of-weighted-residual-squares used for PCH fitting with approximate least-squares fitting
        '''
        return np.sum(((pch_data - pch_model) / pch_sigma)**2)


    # def regularization_MEM(self,
    #                        params,
    #                        spectrum_parameter):
        
    #     # LEGACY CODE - NO LONGER MAINTAINED
        
    #     # Unpack parameters
    #     n_species = self.get_n_species(params)
    #     if spectrum_parameter == 'Amplitude':
    #         reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value * params[f'cpms_{i_spec}'].value**2 for i_spec in range(n_species)])
    #     elif spectrum_parameter == 'N_monomers':
    #         reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value * params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
    #     elif spectrum_parameter == 'N_oligomers':
    #         reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species)])
    #     else:
    #         raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_parameter - must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
        
    #     # Normalize
    #     frequency_array = reg_target / reg_target.sum()
        
    #     # Element-wise "entropy" terms circumventing log(0) errors
    #     entropy_array = frequency_array[frequency_array > 0.] * np.log(frequency_array[frequency_array > 0.])
        
    #     # neg entropy regularizer
    #     regularizer = np.sum(entropy_array)
        
    #     # Normalize by n_species to avoid effects from length
    #     regularizer /= n_species
                
    #     return regularizer


    # def regularization_CONTIN(self,
    #                           params,
    #                           spectrum_parameter):

    #     # LEGACY CODE - NO LONGER MAINTAINED
        
    #     # Unpack parameters
    #     n_species = self.get_n_species(params)
        
    #     if spectrum_parameter == 'Amplitude':
    #         reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value * params[f'cpms_{i_spec}'].value**2 for i_spec in range(n_species)])
    #     elif spectrum_parameter == 'N_monomers':
    #         reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value * params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
    #     elif spectrum_parameter == 'N_oligomers':
    #         reg_target = np.array([params[f'N_avg_pop_{i_spec}'].value for i_spec in range(n_species)])
    #     else:
    #         raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_parameter - must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
            
    #     # Normalize
    #     frequency_array = reg_target / reg_target.sum()
        
    #     # Numerically approximate second derivative of N distribution
    #     second_numerical_diff = np.diff(frequency_array, 2)
        
    #     # Get mean of squared second derivative as single number that reports on non-smoothness of distribution
    #     # Mean rather than sum to avoid effects from length
    #     regularizer = np.mean(second_numerical_diff**2)
        
    #     return regularizer
        
        
    # def simple_pch_penalty(self,
    #                        params,
    #                        pch
    #                        ):
    #     # Depracated
    #     pch_model = self.get_simple_pch_lmfit(params)
    #     return self.negloglik_binomial_simple(n_trials = np.sum(pch),
    #                                           k_successes = pch,
    #                                           probabilities = pch_model)


    def negloglik_pch_single_full_labelling(self,
                                            params,
                                            spectrum_type,
                                            i_bin_time = 0,
                                            numeric_precision = 1e-4,
                                            mp_pool = None):
        
        pch_model = self.get_pch_full_labelling(params,
                                                self.data_PCH_bin_times[i_bin_time],
                                                spectrum_type = spectrum_type,
                                                time_resolved_PCH = False,
                                                crop_output = True,
                                                numeric_precision = numeric_precision,
                                                mp_pool = mp_pool)
        
        # Crop data/model where needed...Can happen for data handling reasons
        pch_data = self.data_PCH_hist[:,i_bin_time]
        n_data_points = pch_data.shape[0]
        n_model_points = pch_model.shape[0]
        
        if self.NLL_funcs_accurate:
            # Binomial MLE
            if n_model_points > n_data_points:
                pch_model = pch_model[:n_data_points]
            elif n_model_points < n_data_points:
                pch_data = pch_data[:n_model_points]
            negloglik = self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                                       k_successes = pch_data,
                                                       probabilities = pch_model)
        else:
            # Least-squares approximation
            if n_model_points > n_data_points:
                pch_model = pch_model[:n_data_points]
                pch_sigma = self.data_PCH_sigma[:n_data_points,i_bin_time]
            elif n_model_points < n_data_points:
                pch_data = pch_data[:n_model_points]
                pch_sigma = self.data_PCH_sigma[:n_model_points,i_bin_time]
            
            negloglik = 0.5 * self.PCH_chisq(pch_data,
                                             pch_model * np.sum(pch_data),
                                             pch_sigma)
        
        # # Normalize by number of "meaningful" data points to avoid effects from dataset length alone
        # negloglik /= np.min([n_data_points, n_model_points])

        return negloglik


    def negloglik_pch_single_partial_labelling(self,
                                               params,
                                               i_bin_time = 0,
                                               numeric_precision = 1e-4,
                                               mp_pool = None):
        
        pch_model = self.get_pch_partial_labelling(params,
                                                   self.data_PCH_bin_times[i_bin_time],
                                                   time_resolved_PCH = False,
                                                   crop_output = True,
                                                   numeric_precision = numeric_precision,
                                                   mp_pool = mp_pool)
        

        # Crop data/model where needed...Can happen for data handling reasons
        pch_data = self.data_PCH_hist[:,i_bin_time]
        n_data_points = pch_data.shape[0]
        n_model_points = pch_model.shape[0]
        
        if self.NLL_funcs_accurate:
            # Binomial MLE
            if n_model_points > n_data_points:
                pch_model = pch_model[:n_data_points]
            elif n_model_points < n_data_points:
                pch_data = pch_data[:n_model_points]
            negloglik = self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                                       k_successes = pch_data,
                                                       probabilities = pch_model)
        else:
            # Least-squares approximation
            if n_model_points > n_data_points:
                pch_model = pch_model[:n_data_points]
                pch_sigma = self.data_PCH_sigma[:n_data_points,i_bin_time]
            elif n_model_points < n_data_points:
                pch_data = pch_data[:n_model_points]
                pch_sigma = self.data_PCH_sigma[:n_model_points,i_bin_time]
            
            negloglik = 0.5 * self.PCH_chisq(pch_data,
                                             pch_model * np.sum(pch_data),
                                             pch_sigma)
        
        # # Normalize by number of "meaningful" data points to avoid effects from dataset length alone
        # negloglik /= np.min([n_data_points, n_model_points])

        return negloglik
        
    
    def negloglik_pcmh_full_labelling(self,
                                      params,
                                      spectrum_type,
                                      numeric_precision = 1e-4,
                                      mp_pool = None):
        
        negloglik = 0
                
        for i_bin_time, t_bin in enumerate(self.data_PCH_bin_times):
            
            pch_model = self.get_pch_full_labelling(params,
                                                    t_bin,
                                                    spectrum_type = spectrum_type,
                                                    time_resolved_PCH = True,
                                                    crop_output = True,
                                                    numeric_precision = numeric_precision,
                                                    mp_pool = mp_pool)
            
            # Crop data/model where needed...Can happen for data handling reasons
            pch_data = self.data_PCH_hist[:,i_bin_time]
            n_data_points = pch_data.shape[0]
            n_model_points = pch_model.shape[0]
            
            if self.NLL_funcs_accurate:
                # Binomial MLE
                if n_model_points > n_data_points:
                    pch_model = pch_model[:n_data_points]
                elif n_model_points < n_data_points:
                    pch_data = pch_data[:n_model_points]
                negloglik_iter = self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                                                k_successes = pch_data,
                                                                probabilities = pch_model)
            else:
                # Least-squares approximation
                if n_model_points > n_data_points:
                    pch_model = pch_model[:n_data_points]
                    pch_sigma = self.data_PCH_sigma[:n_data_points,i_bin_time]
                elif n_model_points < n_data_points:
                    pch_data = pch_data[:n_model_points]
                    pch_sigma = self.data_PCH_sigma[:n_model_points,i_bin_time]
                
                negloglik_iter = self.PCH_chisq(pch_data,
                                                pch_model * np.sum(pch_data),
                                                pch_sigma)
            
            # # Normalize by number of "meaningful" data points to avoid effects from dataset length alone
            # negloglik_iter /= np.min([n_data_points, n_model_points])

            negloglik += negloglik_iter
            
        # Normalize for number of data points along second axis
        negloglik /= self.data_PCH_bin_times.shape[0]
        
        return negloglik

    
    def negloglik_pcmh_partial_labelling(self,
                                         params,
                                         numeric_precision = 1e-4,
                                         mp_pool = None):
        
        negloglik = 0
                
        for i_bin_time, t_bin in enumerate(self.data_PCH_bin_times):
            
            pch_model = self.get_pch_partial_labelling(params,
                                                       t_bin,
                                                       time_resolved_PCH = True,
                                                       crop_output = True,
                                                       numeric_precision = numeric_precision,
                                                       mp_pool = mp_pool)
            
            # Crop data/model where needed...Can happen for data handling reasons
            pch_data = self.data_PCH_hist[:,i_bin_time]
            n_data_points = pch_data.shape[0]
            n_model_points = pch_model.shape[0]
            
                
            if self.NLL_funcs_accurate:
                # Binomial MLE
                if n_model_points > n_data_points:
                    pch_model = pch_model[:n_data_points]
                elif n_model_points < n_data_points:
                    pch_data = pch_data[:n_model_points]
                negloglik_iter = self.negloglik_binomial_simple(n_trials = np.sum(pch_data),
                                                                k_successes = pch_data,
                                                                probabilities = pch_model)
            else:
                # Least-squares approximation
                if n_model_points > n_data_points:
                    pch_model = pch_model[:n_data_points]
                    pch_sigma = self.data_PCH_sigma[:n_data_points,i_bin_time]
                elif n_model_points < n_data_points:
                    pch_data = pch_data[:n_model_points]
                    pch_sigma = self.data_PCH_sigma[:n_model_points,i_bin_time]
                
                negloglik_iter = self.PCH_chisq(pch_data,
                                                pch_model * np.sum(pch_data),
                                                pch_sigma)

            # # Normalize by number of "meaningful" data points to avoid effects from dataset length alone
            # negloglik_iter /= np.min([n_data_points, n_model_points])

            negloglik += negloglik_iter
            
        # Normalize for number of data points along second axis
        negloglik /= self.data_PCH_bin_times.shape[0]
            
        return negloglik

    def negloglik_avg_count_rate(self,
                                 params,
                                 labelling_correction,
                                 incomplete_sampling_correction,
                                 spectrum_type
                                 ):
        
        # Poisson likelihood function for considering average count rate in fitting
        n_species_spec, n_species_disc = self.get_n_species(params)
        
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species_spec)])
        if spectrum_type in ['reg_MEM', 'reg_CONTIN']:    
            N_avg_array = self._N_pop_array
        else: 
            N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])

        cpms_array_d = np.array([params[f'cpms_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        N_avg_array_d = np.array([params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_binwidth_array_d = np.array([params[f'stoichiometry_binwidth_d_{i_spec}'].value for i_spec in range(n_species_disc)])

        if labelling_correction:
            Label_efficiency_obs_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
            Label_efficiency_obs_array_d = np.array([params[f'Label_efficiency_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        else:
            Label_efficiency_obs_array = np.ones_like(cpms_array)
            Label_efficiency_obs_array_d = np.ones_like(cpms_array_d)
        
        avg_count_rate_model = np.sum(cpms_array * N_avg_array * stoichiometry_binwidth_array * Label_efficiency_obs_array) + \
                               np.sum(cpms_array_d * N_avg_array_d * stoichiometry_binwidth_array_d * Label_efficiency_obs_array_d)
        
        # Correct dark-state fraction
        avg_count_rate_model *= 1 - params['F_blink'].value
        
        if self.NLL_funcs_accurate and self.incomplete_sampling_possible:
            # Poisson likelihood
            negloglik = self.negloglik_poisson_simple(rates = avg_count_rate_model * self.acquisition_time_s, 
                                                      observations = self.data_avg_count_rate * self.acquisition_time_s)
            
        else:
            # Either the use of least-squares approximation was specified, or we
            # do not have the acquisition time, in which case we can still work with WLSQ
            negloglik = 0.5 *  self.acquisition_time_s * (avg_count_rate_model - self.data_avg_count_rate)**2 / self.data_avg_count_rate
            
        return negloglik


    def negloglik_global_fit(self,
                             params,
                             use_FCS,
                             use_PCH,
                             time_resolved_PCH,
                             spectrum_type,
                             spectrum_parameter,
                             labelling_correction,
                             incomplete_sampling_correction,
                             use_avg_count_rate = False,
                             i_bin_time = 0,
                             numeric_precision = 1e-4,
                             mp_pool = None
                             ):
        
        negloglik = 0.
        
        n_species_spec, n_species_disc = self.get_n_species(params)
        
        
        if use_FCS:
            # Correlation function is being fitted
            if spectrum_type in ['reg_CONTIN', 'reg_MEM', 'par_Gauss', 'par_Gamma', 'par_LogNorm', 'par_StrExp']:
                if labelling_correction:
                    # Raw likelihood
                    negloglik_FCS = self.fcs_chisq_partial_labelling_par_reg(params) / 2.
                                            
                else: # not labelling_correction
                    negloglik_FCS = self.fcs_chisq_full_labelling_par_reg(params,
                                                                          spectrum_type) / 2.

            else: 
                if labelling_correction:
                    negloglik_FCS = self.fcs_chisq_partial_labelling(params) / 2.
                else: # not labelling_correction
                    negloglik_FCS = self.fcs_chisq_full_labelling(params) / 2.
                    
            # Count local degrees of freedom for weighting = number of FCS data points - 1 - number of FCS-only (!) parameters
            dof_FCS = self.data_FCS_G.shape[0] - 1
            dof_FCS -= 1. if params['acf_offset'].vary else 0.
            
            if not (use_PCH or use_avg_count_rate):
                # Parameters that would be shared with PCH and/or ACR and therefore only count if we do not have those
                dof_FCS -= np.sum([1. for key in params.keys() if key in ['Label_efficiency', 'N_dist_amp',  'N_dist_a', 'N_dist_b'] and params[key].vary])
                dof_FCS -= n_species_spec if spectrum_type in ['reg_CONTIN', 'reg_MEM'] else 0.
                # Spectrum N_obs, cpms, and labelling_efficiency_obs are never parameters of only this likelihood function
                
                for i_spec in range(n_species_disc):
                    # Discrete species can have just about any parameter configuration
                    dof_FCS -= np.sum([1. for key in params.keys() if key in [f'N_avg_obs_d_{i_spec}', f'stoichiometry_d_{i_spec}', f'stoichiometry_binwidth_d_{i_spec}', f'Label_efficiency_obs_d_{i_spec}'] and params[key].vary])
            
            if not time_resolved_PCH:
                # discrete-species tau_diff is shared only if we have time-resolved PCH and FCS
                dof_FCS -= np.sum([1. for i_spec in range(n_species_disc) if f'tau_diff_d_{i_spec}' in params.keys() and params[f'tau_diff_d_{i_spec}'].vary])
                
            # Ensure positive-nonzero DoF
            if dof_FCS <= 0:
                dof_FCS = 1. 
                
            negloglik += negloglik_FCS / dof_FCS
            # negloglik += negloglik_FCS
                
        if use_PCH and (not time_resolved_PCH):
            # Conventional single PCH is being fitted
            if labelling_correction:
                negloglik_PCH = self.negloglik_pch_single_partial_labelling(params,
                                                                            i_bin_time = i_bin_time,
                                                                            numeric_precision = numeric_precision,
                                                                            mp_pool = mp_pool)
            else: # not labelling_correction
                negloglik_PCH = self.negloglik_pch_single_full_labelling(params,
                                                                         spectrum_type = spectrum_type,
                                                                         i_bin_time = i_bin_time,
                                                                         numeric_precision = numeric_precision,
                                                                         mp_pool = mp_pool)
                
            dof_PCH = np.sum(self.data_PCH_hist[:, i_bin_time] > 0) - 1 # We count the nonzero elements as valid data points

                
        elif use_PCH and time_resolved_PCH:
            # PCMH fit
            
            if labelling_correction:
                negloglik_PCH = self.negloglik_pcmh_partial_labelling(params,
                                                                      numeric_precision = numeric_precision,
                                                                      mp_pool = mp_pool)
            else: # not labelling_correction
                negloglik_PCH = self.negloglik_pcmh_full_labelling(params,
                                                                   spectrum_type = spectrum_type,
                                                                   numeric_precision = numeric_precision,
                                                                   mp_pool = mp_pool)
                
            dof_PCH = np.sum(self.data_PCH_hist > 0) - 1 # We count the nonzero elements as valid data points
                
            
        if use_PCH:
            # Count local degrees of freedom for weighting = number of PCH data points - 1 - number of PCH-only (!) parameters
            dof_PCH -= 1. if params['F'].vary else 0.
            
            if not use_avg_count_rate:
                # cpms parameters can be shared between PCH and average count rate
                dof_PCH -= 1. if params['cpms_0'].vary else 0.
                for i_spec in range(n_species_disc):
                    dof_PCH -= 1. if f'cpms_d_{n_species_disc}' in params.keys() and params[f'cpms_d_{n_species_disc}'].vary else 0.
                    
            if not (use_FCS or use_avg_count_rate):
                # Parameters that would be shared with FCS and/or ACR and therefore only count if we do not have those
                dof_PCH -= np.sum([1. for key in params.keys() if key in ['Label_efficiency', 'N_dist_amp',  'N_dist_a', 'N_dist_b'] and params[key].vary])
                dof_PCH -= n_species_spec if spectrum_type in ['reg_CONTIN', 'reg_MEM'] else 0.
                # N_obs and labelling_efficiency_obs are never parameters of only this likelihood function
                
                for i_spec in range(n_species_disc):
                    # Discrete species can have just about any parameter configuration
                    dof_PCH -= np.sum([1. for key in params.keys() if key in [f'N_avg_obs_d_{i_spec}', f'stoichiometry_d_{i_spec}', f'stoichiometry_binwidth_d_{i_spec}', f'Label_efficiency_obs_d_{i_spec}'] and params[key].vary])
    
            if time_resolved_PCH and not use_FCS:
                # discrete-species tau_diff is shared only if we have time-resolved PCH and FCS
                dof_PCH -= np.sum([1. for i_spec in range(n_species_disc) if f'tau_diff_d_{i_spec}' in params.keys() and params[f'tau_diff_d_{i_spec}'].vary])
    
            if dof_PCH <= 0:
                dof_PCH = 1. # Ensure positive-nonzero DoF

            negloglik += negloglik_PCH / dof_PCH
            # negloglik += negloglik_PCH 

        # For the following two likelihood functions we need no DoF counting, 
        # the normalization is part of the functions itself in these cases.
        if incomplete_sampling_correction:
            # Incomplete sampling correction included in fit
            
            if labelling_correction and self.labelling_efficiency_incomp_sampling:
                # Allow deviation of labelling efficiency
                negloglik += self.negloglik_incomplete_sampling_variable_labelling(params,
                                                                                   spectrum_type = spectrum_type,
                                                                                   numeric_precision = numeric_precision) 
            else: 
                # not labelling_correction, or at least no treatment of labelling efficiency fluctuations
                negloglik += self.negloglik_incomplete_sampling_static_labelling(params,
                                                                                  spectrum_type = spectrum_type,
                                                                                  labelling_correction = labelling_correction)
                
                
                
            
        if use_avg_count_rate:
            # Include avg count rate in fit
            negloglik += self.negloglik_avg_count_rate(params, 
                                                       labelling_correction, 
                                                       incomplete_sampling_correction,
                                                       spectrum_type)


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
                          cpm_eff,
                          PCH_n_photons_max):
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
        photon_counts_array = np.arange(1, PCH_n_photons_max+1)
        
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
        pch *=  1 / self.PCH_Q / np.sqrt(np.pi) / sspecial.factorial(photon_counts_array)

        return pch
    
    
    def pch_3dgauss_nonideal_1part(self,
                                   F,
                                   cpm_eff,
                                   PCH_n_photons_max):
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
                                     PCH_n_photons_max)

        # Apply correction
        pch[0] += 2**(-3/2) * cpm_eff * F / self.PCH_Q
        pch /= (1 + F)**2
        
        return pch
        
    
    def pch_get_N_spectrum(self,
                           N_avg,
                           numeric_precision = 1e-4):
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
        N_box_array = np.arange(0, np.max([np.ceil(N_avg_box * 1E3), 3])) # At least inspect 3 elements
        # Clip N_box to useful significant values within precision (on righthand side)
        poisson_sf = poisson_dist.sf(N_box_array)
        
        significant_bins = np.nonzero(poisson_sf > numeric_precision)[0]
        # It can happen that none fulfil criterion, than we need more precise calculation...
        i_iter = 0
        while significant_bins.shape[0] == 0 and i_iter < 10:
            numeric_precision /= 10.
            significant_bins = np.nonzero(poisson_sf > numeric_precision)[0]
            i_iter += 1
        
        if significant_bins.shape[0] > 0: 
            N_box_array_clip = N_box_array[:significant_bins[-1] + 1]
        else:
            # increasing precision did not help - use 3 or all, whichever is smaller
            N_box_array_clip = N_box_array[:np.min([3, N_box_array.shape[0]])]
            
        # Get probability mass function
        p_of_N = poisson_dist.pmf(N_box_array_clip)
        
        return p_of_N


    def pch_get_stoichiometry_spectrum(self,
                                       max_label,
                                       Label_efficiency,
                                       numeric_precision = 1e-4):
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
        n_labels_array_clip = n_labels_array[weights > numeric_precision]
        
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
        # Exception workaround for weird parameter constellations
        if p_of_N.shape[0] == 1:
            p_of_N = np.append(p_of_N, [0.])
        
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
        
        return pch_full
        
    
    def get_pch(self,
                F,
                t_bin,
                cpms,
                N_avg,
                crop_output = True,
                numeric_precision = 1e-4):
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
        
        PCH_n_photons_max = self.PCH_n_photons_max[self.data_PCH_bin_times == t_bin][0]

        # "Fundamental" single-particle PCH
        pch_single_particle = self.pch_3dgauss_nonideal_1part(F, 
                                                              t_bin * cpms,
                                                              PCH_n_photons_max)
        
        # Weights for observation of 1, 2, 3, 4, ... particles
        p_of_N = self.pch_get_N_spectrum(N_avg,
                                         numeric_precision)
        
        # Put weights and fundamental PCH together for full PCH
        pch = self.pch_N_part(pch_single_particle,
                              p_of_N)

        if crop_output:
            pch = pch[:PCH_n_photons_max + 1]
            
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
            tau_blink_avg = tau_blink * sspecial.gamma(1 + 1 / beta_blink)
            
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


    def multi_species_pch_single(self,
                                 F,
                                 t_bin,
                                 cpms_array,
                                 N_avg_array,
                                 crop_output = True,
                                 numeric_precision = 1e-4
                                 ):
                

        for i_spec, cpms_spec in enumerate(cpms_array):
            N_avg_spec = N_avg_array[i_spec]
                        
            if i_spec == 0:
                # Get first species PCH
                pch = self.get_pch(F,
                                   t_bin,
                                   cpms_spec,
                                   N_avg_spec,
                                   crop_output = False,
                                   numeric_precision = numeric_precision)        
            
            else:
                # Colvolve with further species PCH
                pch = np.convolve(pch, 
                                  self.get_pch(F,
                                               t_bin,
                                               cpms_spec,
                                               N_avg_spec,
                                               crop_output = False,
                                               numeric_precision = numeric_precision),
                                  mode = 'full')      
        
        if crop_output:
            PCH_n_photons_max = self.PCH_n_photons_max[self.data_PCH_bin_times == t_bin][0]
            pch = pch[:PCH_n_photons_max + 1]
            
        return pch
    
    
    @staticmethod
    def multi_species_pch_parfunc(F,
                                  t_bin,
                                  cpms_spec,
                                  N_avg_spec,
                                  PCH_Q,
                                  PCH_n_photons_max,
                                  numeric_precision = 1e-4):
        '''Static method copy of multi_species_pch and its child functions for parallel processing'''
        
        ### "Fundamental" single-particle PCH
        
        # Get ideal-Gauss PCH
        # Array with all photon counts > 0 to sample
        photon_counts_array = np.arange(1, PCH_n_photons_max+1)
        
        # Preliminary results container
        pch_single_particle = np.zeros_like(photon_counts_array, 
                                            dtype = np.float64)
        
        # Iterate over photon counts 
        for k in photon_counts_array:
            # Numeric integration over incomplete gamma function
            pch_single_particle[k-1], _ = sintegrate.quad(lambda x: sspecial.gammainc(k, 
                                                                                      t_bin * cpms_spec * np.exp(-2*x**2)),
                                                          0, np.inf)
            
        # Scipy.special actually implements regularized lower incompl. gamma func, so we need to scale it
        pch_single_particle *= sspecial.gamma(photon_counts_array)
        
        # Apply prefactors
        pch_single_particle *=  1 / PCH_Q / np.sqrt(np.pi) / sspecial.factorial(photon_counts_array)
        pch_single_particle[0] += 2**(-3/2) * t_bin * cpms_spec * F / PCH_Q
        pch_single_particle /= (1 + F)**2

        
        ### Weights for observation of 1, 2, 3, 4, ... particles
        # Get scaled <N> as mean for Poisson dist object
        N_avg_box = PCH_Q * N_avg_spec
        poisson_dist = sstats.poisson(N_avg_box)
        
        # x axis array
        N_box_array = np.arange(0, np.max([np.ceil(N_avg_box * 1E3), 3])) # At least inspect 3 elements
        # Clip N_box to useful significant values within precision (on righthand side)
        poisson_sf = poisson_dist.sf(N_box_array)
        
        significant_bins = np.nonzero(poisson_sf > numeric_precision)[0]
        # It can happen that none fulfil criterion, than we need more precise calculation...
        i_iter = 0
        while significant_bins.shape[0] == 0 and i_iter < 10:
            numeric_precision /= 10.
            significant_bins = np.nonzero(poisson_sf > numeric_precision)[0]
            i_iter += 1
            
        if significant_bins.shape[0] > 0: 
            N_box_array_clip = N_box_array[:significant_bins[-1] + 1]
        else:
            # increasing precision did not help - use 3 or all, whichever is smaller
            N_box_array_clip = N_box_array[:np.min([3, N_box_array.shape[0]])]

        
        # Get probability mass function
        p_of_N = poisson_dist.pmf(N_box_array_clip)
        
        # Get probability mass function
        p_of_N = poisson_dist.pmf(N_box_array_clip)
        
        ### Put weights and fundamental PCH together for full PCH
        # Initialize a long array of zeros to contain the full PCH
        pch_new_full = np.zeros((pch_single_particle.shape[0] + 1 ) * (p_of_N.shape[0] + 1))
        
        # Append the probability mass for 0 photons to pch_single_particle
        pch_1 = np.concatenate((np.array([1 - pch_single_particle.sum()]), pch_single_particle))
        
        # Write probability for 0 particles -> 0 photons into full PCH
        pch_new_full[0] += p_of_N[0]
        
        # Write single-particle weighted PCH into full PCH
        pch_new_full[0:pch_1.shape[0]] += p_of_N[1] * pch_1
        
        # Get N-particle PCHs and write them into the full thing through 
        # iterative convolutions
        pch_N = pch_1.copy()
        for pN in p_of_N[2:]:
            pch_N = np.convolve(pch_N, 
                                pch_1, 
                                mode='full')
            pch_new_full[0:pch_N.shape[0]] += pN * pch_N

        pch = pch_new_full[:PCH_n_photons_max + 1]
            
        return pch

    
    def multi_species_pch_parwrap(self,
                                  F,
                                  t_bin,
                                  cpms_array,
                                  N_avg_array,
                                  mp_pool,
                                  crop_output = True,
                                  numeric_precision = 1e-4
                                  ):
        
        PCH_n_photons_max = self.PCH_n_photons_max[self.data_PCH_bin_times == t_bin][0]

        # Define parameter list for parallel processes
        list_of_param_tuples = []
        for i_spec, cpms_spec in enumerate(cpms_array):
            N_avg_spec = N_avg_array[i_spec]
            
            list_of_param_tuples.append((F,
                                         t_bin,
                                         cpms_spec,
                                         N_avg_spec,
                                         self.PCH_Q,
                                         PCH_n_photons_max,
                                         numeric_precision))

        # Run parallel calculation
        list_of_species_pch = [mp_pool.starmap(self.multi_species_pch_parfunc, list_of_param_tuples)]

        # Put results together
        # Get first species PCH
        pch = list_of_species_pch[0]
        
        for i_spec in range(1, cpms_array.shape[0]):
            # Colvolve with further species PCH
            pch = np.convolve(pch,
                              list_of_species_pch[i_spec], 
                              mode = 'full')      
            
        if crop_output:
            pch = pch[:PCH_n_photons_max + 1]
            
        return pch


    def multi_species_pch_handler(self,
                                  F,
                                  t_bin,
                                  cpms_array,
                                  N_avg_array,
                                  crop_output = True,
                                  numeric_precision = 1e-4,
                                  mp_pool = None
                                  ):
        
        if not utils.isfloat(F) or F < 0.:
            raise Exception(f'[{self.job_prefix}] Invalid input for F: Must be float >= 0.') 

        if not utils.isfloat(t_bin) or t_bin <= 0.:
            raise Exception(f'[{self.job_prefix}] Invalid input for t_bin: Must be float > 0.') 

        if not utils.isiterable(cpms_array):
            raise Exception(f'[{self.job_prefix}] Invalid input for cpms_array: Must be array.') 
            
        if not utils.isiterable(N_avg_array):
            raise Exception(f'[{self.job_prefix}] Invalid input for N_avg_array: Must be array.')
           
        if not cpms_array.shape[0] == N_avg_array.shape[0]:
            raise Exception(f'[{self.job_prefix}] cpms_array and N_avg_array must have same length.')
           
        if not type(crop_output) == bool:
            raise Exception(f'[{self.job_prefix}] Invalid input for crop_output: Must be bool.')
            
        if not utils.isfloat(numeric_precision) or numeric_precision <= 0. or numeric_precision >= 1.:
            raise Exception(f'[{self.job_prefix}] Invalid input for numeric_precision: Must be float, 0 < numeric_precision < 1.') 

        if type(mp_pool) == multiprocessing.Pool:
            # Parallel execution
            pch = self.multi_species_pch_parwrap(F,
                                                 t_bin,
                                                 cpms_array,
                                                 N_avg_array,
                                                 mp_pool,
                                                 crop_output = crop_output,
                                                 numeric_precision = numeric_precision
                                                 )
        else:
            # "Normal" execution
            pch = self.multi_species_pch_single(F,
                                                t_bin,
                                                cpms_array,
                                                N_avg_array,
                                                crop_output = crop_output,
                                                numeric_precision = numeric_precision
                                                )
        return pch



    #%% More complete/complex FCS models
                
    
    def get_acf_full_labelling(self, 
                               params):
                
        acf = np.zeros_like(self.data_FCS_G)
        acf_den = 0.
        
        n_species_spec, n_species_disc = self.get_n_species(params)
            
        # Extract parameters
        # Spectrum species
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
        
        # Discrete species
        cpms_array_d = np.array([params[f'cpms_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        N_avg_array_d = np.array([params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_binwidth_array_d = np.array([params[f'stoichiometry_binwidth_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        tau_diff_array_d = np.array([params[f'tau_diff_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        
        # Only relative brightness matters, and normalizing may help avoid overflows etc.
        cpms_norm = (cpms_array.sum() + cpms_array_d.sum()) / (n_species_spec + n_species_disc)
        cpms_array /= cpms_norm 
        cpms_array_d /= cpms_norm


        # Eval spectrum species
        if n_species_spec > 0:
            for i_spec in range(n_species_spec):
                g_norm = self.fcs_3d_diff_single_species(tau_diff_array[i_spec])
                acf += g_norm * N_avg_array[i_spec] * stoichiometry_binwidth_array[i_spec] * cpms_array[i_spec]**2
            acf_den += np.sum(N_avg_array * cpms_array * stoichiometry_binwidth_array)**2
        
        # Eval discrete species
        if n_species_disc > 0:
            for i_spec in range(n_species_disc):
                g_norm = self.fcs_3d_diff_single_species(tau_diff_array_d[i_spec])
                acf += g_norm * N_avg_array_d[i_spec] * stoichiometry_binwidth_array_d[i_spec] * cpms_array_d[i_spec]**2
            acf_den += np.sum(N_avg_array_d * cpms_array_d * stoichiometry_binwidth_array_d)**2

        # Put together and further terms
        acf *= 2**(-3/2) 
        acf /= acf_den
        if params['F_blink'].value > 0:
            acf *= 1 + params['F_blink'].value / (1 - params['F_blink'].value) * self.fcs_blink_stretched_exp(params['tau_blink'].value,
                                                                                                              params['beta_blink'].value)        
        acf += params['acf_offset'].value # Offset

        return acf
    
    
    def get_acf_full_labelling_par(self, 
                                   params):
                
        n_species_spec, n_species_disc = self.get_n_species(params)
            
        # Extract parameters
        # Spectrum species
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        
        # Discrete species
        cpms_array_d = np.array([params[f'cpms_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        N_avg_array_d = np.array([params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_binwidth_array_d = np.array([params[f'stoichiometry_binwidth_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        tau_diff_array_d = np.array([params[f'tau_diff_d_{i_spec}'].value for i_spec in range(n_species_disc)])

        # Only relative brightness matters, and normalizing may help avoid overflows etc.
        cpms_norm = (cpms_array.sum() + cpms_array_d.sum()) / (n_species_spec + n_species_disc)
        cpms_array /= cpms_norm 
        cpms_array_d /= cpms_norm

        acf_den = 0.

        # Eval spectrum species
        if n_species_spec > 0:
            spec_weights =  N_avg_array * stoichiometry_binwidth_array * cpms_array**2
            acf = np.dot(self.tau__tau_diff_array, 
                         spec_weights)
            acf_den += np.sum(N_avg_array * cpms_array * stoichiometry_binwidth_array)**2

        # Eval discrete species
        if n_species_disc > 0:
            for i_spec in range(n_species_disc):
                g_norm = self.fcs_3d_diff_single_species(tau_diff_array_d[i_spec])
                acf += g_norm * N_avg_array_d[i_spec] * stoichiometry_binwidth_array_d[i_spec] * cpms_array_d[i_spec]**2
            acf_den += np.sum(N_avg_array_d * cpms_array_d * stoichiometry_binwidth_array_d)**2
     
        # Put together and further terms
        acf *= 2**(-3/2) 
        acf /= acf_den
        if params['F_blink'].value > 0:
            acf *= 1 + params['F_blink'].value / (1 - params['F_blink'].value) * self.fcs_blink_stretched_exp(params['tau_blink'].value,
                                                                                                              params['beta_blink'].value)
        acf += params['acf_offset'].value # Offset

        return acf


    def get_acf_full_labelling_reg(self, 
                                   params):
        n_species_spec, n_species_disc = self.get_n_species(params)
            
        # Extract parameters
        
        # Spectrum species
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])

        # Discrete species
        cpms_array_d = np.array([params[f'cpms_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        N_avg_array_d = np.array([params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_binwidth_array_d = np.array([params[f'stoichiometry_binwidth_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        tau_diff_array_d = np.array([params[f'tau_diff_d_{i_spec}'].value for i_spec in range(n_species_disc)])

        # Only relative brightness matters, and normalizing may help avoid overflows etc.
        cpms_norm = (cpms_array.sum() + cpms_array_d.sum()) / (n_species_spec + n_species_disc)
        cpms_array /= cpms_norm 
        cpms_array_d /= cpms_norm

        acf_den = 0.

        # Eval spectrum species
        if n_species_spec > 0:
            spec_weights = self._N_pop_array * stoichiometry_binwidth_array * cpms_array**2
            acf = np.dot(self.tau__tau_diff_array, 
                         spec_weights)
            acf_den += np.sum(self._N_pop_array * cpms_array * stoichiometry_binwidth_array)**2

        # Eval discrete species
        if n_species_disc > 0:
            for i_spec in range(n_species_disc):
                g_norm = self.fcs_3d_diff_single_species(tau_diff_array_d[i_spec])
                acf += g_norm * N_avg_array_d[i_spec] * stoichiometry_binwidth_array_d[i_spec] * cpms_array_d[i_spec]**2
            acf_den += np.sum(N_avg_array_d * cpms_array_d * stoichiometry_binwidth_array_d)**2
        
        # Put together and further terms
        acf /= acf_den
        acf *= 2**(-3/2) 
        if params['F_blink'].value > 0:
            acf *= 1 + params['F_blink'].value / (1 - params['F_blink'].value) * self.fcs_blink_stretched_exp(params['tau_blink'].value,
                                                                                                              params['beta_blink'].value)
        acf += params['acf_offset'].value # Offset

        return acf


    def get_acf_partial_labelling(self, 
                                  params):
                
        acf = np.zeros_like(self.data_FCS_G)
        
        n_species_spec, n_species_disc = self.get_n_species(params)
            
        # Extract parameters
        # Spectrum species
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species_spec)])
        labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        tau_diff_array = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_array = np.array([params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species_spec)])

        # Discrete species
        cpms_array_d = np.array([params[f'cpms_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        N_avg_array_d = np.array([params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_binwidth_array_d = np.array([params[f'stoichiometry_binwidth_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_array_d = np.array([params[f'stoichiometry_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        tau_diff_array_d = np.array([params[f'tau_diff_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        labelling_efficiency_array_d = np.array([params[f'Label_efficiency_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        
        # Only relative brightness matters, and normalizing may help avoid overflows etc.
        cpms_norm = (cpms_array.sum() + cpms_array_d.sum()) / (n_species_spec + n_species_disc)
        cpms_array /= cpms_norm 
        cpms_array_d /= cpms_norm
        
        acf_den = 0.
        
        # Eval spectrum species
        if n_species_spec > 0:
            for i_spec in range(n_species_spec):
                g_norm = self.fcs_3d_diff_single_species(tau_diff_array[i_spec])
                acf += g_norm * N_avg_array[i_spec] * stoichiometry_binwidth_array[i_spec] * cpms_array[i_spec]**2 * (1 + (1 - labelling_efficiency_array[i_spec]) / stoichiometry_array[i_spec] / labelling_efficiency_array[i_spec])
            acf_den += np.sum(self._N_pop_array * cpms_array * stoichiometry_binwidth_array)**2
        
        # Eval discrete species
        if n_species_disc > 0:
            for i_spec in range(n_species_disc):
                g_norm = self.fcs_3d_diff_single_species(tau_diff_array_d[i_spec])
                acf += g_norm * N_avg_array_d[i_spec] * stoichiometry_binwidth_array_d[i_spec] * cpms_array_d[i_spec]**2 * (1 + (1 - labelling_efficiency_array_d[i_spec]) / stoichiometry_array_d[i_spec] / labelling_efficiency_array_d[i_spec])
            acf_den += np.sum(N_avg_array_d * cpms_array_d * stoichiometry_binwidth_array_d)**2

        # Put together and further terms
        acf /= acf_den
        acf *= 2**(-3/2) 
        if params['F_blink'].value > 0:
            acf *= 1 + params['F_blink'].value / (1 - params['F_blink'].value) * self.fcs_blink_stretched_exp(params['tau_blink'].value,
                                                                                                              params['beta_blink'].value)
        acf += params['acf_offset'].value # Offset

        return acf


    def get_acf_partial_labelling_par_reg(self, 
                                          params):
                        
        n_species_spec, n_species_disc = self.get_n_species(params)
            
        # Spectrum species
        cpms_array = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species_spec)])
        labelling_efficiency_array = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_array = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_array = np.array([params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species_spec)])

        # Discrete species
        cpms_array_d = np.array([params[f'cpms_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        N_avg_array_d = np.array([params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_binwidth_array_d = np.array([params[f'stoichiometry_binwidth_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        tau_diff_array_d = np.array([params[f'tau_diff_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        labelling_efficiency_array_d = np.array([params[f'Label_efficiency_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_array_d = np.array([params[f'stoichiometry_d_{i_spec}'].value for i_spec in range(n_species_disc)])

        # Only relative brightness matters, and normalizing may help avoid overflows etc.
        cpms_norm = (cpms_array.sum() + cpms_array_d.sum()) / (n_species_spec + n_species_disc)
        cpms_array /= cpms_norm 
        cpms_array_d /= cpms_norm

        acf_den = 0.

        # Eval spectrum species
        if n_species_spec > 0:
            spec_weights = N_avg_array * stoichiometry_binwidth_array * cpms_array**2 * (1 + (1 - labelling_efficiency_array) / stoichiometry_array / labelling_efficiency_array)
            acf = np.dot(self.tau__tau_diff_array, 
                         spec_weights)
            acf_den += np.sum(N_avg_array * cpms_array * stoichiometry_binwidth_array)**2

        # Eval discrete species
        if n_species_disc > 0:
            for i_spec in range(n_species_disc):
                g_norm = self.fcs_3d_diff_single_species(tau_diff_array_d[i_spec])
                acf += g_norm * N_avg_array_d[i_spec] * stoichiometry_binwidth_array_d[i_spec] * cpms_array_d[i_spec]**2 * (1 + (1 - labelling_efficiency_array_d[i_spec]) / stoichiometry_array_d[i_spec] / labelling_efficiency_array_d[i_spec])
            acf_den += np.sum(N_avg_array_d * cpms_array_d * stoichiometry_binwidth_array_d)**2

        # Put together and further terms
        acf /= acf_den
        acf *= 2**(-3/2) 
        if params['F_blink'].value > 0:
            acf *= 1 + params['F_blink'].value / (1 - params['F_blink'].value) * self.fcs_blink_stretched_exp(params['tau_blink'].value,
                                                                                                              params['beta_blink'].value)
        acf += params['acf_offset'].value # Offset

        return acf




    #%% More complete/complex PCH models
    # def get_simple_pch_lmfit(self,
    #                          params,
    #                          ):
    #     '''
    #     Obsolete wrapper for get_pch using a syntax that works easily with 
    #     lmfit.minimize() from early development
    #
        
    #     Parameters
    #     ----------
    #     params : 
    #         lmfit.parameters object containing all arguments of self.get_pch()
    #         as parameters

    #     Returns
    #     -------
    #     pch
    #         np.array (1D) with normalized pch model.

    #     '''
        
    #     return self.get_pch(params['F'].value,
    #                         params['t_bin'].value,
    #                         params['cpms'].value,
    #                         params['N_avg'].value)
        
    
   
    
    def get_pch_full_labelling(self,
                               params,
                               t_bin,
                               spectrum_type,
                               time_resolved_PCH = False,
                               crop_output = False,
                               numeric_precision = 1E-4,
                               mp_pool = None
                               ):
        
        n_species_spec, n_species_disc = self.get_n_species(params)
            
        # Extract parameters for discrete and spectrum species, and concatenate them
        cpms_array_s = np.array([params[f'cpms_{i_spec}'].value for i_spec in range(n_species_spec)])
        cpms_array_d = np.array([params[f'cpms_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        cpms_array = np.concatenate((cpms_array_s, cpms_array_d), 
                                    axis = 0) * (1 + params['F'].value) * 2**(-3/2) # Correct for non-Gaussian PSF elements and gamma
        
        stoichiometry_binwidth_array_s = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array_d = np.array([params[f'stoichiometry_binwidth_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_binwidth_array = np.concatenate((stoichiometry_binwidth_array_s, stoichiometry_binwidth_array_d), 
                                                      axis = 0)

        if spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            N_avg_array_s = self._N_pop_array
        else:
            N_avg_array_s = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_array_d = np.array([params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        N_eff_array = np.concatenate((N_avg_array_s, N_avg_array_d), 
                                     axis = 0) * stoichiometry_binwidth_array * 2**(3/2) # Binwidth and gamma correction
        
        if time_resolved_PCH:
            # Bin time correction
            tau_diff_array_s = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
            tau_diff_array_d = np.array([params[f'tau_diff_d_{i_spec}'].value for i_spec in range(n_species_disc)])
            tau_diff_array = np.concatenate((tau_diff_array_s, tau_diff_array_d), 
                                            axis = 0)

            cpms_array, N_eff_array = self.pch_bin_time_correction(t_bin = t_bin, 
                                                                   cpms = cpms_array, 
                                                                   N_avg = N_eff_array,
                                                                   tau_diff = tau_diff_array, 
                                                                   tau_blink = params['tau_blink'].value, 
                                                                   beta_blink = params['beta_blink'].value, 
                                                                   F_blink = params['F_blink'].value)
        
        pch = self.multi_species_pch_handler(F = params['F'].value,
                                             t_bin = t_bin,
                                             cpms_array = cpms_array,
                                             N_avg_array = N_eff_array,
                                             crop_output = True,
                                             numeric_precision = numeric_precision,
                                             mp_pool = mp_pool
                                             )
        
        if crop_output:
            PCH_n_photons_max = self.PCH_n_photons_max[self.data_PCH_bin_times == t_bin][0]
            pch = pch[:PCH_n_photons_max + 1]

        return pch


    def get_pch_partial_labelling(self,
                                  params,
                                  t_bin,
                                  time_resolved_PCH = False,
                                  crop_output = False,
                                  numeric_precision = 1e-4,
                                  mp_pool = None):
        
        # Unpack parameters
        cpms_0 = params['cpms_0'].value * (1 + params['F'].value) * 2**(-3/2) # Correct for non-Gaussian PSF elements and gamma
        
        n_species_spec, n_species_disc = self.get_n_species(params)
            
        # Extract parameters for discrete and spectrum species, and concatenate them
        stoichiometry_array_s = np.array([params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_array_d = np.array([params[f'stoichiometry_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_array = np.concatenate((stoichiometry_array_s, stoichiometry_array_d), 
                                             axis = 0).astype(np.float64)

        stoichiometry_binwidth_array_s = np.array([params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth_array_d = np.array([params[f'stoichiometry_binwidth_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        stoichiometry_binwidth_array = np.concatenate((stoichiometry_binwidth_array_s, stoichiometry_binwidth_array_d), 
                                                      axis = 0)
        
        labelling_efficiency_array_s = np.array([params[f'Label_efficiency_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        labelling_efficiency_array_d = np.array([params[f'Label_efficiency_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        labelling_efficiency_array = np.concatenate((labelling_efficiency_array_s, labelling_efficiency_array_d), 
                                             axis = 0)

        N_avg_array_s = np.array([params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_avg_array_d = np.array([params[f'N_avg_obs_d_{i_spec}'].value for i_spec in range(n_species_disc)])
        N_eff_array = np.concatenate((N_avg_array_s, N_avg_array_d), 
                                     axis = 0) * stoichiometry_binwidth_array * 2**(3/2) # Binwidth and gamma correction
        
        if time_resolved_PCH:
            # Bin time correction
            tau_diff_array_s = np.array([params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
            tau_diff_array_d = np.array([params[f'tau_diff_d_{i_spec}'].value for i_spec in range(n_species_disc)])
            tau_diff_array = np.concatenate((tau_diff_array_s, tau_diff_array_d), 
                                            axis = 0)
            tau_blink = params['tau_blink'].value, 
            beta_blink = params['beta_blink'].value, 
            F_blink = params['F_blink'].value
        
        # Iterate over species in spectrum
        for i_spec in range(n_species_spec + n_species_disc):
            
            # Get probabilities for different label stoichiometries, 
            # clipping away extremely unlikely ones for computational feasibility
            n_labels_array, p_of_n_labels = self.pch_get_stoichiometry_spectrum(stoichiometry_array[i_spec],
                                                                                labelling_efficiency_array[i_spec],
                                                                                numeric_precision = numeric_precision)
            
            # Use parameters to get full array of PCH parameters
            cpms_array_spec = n_labels_array * cpms_0 #  frequency of 0,1,2,3,... labels
            N_array_spec = N_eff_array[i_spec] * p_of_n_labels # N_eff_array[i_spec] as a kind of global amplitude for the frequencies
            
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
                pch = self.multi_species_pch_handler(F = params['F'].value,
                                                     t_bin = t_bin,
                                                     cpms_array = cpms_array_spec,
                                                     N_avg_array = N_array_spec,
                                                     crop_output = True,
                                                     numeric_precision = numeric_precision,
                                                     mp_pool = mp_pool
                                                     )
            else:
                # Colvolve with further species PCH
                pch = np.convolve(pch,
                                 self.multi_species_pch_handler(F = params['F'].value,
                                                                t_bin = t_bin,
                                                                cpms_array = cpms_array_spec,
                                                                N_avg_array = N_array_spec,
                                                                crop_output = True,
                                                                numeric_precision = numeric_precision,
                                                                mp_pool = mp_pool
                                                                ),
                                 mode = 'full')
            if crop_output:
                PCH_n_photons_max = self.PCH_n_photons_max[self.data_PCH_bin_times == t_bin][0]
                pch = pch[:PCH_n_photons_max + 1]

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


    def wlc_tau_diff_fold_change_deviation(self,
                                           log_j,
                                           tau_diff_fold_change,
                                           oligomer_model_params):
        for param_name in ['r_mono', 'l_Kuhn']:
            if not param_name in oligomer_model_params.keys():
                raise Exception(f"[{self.job_prefix}] Invalid input for oligomer_model_params - must be dict containing key words r_mono and l_Kuhn. Got: {oligomer_model_params}")

        tau_diff_fold_change_model = self.wlc_tau_diff_fold_change(np.exp(log_j), 
                                                                   oligomer_model_params['r_mono'],
                                                                   oligomer_model_params['l_Kuhn'])
        return (tau_diff_fold_change_model - tau_diff_fold_change) ** 2


    def helical_wlc_tau_diff_fold_change_deviation(self,
                                                   log_j,
                                                   tau_diff_fold_change,
                                                   oligomer_model_params):
        for param_name in ['helix_radius', 'helix_pitch', 'r_mono', 'l_Kuhn']:
            if not param_name in oligomer_model_params.keys():
                raise Exception(f"[{self.job_prefix}] Invalid input for oligomer_model_params - must be dict containing key words helix_radius, helix_pitch, r_mono and l_Kuhn. Got: {oligomer_model_params}")

        tau_diff_fold_change_model = self.helical_wlc_tau_diff_fold_change(np.exp(log_j), 
                                                                           oligomer_model_params['helix_radius'],
                                                                           oligomer_model_params['helix_pitch'],
                                                                           oligomer_model_params['r_mono'],
                                                                           oligomer_model_params['l_Kuhn'])
        return (tau_diff_fold_change_model - tau_diff_fold_change) ** 2


    def stoichiometry_bindwiths_from_stoichiometries(self,
                                                     stoichiometry):
        # Each stoichiometry is assigned a binwidth
        stoichiometry_binwidth = np.ones_like(stoichiometry)
        
        # Geometric mean between data points as log-spaced bin edges
        for i_stoi in range(0, stoichiometry.shape[0]-1):
            stoichiometry_binwidth[i_stoi] = np.sqrt(stoichiometry[i_stoi] * stoichiometry[i_stoi + 1])
            
        # Last right bin edge is extrapolated
        stoichiometry_binwidth[-1] = stoichiometry[-1] * np.mean(stoichiometry_binwidth[:-1] / stoichiometry[:-1])
        
        # Binwidth as distance between bin edges
        stoichiometry_binwidth[1:] = np.diff(stoichiometry_binwidth)
        
        return stoichiometry_binwidth


    def stoichiometry_from_tau_diff_array(self,
                                          tau_diff_array,
                                          oligomer_type,
                                          oligomer_model_params = {}
                                          ):
        
        if not utils.isiterable(tau_diff_array):
            raise Exception(f"[{self.job_prefix}] Invalid input for tau_diff_array - must be np.array. Got: {tau_diff_array}")
            
        if not type(oligomer_model_params) == dict:
            raise Exception(f"[{self.job_prefix}] Invalid input for oligomer_model_params - must be dict. Got: {oligomer_model_params}")

        # The monomer by definition has stoichiometry 1, so we start with the second element
        fold_changes = tau_diff_array[1:] / tau_diff_array[0]
        stoichiometry = np.ones_like(tau_diff_array)
        
        if oligomer_type in ['sherical_dense', 'Spherical_dense', 'Spherical_Dense', 'dense_sphere', 'Dense_sphere', 'Dense_Sphere']:
            # tau_diff proportional hydrodyn. radius
            # Stoichiometry proportional volume
            stoichiometry[1:] = fold_changes ** 3
            
            # Round to integer stoichiometries and remove redundant elements
            stoichiometry, _ = np.unique(np.round(stoichiometry),
                                         return_index = True)
            stoichiometry = stoichiometry[stoichiometry > 0]

            # Recalculate exact diffusion times
            tau_diff_array = tau_diff_array[0] * stoichiometry ** (1/3)
            
            # Get (log-space-centered) bin widths
            stoichiometry_binwidth = self.stoichiometry_bindwiths_from_stoichiometries(stoichiometry)
                    
        elif oligomer_type in ['spherical_shell', 'Spherical_shell', 'Spherical_Shell', 'gaussian_chain', 'Gaussian_chain', 'Gaussian_Chain', 'GC', 'gc', 'Gc', 'worm_like_chain', 'Worm_like_chain', 'Worm_Like_Chain', 'wlc', 'WLC', 'Wlc']:
            # tau_diff proportional hydrodyn. radius
            # Stoichiometry proportional surface area or sqrt of Kuhn segment number
            stoichiometry[1:] = fold_changes ** 2

            stoichiometry, _ = np.unique(np.round(stoichiometry),
                                         return_index = True)
            tau_diff_array = tau_diff_array[0] * stoichiometry ** (1/2)
            stoichiometry = stoichiometry[stoichiometry > 0]
            stoichiometry_binwidth = self.stoichiometry_bindwiths_from_stoichiometries(stoichiometry)

        elif oligomer_type in ['single_filament', 'Single_filament', 'Single_Filament']:
            # For the filament models, we have more complicated expressions based
            # on Seils & Pecora 1995. We numerically solve the expression.
            
            for i_spec, tau_diff_fold_change in enumerate(fold_changes[1:]):
                res = sminimize_scalar(fun = self.single_filament_tau_diff_fold_change_deviation, 
                                       args = (tau_diff_fold_change,))
                stoichiometry[i_spec + 1] = np.exp(res.x)
        
            stoichiometry, _ = np.unique(np.round(stoichiometry),
                                         return_index = True)
            stoichiometry = stoichiometry[stoichiometry > 0]
            tau_diff_array = np.append(tau_diff_array[0],
                                       tau_diff_array[0] * self.single_filament_tau_diff_fold_change(stoichiometry[1:]))
            stoichiometry_binwidth = self.stoichiometry_bindwiths_from_stoichiometries(stoichiometry)

        elif oligomer_type in ['double_filament', 'Double_filament', 'Double_Filament']:
        
            stoichiometry = np.zeros_like(fold_changes)
            
            for i_spec, tau_diff_fold_change in enumerate(fold_changes[1:]):

                res = sminimize_scalar(fun = self.double_filament_tau_diff_fold_change_deviation, 
                                       args = (tau_diff_fold_change,))
                stoichiometry[i_spec + 1] = np.exp(res.x)
                
            stoichiometry, _ = np.unique(np.round(stoichiometry),
                                         return_index = True)
            stoichiometry = stoichiometry[stoichiometry > 0]
            tau_diff_array = np.append(tau_diff_array[0],
                                       tau_diff_array[0] * self.double_filament_tau_diff_fold_change(stoichiometry[1:]))
            stoichiometry_binwidth = self.stoichiometry_bindwiths_from_stoichiometries(stoichiometry)

        elif oligomer_type in ['worm_like_chain', 'Worm_like_chain', 'Worm_Like_Chain', 'wlc', 'WLC', 'Wlc']:
            # This is a simple worm-like chain model parameterized by the Kuhn segment length
            stoichiometry = np.zeros_like(fold_changes)
            
            for i_spec, tau_diff_fold_change in enumerate(fold_changes[1:]):

                res = sminimize_scalar(fun = self.wlc_tau_diff_fold_change_deviation, 
                                       args = (tau_diff_fold_change,
                                               oligomer_model_params,))
                stoichiometry[i_spec + 1] = np.exp(res.x)
                
            stoichiometry, _ = np.unique(np.round(stoichiometry),
                                         return_index = True)
            stoichiometry = stoichiometry[stoichiometry > 0]
            tau_diff_array = np.append(tau_diff_array[0],
                                       tau_diff_array[0] * self.wlc_tau_diff_fold_change(stoichiometry[1:],
                                                                                         oligomer_model_params['r_mono'], 
                                                                                         oligomer_model_params['l_Kuhn']))
            stoichiometry_binwidth = self.stoichiometry_bindwiths_from_stoichiometries(stoichiometry)

        elif oligomer_type in ['helical_worm_like_chain', 'Helical_worm_like_chain', 'Helical_Worm_Like_Chain', 'hwlc', 'HWLC', 'Hwlc']:
            # This is the "helical worm-like chain" described in Yamakawa & Yoshizaki 1981 and papers referenced therein
            stoichiometry = np.zeros_like(fold_changes)
            
            for i_spec, tau_diff_fold_change in enumerate(fold_changes[1:]):

                res = sminimize_scalar(fun = self.helical_wlc_tau_diff_fold_change_deviation, 
                                       args = (tau_diff_fold_change,
                                               oligomer_model_params,))
                stoichiometry[i_spec + 1] = np.exp(res.x)
                
            stoichiometry, _ = np.unique(np.round(stoichiometry),
                                         return_index = True)
            stoichiometry = stoichiometry[stoichiometry > 0]
            tau_diff_array = np.append(tau_diff_array[0],
                                       tau_diff_array[0] * self.helical_wlc_tau_diff_fold_change(stoichiometry[1:],
                                                                                                 oligomer_model_params['helix_radius'],
                                                                                                 oligomer_model_params['helix_pitch'],
                                                                                                 oligomer_model_params['r_mono'], 
                                                                                                 oligomer_model_params['l_Kuhn']))
            stoichiometry_binwidth = self.stoichiometry_bindwiths_from_stoichiometries(stoichiometry)

        elif oligomer_type in ['naive', 'Naive']:
            # Dummy ones
            stoichiometry = np.arange(1, tau_diff_array.shape[0] + 1)
            stoichiometry_binwidth = 1. / stoichiometry
            # tau_diff_array remains unchanged
            
        else:
            raise Exception(f"[{self.job_prefix}] Invalid input for oligomer_type - oligomer_type must be one out of 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', 'double_filament', 'Gaussian_chain', 'worm_like_chain', 'helical_worm_like_chain', or certain allowed synonyms. Got: {oligomer_type}")

                    
        return stoichiometry, tau_diff_array, stoichiometry_binwidth
    
       
    def set_blinking_initial_params(self,
                                    initial_params,
                                    use_blinking,
                                    tau_diff_min):
        
        # Blinking parameters 
        if use_blinking:
            initial_params.add('tau_blink', 
                               value = tau_diff_min / 10., 
                               min = self.data_FCS_tau_s.min() / 10. , 
                               max = tau_diff_min, 
                               vary = True)

            initial_params.add('F_blink', 
                               value = 0.1, 
                               min = 0., 
                               max = 1., 
                               vary = True)

            initial_params.add('beta_blink', 
                               value = 1., 
                               min = 0.5, 
                               max = 2., 
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
                               use_blinking,
                               use_avg_count_rate = False,
                               fit_label_efficiency = False
                               ):
        
        if use_FCS and not self.FCS_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run PCH fit - not all required attributes set in class')
            
        if not (use_FCS or use_PCH):
            raise Exception(f'[{self.job_prefix}] Cannot run fit - at least one out of use_FCS or use_PCH must be True, got False for both!')

        if use_PCH and time_resolved_PCH and self.FCS_psf_aspect_ratio == None:
            raise Exception(f'[{self.job_prefix}] Cannot run PCMH fit - PSF aspect ratio must be set')

        if not (utils.isint(n_species) and n_species > 0):
            raise Exception(f"[{self.job_prefix}] Invalid input for n_species - must be int > 0. Got {n_species}")
        
        if not (utils.isfloat(tau_diff_min) and tau_diff_min > 0):
            raise Exception(f"[{self.job_prefix}] Invalid input for tau_diff_min - must be float > 0. Got {tau_diff_min}")

        if not (utils.isfloat(tau_diff_max) and tau_diff_max > 0):
            raise Exception(f"[{self.job_prefix}] Invalid input for tau_diff_max - must be float > 0. Got {tau_diff_max}")

        if type(use_blinking) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for use_blinking - must be bool. Got {use_blinking}")

        if type(use_avg_count_rate) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for use_avg_count_rate - must be bool. Got {use_avg_count_rate}")

        if use_avg_count_rate and not self.avg_count_rate_given:
            raise Exception(f"[{self.job_prefix}] Average count rate not specified and thus cannot be included in fit.")
            
        if type(fit_label_efficiency) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for fit_label_efficiency - must be bool. Got {fit_label_efficiency}")
        elif fit_label_efficiency and not labelling_correction:
            raise Exception(f"[{self.job_prefix}] Invalid input of labelling_correction==False and fit_label_efficiency==True: The labelling efficiency cannot be fitted without labelling correction!")

        # We include a reference to a custom function
        initial_params = lmfit.Parameters(usersyms = {'invgamma_helper':self.invgamma_helper})
        
        # More technical parameters
        initial_params.add('Label_efficiency', 
                           value = self.labelling_efficiency if labelling_correction else 1.,
                           min = 0.,
                           max = 1.,
                           vary = fit_label_efficiency)
        
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
            initial_params.add(f'N_avg_obs_d_{i_spec}', 
                               value = 1., 
                               min = 0., 
                               vary = True)

            if use_FCS or (use_PCH and time_resolved_PCH):
                # Diffusion time only for FCS and PCMH
                initial_params.add(f'tau_diff_d_{i_spec}', 
                                   value = np.sqrt(tau_diff_min * tau_diff_max) *10**(i_spec + 0.5 - n_species / 2), # Initialize around geo mean of bounds
                                   min = tau_diff_min,
                                   max = tau_diff_max,
                                   vary = True)
            
            if use_PCH or use_avg_count_rate:
                initial_params.add(f'cpms_d_{i_spec}', 
                                   value = 1E4 * 10**(i_spec + 0.5 - n_species / 2), # Initialize around 10000
                                   min = 1E-3, 
                                   vary = True)
            else:
                # If we do not use PCH or the avg count rate, use a dummy, as we won't be able to tell from FCS alone
                initial_params.add(f'cpms_d_{i_spec}', 
                                   value = 1., 
                                   min = 1E-3, 
                                   vary = False)
                
            # Additional parameters that are included only as dummies for 
            # consistency with spectrum models, to be able to use the same 
            # functions without too much added logic
            initial_params.add(f'Label_obs_factor_d_{i_spec}', 
                               value = 1.,
                               vary = False)

            initial_params.add(f'Label_efficiency_obs_d_{i_spec}', 
                               expr = f'Label_efficiency * Label_obs_factor_d_{i_spec}',
                               vary = False)

            initial_params.add(f'stoichiometry_binwidth_d_{i_spec}', 
                               value = 1., 
                               vary = False)

            initial_params.add(f'stoichiometry_d_{i_spec}', 
                               value = 1., 
                               vary = False)

                
        # Add blinking parameters for FCS and PCMH - real or dummy
        initial_params = self.set_blinking_initial_params(initial_params,
                                                          use_blinking and (use_FCS or (use_PCH and time_resolved_PCH)),
                                                          tau_diff_min)

        return initial_params

    def add_discrete_species(self,
                             initial_params,
                             N_avg_obs = 1.,
                             vary_N_avg_obs = True,                             
                             tau_diff = 1E-3,
                             vary_tau_diff = False,
                             cpms = 1.,
                             vary_cpms = False,
                             link_brightness_to_spectrum_monomer = True,
                             stoichiometry = 1.,
                             vary_stoichiometry = False,
                             stoichiometry_binwidth = 1.,
                             vary_stoichiometry_binwidth = False,
                             labelling_efficiency = 1.,
                             vary_labelling_efficiency = False,
                             ):
        '''
        Handle for adding a single species to an existing initial_params object.
        Here, you have free access to all the parameters!

        Parameters
        ----------
        initial_params : 
            lmfit.Parameters() object to be appended. Should be created with 
            set_up_params_par() or similar.
        N_avg_obs, cpms, tau_diff, stoichiometry, stoichiometry_binwidth, labelling_efficiency :
            OPITONAL floats as initial estimates for the 6 parameters that
            characterize a species in this software. tau_diff has default 1E-3,
            the others default to 1.
        vary_N_avg_obs, vary_cpms, vary_tau_diff, vary_stoichiometry, vary_stoichiometry_binwidth, vary_labelling_efficiency : 
            OPTIONAL bools to choose whether or not to vary the parameters in 
            fitting, i.e., this is the handle for deciding what is prior 
            knowledge and what is fitted. vary_N_avg_obs has default True, the 
            others have default False.
        link_brightness_to_spectrum_monomer : 
            OPTIONAL bool with default True. This is a key parameter that 
            determines whether or not this species is linked to the monomer of 
            an oligomer spectrum. This is a bit more complicated:
            link_brightness_to_spectrum_monomer == False: The brightness of the 
                new species is handled trivially according to inputs for cpms
                and vary_cpms, like the other parameters, with no connection 
                to the spectrum.
            link_brightness_to_spectrum_monomer == True and vary_cpms == True: 
                In this case, the input for cpms into this function is ignored!
                Instead, this species' brightness becomes a dependent parameter
                tethered to the spectrum monomer brightness cpms_0
            link_brightness_to_spectrum_monomer == True and vary_cpms == False:
                In this case, the input for cpms is treated as a known input
                constant (e.g., from a monomer-only measurement) and both the
                new species's brightness and the spectrum monomer brightness
                become fixed at this value.

        Returns
        -------
        initial_params : 
            lmfit.Parameters() with additional species.

        '''
        
        # Find out what is the highest index of exisitng discrete species
        n_species_spec, n_species_disc = self.get_n_species(initial_params)
        
        # With the brightness we can do a few interesting things...
        
        if link_brightness_to_spectrum_monomer:
            # We use this species to describe a free-monomer or free-dye species with linked brightness
            
            if n_species_spec == 0:
                raise Exception('No spectrum of species defined, therefore cannot add a discrete species with brightness linked to spectrum monomer brightness!')
                
            if vary_cpms:
                # In this configuration, we vary the spectrum cpms_0 and link this species to it
                initial_params.add(f'cpms_d_{n_species_disc}',
                                   expr = 'cpms_0',
                                   vary = False)
                
            else:
                 # In this configuration, we take the input value specified for this species to express the spectrum cpms_0  
                 initial_params.add(f'cpms_d_{n_species_disc}',
                                    value = cpms,
                                    min = 1E-3,
                                    vary = False)
                 
                 initial_params['cpms_0'].vary = False
                 initial_params['cpms_0'].value = None
                 initial_params['cpms_0'].expr = f'cpms_d_{n_species_disc}'

        else:
            # No linking this species to spectrum monomer - then it's trivial
            initial_params.add(f'cpms_d_{n_species_disc}',
                               value = cpms,
                               min = 1E-3,
                               vary = vary_cpms)
            
        # The other parameter are all straightforward
        initial_params.add(f'N_avg_obs_d_{n_species_disc}',
                           value = N_avg_obs,
                           min = 0.,
                           vary = vary_N_avg_obs)

        initial_params.add(f'tau_diff_d_{n_species_disc}',
                           value = tau_diff,
                           min = self.data_FCS_tau_s.min() if self.FCS_possible else 0.,
                           max = self.data_FCS_tau_s.max() if self.FCS_possible else np.inf,
                           vary = vary_tau_diff)

        initial_params.add(f'stoichiometry_d_{n_species_disc}',
                           value = stoichiometry,
                           min = 1.,
                           vary = vary_stoichiometry)

        initial_params.add(f'stoichiometry_binwidth_d_{n_species_disc}',
                           value = stoichiometry_binwidth,
                           min = 0.,
                           vary = vary_stoichiometry_binwidth)

        initial_params.add(f'Label_efficiency_obs_d_{n_species_disc}',
                           value = labelling_efficiency,
                           min = 0.,
                           max = 1.,
                           vary = vary_labelling_efficiency)
        
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
                          use_blinking,
                          oligomer_model_params = {},
                          use_avg_count_rate = False,
                          fit_label_efficiency = False
                          ):
    
        if use_FCS and not self.FCS_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run PCH fit - not all required attributes set in class')
                    
        if not (use_FCS or use_PCH):
            raise Exception(f'[{self.job_prefix}] Cannot run fit - at least one out of use_FCS or use_PCH must be True, got False for both!')

        if use_PCH and time_resolved_PCH and self.FCS_psf_aspect_ratio == None:
            raise Exception(f'[{self.job_prefix}] Cannot run PCMH fit - PSF aspect ratio must be set')
            
        if not spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_type for set_up_params_reg - must be one out of 'reg_MEM', 'reg_CONTIN'. Got: {spectrum_type}")

        if not (utils.isint(n_species) and n_species >= 10):
            raise Exception(f"[{self.job_prefix}] Invalid input for n_species - must be int >= 10 for regularized fitting. Got: {n_species}")
            
        if type(use_avg_count_rate) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for use_avg_count_rate - must be bool. Got {use_avg_count_rate}")

        if use_avg_count_rate and not self.avg_count_rate_given:
            raise Exception(f"[{self.job_prefix}] Average count rate not specified and thus cannot be included in fit.")

        if type(fit_label_efficiency) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for fit_label_efficiency - must be bool. Got {fit_label_efficiency}")
        elif fit_label_efficiency and not labelling_correction:
            raise Exception(f"[{self.job_prefix}] Invalid input of labelling_correction==False and fit_label_efficiency==True: The labelling efficiency cannot be fitted without labelling correction!")

            
        tau_diff_array = self.get_tau_diff_array(tau_diff_min, 
                                                 tau_diff_max, 
                                                 n_species)
        
        stoichiometry, tau_diff_array, stoichiometry_binwidth = self.stoichiometry_from_tau_diff_array(tau_diff_array, 
                                                                                                       oligomer_type,
                                                                                                       oligomer_model_params)
        # Can be fewer here than originally intended, depending on settings
        n_species = stoichiometry.shape[0]
        
        # We include a reference to a custom function
        initial_params = lmfit.Parameters(usersyms = {'invgamma_helper':self.invgamma_helper})

        # More technical parameters
        if use_FCS:
            initial_params.add('acf_offset', 
                                value = 0., 
                                vary = False)

        if use_PCH:
            initial_params.add('F', 
                               value = 0.4, 
                               min = 0, 
                               max = 1.,
                               vary=True)

        initial_params.add('Label_efficiency', 
                           value = self.labelling_efficiency if labelling_correction else 1.,
                           min = 0.,
                           max = 1.,
                           vary = fit_label_efficiency)


        # Species-wise parameters
        for i_spec, tau_diff_i in enumerate(tau_diff_array):

            if incomplete_sampling_correction:
                # Allow fluctuations of observed apparent particle count
                initial_params.add(f'N_avg_obs_{i_spec}', 
                                   value = 1., 
                                   min = 0., 
                                   vary = True)
            else:
                pass
                # in that case, this variable is not accessed at all

            if use_FCS or (use_PCH and time_resolved_PCH):
                # Diffusion time only for FCS and PCMH
                initial_params.add(f'tau_diff_{i_spec}', 
                                   value = tau_diff_array[i_spec], 
                                   vary = False)
            
            initial_params.add(f'stoichiometry_{i_spec}', 
                               value = stoichiometry[i_spec], 
                               vary = False)

            initial_params.add(f'stoichiometry_binwidth_{i_spec}', 
                               value = stoichiometry_binwidth[i_spec], 
                               vary = False)

            # An additional factor for translating between "sample-level" and
            # "population-level" observed label efficiency if and only if we 
            # use both incomplete_sampling_correction and labelling_correction
            initial_params.add(f'Label_obs_factor_{i_spec}', 
                               value = 1.,
                               vary = incomplete_sampling_correction and labelling_correction and self.labelling_efficiency_incomp_sampling)

            initial_params.add(f'Label_efficiency_obs_{i_spec}', 
                               expr = f'Label_efficiency * Label_obs_factor_{i_spec}',
                               vary = False)

            if i_spec == 0:
                
                # Monomer brightness
                if use_PCH or use_avg_count_rate:
                    initial_params.add('cpms_0', 
                                       value = 1E3, 
                                       min =  1E-3, 
                                       vary = True)
                else:
                    # If we do not use PCH or avg count rate, use a dummy, as we won't be able to tell from FCS alone
                    initial_params.add('cpms_0', 
                                       value = 1., 
                                       min =  1E-3,
                                       vary = False)
            else: # i_spec >= 1
            
                # Oligomer cpms is defined by monomer and stoichiometry factor
                initial_params.add(f'cpms_{i_spec}', 
                                   expr = f'cpms_0 * stoichiometry_{i_spec}', 
                                   vary = False)
            
        # Add blinking parameters for FCS and PCMH - real or dummy
        initial_params = self.set_blinking_initial_params(initial_params,
                                                          use_blinking and (use_FCS or (use_PCH and time_resolved_PCH)),
                                                          tau_diff_min)


        # Pre-calculate Normalized species correlation functions that we would 
        # otherwise spend a significant amount of time on during fitting
        if use_FCS:
            n_lag_times = self.data_FCS_G.shape[0]
            tau_diff__tau_array = np.zeros((n_species, n_lag_times))
            for i_spec in range(n_species):
                tau_diff__tau_array[i_spec,:] = self.fcs_3d_diff_single_species(tau_diff_array[i_spec])
            self.tau__tau_diff_array = np.transpose(tau_diff__tau_array)

        return initial_params



    def set_up_params_reg_from_parfit(self,
                                      gauss_fit_params,
                                      use_FCS,
                                      use_PCH,
                                      time_resolved_PCH,
                                      spectrum_type,
                                      oligomer_type,
                                      incomplete_sampling_correction,
                                      labelling_correction,
                                      tau_diff_min,
                                      tau_diff_max,
                                      use_blinking,
                                      use_avg_count_rate = False,
                                      fit_label_efficiency = False
                                      ):
    
        # OLD STUFF, NOT FUNCTIONAL ANY MORE
        if use_FCS and not self.FCS_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run PCH fit - not all required attributes set in class')
                    
        if use_PCH and time_resolved_PCH and self.FCS_psf_aspect_ratio == None:
            raise Exception(f'[{self.job_prefix}] Cannot run PCMH fit - PSF aspect ratio must be set')
            
        if not spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_type for set_up_params_reg_from_parfit - must be one out of 'reg_MEM', 'reg_CONTIN'. Got: {spectrum_type}")

        if not (oligomer_type in ['naive', 'spherical_shell', 'sherical_dense', 'single_filament', 'double_filament', 'Gaussian_chain']):
            raise Exception(f"[{self.job_prefix}] Invalid input for oligomer_type - oligomer_type must be one out of 'naive', 'spherical_shell', 'sherical_dense', 'single_filament', 'double_filament', or 'Gaussian_chain'. Got: {oligomer_type}")
            
        if type(use_avg_count_rate) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for use_avg_count_rate - must be bool. Got {use_avg_count_rate}")

        if use_avg_count_rate and not self.avg_count_rate_given:
            raise Exception(f"[{self.job_prefix}] Average count rate not specified and thus cannot be included in fit.")

        if type(fit_label_efficiency) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for fit_label_efficiency - must be bool. Got {fit_label_efficiency}")
        elif fit_label_efficiency and not labelling_correction:
            raise Exception(f"[{self.job_prefix}] Invalid input of labelling_correction==False and fit_label_efficiency==True: The labelling efficiency cannot be fitted without labelling correction!")

            
        # Extract a bunch of arrays from Gauss fit results
        n_species_spec, _ = self.get_n_species(gauss_fit_params)
        tau_diff_array = np.array([gauss_fit_params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry = np.array([gauss_fit_params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species_spec)])
        stoichiometry_binwidth = np.array([gauss_fit_params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species_spec)])
        N_pop_array = np.array([gauss_fit_params[f'N_avg_obs_{i_spec}'].value for i_spec in range(n_species_spec)])
                    
        initial_params = lmfit.Parameters()
        
        # More technical parameters
        if use_FCS:

            initial_params.add('acf_offset', 
                                value = 0., 
                                vary = False)

        if use_PCH:
            initial_params.add('F', 
                                value = gauss_fit_params['F'].value, 
                                min = 0, 
                                max = 1.,
                                vary=True)

        initial_params.add('Label_efficiency', 
                            value = self.labelling_efficiency if labelling_correction else 1.,
                            min = 0.,
                            max = 1.,
                            vary = fit_label_efficiency)


        # Species-wise parameters
        
        
        # Other parameters tuned in the nested Nelder minimization
        for i_spec, tau_diff_i in enumerate(tau_diff_array):
            

            if incomplete_sampling_correction:
                # Allow fluctuations of observed apparent particle count
                initial_params.add(f'N_avg_obs_{i_spec}', 
                                    value = gauss_fit_params[f'N_avg_obs_{i_spec}'].value, 
                                    min = 0., 
                                    vary = True)
            else:
                # Dummy
                initial_params.add(f'N_avg_obs_{i_spec}', 
                                    value = 1., 
                                    vary = False)

            if use_FCS or (use_PCH and time_resolved_PCH):
                # Diffusion time only for FCS and PCMH
                initial_params.add(f'tau_diff_{i_spec}', 
                                    value = tau_diff_array[i_spec], 
                                    vary = False)
            
            initial_params.add(f'stoichiometry_{i_spec}', 
                                value = stoichiometry[i_spec], 
                                vary = False)

            initial_params.add(f'stoichiometry_binwidth_{i_spec}', 
                                value = stoichiometry_binwidth[i_spec], 
                                vary = False)

            # An additional factor for translating between "sample-level" and
            # "population-level" observed label efficiency if and only if we 
            # use both incomplete_sampling_correction and labelling_correction
            initial_params.add(f'Label_obs_factor_{i_spec}', 
                                value = gauss_fit_params[f'Label_obs_factor_{i_spec}'].value,
                                vary = incomplete_sampling_correction and labelling_correction and self.labelling_efficiency_incomp_sampling)

            initial_params.add(f'Label_efficiency_obs_{i_spec}', 
                                expr = f'Label_efficiency * Label_obs_factor_{i_spec}',
                                vary = False)

            if i_spec == 0:
                
                # Monomer brightness
                if use_PCH or use_avg_count_rate:
                    initial_params.add('cpms_0', 
                                        value = gauss_fit_params['cpms_0'].value, 
                                        min =  1E-3,
                                        vary = True)
                else:
                    # If we do not use PCH or avg_count_rate, use a dummy, as we won't be able to tell from FCS alone
                    initial_params.add('cpms_0', 
                                        value = 1., 
                                        min =  1E-3,
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

        return initial_params, N_pop_array


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
                          use_blinking,
                          oligomer_model_params = {},
                          previous_params = None,
                          previous_N_obs_array = None,
                          skip_species_mask = None,
                          use_avg_count_rate = False,
                          fixed_spectrum_params = {},
                          fit_label_efficiency = False
                          ):
    
        if use_FCS and not self.FCS_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run PCH fit - not all required attributes set in class')

        if not (use_FCS or use_PCH):
            raise Exception(f'[{self.job_prefix}] Cannot run fit - at least one out of use_FCS or use_PCH must be True, got False for both!')

        if use_PCH and time_resolved_PCH and self.FCS_psf_aspect_ratio == None:
            raise Exception(f'[{self.job_prefix}] Cannot run PCMH fit - PSF aspect ratio must be set')

        if not spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_type for set_up_params_par - must be one out of 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'")

        if not (utils.isint(n_species) and n_species >= 10):
            raise Exception(f"[{self.job_prefix}] Invalid input for n_species - must be int >= 10 for parameterized spectrum fitting")
        
        if type(use_avg_count_rate) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for use_avg_count_rate - must be bool. Got {use_avg_count_rate}")

        if use_avg_count_rate and not self.avg_count_rate_given:
            raise Exception(f"[{self.job_prefix}] Average count rate not specified and thus cannot be included in fit.")

        if type(fixed_spectrum_params) != dict:
            raise Exception(f"[{self.job_prefix}] Invalid input for fixed_spectrum_params - must be dict. Got {fixed_spectrum_params}")
        elif np.any([1 for key in fixed_spectrum_params.keys() if not key in ['N_dist_amp', 'N_dist_a', 'N_dist_b', 'j_avg_N_oligo']]):
            raise Exception(f"[{self.job_prefix}] Invalid key fixed_spectrum_params - can only be 'N_dist_amp', 'N_dist_a', 'N_dist_b', 'j_avg_N_oligo'. Got {fixed_spectrum_params.keys}")
        # Set np.nan defaults for unused parameters
        if not ('N_dist_amp' in fixed_spectrum_params.keys()) or ('N_dist_amp' in fixed_spectrum_params.keys() and not fixed_spectrum_params['N_dist_amp'] > 0.):
            fixed_spectrum_params['N_dist_amp'] = np.nan
        if not ('N_dist_a' in fixed_spectrum_params.keys()) or ('N_dist_a' in fixed_spectrum_params.keys() and not fixed_spectrum_params['N_dist_a'] > 0.):
            fixed_spectrum_params['N_dist_a'] = np.nan
        if not ('N_dist_b' in fixed_spectrum_params.keys()) or ('N_dist_b' in fixed_spectrum_params.keys() and not fixed_spectrum_params['N_dist_b'] > 0.):
            fixed_spectrum_params['N_dist_b'] = np.nan
        if not ('j_avg_N_oligo' in fixed_spectrum_params.keys()) or ('j_avg_N_oligo' in fixed_spectrum_params.keys() and not fixed_spectrum_params['j_avg_N_oligo'] > 0.):
            fixed_spectrum_params['j_avg_N_oligo'] = np.nan
            N_dist_b_expr = ''
            
        else: # fixed_spectrum_params['j_avg_N_oligo'] > 0.
            # We have a j_avg_N_oligo, more complex and we overwrite a and b based on this parameter
            if spectrum_type == 'par_Gauss':
                # a is mean here, so that's trivial
                fixed_spectrum_params['N_dist_a'] = fixed_spectrum_params['j_avg_N_oligo']
                fixed_spectrum_params['N_dist_b'] = np.nan
                N_dist_b_expr = ''
                
            elif spectrum_type == 'par_LogNorm':
                # For this and all that follow, mean and a define b
                fixed_spectrum_params['N_dist_a'] = np.nan
                fixed_spectrum_params['N_dist_b'] = np.nan
                N_dist_b_expr = f'exp(sqrt(2*(log({fixed_spectrum_params["j_avg_N_oligo"]}) - log(N_dist_a))))'
            elif spectrum_type == 'par_Gamma':
                fixed_spectrum_params['N_dist_a'] = np.nan
                fixed_spectrum_params['N_dist_b'] = np.nan
                N_dist_b_expr = f'N_dist_a /{fixed_spectrum_params["j_avg_N_oligo"]}'
                
            else: # Implies spectrum_type == 'par_StrExp'
                fixed_spectrum_params['N_dist_a'] = np.nan
                fixed_spectrum_params['N_dist_b'] = np.nan
                N_dist_b_expr = f'invgamma_helper({fixed_spectrum_params["j_avg_N_oligo"]}, N_dist_a)'


        if type(fit_label_efficiency) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for fit_label_efficiency - must be bool. Got {fit_label_efficiency}")
        elif fit_label_efficiency and not labelling_correction:
            raise Exception(f"[{self.job_prefix}] Invalid input of labelling_correction==False and fit_label_efficiency==True: The labelling efficiency cannot be fitted without labelling correction!")

        
        if type(previous_params) == lmfit.Parameters:

            from_scratch = False
            
            # If we do have previous_params, do we also use previous_N_pop_array?
            if utils.isiterable(previous_N_obs_array):
                previous_N_obs_array = np.array(previous_N_obs_array)
                use_previous_N_obs_array = True
            else:
                use_previous_N_obs_array = False
                
            # If we use previous information, we can also use a boolean array 
            # skip_species to indicate which species to remove in the re-fit. Do we have that?
            skip_species = utils.isiterable(skip_species_mask)
            if skip_species:
                skip_species_mask = np.array(skip_species_mask, dtype = np.bool8)

        else:

            from_scratch = True
            use_previous_N_obs_array = False # Actually unused if from_scratch, but whatever
            skip_species = False # Actually unused if from_scratch, but whatever
        
        # We include a reference to a custom function
        initial_params = lmfit.Parameters(usersyms = {'invgamma_helper':self.invgamma_helper})

        if from_scratch:

            tau_diff_array = self.get_tau_diff_array(tau_diff_min, 
                                                     tau_diff_max, 
                                                     n_species)        
            stoichiometry, tau_diff_array, stoichiometry_binwidth = self.stoichiometry_from_tau_diff_array(tau_diff_array, 
                                                                                                           oligomer_type,
                                                                                                           oligomer_model_params)
            stoichiometry = stoichiometry.astype(np.float64)
            
            # Can be fewer here than originally intended, depending on settings
            n_species = stoichiometry.shape[0]
            
        
            
    
            # More technical parameters
            if use_FCS:
                initial_params.add('acf_offset', 
                                    value = 0., 
                                    vary=False)
    
            if use_PCH:
                initial_params.add('F', 
                                   value = 0.4, 
                                   min = 0, 
                                   max = 1.,
                                   vary=True)
                
            initial_params.add('Label_efficiency', 
                               value = self.labelling_efficiency if labelling_correction else 1.,
                               min = 0.,
                               max = 1.,
                               vary = fit_label_efficiency)
            
            # N distribution parameters
            N_dist_amp = 1. if np.isnan(fixed_spectrum_params['N_dist_amp']) else fixed_spectrum_params['N_dist_amp']
            initial_params.add('N_dist_amp', 
                               value = N_dist_amp, 
                               min = 0., 
                               vary = np.isnan(fixed_spectrum_params['N_dist_amp']))
            N_dist_a = 0.5 if np.isnan(fixed_spectrum_params['N_dist_a']) else fixed_spectrum_params['N_dist_a']
            initial_params.add('N_dist_a', 
                               value = N_dist_a, 
                               min = 0., 
                               vary = np.isnan(fixed_spectrum_params['N_dist_a']) and len(N_dist_b_expr) == 0) 
    
            
            if  len(N_dist_b_expr) == 0:
                N_dist_b = 10. if np.isnan(fixed_spectrum_params['N_dist_b']) else fixed_spectrum_params['N_dist_b']
                initial_params.add('N_dist_b', 
                                   value = N_dist_b, 
                                   min = 0.01,
                                   vary = np.isnan(fixed_spectrum_params['N_dist_b']))
            else:
                initial_params.add('N_dist_b', 
                                   expr = N_dist_b_expr,
                                   vary = False)

            print(f'N params setup: initial_params["N_dist_amp"]: {initial_params["N_dist_amp"].value}')
            print(f'N params setup: initial_params["N_dist_a"]: {initial_params["N_dist_a"].value}')
            print(f'N params setup: initial_params["N_dist_b"]: {initial_params["N_dist_b"].value}')
            
            for i_spec, tau_diff_i in enumerate(tau_diff_array):
                if use_FCS or (use_PCH and time_resolved_PCH):
                    # Diffusion time only for FCS and PCMH
                    initial_params.add(f'tau_diff_{i_spec}', 
                                       value = tau_diff_array[i_spec], 
                                       vary = False)
                
                initial_params.add(f'stoichiometry_{i_spec}', 
                                   value = stoichiometry[i_spec], 
                                   vary = False)
                
                initial_params.add(f'stoichiometry_binwidth_{i_spec}', 
                                   value = stoichiometry_binwidth[i_spec], 
                                   vary = False)
    
                # An additional factor for translating between "sample-level" and
                # "population-level" observed label efficiency if and only if we 
                # use both incomplete_sampling_correction and labelling_correction
                # (if not, this is a dummy variable kept for consistency)
                initial_params.add(f'Label_obs_factor_{i_spec}', 
                                   value = 1.,
                                   vary = incomplete_sampling_correction and labelling_correction and self.labelling_efficiency_incomp_sampling)
    
                initial_params.add(f'Label_efficiency_obs_{i_spec}', 
                                   expr = f'Label_efficiency * Label_obs_factor_{i_spec}',
                                   vary = False)

                # Weighting function that essentially decides which number the parameterization acts on
                if spectrum_parameter == 'Amplitude' and not oligomer_type == 'naive':
                    spectrum_weight_str = f'stoichiometry_{i_spec}**(-2) / stoichiometry_binwidth_{i_spec}'
                    if labelling_correction:
                        spectrum_weight_str += f' / ( 1. + (1 - Label_efficiency_obs_{i_spec}) / (Label_efficiency_obs_{i_spec} * stoichiometry_{i_spec}))'
                elif spectrum_parameter == 'N_monomers' and not oligomer_type == 'naive':
                    spectrum_weight_str = f'stoichiometry_{i_spec}**(-1)'
                    
                elif spectrum_parameter == 'N_oligomers' or oligomer_type == 'naive':
                    spectrum_weight_str = '1.'
                else:
                    raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_parameter - must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
                initial_params.add(f'spectrum_weight_{i_spec}', 
                                   expr = spectrum_weight_str, 
                                   vary = False)
                
                
                # Define particle numbers via parameterized distributions
                if spectrum_type == 'par_Gauss':
                    initial_params.add(f'N_avg_pop_{i_spec}', 
                                       expr = f'N_dist_amp * spectrum_weight_{i_spec} * exp(-0.5 * ((stoichiometry_{i_spec} - N_dist_a) / N_dist_b) ** 2)', 
                                       vary = False)
                    
                    if incomplete_sampling_correction:
                        # Allow fluctuations of observed apparent particle count
                        # We increment the value a bit to avoid starting too close to 0
                        initial_params.add(f'N_avg_obs_{i_spec}', 
                                           value = N_dist_amp  * initial_params[f'spectrum_weight_{i_spec}'].value * np.exp(-0.5 * ((stoichiometry[i_spec] - N_dist_a) / N_dist_b) ** 2) + 1E-9,
                                           min = 0., 
                                           vary = True)
                        
                if spectrum_type == 'par_LogNorm':
                    initial_params.add(f'N_avg_pop_{i_spec}', 
                                       expr = f'N_dist_amp * spectrum_weight_{i_spec} / stoichiometry_{i_spec} * exp(-0.5 * ((log(stoichiometry_{i_spec}) - log(N_dist_a)) / log(N_dist_b)) ** 2)',
                                       vary = False)
                    
                    if incomplete_sampling_correction:
                        initial_params.add(f'N_avg_obs_{i_spec}', 
                                           value = N_dist_amp * initial_params[f'spectrum_weight_{i_spec}'].value / stoichiometry[i_spec] * np.exp(-0.5 * ((np.log(stoichiometry[i_spec]) - np.log(N_dist_a)) / np.log(N_dist_b)) ** 2) + 1E-9,
                                           min = 0., 
                                           vary = True)
                if spectrum_type == 'par_Gamma':
                    initial_params.add(f'N_avg_pop_{i_spec}', 
                                       expr = f'N_dist_amp * spectrum_weight_{i_spec} * stoichiometry_{i_spec}**(N_dist_a - 1) * exp(- N_dist_b * stoichiometry_{i_spec})', 
                                       vary = False)
                    
                    if incomplete_sampling_correction:
                        initial_params.add(f'N_avg_obs_{i_spec}', 
                                           value = N_dist_amp * initial_params[f'spectrum_weight_{i_spec}'].value * stoichiometry[i_spec]**(N_dist_a - 1) * np.exp(- N_dist_b * stoichiometry[i_spec]) + 1E-9,
                                           min = 0., 
                                           vary = True)
    
                if spectrum_type == 'par_StrExp':
                    initial_params.add(f'N_avg_pop_{i_spec}', 
                                       expr = f'N_dist_amp * spectrum_weight_{i_spec} * exp(-(stoichiometry_{i_spec} * N_dist_a) ** N_dist_b)', 
                                       vary = False)
    
                    if incomplete_sampling_correction:
                        initial_params.add(f'N_avg_obs_{i_spec}', 
                                           value = N_dist_amp * initial_params[f'spectrum_weight_{i_spec}'].value * np.exp(-(stoichiometry[i_spec] * N_dist_a) ** N_dist_b) + 1E-9,
                                           min = 0., 
                                           vary = True)
                        
                if not incomplete_sampling_correction:
                    # Dummy expression does not depend on model
                    initial_params.add(f'N_avg_obs_{i_spec}', 
                                       expr = f'N_avg_pop_{i_spec}', 
                                       vary = False)
    
                    
                if i_spec == 0:
                    
                    # Monomer brightness
                    if use_PCH or use_avg_count_rate:
                        initial_params.add('cpms_0', 
                                           value = 1E3, 
                                           min =  1E-3, 
                                           vary = True)
                    else:
                        # If we do not use PCH or avg count rate, use a dummy, as we won't be able to tell from FCS alone
                        initial_params.add('cpms_0', 
                                           value = 1., 
                                           min =  1E-3,
                                           vary = False)
                        
                else: # i_spec >= 1
                    # Oligomer cpms is defined by monomer and stoichiometry factor
                    initial_params.add(f'cpms_{i_spec}', 
                                       expr = f'cpms_0 * stoichiometry_{i_spec}', 
                                       vary = False)
                
            # Add blinking parameters for FCS and PCMH - real or dummy
            initial_params = self.set_blinking_initial_params(initial_params,
                                                              use_blinking and (use_FCS or (use_PCH and time_resolved_PCH)),
                                                              tau_diff_min)
            
            # Pre-calculate Normalized species correlation functions that we would 
            # otherwise spend a significant amount of time on during fitting
            if use_FCS:

                n_lag_times = self.data_FCS_G.shape[0]
                tau_diff__tau_array = np.zeros((n_species, n_lag_times))
                for i_spec in range(n_species):
                    tau_diff__tau_array[i_spec,:] = self.fcs_3d_diff_single_species(tau_diff_array[i_spec])
                self.tau__tau_diff_array = np.transpose(tau_diff__tau_array)
            
            
            
        else: # not from_scratch - use previous        
            
            n_species, _ = self.get_n_species(previous_params)
        
            if use_FCS:
                initial_params.add('acf_offset', 
                                    value = previous_params['acf_offset'].value,
                                    vary=True)
    
            if use_PCH:
                initial_params.add('F', 
                                   value = previous_params['F'].value if 'F' in previous_params.keys() else 0.4,
                                   min = 0, 
                                   max = 1.,
                                   vary=True)
                
            initial_params.add('Label_efficiency', 
                               value = self.labelling_efficiency if labelling_correction else 1.,
                               min = 0.,
                               max = 1.,
                               vary = fit_label_efficiency)
            
            
            
            if not use_previous_N_obs_array:
                # Read out from previous_params or set to default
                N_dist_amp = previous_params['N_dist_amp'].value if 'N_dist_amp' in previous_params.keys() else 1.
                N_dist_a = previous_params['N_dist_a'].value if 'N_dist_a' in previous_params.keys() else 10.
                N_dist_b = previous_params['N_dist_b'].value if 'N_dist_b' in previous_params.keys() else 0.5

            else:
                # Recalculate from previous_N_obs_array by least-squares fitting the parameterized model onto the array
                
                # Get spectrum weights that "encode" the type of spectrum to be fitted
                # stoichiometry_array = np.array([previous_params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
                
                if skip_species:
                    # There are species to discard - in that case we also need to recalculate stoichiometry parameters!
                    tau_diff_array = np.array([previous_params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])
                    tau_diff_array = tau_diff_array[np.logical_not(skip_species_mask)]
                    stoichiometry_array, tau_diff_array, stoichiometry_binwidth_array = self.stoichiometry_from_tau_diff_array(tau_diff_array,
                                                                                                                               oligomer_type,
                                                                                                                               oligomer_model_params)

                else:
                    stoichiometry_array = np.array([previous_params[f'stoichiometry_{i_spec}'].value for i_spec in range(n_species)])
                    tau_diff_array = np.array([previous_params[f'tau_diff_{i_spec}'].value for i_spec in range(n_species)])
                    stoichiometry_binwidth_array = np.array([previous_params[f'stoichiometry_binwidth_{i_spec}'].value for i_spec in range(n_species)])
                    
                    
                if spectrum_parameter == 'Amplitude':
                    spectrum_weight_array = stoichiometry_array**(-2) * stoichiometry_binwidth_array**(-1)
                    if labelling_correction:
                        spectrum_weight_array *= (1. + (1 - self.labelling_efficiency) / (self.labelling_efficiency * stoichiometry_array))**(-1)
                        
                elif spectrum_parameter == 'N_monomers':
                    spectrum_weight_array = stoichiometry_array**(-1)
                    
                else: # spectrum_parameter == 'N_oligomers'
                    spectrum_weight_array = np.ones_like(stoichiometry_array, dtype = np.float64)
                    
                # Initial parameter array 
                spectrum_fit_init_params = np.array([previous_params['N_dist_amp'].value if 'N_dist_amp' in previous_params.keys() else 1.,
                                                     previous_params['N_dist_a'].value if 'N_dist_a' in previous_params.keys() else 1.,
                                                     previous_params['N_dist_b'].value if 'N_dist_b' in previous_params.keys() else 1.])
                
                # Poisson weighting (pseudo-variance), eliminating zeros
                previous_N_obs_array_fit = previous_N_obs_array[np.logical_not(skip_species_mask)]
                # spectrum_fit_weights = np.where(previous_N_obs_array_fit > 0,
                #                                 previous_N_obs_array_fit,
                #                                 np.max(previous_N_obs_array_fit))
                spectrum_fit_weights = np.ones_like(previous_N_obs_array_fit)
                
                # Fit model
                if spectrum_type == 'par_Gauss':
                    spectrum_fit_params = sminimize(lambda x: np.sum((x[0] * spectrum_weight_array * np.exp(-0.5 * ((stoichiometry_array - x[1]) / x[2]) ** 2) - previous_N_obs_array_fit)**2 / spectrum_fit_weights),
                                                    spectrum_fit_init_params,
                                                    ).x
                        
                if spectrum_type == 'par_LogNorm':
                    spectrum_fit_params = sminimize(lambda x: np.sum((x[0] * spectrum_weight_array / stoichiometry_array * np.exp(-0.5 * ((np.log(stoichiometry_array) - np.log(x[1])) / np.log(x[2])) ** 2) - previous_N_obs_array_fit)**2 / spectrum_fit_weights),
                                                    spectrum_fit_init_params,
                                                    ).x
                    
                if spectrum_type == 'par_Gamma':
                    spectrum_fit_params = sminimize(lambda x: np.sum((x[0] * spectrum_weight_array * stoichiometry_array**(x[1] - 1) * np.exp(x[1] - x[2] * stoichiometry_array) - previous_N_obs_array_fit)**2 / spectrum_fit_weights),
                                                    spectrum_fit_init_params,
                                                    ).x
    
                if spectrum_type == 'par_StrExp':
                    spectrum_fit_params = sminimize(lambda x: np.sum((x[0] * spectrum_weight_array * np.exp(- (stoichiometry_array * x[1]) ** x[2])- previous_N_obs_array_fit)**2 / spectrum_fit_weights),
                                                    spectrum_fit_init_params,
                                                    ).x

                # Read out parameters - if not user-defined
                if np.isnan(fixed_spectrum_params['N_dist_amp']):
                    N_dist_amp = spectrum_fit_params[0]
                else:
                    N_dist_amp = fixed_spectrum_params['N_dist_amp']

                if np.isnan(fixed_spectrum_params['N_dist_a']):
                    N_dist_a = spectrum_fit_params[1]
                else:
                    N_dist_a = fixed_spectrum_params['N_dist_a']

                if np.isnan(fixed_spectrum_params['N_dist_b']):
                    N_dist_b = spectrum_fit_params[2]
                else:
                    N_dist_b = fixed_spectrum_params['N_dist_b']
                                    

            # Whatever method we used, now we have the parameters and can define the spectrum in the new initial_params
            initial_params.add('N_dist_amp', 
                               value = N_dist_amp, 
                               min = 0., 
                               vary = np.isnan(fixed_spectrum_params['N_dist_amp']))
    
            initial_params.add('N_dist_a', 
                               value = N_dist_a, 
                               min = 0., 
                               vary = np.isnan(fixed_spectrum_params['N_dist_a']) and len(N_dist_b_expr) == 0) 
        
            if  len(N_dist_b_expr) == 0:
                N_dist_b = 10. if np.isnan(fixed_spectrum_params['N_dist_b']) else fixed_spectrum_params['N_dist_b']
                initial_params.add('N_dist_b', 
                                   value = N_dist_b, 
                                   min = 0.01,
                                   vary = np.isnan(fixed_spectrum_params['N_dist_b']))
            else:
                initial_params.add('N_dist_b', 
                                   expr = N_dist_b_expr,
                                   vary = False)

            skip_counter = 0
            for i_spec in range(n_species):
                if skip_species:
                    if skip_species_mask[i_spec]:
                        # skip this species!
                        skip_counter += 1
                        continue
                    
                if use_FCS or (use_PCH and time_resolved_PCH):
                    # Diffusion time only for FCS and PCMH
                    initial_params.add(f'tau_diff_{i_spec - skip_counter}', 
                                       value = tau_diff_array[i_spec - skip_counter] if use_previous_N_obs_array else previous_params[f'tau_diff_{i_spec}'].value, 
                                       vary = False)
                
                initial_params.add(f'stoichiometry_{i_spec - skip_counter}', 
                                   value = stoichiometry_array[i_spec - skip_counter] if use_previous_N_obs_array else previous_params[f'stoichiometry_{i_spec}'].value, 
                                   vary = False)
                
                initial_params.add(f'stoichiometry_binwidth_{i_spec - skip_counter}', 
                                   value = stoichiometry_binwidth_array[i_spec - skip_counter] if use_previous_N_obs_array else previous_params[f'stoichiometry_binwidth_{i_spec}'].value, 
                                   vary = False)
                    
                initial_params.add(f'Label_obs_factor_{i_spec - skip_counter}', 
                                   value = 1.,
                                   vary = incomplete_sampling_correction and labelling_correction and self.labelling_efficiency_incomp_sampling)
    
                initial_params.add(f'Label_efficiency_obs_{i_spec - skip_counter}', 
                                   expr = f'Label_efficiency * Label_obs_factor_{i_spec - skip_counter}',
                                   vary = False)

                # Weighting function that essentially decides which number the parameterization acts on
                if spectrum_parameter == 'Amplitude':
                    spectrum_weight_str = f'stoichiometry_{i_spec - skip_counter}**(-2) / stoichiometry_binwidth_{i_spec - skip_counter} '
                    if labelling_correction:
                        spectrum_weight_str += f' / ( 1. + (1 - Label_efficiency_obs_{i_spec - skip_counter}) / (Label_efficiency_obs_{i_spec - skip_counter} * stoichiometry_{i_spec - skip_counter})) '

                elif spectrum_parameter == 'N_monomers' and not oligomer_type == 'naive':
                    spectrum_weight_str = f'stoichiometry_{i_spec - skip_counter}**(-1) '
                    
                elif spectrum_parameter == 'N_oligomers' or oligomer_type == 'naive':
                    spectrum_weight_str = '1.'
                    
                else:
                    raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_parameter - must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'")
                    
                initial_params.add(f'spectrum_weight_{i_spec - skip_counter}', 
                                   expr = spectrum_weight_str, 
                                   vary = False)
                
                
                # N_avg_pop
                if spectrum_type == 'par_Gauss':
                    initial_params.add(f'N_avg_pop_{i_spec - skip_counter}', 
                                       expr = f'N_dist_amp * spectrum_weight_{i_spec - skip_counter} * exp(-0.5 * ((stoichiometry_{i_spec - skip_counter} - N_dist_a) / N_dist_b) ** 2)', 
                                       vary = False)
                        
                if spectrum_type == 'par_LogNorm':
                    initial_params.add(f'N_avg_pop_{i_spec - skip_counter}', 
                                       expr = f'N_dist_amp * spectrum_weight_{i_spec - skip_counter} / stoichiometry_{i_spec - skip_counter} * exp(-0.5 * ((log(stoichiometry_{i_spec - skip_counter}) - log(N_dist_a)) / log(N_dist_b)) ** 2)',
                                       vary = False)
                                            
                if spectrum_type == 'par_Gamma':
                    initial_params.add(f'N_avg_pop_{i_spec - skip_counter}', 
                                       expr = f'N_dist_amp * spectrum_weight_{i_spec - skip_counter} * stoichiometry_{i_spec - skip_counter}**(N_dist_a - 1) * exp(- N_dist_b * stoichiometry_{i_spec - skip_counter})', 
                                       vary = False)
                                                
                if spectrum_type == 'par_StrExp':
                    initial_params.add(f'N_avg_pop_{i_spec - skip_counter}', 
                                       expr = f'N_dist_amp * spectrum_weight_{i_spec - skip_counter} * exp(- (stoichiometry_{i_spec - skip_counter} * N_dist_a) ** N_dist_b)', 
                                       vary = False)
    

                # N_avg_obs
                if incomplete_sampling_correction:
                    if use_previous_N_obs_array:
                        # from previous_N_obs_array
                        N_avg_obs_spec = previous_N_obs_array[i_spec]
                        
                    elif f'N_avg_pop_{i_spec}' in previous_params.keys(): 
                        # From previous_params
                        N_avg_obs_spec = previous_params[f'N_avg_pop_{i_spec}'].value
                        
                    else:
                        # From model (initial assumption N_obs = N_pop)
                        if spectrum_type == 'par_Gauss':
                            N_avg_obs_spec = N_dist_amp * initial_params[f'spectrum_weight_{i_spec - skip_counter}'].value * np.exp(-0.5 * ((previous_params[f'stoichiometry_{i_spec}'].value - N_dist_a) / N_dist_b) ** 2) + 1E-15
                        elif spectrum_type == 'par_LogNorm':
                            N_avg_obs_spec = N_dist_amp * initial_params[f'spectrum_weight_{i_spec - skip_counter}'].value / previous_params[f'stoichiometry_{i_spec}'].value * np.exp(-0.5 * ((np.log(previous_params[f'stoichiometry_{i_spec}'].value) - np.log(N_dist_a)) / np.log(N_dist_b)) ** 2) + 1E-15
                        elif spectrum_type == 'par_Gamma':
                            N_avg_obs_spec = N_dist_amp * initial_params[f'spectrum_weight_{i_spec - skip_counter}'].value * previous_params[f'stoichiometry_{i_spec}'].value**(N_dist_a - 1) * np.exp(- N_dist_b * previous_params[f'stoichiometry_{i_spec}'].value) + 1E-15
                        else: # spectrum_type == 'par_StrExp'
                            N_avg_obs_spec = N_dist_amp * initial_params[f'spectrum_weight_{i_spec - skip_counter}'].value * np.exp(- (previous_params[f'stoichiometry_{i_spec}'].value * N_dist_a) ** N_dist_b) + 1E-15
                                    
                    initial_params.add(f'N_avg_obs_{i_spec - skip_counter}', 
                                       value = N_avg_obs_spec,
                                       min = 0., 
                                       vary = True)

                else:
                    # Not used - dummy
                    initial_params.add(f'N_avg_obs_{i_spec - skip_counter}', 
                                       expr = f'N_avg_pop_{i_spec}', 
                                       vary = False)
    
    
                    
                if i_spec == 0:
                    
                    if use_PCH or use_avg_count_rate:
                        initial_params.add('cpms_0', 
                                           value = previous_params['cpms_0'].value if previous_params['cpms_0'].vary else 1E3, 
                                           min = 1E-3,
                                           vary = True)
                    else:
                        initial_params.add('cpms_0', 
                                           value = 1., 
                                           min = 1E-3,
                                           vary = False)
                        
                else: # i_spec >= 1
                    initial_params.add(f'cpms_{i_spec - skip_counter}', 
                                       expr = f'cpms_0 * stoichiometry_{i_spec - skip_counter}', 
                                       vary = False)
                
            # We re-init those, whatever
            initial_params = self.set_blinking_initial_params(initial_params,
                                                              use_blinking and (use_FCS or (use_PCH and time_resolved_PCH)),
                                                              tau_diff_min)
            
            # Pre-calculate Normalized species correlation functions that we would 
            # otherwise spend a significant amount of time on during fitting
            if use_FCS:
                n_lag_times = self.data_FCS_G.shape[0]
                tau_diff__tau_array = np.zeros((n_species - skip_counter, n_lag_times))
                skip_counter = 0
                for i_spec in range(n_species):
                    if skip_species:
                        if skip_species_mask[i_spec]:
                            # Skip this species
                            skip_counter += 1
                            continue
                    tau_diff__tau_array[i_spec - skip_counter,:] = self.fcs_3d_diff_single_species(initial_params[f'tau_diff_{i_spec}'].value)
                self.tau__tau_diff_array = np.transpose(tau_diff__tau_array)


        return initial_params


    #%% Complete single-call fit routine
    


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
                use_blinking, # bool
                oligomer_model_params = {}, # Dict
                use_avg_count_rate = False, # Bool
                fit_label_efficiency = False, # bool
                fixed_spectrum_params = {}, # Dict
                two_step_fit = True,
                discrete_species = [{}], # list of dicts
                i_bin_time = 0, # int
                use_parallel = False # Bool
                ):
        
        
        # self.avg_count_rate_given
        
        
        # A bunch of input and compatibility checks
        if use_FCS and not self.FCS_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run FCS fit - not all required attributes set in class')
        
        if use_PCH and not self.PCH_possible:
            raise Exception(f'[{self.job_prefix}] Cannot run PCH fit - not all required attributes set in class')

        if not (use_FCS or use_PCH):
            raise Exception(f'[{self.job_prefix}] Cannot run fit - at least one out of use_FCS or use_PCH must be True, got False for both!')

        if use_PCH and time_resolved_PCH and self.FCS_psf_aspect_ratio == None:
            raise Exception(f'[{self.job_prefix}] Cannot run PCMH fit - PSF aspect ratio must be set')

        if not spectrum_type in ['discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
            raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_type - must be one out of 'discrete', 'reg_MEM', 'reg_CONTIN', 'par_Gauss', 'par_LogNorm', 'par_Gamma', or 'par_StrExp'. Got {spectrum_type}")

        if not (spectrum_parameter in ['Amplitude', 'N_monomers', 'N_oligomers'] or  spectrum_type == 'discrete'):
            raise Exception(f"[{self.job_prefix}] Invalid input for spectrum_parameter - unless spectrum_type is 'discrete', spectrum_parameter must be one out of 'Amplitude', 'N_monomers', or 'N_oligomers'. Got {spectrum_parameter}")
            
        if type(labelling_correction) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for labelling_correction - must be bool. Got {labelling_correction}")

        if type(incomplete_sampling_correction) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for incomplete_sampling_correction - must be bool. Got {incomplete_sampling_correction}")

        if incomplete_sampling_correction and spectrum_type == 'discrete':
            raise Exception(f"[{self.job_prefix}] incomplete_sampling_correction does not work for spectrum_type 'discrete' and must be set to false")

        if not (utils.isint(n_species) and n_species > 0):
            raise Exception(f"[{self.job_prefix}] Invalid input for n_species - must be int > 0. Got {n_species}")
        
        if n_species < 10 and spectrum_type != 'discrete':
            raise Exception(f"[{self.job_prefix}] For any spectrum_type other than 'discrete', use n_species >= 10")

        if not (utils.isfloat(tau_diff_min) and tau_diff_min > 0):
            raise Exception(f"[{self.job_prefix}] Invalid input for tau_diff_min - must be float > 0. Got {tau_diff_min}")

        if not (utils.isfloat(tau_diff_max) and tau_diff_max > 0):
            raise Exception(f"[{self.job_prefix}] Invalid input for tau_diff_max - must be float > 0. Got {tau_diff_max}")

        if type(use_blinking) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for use_blinking - must be bool. Got {use_blinking}")

        if not (utils.isint(i_bin_time) and i_bin_time >= 0):
            raise Exception(f"[{self.job_prefix}] Invalid input for i_bin_time - must be int >= 0. Got {i_bin_time}")

        if not spectrum_type == 'discrete':
            if type(two_step_fit) != bool:
                raise Exception(f"[{self.job_prefix}] Invalid input for two_step_fit - must be bool if spectrum_type is not 'discrete'. Got {two_step_fit}")

        if not type(discrete_species) == list and np.all([type(element) == dict for element in discrete_species]):
            raise Exception(f"[{self.job_prefix}] Invalid input for discrete_species - must be list of dict, where each dict contains the keywords to define one discrete discrete species to add (see method add_discrete_species()) {two_step_fit}")
        elif discrete_species == [{}]:
            has_discrete_species = False
        else:
            has_discrete_species = True
            
        if use_avg_count_rate and not self.avg_count_rate_given:
            raise Exception(f"[{self.job_prefix}] Average count rate not specified and thus cannot be included in fit.")
                 
        if type(fit_label_efficiency) != bool:
            raise Exception(f"[{self.job_prefix}] Invalid input for fit_label_efficiency - must be bool. Got {fit_label_efficiency}")
        elif fit_label_efficiency and not labelling_correction:
            raise Exception(f"[{self.job_prefix}] Invalid input of labelling_correction==False and fit_label_efficiency==True: The labelling efficiency cannot be fitted without labelling correction!")

        if type(fixed_spectrum_params) != dict:
            raise Exception(f"[{self.job_prefix}] Invalid input for fixed_spectrum_params - must be dict. Got {fixed_spectrum_params}")
        elif np.any([1 for key in fixed_spectrum_params.keys() if not key in ['N_dist_amp', 'N_dist_a', 'N_dist_b', 'j_avg_N_oligo']]):
            raise Exception(f"[{self.job_prefix}] Invalid key fixed_spectrum_params - can only be 'N_dist_amp', 'N_dist_a', 'N_dist_b', 'j_avg_N_oligo'. Got {fixed_spectrum_params.keys}")
        elif np.any([fixed_spectrum_params[key] > 0. for key in fixed_spectrum_params.keys()]) and spectrum_type in ['reg_MEM', 'reg_CONTIN']:
            raise Exception(f"[{self.job_prefix}] fixed_spectrum_params cannot be used with spectrum_type reg_MEM or reg_CONTIN. Please give empty, or all-NaN, dict.")

        # Parameter setup
        if spectrum_type == 'discrete':
            initial_params = self.set_up_params_discrete(use_FCS, 
                                                         use_PCH,
                                                         time_resolved_PCH,
                                                         labelling_correction,
                                                         n_species, 
                                                         tau_diff_min, 
                                                         tau_diff_max, 
                                                         use_blinking,
                                                         use_avg_count_rate = use_avg_count_rate,
                                                         fit_label_efficiency = fit_label_efficiency
                                                         )

        elif (spectrum_type in ['par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp'] or
              (spectrum_type in ['reg_MEM', 'reg_CONTIN'] and two_step_fit)):

            initial_params = self.set_up_params_par(use_FCS, 
                                                    use_PCH, 
                                                    time_resolved_PCH,
                                                    ('par_Gauss' if spectrum_type in ['reg_MEM', 'reg_CONTIN'] else spectrum_type),
                                                    spectrum_parameter, 
                                                    oligomer_type, 
                                                    incomplete_sampling_correction and not two_step_fit, # we never do incomplete_sampling_correction in step 1 of a two-step fit
                                                    labelling_correction, 
                                                    n_species, 
                                                    tau_diff_min, 
                                                    tau_diff_max, 
                                                    use_blinking,
                                                    oligomer_model_params = oligomer_model_params,
                                                    use_avg_count_rate = use_avg_count_rate and not two_step_fit,
                                                    fit_label_efficiency = fit_label_efficiency and not two_step_fit, # Label efficiency is also optimized in second fit round
                                                    fixed_spectrum_params = fixed_spectrum_params)   

        else: # spectrum_type in ['reg_MEM', 'reg_CONTIN'] and not two_step_fit:

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
                                                    use_blinking,
                                                    oligomer_model_params = oligomer_model_params,
                                                    use_avg_count_rate = use_avg_count_rate,
                                                    fit_label_efficiency = fit_label_efficiency
                                                    )
        if has_discrete_species:
            for spec_dict in discrete_species:
                # Use keyword arg unpacking to pass through the user-defined parameters for each species, leaving others at defaults
                initial_params = self.add_discrete_species(initial_params, **spec_dict)


            
        if self.verbosity > 0:

            if spectrum_type in ['reg_MEM', 'reg_CONTIN'] and two_step_fit:
                print(f'[{self.job_prefix}]    --- Initial Gauss fit ---')
            print(f'[{self.job_prefix}]    Initial parameters:')
            [print(f'[{self.job_prefix}] {key}: {initial_params[key].value}') for key in initial_params.keys() if initial_params[key].vary]
            
        if self.verbosity > 2:
            print(f'[{self.job_prefix}]    Constants & dep. variables:')
            [print(f'[{self.job_prefix}] {key}: {initial_params[key].value}') for key in initial_params.keys() if not initial_params[key].vary]

        if use_parallel:
            mp_pool = multiprocessing.Pool(processes = os.cpu_count() - 1)
        else:
            mp_pool = None
        
        try:
            if (not self.precision_incremental) or (not (use_PCH or (labelling_correction and incomplete_sampling_correction))):
                # Fit with a single numeric precision value (or with settings where the precision parameter is irrelevant)
                
                if spectrum_type in ['discrete', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:

                    fit_result = lmfit.minimize(fcn = self.negloglik_global_fit, 
                                                params = initial_params, 
                                                method = 'nelder',
                                                args = (use_FCS, 
                                                        use_PCH, 
                                                        time_resolved_PCH,
                                                        spectrum_type, 
                                                        spectrum_parameter,
                                                        labelling_correction, 
                                                        incomplete_sampling_correction and not two_step_fit
                                                        ),
                                                kws = {'i_bin_time': i_bin_time,
                                                       'numeric_precision': self.numeric_precision,
                                                       'mp_pool': mp_pool,
                                                       'use_avg_count_rate': use_avg_count_rate and (not two_step_fit or spectrum_type == 'discrete')
                                                       },
                                                calc_covar = True)

                    if two_step_fit and \
                        (incomplete_sampling_correction or fit_label_efficiency or use_avg_count_rate) and \
                        spectrum_type != 'discrete': 
                        # Re-fit - currently only does anything if incomplete_sampling_correction or fit_label_efficiency are True, that's why we set the cond
                        # Re-fit with with fuller model
                        if self.verbosity > 0:
                            print(f'[{self.job_prefix}]    --- Parameterized-spectrum re-fit---')                        
                        
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
                                                                use_blinking,
                                                                oligomer_model_params = oligomer_model_params,
                                                                previous_params = fit_result.params,
                                                                use_avg_count_rate = use_avg_count_rate,
                                                                fit_label_efficiency = fit_label_efficiency,
                                                                fixed_spectrum_params = fixed_spectrum_params
                                                                )
                        
                        if has_discrete_species:
                            for spec_dict in discrete_species:
                                # Use keyword arg unpacking to pass through the user-defined parameters for each species, leaving others at defaults
                                initial_params = self.add_discrete_species(initial_params, **spec_dict)

                        
                        fit_result = lmfit.minimize(fcn = self.negloglik_global_fit, 
                                                    params = initial_params, 
                                                    method = 'nelder',
                                                    args = (use_FCS, 
                                                            use_PCH, 
                                                            time_resolved_PCH,
                                                            spectrum_type, 
                                                            spectrum_parameter,
                                                            labelling_correction, 
                                                            incomplete_sampling_correction
                                                            ),
                                                    kws = {'i_bin_time': i_bin_time,
                                                           'numeric_precision': self.numeric_precision,
                                                           'mp_pool': mp_pool,
                                                           'use_avg_count_rate': use_avg_count_rate},
                                                    calc_covar = True)

                 
                
                elif spectrum_type in ['reg_MEM', 'reg_CONTIN'] and two_step_fit:
                    # Run Gauss fit
                    fit_result = lmfit.minimize(fcn = self.negloglik_global_fit, 
                                                params = initial_params, 
                                                method = 'nelder',
                                                args = (use_FCS, 
                                                        use_PCH, 
                                                        time_resolved_PCH,
                                                        'par_Gauss', 
                                                        spectrum_parameter,
                                                        labelling_correction, 
                                                        False
                                                        ),
                                                kws = {'i_bin_time': i_bin_time,
                                                       'numeric_precision': self.numeric_precision,
                                                       'mp_pool': mp_pool,
                                                       'use_avg_count_rate': True},
                                                calc_covar = False)
                    
                    # Recalculate initial parameters to those for regularized fit
                    initial_params, N_pop_array = self.set_up_params_reg_from_parfit(fit_result.params,
                                                                                     use_FCS, 
                                                                                     use_PCH, 
                                                                                     time_resolved_PCH,
                                                                                     spectrum_type,
                                                                                     oligomer_type, 
                                                                                     incomplete_sampling_correction,
                                                                                     labelling_correction,
                                                                                     tau_diff_min,
                                                                                     tau_diff_max, 
                                                                                     use_blinking,
                                                                                     use_avg_count_rate = use_avg_count_rate,
                                                                                     fit_label_efficiency = fit_label_efficiency
                                                                                     )
                    if has_discrete_species:
                        for spec_dict in discrete_species:
                            # Use keyword arg unpacking to pass through the user-defined parameters for each species, leaving others at defaults
                            initial_params = self.add_discrete_species(initial_params, **spec_dict)

                    # Re-fit with regularization
                    if self.verbosity > 0:
                        print(f'[{self.job_prefix}]    --- Regularized-spectrum re-fit ---')                        
                    fit_result, N_pop_array, lagrange_mul = self.regularized_minimization_fit(initial_params,
                                                                                              use_FCS,
                                                                                              use_PCH,
                                                                                              time_resolved_PCH,
                                                                                              spectrum_type,
                                                                                              spectrum_parameter,
                                                                                              labelling_correction,
                                                                                              incomplete_sampling_correction,
                                                                                              i_bin_time = i_bin_time,
                                                                                              N_pop_array = N_pop_array,
                                                                                              numeric_precision = self.numeric_precision,
                                                                                              mp_pool = mp_pool,
                                                                                              use_avg_count_rate = use_avg_count_rate
                                                                                              )
                    
        
                else: # spectrum_type in ['reg_MEM', 'reg_CONTIN'] and not two_step_fit
     
                    fit_result, N_pop_array, lagrange_mul = self.regularized_minimization_fit(initial_params,
                                                                                              use_FCS,
                                                                                              use_PCH,
                                                                                              time_resolved_PCH,
                                                                                              spectrum_type,
                                                                                              spectrum_parameter,
                                                                                              labelling_correction,
                                                                                              incomplete_sampling_correction,
                                                                                              i_bin_time = i_bin_time,
                                                                                              N_pop_array = None,
                                                                                              numeric_precision = self.numeric_precision,
                                                                                              mp_pool = mp_pool,
                                                                                              use_avg_count_rate = use_avg_count_rate
                                                                                              )

                
            else:
                if self.NLL_funcs_accurate:
                    # For early, lower-precision fits we also go to the fast least-squares approximation!
                    NLL_funcs_accurate = True
                    self.NLL_funcs_accurate = False
                else:
                    # Leave a marker that we stick to least-squares approximation anyway
                    NLL_funcs_accurate = False
                    

                # we use incremental precision fitting
                for i_inc, inc_precision in enumerate(self.numeric_precision):
                    

                    
                    if self.verbosity > 0:
                        print(f'[{self.job_prefix}] Numeric precision increment {i_inc+1} of {self.numeric_precision.shape[0]}')
                        
                    if i_inc == self.numeric_precision.shape[0] - 1:
                        # Last iteration: Switch from least-squares to MLE fitting now if needed
                        self.NLL_funcs_accurate = NLL_funcs_accurate

                    if spectrum_type in ['discrete', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:

                        
                        if (i_inc == self.numeric_precision.shape[0] - 1) and two_step_fit and (incomplete_sampling_correction or fit_label_efficiency or use_avg_count_rate):
                            # Last iteration - switch on incomplete_sampling_correction if needed
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
                                                                    use_blinking,
                                                                    oligomer_model_params = oligomer_model_params,
                                                                    previous_params = fit_result.params,
                                                                    use_avg_count_rate = use_avg_count_rate,
                                                                    fit_label_efficiency = fit_label_efficiency,
                                                                    fixed_spectrum_params = fixed_spectrum_params
                                                                    ) 
                            if has_discrete_species:
                                for spec_dict in discrete_species:
                                    # Use keyword arg unpacking to pass through the user-defined parameters for each species, leaving others at defaults
                                    initial_params = self.add_discrete_species(initial_params, **spec_dict)


                            
                        fit_result = lmfit.minimize(fcn = self.negloglik_global_fit, 
                                                    params = initial_params, 
                                                    method = 'nelder',
                                                    args = (use_FCS, 
                                                            use_PCH, 
                                                            time_resolved_PCH,
                                                            spectrum_type, 
                                                            spectrum_parameter,
                                                            labelling_correction, 
                                                            (incomplete_sampling_correction and (i_inc == self.numeric_precision.shape[0] - 1))),
                                                    kws = {'i_bin_time': i_bin_time,
                                                           'numeric_precision': inc_precision,
                                                           'mp_pool': mp_pool,
                                                           'use_avg_count_rate': (use_avg_count_rate and (i_inc == self.numeric_precision.shape[0] - 1))},
                                                    calc_covar = i_inc == self.numeric_precision.shape[0] - 1) # Covar only needed at least step
                        


                        
                    elif spectrum_type in ['reg_MEM', 'reg_CONTIN'] and two_step_fit:
                        
                        if i_inc == 0:
                            # Run Gauss fit in first iteration only
                            fit_result = lmfit.minimize(fcn = self.negloglik_global_fit, 
                                                        params = initial_params, 
                                                        method = 'nelder',
                                                        args = (use_FCS, 
                                                                use_PCH, 
                                                                time_resolved_PCH,
                                                                'par_Gauss', 
                                                                spectrum_parameter,
                                                                labelling_correction, 
                                                                False
                                                                ),
                                                        kws = {'i_bin_time': i_bin_time,
                                                               'numeric_precision': inc_precision,
                                                               'mp_pool': mp_pool,
                                                               'use_avg_count_rate': False},
                                                        calc_covar = False)
                        
                            # Recalculate initial parameters to those for regularized fit
                            initial_params, N_pop_array = self.set_up_params_reg_from_parfit(fit_result.params,
                                                                                             use_FCS, 
                                                                                             use_PCH, 
                                                                                             time_resolved_PCH,
                                                                                             spectrum_type,
                                                                                             oligomer_type, 
                                                                                             incomplete_sampling_correction,
                                                                                             labelling_correction,
                                                                                             tau_diff_min,
                                                                                             tau_diff_max, 
                                                                                             use_blinking,
                                                                                             use_avg_count_rate = use_avg_count_rate,
                                                                                             fit_label_efficiency = fit_label_efficiency
                                                                                             )
                            if has_discrete_species:
                                for spec_dict in discrete_species:
                                    # Use keyword arg unpacking to pass through the user-defined parameters for each species, leaving others at defaults
                                    initial_params = self.add_discrete_species(initial_params, **spec_dict)

                        
                        # Re-fit with regularization
                        if self.verbosity > 0 and i_inc == 0:
                            print(f'[{self.job_prefix}]    --- Regularized-spectrum re-fit ---')                            
                        fit_result, N_pop_array, lagrange_mul = self.regularized_minimization_fit(initial_params,
                                                                                                  use_FCS,
                                                                                                  use_PCH,
                                                                                                  time_resolved_PCH,
                                                                                                  spectrum_type,
                                                                                                  spectrum_parameter,
                                                                                                  labelling_correction,
                                                                                                  incomplete_sampling_correction,
                                                                                                  i_bin_time = i_bin_time,
                                                                                                  N_pop_array = N_pop_array,
                                                                                                  numeric_precision = inc_precision,
                                                                                                  mp_pool = mp_pool,
                                                                                                  use_avg_count_rate = use_avg_count_rate
                                                                                                  )

                        
                    else: # spectrum_type in ['reg_MEM', 'reg_CONTIN'] and not two_step_fit
                        if i_inc == 0:
                            N_pop_array = None

                        fit_result, N_pop_array, lagrange_mul = self.regularized_minimization_fit(initial_params,
                                                                                                  use_FCS,
                                                                                                  use_PCH,
                                                                                                  time_resolved_PCH,
                                                                                                  spectrum_type,
                                                                                                  spectrum_parameter,
                                                                                                  labelling_correction,
                                                                                                  incomplete_sampling_correction,
                                                                                                  i_bin_time = i_bin_time,
                                                                                                  N_pop_array = N_pop_array,
                                                                                                  numeric_precision = inc_precision,
                                                                                                  mp_pool = mp_pool,
                                                                                                  use_avg_count_rate = use_avg_count_rate
                                                                                                  )

                    if i_inc < self.numeric_precision.shape[0] - 1:
                        # At least one fit more to run, use output of previous fit as input for next round
                        initial_params = fit_result.params

        except:
            # Something went wrong - whatever, not too much we can do
            traceback.print_exc()
            Warning('Cade ran into error. You will likely see a "variable referenced before assignment" error message below. That is an artifact from the error handling logic here, ignore that! The real error message is above this warning.')
        finally:
            # In any case, close parpool at the end if possible
            try:       

                if not mp_pool == None:
                    mp_pool.close()
                if spectrum_type in ['discrete', 'par_Gauss', 'par_LogNorm', 'par_Gamma', 'par_StrExp']:
                    return fit_result
                
                else: # spectrum_type in ['reg_MEM', 'reg_CONTIN']

                    if not fit_result == None:
                        return fit_result, N_pop_array, lagrange_mul
                    else:
                        return initial_params, N_pop_array, lagrange_mul
            except:
                return None
        
        
        