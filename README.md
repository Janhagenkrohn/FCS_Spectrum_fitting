
# FCS_Spectrum_fitting
## Collection of experimental code for fitting of polydisperse FCS data 

FCS of polydisperse systems is challenging for many reasons, one being the challenge of fitting meaningful distributions of concentrations over particle size to the data. This is a public repo of  *purely experimental* Python code that was employed in a project exploring many ideas what could be useful in that context. The key ideas of the code as deposited in this repo was testing out many methods for data fitting, individually and in combination, while joining them into a single modular framework.  
### Techniques/concepts implemented

 -  **Standard FCS fitting with 1, 2, 3, ... n independent species**
Here, the basic model is that of three-dimensional diffusion in a 3D-Gaussian detection volume, as well as extension with a blinking term ([Aragón and Pecora 1976](http://aip.scitation.org/doi/10.1063/1.432357), [Widengren and Mets 2002](http://doi.wiley.com/10.1002/3527600809.ch3)). The fit is based on weighted least-squares fitting, expecting the correlation times, the autocorrelation function, and the uncertainty as input. The user can additionally supply the average count rate as input for on-the-fly calculation of molecular brightness.
 - **Photon Counting (Multiple) Histograms fitting for 1, 2, 3, ..., n species**
 PC(M)H is essentially based on [Huang, Perroud, Zare 2004](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/cphc.200400176) and [Perroud, Huang, and Zare 2005](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/cphc.200400547). An arbitrary number of (non-normalized) PCHs with different (specified) bin sizes is accepted for global fitting. The user can chose whether to perform a weighted least-squares fit, or a Poisson count maximum likelihood fit. Usually that does not make a big difference, although least-squares is significantly faster, while maximum likelihood in some instances (not many) converges to a better result.
 -  **Global fit of FCS and PC(M)H** 
Global fitting of FCS and PC(M)H representation of the same input data is based on [Skakun et al. 2011](https://imrpress.com/journal/FBE/3/2/10.2741/E264). In that case, make sure to supply proper weights for the correlation function! The code does not contain a particularly smart parameter initialization to be honest, so in the current form this works so-so all things considered.
 -  **Fitting of a concentration spectrum over particle sizes using few-parameter functions**
This is compatible with autocorrelation functions and in principle PC(M)H the same way as fitting of few independent species, and based on the same underlying diffusion+blinking model. However, it tends to be computationally prohibitively expensive for PC(M)H. The difference is that a large number of species is linked on the one hand by physical models describing the scaling of hydrodynamic radius (and hence diffusion time) with stoichiometry (molecular brightness), and on the other hand enforcing a fixed shape of the spectrum of concentration over stoichiometry. Available particle size scaling models include approximations for solid spheres, spherical shells, and filamentout structures. Spectrum shape fucntions supported are a Gaussian distribution, a lognormal distribution, a Gamma distribution, and a stretched-exponential distribution. Each distribution is characterized by only three parameters. Note that at small stoichiometries <~10, the software produces some discretization artifacts, but these should generally not have a huge impact. The user can chose wether the oligomer number (particle concentration), the monomer number (mass fraction), or the correlation function amplitudes follow the chosen distribution. The other two parameters will then follow accordingly. One can also specify one or multiple discrete, independent species to consider in addition to the spectrum, e.g. to treat residual free dye. For those additional discrete species, one has a lot of freedom to fix or fit parameters. 
 -  **Fitting of a concentration spectrum over particle sizes using regularization methods**
This is similar to spectra fitting of using few-parameter functions, but uses the maximum entropy method (see [Sengupta et al. 2003](https://www.sciencedirect.com/science/article/pii/S0006349503750061)) or CONTIN ([Pánek et al. 2018](https://doi.org/10.1021/acs.macromol.7b02158)) to enforce smooth spectra without *a priori* assumptions about the shape of the spectrum. At the time of writing, the implementation that is in the common framework with few-parameter spectrum fitting is broken (I hope I will find time to fix that soon), but a relatively lean standalone-script is also provided. Different from earlier implementations for FCS that we are aware of, in our implementation the Lagrange multiplier (regularization weight) is optimized to match the signal to noise ratio of the provided data. For maximum-entropy fitting this usually converges well using the "test" criterion of [Steinbach et al. 1992](https://linkinghub.elsevier.com/retrieve/pii/S0006349592818301), for CONTIN this does not work well and usually our current implementation runs until it reaches the hard-coded iteration limit.
 - **Correction of the impact of labelling efficiency on the FCS data**
Spectrum fits can be combined with assumption of a labelling efficiency <= 1. For polydisperse systems, this changes the relation between correlation function amplitudes and particle concentrations (see [Petersen 1986](https://www.sciencedirect.com/science/article/pii/S0006349586837092)).
 - **Experimental but turned out only moderately useful: Correction of imcomplete sampling in FCS**
 FCS of polydisperse systems often deals with significant correlation function amplitude contributions of large but rare particles, which tend to be statistically weakly represented and thus noisy. We implemented a sort of hierarchical model of FCS of a polydisperse concentration spectrum that allows deviations between the "actual", population-level, concentrations, and the "observed" concentrations. The idea is that if the FCS fit tells you the diffusion time and average particle number of a species and you know the total acquisition time, you can estimate the total number of particles you should have seen during the measurement. Knowing that number and exploiting Poisson statistics, you can define confidence intervals for deviations between population-level and observation-level particle numbers. The former should follow a smooth distribution (e.g., lognormal), the latter can, especially for rare species, show significant spikes from one particle size to the next. So far the idea. In practice, it sort of works, but not very convincingly. We are still putting it here, in case someone wants to pick up the idea.


#### Python environment requirements
The functions used are not especially fancy, the version requirements should be quite relaxed. We ran the code on various machines with Anaconda python environments that were inconsistent in module versions (both Windows and Linux).

The required/recommended modules (with versions as used productively on one of our Linux machines) are:
```
python 3.7.11
numpy 1.20.3
scipy 1.7.3
pandas 1.3.5
lmfit 1.2.2
matplotlilb 3.5.1
uncertainties 3.1.6
numdifftools 0.9.41 
```
`lmfit`, `numdifftools`, and `uncertainties` are most likely to cause trouble: Most problems we encounter with the analysis pipeline relate to uncertainty calculations.

#### License
[MIT License](https://github.com/Janhagenkrohn/FCS_Spectrum_fitting/tree/main/LICENSE)

#### Citation
```
Will be added soon...
@ARTICLE
{Krohn2025,
author={Krohn, Jan-Hagen and Schwille, Petra},
journal={TBD},
title={TBD},
year={2024},
volume={TBD},
number={TBD},
doi={TBD}}
```
