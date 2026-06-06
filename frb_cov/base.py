import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RegularGridInterpolator
import astropy.cosmology
from astropy import units as u
from astropy import constants
from scipy.special import jv
from scipy.special import eval_legendre
from tqdm import tqdm
import time as time
import cosmopower as cp

import os

class FRBCovariance():
    """Mean DM and LSS covariance matrix for a catalogue of localised FRBs.

    Given a set of FRB host redshifts and pairwise sky separations, this class
    computes the mean intergalactic dispersion measure expected from large-scale
    structure along each line of sight, as well as the full N×N covariance
    matrix of those DMs arising from correlated electron density fluctuations.

    The calculation follows a Limber-approximation angular power spectrum
    approach.  Baryon physics (AGN feedback, modified gravity, EP breaking) are
    encoded through cosmopower emulators.

    Parameters
    ----------
    cosmo_dict : dict
        Cosmological and model parameters.  Required keys:

        ============  ==============================================================
        Key           Description
        ============  ==============================================================
        sigma_8       Matter fluctuation amplitude σ_8
        h             Dimensionless Hubble constant (H_0 / 100 km s⁻¹ Mpc⁻¹)
        omega_m       Total matter density Ω_m
        omega_b       Baryon density Ω_b
        omega_de      Dark-energy density Ω_de
        w0            Dark-energy equation-of-state parameter w_0
        wa            Dark-energy equation-of-state running w_a (CPL)
        ns            Primordial scalar spectral index n_s
        neff          Effective number of relativistic species N_eff
        m_nu          Sum of neutrino masses [eV]
        Tcmb0         CMB temperature today [K]
        alpha_B       Horndeski braiding parameter α_B
        alpha_M       Planck-mass running rate α_M
        logTAGN       log₁₀(T_AGN / K) — AGN feedback strength (Mead 2020)
        ks            Modified-gravity screening scale [h Mpc⁻¹]
        delta_gamma   Equivalence-principle breaking parameter δγ; 0 = standard
        diagonal      If True only auto-variances are computed (much faster)
        path_to_f_iGM Path to two-column ASCII file tabulating f_IGM(z);
                      falls back to f_IGM = 0.9 if the file cannot be loaded
        mass_photon   (optional) Photon mass [eV] for a massive-photon DM term
        ============  ==============================================================

    bias_emu : cosmopower_NN
        Emulator for the electron-bias contribution to the power spectrum,
        P_ee(k, z) − P_mm(k, z).
    power_spec_emu : cosmopower_NN
        Emulator for the non-linear matter power spectrum P_mm(k, z).
    zet : array_like of shape (N,)
        Host redshifts of the N FRBs, **sorted in ascending order**.
    delta_theta : array_like of shape (N, N)
        Pairwise angular separations on the sky [radians].
    flat_sky : bool, optional
        If True, use the flat-sky Hankel-transform method to build the
        covariance matrix (default False).  Automatically enabled for N ≥ 100.
    frequency_width : array_like of shape (N,), optional
        Bandwidth of each FRB observation [GHz].  Required when delta_gamma ≠ 0.
    frequency_band : array_like of shape (N,), optional
        Central observing frequency of each FRB [GHz].  Required when
        delta_gamma ≠ 0.
    plin_emu : cosmopower_NN, optional
        Emulator for the linear matter power spectrum.  When provided, the
        non-linear spectrum is replaced by the linear one below k = 0.01 h/Mpc,
        which improves large-scale accuracy.
    non_verbose : bool, optional
        Suppress the tqdm progress bar when True (default True).
    bg_only : bool, optional
        If True, compute only the mean DM (`self.DM`) and skip the covariance
        calculation entirely (default False).

    Attributes
    ----------
    DM : ndarray of shape (N,)
        Mean LSS/IGM dispersion measure for each FRB [pc cm⁻³].
    covariance : ndarray of shape (N, N)
        DM covariance matrix from correlated large-scale structure [pc² cm⁻⁶].
        Off-diagonal elements are zero when ``diagonal=True``.
    C_ell : ndarray
        Angular power spectra evaluated on ``self.ell``, used internally to
        build the covariance matrix.
    """

    def __init__(self,
                 cosmo_dict,
                 bias_emu,
                 power_spec_emu,
                 zet,
                 delta_theta,
                 flat_sky = False,
                 frequency_width = None,
                 frequency_band = None,
                 plin_emu = None,
                 non_verbose = True,
                 bg_only = False,
                 n_chi_limber = 1000,
                 n_chi_interp = 200):
        self.verbose = non_verbose
        self.n_chi_limber = n_chi_limber
        self.frequency_width = frequency_width
        self.frequency_bands = frequency_band
        if cosmo_dict['delta_gamma'] != 0 and (frequency_width is None or frequency_band is None):
            print("You are asking for the EP breaking term, 'delta_gamma != 0'.")
            print("Make sure that you specify the frequency width and frequency band (centra value of the frequency) of each FRB in GHz!")
            print("Specifically specify 'frequency_width = ' and 'frequency_band = ' as keyword arguments in the constructor of the covariance.")
            print("Both must be arrays of the same length as the number of FRBs.")
            print("I will proceed by assuming that all FRBs are observed at 1GHz with a bandwidth of 0.5GHz")
            self.frequency_width = 0.5*np.ones(len(zet))
            self.frequency_bands = 1.0*np.ones(len(zet))

        try:
            aux = np.array(np.loadtxt(cosmo_dict['path_to_f_iGM']))
            self.f_igm_spline = UnivariateSpline(aux[:,0], aux[:,1], k = 3, s = 0)
        except:
            self.f_igm_spline = None

        self.mass_gamma = 0.0
        if 'mass_photon' in cosmo_dict.keys():
            self.mass_gamma = cosmo_dict['mass_photon']

        self.frequncies_high = np.ones(len(zet))
        self.frequncies_low = 1.1*np.ones(len(zet))
        if cosmo_dict['delta_gamma'] != 0:
            self.frequency_width = np.array(self.frequency_width)
            self.frequency_bands = np.array(self.frequency_bands)
            self.frequncies_high = (self.frequency_bands + self.frequency_width/2)
            self.frequncies_low = (self.frequency_bands - self.frequency_width/2)
        N_interp_chi = n_chi_interp
        self.flat_sky = flat_sky
        self.delta_theta = delta_theta
        self.deg2torad2 = 180 / np.pi * 180 / np.pi
        self.arcmin2torad2 = 60*60 * self.deg2torad2
        self.cosmo_dict_fiducial = cosmo_dict
        self.cosmology = astropy.cosmology.w0waCDM(
            H0=cosmo_dict['h']*100.0,
            Om0=cosmo_dict['omega_m'],
            Ob0=cosmo_dict['omega_b'],
            Ode0=cosmo_dict['omega_de'],
            w0=cosmo_dict['w0'],
            wa=cosmo_dict['wa'],
            Neff=cosmo_dict['neff'],
            m_nu=cosmo_dict['m_nu']*u.eV,
            Tcmb0=cosmo_dict['Tcmb0'])
        self.chi_H = self.cosmology.hubble_distance.value*self.cosmology.h            
        m2 = 1/u.m/u.m
        self.prefac = 3.0*constants.c.value / \
            (constants.m_p.value*constants.G.value*8.0*np.pi) * \
            (self.cosmology.H(0).to(1/u.s)).value * \
            m2.to(u.parsec/u.cm**3).value*self.cosmology.Ob0
        self.los_integration_z = np.linspace(
            0,
            np.amax(zet),
            100)
        self.spline_z_of_chi = UnivariateSpline(self.cosmology.comoving_distance(
            self.los_integration_z).value*self.cosmology.h, self.los_integration_z, k=2, s=0, ext=0)
        self.chi = self.cosmology.comoving_distance(
            zet).value*self.cosmology.h
        self.zet = zet
        self.chi_interp = np.geomspace(10, self.chi[-1], N_interp_chi)
        self.zet_interp = np.linspace(0, np.amax(zet), N_interp_chi)
        try:
            self.diagonal = cosmo_dict["diagonal"]
        except:
            self.diagonal = False
        params = {'Omega_b': [cosmo_dict['omega_b']]*np.ones_like(self.chi_interp),
          'Omega_cdm' : [cosmo_dict['omega_m'] - cosmo_dict['omega_b']]*np.ones_like(self.chi_interp),
          'h' : [cosmo_dict['h']]*np.ones_like(self.chi_interp),
          'n_s' : [cosmo_dict['ns']]*np.ones_like(self.chi_interp),
          'log10_T_heat' : [cosmo_dict['logTAGN']]*np.ones_like(self.chi_interp),
          'm_nu' : [cosmo_dict['m_nu']]*np.ones_like(self.chi_interp),
          'sigma8' : [cosmo_dict['sigma_8']]*np.ones_like(self.chi_interp),
          'alpha_B' : [cosmo_dict['alpha_B']]*np.ones_like(self.chi_interp),
          'alpha_M' : [cosmo_dict['alpha_M']]*np.ones_like(self.chi_interp),
          'log10_k_screen' : [np.log10(cosmo_dict['ks'])]*np.ones_like(self.chi_interp),
          'z_val' : self.spline_z_of_chi(self.chi_interp),}
        self.power_at_z = power_spec_emu.predictions_np(params)
        log_modes = np.log(power_spec_emu.modes)
        bias_power = bias_emu.predictions_np(params)
        if plin_emu is not None:
            plin_aux = plin_emu.predictions_np(params)
            index = np.argmax(power_spec_emu.modes > 1e-2)
            self.power_at_z[:,:index] = plin_aux[:,:index]
            self.spline_Pmm_lin = RegularGridInterpolator((self.chi_interp, log_modes), plin_aux, bounds_error=False, fill_value=None)
        self.spline_Pee = RegularGridInterpolator((self.chi_interp, log_modes), bias_power + self.power_at_z, bounds_error=False, fill_value=None)
        self.spline_Pmm = RegularGridInterpolator((self.chi_interp, log_modes), self.power_at_z, bounds_error=False, fill_value=None)
        self.spline_potential = RegularGridInterpolator((self.chi_interp, log_modes), self.power_at_z - 4*np.log10(power_spec_emu.modes), bounds_error=False, fill_value=None)
        self.spline_potential_ee = RegularGridInterpolator((self.chi_interp, log_modes), 0.5*bias_power + self.power_at_z - 2*np.log10(power_spec_emu.modes), bounds_error=False, fill_value=None)
        self.DM = np.zeros_like(self.zet)
        self.hbar = constants.hbar.to(u.eV * u.s).value
        e = 4.803e-10*u.cm**(3/2) * u.g**(1/2) / u.s
        me = constants.m_e.to(u.g)
        c = constants.c.to(u.cm / u.s)
        self.K = (e**2 / (2 * np.pi * me*c)).to(1/u.parsec*u.cm**3/u.s)
        self.dispersion_constant_times_c = (self.K*c).to(u.cm**3*u.GHz**2*u.Mpc/u.pc).value#4.03068e-17 # in cm3 GHz2 Mpc / pc
        # Cumulative trapezoid on a fine grid — one vectorised evaluation
        # replaces N independent quad() calls.
        z_fine = np.linspace(0, np.amax(self.zet) * 1.001, 500)
        w_fine = self.weight_all(z_fine)
        dm_fine = np.zeros_like(z_fine)
        dm_fine[1:] = np.cumsum((w_fine[:-1] + w_fine[1:]) / 2.0 * np.diff(z_fine))
        self.DM = np.interp(self.zet, z_fine, dm_fine)
        self.ell = np.geomspace(1,50000,100)
        if not bg_only:
            self.limber_cell(self.ell)
            if self.flat_sky or len(self.zet) >= 100:
                self.get_covariance_flat_sky()
            else:
                self.get_covariance()



    def weight(self, z):
        """LSS line-of-sight weight function for the mean DM integrand.

        Evaluates the kernel W(z) such that DM_IGM = ∫₀^z W(z') dz'.
        Accounts for the IGM baryon fraction f_IGM(z), the helium ionisation
        correction, and the cosmological expansion factor.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate the weight.

        Returns
        -------
        float or ndarray
            Weight function value(s) [pc cm⁻³ per unit redshift].
        """
        if self.f_igm_spline:
            f_IGM = self.f_igm_spline(z)
        else:     
            f_IGM =  0.9 #np.interp(z,aux[:,0], aux[:,1])
        f_He = 0.245  # Helium fraction
        f_H = 1. - f_He
        f_e = (f_H + 1/2. * f_He)  # electron fraction
        result = f_e * f_IGM * self.prefac*(1+z)/self.cosmology.efunc(z)
        return result

    def weight_wep(self):
        """Amplitude of the equivalence-principle (EP) breaking contribution to C_ell.

        Non-zero only when ``delta_gamma ≠ 0`` in the cosmology dictionary.
        This term arises from a frequency-dependent photon propagation speed
        and couples the observed DM to the gravitational potential along the
        line of sight.

        Returns
        -------
        float
            EP-breaking weight (scalar, frequency-independent prefactor).
        """
        return 1.5*self.cosmology.Om0/(self.dispersion_constant_times_c*self.chi_H**2)*self.cosmo_dict_fiducial['delta_gamma']

    def weight_massive_photon(self, z):
        """DM weight function for a non-zero photon mass.

        Returns zero at all redshifts when ``mass_photon`` is not set in the
        cosmology dictionary (the default).  When a photon mass is specified,
        this term adds a redshift-dependent correction to the dispersion
        measure that scales as m_γ² / (1 + z)².

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate the weight.

        Returns
        -------
        float or ndarray
            Massive-photon weight function value(s) [pc cm⁻³ per unit redshift].
        """
        if self.mass_gamma == 0.0:
            return np.zeros_like(z)
        else:
            result = self.K.value*(self.mass_gamma/(4*np.pi*self.hbar))**2/(self.cosmology.efunc(z))/(1+z)**2
            return result
    
    def weight_all(self, z):
        """Total DM weight function including all physical contributions.

        Sums the standard IGM weight and the massive-photon correction.
        This is the integrand used to compute `self.DM`.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate the weight.

        Returns
        -------
        float or ndarray
            Combined weight function value(s) [pc cm⁻³ per unit redshift].
        """
        return self.weight(z) + self.weight_massive_photon(z)

    def limber_cell(self, ells):
        """Compute the tomographic angular power spectra C_ell under the Limber approximation.

        Evaluates C_ell(ℓ, z_i, z_j) for all FRB pairs by integrating the
        electron power spectrum P_ee(k, z) along the line of sight, weighted
        by the DM kernel squared.  Three contributions are included:

        1. **Standard LSS term** — correlated free electrons traced by P_ee.
        2. **EP-breaking term** — couples DM to the gravitational potential
           when ``delta_gamma ≠ 0``; frequency-dependent via ``freq_weight``.
        3. **Cross term** — geometric mean of the LSS and EP-breaking kernels.

        When ``diagonal=True``, only auto-spectra C_ell(ℓ, z_i, z_i) are
        computed, halving the memory footprint and skipping the full N×N
        tomographic weighting.  For N ≥ 110 FRBs the spectra are stored on a
        coarser redshift grid and interpolated on demand.

        The result is stored in ``self.C_ell`` and used immediately by
        `get_covariance` or `get_covariance_flat_sky`.

        Parameters
        ----------
        ells : array_like
            Multipole moments ℓ at which to evaluate the power spectra.
        """
        chi_max = self.cosmology.comoving_distance(np.max(self.zet)).value*self.cosmology.h
        chi = np.geomspace(1, chi_max, self.n_chi_limber)
        # Build (chi, log-k) evaluation grid with meshgrid instead of a double loop.
        chi_2d = np.repeat(chi, len(ells))
        ell_2d = np.tile(ells + 0.5, len(chi))
        x_values = np.array([chi_2d, np.log(ell_2d / chi_2d * self.cosmology.h)])
        self.exp = self.spline_Pee((x_values[0,:],x_values[1,:])).reshape((len(chi),len(ells)))
        z_of_chi = self.spline_z_of_chi(chi)
        chi_integrand = self.weight(z_of_chi)**2/chi**2*(self.cosmology.efunc(z_of_chi)/self.chi_H)**2
        freq_weight = 1.0/(1/self.frequncies_low**2 - 1/self.frequncies_high**2)
        freq_weight_squared = freq_weight[:,None] * freq_weight[None, :]

        use_ep = (self.cosmo_dict_fiducial['delta_gamma'] != 0)
        if use_ep:
            self.exp_ep    = self.spline_potential((x_values[0,:],x_values[1,:])).reshape((len(chi),len(ells)))
            self.exp_ep_ee = self.spline_potential_ee((x_values[0,:],x_values[1,:])).reshape((len(chi),len(ells)))
            chi_integrand_ep = self.weight_wep()**2/chi**2/(1+z_of_chi)

        if self.diagonal:
            # LSS contribution — shape (n_ell, n_chi), FRB-independent.
            lss = 10**self.exp.T * chi_integrand[None, :] * self.cosmology.h**3

            if not use_ep:
                # Fast path: cumulative trapezoid avoids the O(n_ell×N×n_chi) broadcast.
                # Each FRB i integrates from chi[0] to the last chi where z(chi) <= zet[i].
                d_chi = np.diff(chi)
                cumint = np.empty((len(ells), len(chi)))
                cumint[:, 0] = 0.0
                cumint[:, 1:] = np.cumsum(
                    0.5 * (lss[:, :-1] + lss[:, 1:]) * d_chi[None, :], axis=-1
                )
                chi_at_zet = np.interp(self.zet, z_of_chi, chi)
                cutoff = np.clip(np.searchsorted(chi, chi_at_zet) - 1, 0, len(chi) - 1)
                self.C_ell = cumint[:, cutoff]  # (n_ell, n_zet)
            else:
                integrand = (lss[:, None, :]
                    + (10**self.exp_ep.T[:, None, :] * np.diag(freq_weight_squared)[None, :, None]) * chi_integrand_ep[None, None, :]
                    + (10**self.exp_ep_ee.T[:, None, :] * (np.diag(freq_weight_squared)**.5)[None, :, None]) * (2*(chi_integrand_ep*chi_integrand)**.5)[None, None, :])
                tomo_weighting = (z_of_chi[None, :] <= self.zet[:, None]).astype(float)
                self.C_ell = simpson(tomo_weighting[None, :, :]*integrand, x=chi, axis=-1)

        else:
            if use_ep:
                ep_term = ((10**self.exp_ep.T[:, None, None, :]*freq_weight_squared[None, :, :, None])*chi_integrand_ep[None,None, None,:]
                           + (10**self.exp_ep_ee.T[:, None, None, :]*(freq_weight_squared**.5)[None, :, :, None])*(2*(chi_integrand_ep*chi_integrand)**.5)[None,None,None,:])
            else:
                ep_term = 0.0

            if len(self.zet) < 110:
                integrand = (10**self.exp.T[:, None, None, :]*chi_integrand[None, None, None, :] + ep_term) * self.cosmology.h**3
                # tomo_weighting[i, j, k] = 1 iff z(chi[k]) <= min(zet[i], zet[j])
                z_min = np.minimum(self.zet[:, None], self.zet[None, :])
                tomo_weighting = (z_of_chi[None, None, :] <= z_min[:, :, None]).astype(float)
                self.C_ell = simpson(tomo_weighting[None, :, :, :]*integrand,x = chi, axis = -1)
            else:
                integrand = (10**self.exp.T[:, None, None, :]*chi_integrand[None, None, None, :] + ep_term) * self.cosmology.h**3
                zet_interpolation = np.linspace(self.zet[0], self.zet[-1], 100)
                z_min = np.minimum(zet_interpolation[:, None], zet_interpolation[None, :])
                tomo_weighting = (z_of_chi[None, None, :] <= z_min[:, :, None]).astype(float)
                self.C_ell = simpson(tomo_weighting[None, :, :, :]*integrand, x = chi, axis = -1)
                self.spline_C_ell = RegularGridInterpolator((np.log(self.ell),zet_interpolation,zet_interpolation), np.log(self.C_ell) ,bounds_error= False, fill_value = None)

    def get_covariance(self):
        """Build the full-sky DM covariance matrix via a Legendre series sum.

        Used when ``flat_sky=False`` and the catalogue contains fewer than 100
        FRBs.  For each pair (i, j) the angular correlation function is
        evaluated at the true angular separation δθ_{ij} by summing the
        Legendre expansion

            C(δθ) = Σ_ℓ (2ℓ+1)/(4π) · C_ell(ℓ) · P_ℓ(cos δθ).

        The upper multipole cutoff for off-diagonal pairs is chosen adaptively
        as ℓ_max ~ 1000 / δθ to avoid oscillatory cancellation; it is capped at
        20 000.  The diagonal (auto-variance) uses a fixed ℓ_max = 20 000.

        Sets ``self.covariance`` in-place and returns it.

        Returns
        -------
        ndarray of shape (N, N)
            Symmetric DM covariance matrix [pc² cm⁻⁶].
        """
        MAX_ELL = 20000
        ell_fac = 1000
        self.covariance = np.zeros((len(self.zet), len(self.zet)))
        for z_idx_i in tqdm(np.arange(len(self.zet)),disable=self.verbose):
            for z_idx_j, z_val_j in enumerate(self.zet[z_idx_i:]):
                spline = UnivariateSpline(np.log(self.ell[1:]), np.log(
                    self.C_ell[1:, z_idx_i, z_idx_j+z_idx_i]), k= 3, s =0, ext = 0)
                if (z_idx_i != z_idx_j + z_idx_i):
                    N_ell = int(
                        ell_fac/self.delta_theta[z_idx_i, z_idx_i+z_idx_j])
                    if (N_ell >= MAX_ELL):
                        N_ell = MAX_ELL
                    ells = np.arange(1,N_ell)
                    cov_ell = np.exp(spline(np.log(ells)))
                    sph0 = np.sqrt((2.*ells + 1)/4/np.pi)
                    product = sph0*eval_legendre(ells,np.cos(self.delta_theta[z_idx_i,z_idx_j])) 
                    self.covariance[z_idx_i, z_idx_j+z_idx_i] = np.sum(product*cov_ell)                    
                    self.covariance[z_idx_j+z_idx_i,
                           z_idx_i] = self.covariance[z_idx_i, z_idx_j+z_idx_i]
                else:
                    ells_diag = np.arange(1,MAX_ELL)
                    cov_ell = np.exp(spline(np.log(ells_diag)))
                    product = (2.*ells_diag + 1)/4/np.pi
                    self.covariance[z_idx_i, z_idx_j+z_idx_i] = np.sum(product*cov_ell)
                    self.covariance[z_idx_j+z_idx_i,
                           z_idx_i] = self.covariance[z_idx_i, z_idx_j+z_idx_i]
        return self.covariance

    def get_covariance_flat_sky(self):
        """Build the DM covariance matrix using the flat-sky Hankel transform.

        Used when ``flat_sky=True`` or the catalogue contains 100 or more FRBs.
        Each off-diagonal element is computed as the zeroth-order Hankel
        transform of C_ell:

            C(δθ) = (1/2π) ∫ dℓ ℓ · C_ell(ℓ) · J₀(ℓ δθ).

        The integral is evaluated on a fine ℓ grid of 10⁵ points spanning
        ``self.ell``.  Pairs with δθ > π are skipped (unphysical separation).

        When ``diagonal=True`` all off-diagonal entries remain zero and only
        the diagonal (auto-variance) is computed by setting J₀ = 1.

        For N ≥ 110 FRBs ``self.C_ell`` is stored on a coarser redshift grid
        and evaluated via ``self.spline_C_ell`` during the pair loop.

        Sets ``self.covariance`` in-place.
        """
        self.covariance = np.zeros((len(self.zet), len(self.zet)))
        ell_integral = np.geomspace(self.ell[0], self.ell[-1], int(1e5))
        x_values = np.zeros((3,len(ell_integral)))
        x_values[0,:] = np.log(ell_integral)
        
        if self.diagonal:
            # C_ell is piecewise-linear between its 100 knots.  Integrate
            # ell * C_ell(ell) analytically on each segment — exact for
            # piecewise-linear C_ell and avoids allocating the 1e5-point grid.
            #
            # For segment [a, b] with endpoint values Ca, Cb:
            #   ∫_a^b ell * C(ell) d(ell) = (b-a)/6 * [(2a+b)*Ca + (a+2b)*Cb]
            a  = self.ell[:-1]                 # (n_ell-1,)
            b  = self.ell[1:]                  # (n_ell-1,)
            Ca = self.C_ell[:-1, :]            # (n_ell-1, n_zet)
            Cb = self.C_ell[1:,  :]            # (n_ell-1, n_zet)
            W  = ((b - a) / 6.0)[:, None] * ((2*a + b)[:, None] * Ca + (a + 2*b)[:, None] * Cb)
            np.fill_diagonal(self.covariance, np.sum(W, axis=0) / (2.0 * np.pi))
        else:
            for z_idx_i in tqdm(np.arange(len(self.zet)),disable=self.verbose):
                for z_idx_j in range (z_idx_i, len(self.zet)):    
                    if self.delta_theta[z_idx_i,z_idx_j] > 180./180*np.pi:
                        continue
                    else:
                        bessel = jv(0,ell_integral*self.delta_theta[z_idx_i,z_idx_j])
                        if len(self.zet) < 110:
                            integrand = np.interp(ell_integral, self.ell, self.C_ell[:,z_idx_i,z_idx_j])*ell_integral*bessel
                            self.covariance[z_idx_i,z_idx_j] = simpson(integrand,x = ell_integral)/2.0/np.pi
                            self.covariance[z_idx_j,z_idx_i] = self.covariance[z_idx_i,z_idx_j]
                        else:
                            x_values[1,:] = self.zet[z_idx_i]*np.ones_like(ell_integral)
                            x_values[2,:] = self.zet[z_idx_j]*np.ones_like(ell_integral)
                            integrand = np.exp(self.spline_C_ell((x_values[0,:],x_values[1,:], x_values[2,:]))).reshape(len(ell_integral))*ell_integral*bessel
                            self.covariance[z_idx_i,z_idx_j] = simpson(integrand,x =ell_integral)/2.0/np.pi
                            self.covariance[z_idx_j,z_idx_i] = self.covariance[z_idx_i,z_idx_j]



# backward-compatibility alias
covariance_frb_background = FRBCovariance
