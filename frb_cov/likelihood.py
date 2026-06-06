import numpy as np
from scipy import integrate

from .base import FRBCovariance


class FRBLikelihood:
    """Log-posterior for an FRB catalogue given cosmological and astrophysical parameters.

    The class wraps five variants of the marginalised log-likelihood
    ``ln P(DM_obs | params)``.  All variants share the same parameter
    vector convention and the same expensive sub-step — building the
    LSS DM covariance via :class:`~frb_cov.FRBCovariance` — so the
    emulators and catalogue are stored once and reused.

    Parameters
    ----------
    DM_obs : array_like of shape (N,)
        Observed total dispersion measures [pc cm⁻³].
    DM_MW : array_like of shape (N,)
        Milky Way ISM contribution for each FRB [pc cm⁻³].
    redshifts : array_like of shape (N,)
        Host redshifts.
    ra : array_like of shape (N,)
        Right ascensions [radians].
    dec : array_like of shape (N,)
        Declinations [radians].
    bias_emu : cosmopower_NN
        Trained emulator for the electron-bias power spectrum
        P_ee(k, z) − P_mm(k, z).
    power_emu : cosmopower_NN
        Trained emulator for the non-linear matter power spectrum P_mm(k, z).
    power_lin_emu : cosmopower_NN, optional
        Trained emulator for the linear matter power spectrum.  When
        provided, the non-linear spectrum is replaced by the linear one
        below k = 0.01 h/Mpc.
    planck_mean : array_like of shape (5,), optional
        Mean of the Gaussian Planck prior on the cosmological parameters
        [ω_b, ω_c, σ_8, n_s, h].  Both *planck_mean* and *planck_cov*
        must be supplied for the prior to be active.
    planck_cov : array_like of shape (5, 5), optional
        Covariance matrix of the Planck prior.
    cosmo_template : dict, optional
        Fixed cosmological parameters that are not sampled (e.g.
        ``w0``, ``wa``, ``neff``).  Keys and values follow the convention
        of :class:`~frb_cov.FRBCovariance`.  Defaults are set to Planck
        2018 best-fit values; any key present here overrides the default.
    n_chi_limber : int, optional
        Number of comoving-distance points used in the Limber integral
        inside :class:`~frb_cov.FRBCovariance` (default 1000).
        Reducing this speeds up every likelihood call at the cost of
        accuracy in the angular power spectra.
    n_chi_interp : int, optional
        Number of points on which the cosmopower emulators are evaluated
        inside :class:`~frb_cov.FRBCovariance` (default 200).

    Attributes
    ----------
    DM_obs, DM_MW, z, ra, dec : ndarray of shape (N,)
        Catalogue arrays as stored (original input order).
    delta_theta : ndarray of shape (N, N)
        Pre-computed pairwise great-circle separations [radians].
    n_chi_limber, n_chi_interp : int
        Mutable grid-size parameters passed to every :class:`FRBCovariance`
        call.  Can be changed after construction to trade speed for accuracy.

    Notes
    -----
    **Parameter vector convention** used by all public likelihood methods:

    =====  ============  =====================================================
    Index  Symbol        Description
    =====  ============  =====================================================
    0      DM_host       Mean host-galaxy DM [pc cm⁻³]
    1      σ_host        Scatter of host DM [pc cm⁻³]
    2      log T_AGN     Log₁₀ of AGN feedback temperature
    3      ω_b           Physical baryon density Ω_b h²
    4      ω_c           Physical cold-dark-matter density Ω_c h²
    5      σ_8           Matter fluctuation amplitude
    6      n_s           Primordial spectral index
    7      h             Dimensionless Hubble constant H₀/100
    (8)    f_IGM         IGM baryon fraction — ``log_likelihood_figm_marginalized`` only
    =====  ============  =====================================================

    The host DM is modelled as a log-normal distribution with mean *DM_host*
    and standard deviation *σ_host*.  The LSS contribution is also modelled
    as a log-normal whose parameters are derived from the mean DM and
    variance returned by :class:`~frb_cov.FRBCovariance`.

    All likelihood methods return ``−1e200`` when ``params`` fails the
    hard prior ``50 ≤ DM_host ≤ 1000``, ``50 ≤ σ_host ≤ 1000``,
    ``7.0 ≤ log T_AGN ≤ 8.6``.
    """

    # ------------------------------------------------------------------ #
    # Class-level constants (computed once, shared across all instances). #
    # ------------------------------------------------------------------ #
    _TINY         = np.finfo(float).tiny
    _SQRT_2PI     = np.sqrt(2.0 * np.pi)
    _LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)

    _DEFAULT_COSMO = dict(
        omega_de=0.7,
        w0=-1.0,
        wa=0.0,
        neff=3.046,
        m_nu=0.06,
        Tcmb0=2.725,
        alpha_B=0.05,
        alpha_M=0.05,
        ks=0.1,
        delta_gamma=0.0,
        diagonal=True,
        path_to_f_iGM='./../data/figm',
    )

    # ------------------------------------------------------------------ #
    # Constructor                                                          #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        DM_obs,
        DM_MW,
        redshifts,
        ra,
        dec,
        bias_emu,
        power_emu,
        power_lin_emu=None,
        planck_mean=None,
        planck_cov=None,
        cosmo_template=None,
        n_chi_limber=1000,
        n_chi_interp=200,
    ):
        self.DM_obs = np.asarray(DM_obs, dtype=float)
        self.DM_MW  = np.asarray(DM_MW,  dtype=float)
        self.z      = np.asarray(redshifts, dtype=float)
        self.ra     = np.asarray(ra,  dtype=float)
        self.dec    = np.asarray(dec, dtype=float)

        self.bias_emu      = bias_emu
        self.power_emu     = power_emu
        self.power_lin_emu = power_lin_emu
        self.n_chi_limber  = n_chi_limber
        self.n_chi_interp  = n_chi_interp

        self._cosmo_base = dict(self._DEFAULT_COSMO)
        if cosmo_template is not None:
            self._cosmo_base.update(cosmo_template)

        # Planck prior: pre-factorise the covariance via Cholesky so every
        # evaluation is a single triangular solve + dot product.
        if planck_mean is not None and planck_cov is not None:
            self._planck_mean = np.asarray(planck_mean, dtype=float)
            L = np.linalg.cholesky(np.asarray(planck_cov, dtype=float))
            self._planck_L_inv    = np.linalg.inv(L)
            self._planck_log_norm = (
                -0.5 * len(planck_mean) * np.log(2.0 * np.pi)
                - np.sum(np.log(np.diag(L)))
            )
        else:
            self._planck_L_inv = None

        # Sort index and sorted arrays for methods that require z ascending.
        self._z_sort_idx    = np.argsort(self.z)
        self._z_sorted      = self.z[self._z_sort_idx]
        self._DM_MW_sorted  = self.DM_MW[self._z_sort_idx]
        self._DM_obs_sorted = self.DM_obs[self._z_sort_idx]

        # Pairwise great-circle separations — vectorised O(N²) computation.
        sin_ra       = np.sin(self.ra)
        cos_ra       = np.cos(self.ra)
        cos_dec_diff = np.cos(self.dec[:, None] - self.dec[None, :])
        dot = (sin_ra[:, None] * sin_ra[None, :]
               + cos_ra[:, None] * cos_ra[None, :] * cos_dec_diff)
        self.delta_theta = np.arccos(np.clip(dot, -1.0, 1.0))
        np.fill_diagonal(self.delta_theta, 0.0)

        # delta_theta re-indexed to match the sorted-z ordering.
        self._delta_theta_sorted = self.delta_theta[
            np.ix_(self._z_sort_idx, self._z_sort_idx)
        ]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _planck_prior(self, params):
        """Evaluate the Gaussian Planck log-prior at ``params``.

        Parameters
        ----------
        params : array_like of shape (5,)
            Cosmological parameters [ω_b, ω_c, σ_8, n_s, h].

        Returns
        -------
        float
            Log-prior value, or 0.0 if no prior was supplied at construction.
        """
        if self._planck_L_inv is None:
            return 0.0
        diff = params - self._planck_mean
        x    = self._planck_L_inv @ diff
        return self._planck_log_norm - 0.5 * (x @ x)

    def _build_cosmo(self, params):
        """Construct the cosmology dictionary from the sampled ``params`` vector.

        Converts physical densities (ω_b, ω_c) and the Hubble constant h to
        the density parameters Ω_b, Ω_m, Ω_de expected by
        :class:`~frb_cov.FRBCovariance`, and merges with the fixed parameters
        in ``self._cosmo_base``.

        Parameters
        ----------
        params : array_like
            Full parameter vector (indices 2–7 are used).

        Returns
        -------
        dict
            Cosmology dictionary ready to be passed to
            :class:`~frb_cov.FRBCovariance`.
        """
        cosmo = dict(self._cosmo_base)
        h2    = params[7] ** 2
        omega_m = (params[3] + params[4]) / h2
        cosmo['omega_b']  = params[3] / h2
        cosmo['omega_m']  = omega_m
        cosmo['omega_de'] = 1.0 - omega_m
        cosmo['sigma_8']  = params[5]
        cosmo['ns']       = params[6]
        cosmo['h']        = params[7]
        cosmo['logTAGN']  = params[2]
        return cosmo

    @staticmethod
    def _to_log_normal(mean, variance):
        """Convert a mean and variance to log-normal shape parameters.

        Derives the log-normal location parameter μ_ln and scale parameter
        σ_ln from the first two moments of the distribution:

            μ_ln   = ln(mean) − ½ ln(1 + variance/mean²)
            σ_ln   = sqrt(ln(1 + variance/mean²))

        Uses ``numpy.log1p`` for numerical stability when
        ``variance << mean²`` (typical for low-redshift FRBs).

        Parameters
        ----------
        mean : float or ndarray
            Distribution mean.
        variance : float or ndarray
            Distribution variance.

        Returns
        -------
        mu_ln, sigma_ln : float or ndarray
            Log-normal parameters matching the supplied moments.
        """
        r      = variance / mean ** 2
        log1pr = np.log1p(r)
        return np.log(mean) - 0.5 * log1pr, np.sqrt(log1pr)

    @classmethod
    def _pdf_log_normal(cls, x, mu, sigma, diag=False):
        """Evaluate the log-normal PDF with safe handling of x ≤ 0.

        Returns the log-normal probability density for each element of ``x``,
        setting the result to zero wherever ``x ≤ 0`` (outside the support).
        Uses a log-space formula to minimise intermediate allocations.

        Parameters
        ----------
        x : ndarray
            Evaluation points.  May contain non-positive values; these yield 0.
        mu : float or ndarray
            Log-normal location parameter μ_ln.
        sigma : float or ndarray
            Log-normal scale parameter σ_ln > 0.
        diag : bool, optional
            Broadcasting convention when ``mu`` is an array:

            * ``True``  — FRB axis is **axis 0** of ``x``
              (standard PDF, σ in denominator).
            * ``False`` — FRB axis is **axis 1** of ``x``
              (non-standard convention with σ in numerator, used for the
              pdetect methods where the result is always renormalised).

        Returns
        -------
        ndarray
            PDF values, same shape as ``x``.  NaN and Inf are replaced by 0.
        """
        xc = np.maximum(x, cls._TINY)
        with np.errstate(divide='ignore', invalid='ignore'):
            if isinstance(mu, np.ndarray):
                if diag:
                    trailing = (1,) * (xc.ndim - 1)
                    mu_bc    = mu.reshape(mu.shape + trailing)
                    sigma_bc = sigma.reshape(sigma.shape + trailing)
                else:
                    trailing = (1,) * (xc.ndim - 2)
                    mu_bc    = mu.reshape((1,) + mu.shape + trailing)
                    sigma_bc = sigma.reshape((1,) + sigma.shape + trailing)
                lx = np.log(xc)
                z  = (lx - mu_bc) / sigma_bc
                if diag:
                    result = np.exp(-0.5*z*z - lx - np.log(sigma_bc) - cls._LOG_SQRT_2PI)
                else:
                    result = np.exp(-0.5*z*z - lx + np.log(sigma_bc) - cls._LOG_SQRT_2PI)
            else:
                lx     = np.log(xc)
                z      = (lx - mu) / sigma
                result = np.exp(-0.5*z*z - lx - np.log(sigma) - cls._LOG_SQRT_2PI)
        return np.where(x > 0, np.nan_to_num(result, nan=0.0), 0.0)

    @classmethod
    def _gaussian_pdf(cls, x, mu, sigma):
        """Evaluate the Gaussian PDF.

        Parameters
        ----------
        x : float or ndarray
            Evaluation point(s).
        mu : float or ndarray
            Mean.  If an array of shape (N,), ``x`` is broadcast against it
            as ``x[:, None]``, producing an output of shape
            ``(len(x), N)``.
        sigma : float or ndarray
            Standard deviation, same shape as ``mu``.

        Returns
        -------
        float or ndarray
            Gaussian probability density.
        """
        if isinstance(mu, np.ndarray):
            z = (x[:, None] - mu[None, :]) / sigma[None, :]
            return np.exp(-0.5 * z * z) / (cls._SQRT_2PI * sigma[None, :])
        z = (x - mu) / sigma
        return np.exp(-0.5 * z * z) / (cls._SQRT_2PI * sigma)

    def _lss_params(self, cosmo, sorted_z=False, figm=1.0):
        """Compute log-normal parameters for the LSS DM distribution.

        Runs :class:`~frb_cov.FRBCovariance` to obtain the mean DM and
        diagonal variance for each FRB, then converts them to log-normal
        parameters via :meth:`_to_log_normal`.

        Parameters
        ----------
        cosmo : dict
            Cosmology dictionary (output of :meth:`_build_cosmo`).
        sorted_z : bool, optional
            If True, use the pre-sorted redshift array and the correspondingly
            re-indexed ``delta_theta`` matrix.  Required for pdetect methods,
            which need redshifts in ascending order for the normalisation step.
        figm : float, optional
            IGM baryon fraction.  Scales the mean DM by ``figm`` and the
            variance by ``figm²`` before converting to log-normal parameters.

        Returns
        -------
        mu_ln, sigma_ln : ndarray of shape (N,)
            Log-normal location and scale parameters for each FRB.
        """
        z           = self._z_sorted      if sorted_z else self.z
        delta_theta = self._delta_theta_sorted if sorted_z else self.delta_theta
        cov_obj = FRBCovariance(
            cosmo, self.bias_emu, self.power_emu, z, delta_theta,
            flat_sky=True, plin_emu=self.power_lin_emu,
            n_chi_limber=self.n_chi_limber, n_chi_interp=self.n_chi_interp,
        )
        return self._to_log_normal(
            cov_obj.DM * figm,
            np.diag(cov_obj.covariance) * figm ** 2,
        )

    def _host_ln_params(self, DM_host, sigma_host):
        """Convert host DM mean and scatter to log-normal parameters.

        Parameters
        ----------
        DM_host : float
            Mean host DM [pc cm⁻³].
        sigma_host : float
            Standard deviation of host DM [pc cm⁻³].

        Returns
        -------
        mu_ln_host, sigma_ln_host : float
            Log-normal location and scale parameters.
        """
        return self._to_log_normal(DM_host, sigma_host ** 2)

    def _valid_params(self, params):
        """Return True if ``params`` satisfies the hard parameter bounds.

        Bounds:  ``50 ≤ DM_host ≤ 1000``,  ``50 ≤ σ_host ≤ 1000``,
        ``7.0 ≤ log T_AGN ≤ 8.6``.  Evaluations outside these bounds
        return ``−1e200`` immediately without computing the covariance.

        The upper bounds on DM_host and σ_host are essential for the pdetect
        likelihood variants.  Those methods compute P(DM_obs | DM < DMmax)
        = P(DM_obs) / P(DM < DMmax).  As σ_host → ∞ both numerator and
        denominator shrink to zero but their ratio converges to a finite
        constant, creating a flat plateau along which MCMC walkers drift
        without penalty.  The upper bounds close that plateau.
        """
        return (50 <= params[0] <= 1000
                and 50 <= params[1] <= 1000
                and 7.0 <= params[2] <= 8.6)

    @staticmethod
    def _interp_pdf_at_obs(DM_obs_sorted, DM_int, result):
        """Vectorised linear interpolation of a PDF table at observed DM values.

        For each FRB i, evaluates ``result[:, i]`` at ``DM_obs_sorted[i]``
        using a single ``numpy.searchsorted`` call across all FRBs.

        Parameters
        ----------
        DM_obs_sorted : ndarray of shape (N,)
            Observed DMs in sorted-z order.
        DM_int : ndarray of shape (M,)
            Sorted DM grid on which ``result`` is tabulated.
        result : ndarray of shape (M, N)
            Tabulated PDF values; ``result[:, i]`` is the PDF for FRB i.

        Returns
        -------
        ndarray of shape (N,)
            PDF evaluated at each observed DM.
        """
        j       = np.clip(np.searchsorted(DM_int, DM_obs_sorted) - 1, 0, len(DM_int) - 2)
        t       = (DM_obs_sorted - DM_int[j]) / (DM_int[j + 1] - DM_int[j])
        t       = np.clip(t, 0.0, 1.0)
        frb_idx = np.arange(len(DM_obs_sorted))
        return result[j, frb_idx] * (1.0 - t) + result[j + 1, frb_idx] * t

    # ------------------------------------------------------------------ #
    # FFT convolution helper shared by pdetect methods                    #
    # ------------------------------------------------------------------ #

    def _host_lss_convolution(self, mu_ln, sigma_ln, mu_ln_host, sigma_ln_host,
                               dV, V_max):
        """Compute the host × LSS marginal PDF via FFT convolution.

        For each FRB i the marginal PDF of the LSS+host DM residual is

            P_i(Y) = ∫ h_i(V) · g_i(Y − V) dV

        where ``h_i(V) = (1+z_i) · f_host(V·(1+z_i))`` is the host PDF
        expressed in effective-DM space and ``g_i(V) = f_LSS_i(V)`` is
        the LSS PDF.  The convolution is computed with a batched real FFT
        (zero-padded to the next power of two) on a uniform grid
        [0, ``V_max``] with spacing ``dV``.

        Parameters
        ----------
        mu_ln, sigma_ln : ndarray of shape (N,)
            Log-normal parameters of the LSS PDF (sorted-z order).
        mu_ln_host, sigma_ln_host : float
            Log-normal parameters of the host DM PDF.
        dV : float
            Grid spacing [pc cm⁻³].
        V_max : float
            Upper limit of the convolution grid [pc cm⁻³].

        Returns
        -------
        P_conv : ndarray of shape (N_V, N)
            ``P_conv[k, i]`` ≈ P_i(V_grid[k]).
        V_grid : ndarray of shape (N_V,)
            Uniform evaluation grid.
        """
        N_V    = int(V_max / dV) + 1
        V_grid = np.arange(N_V) * dV                                   # (N_V,)

        # h_i(V) = (1+z_i) · f_host(V·(1+z_i))
        X_host = V_grid[:, None] * (1.0 + self._z_sorted[None, :])    # (N_V, N)
        Xc     = np.maximum(X_host, self._TINY)
        with np.errstate(divide='ignore', invalid='ignore'):
            lx = np.log(Xc)
            z  = (lx - mu_ln_host) / sigma_ln_host
            h  = np.exp(-0.5*z*z - lx - np.log(sigma_ln_host) - self._LOG_SQRT_2PI)
        h = np.where(X_host > 0, np.nan_to_num(h, 0.0), 0.0) * (1.0 + self._z_sorted[None, :])

        # g_i(V) = f_LSS_i(V) — standard log-normal, FRBs on axis 1
        Vc = np.maximum(V_grid[:, None], self._TINY)                   # (N_V, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            lv  = np.log(Vc)
            z_g = (lv - mu_ln[None, :]) / sigma_ln[None, :]
            g   = np.exp(-0.5*z_g*z_g - lv - np.log(sigma_ln[None, :]) - self._LOG_SQRT_2PI)
        g = np.where(V_grid[:, None] > 0, np.nan_to_num(g, 0.0), 0.0)  # (N_V, N)

        N_fft  = 1 << (2 * N_V - 1).bit_length()                      # next power of 2
        H      = np.fft.rfft(h, n=N_fft, axis=0)
        G      = np.fft.rfft(g, n=N_fft, axis=0)
        P_conv = np.maximum(np.fft.irfft(H * G, n=N_fft, axis=0)[:N_V] * dV, 0.0)
        return P_conv, V_grid

    # ------------------------------------------------------------------ #
    # Public likelihood methods                                           #
    # ------------------------------------------------------------------ #

    def log_likelihood(self, params, DM_MW_halo=50, N_int=2000):
        """Log-posterior with fixed MW DM and no selection function.

        Marginalises analytically over the host DM by numerically integrating

            P_i(DM_obs) = ∫ f_host(X) · f_LSS_i(DM_obs − DM_MW_i − halo − X/(1+z_i)) dX

        on a log-spaced host-DM grid.  The host PDF is normalised over the
        integration range so that a truncated grid does not bias the posterior.

        Parameters
        ----------
        params : array_like of shape (8,)
            Parameter vector — see class docstring.
        DM_MW_halo : float, optional
            Fixed MW halo DM contribution [pc cm⁻³] (default 50).
        N_int : int, optional
            Number of host-DM integration points on
            ``geomspace(1, 1000)`` (default 2000).

        Returns
        -------
        float
            Log-posterior value, or ``−1e200`` if parameters are out of
            bounds or the result is not finite.
        """
        if not self._valid_params(params):
            return -1e200
        cosmo = self._build_cosmo(params)

        mu_ln, sigma_ln         = self._lss_params(cosmo)
        mu_ln_host, sigma_ln_host = self._host_ln_params(params[0], params[1])

        X_DM_host = np.geomspace(1e-2, 5000, N_int)
        pdf_ln    = self._pdf_log_normal(X_DM_host, mu_ln_host, sigma_ln_host)
        pdf_ln    = pdf_ln / integrate.simpson(pdf_ln, x=X_DM_host)
        integrand = pdf_ln[None, :] * self._pdf_log_normal(
            self.DM_obs[:, None] - self.DM_MW[:, None] - DM_MW_halo
            - X_DM_host[None, :] / (1.0 + self.z[:, None]),
            mu=mu_ln, sigma=sigma_ln, diag=True,
        )
        result = integrate.simpson(integrand, x=X_DM_host, axis=-1)
        lnl    = np.sum(np.log(result)) + self._planck_prior(params[3:])
        return lnl if np.isfinite(lnl) else -1e200

    def log_likelihood_mw_marginalized(self, params, DM_MW_halo=50,
                                        rel_error_DM_MW=0.2, N_int=200, n_gh=10):
        """Log-posterior marginalising over the MW DM uncertainty.

        The MW DM is uncertain at the ``rel_error_DM_MW`` fractional level,
        modelled as a Gaussian ``DM_MW_true ~ N(DM_MW_est, (rel · DM_MW_est)²)``.
        The Gaussian integral is evaluated with Gauss-Hermite quadrature
        (``n_gh`` nodes) rather than a uniform grid, reducing the dominant
        array from shape ``(N, N_int, N_int_MW)`` to ``(N, N_int, n_gh)``.

        For a smooth log-normal integrand ``n_gh = 10`` matches a 200-point
        uniform grid to better than ``|Δ ln L| < 10⁻⁴``.

        Parameters
        ----------
        params : array_like of shape (8,)
            Parameter vector — see class docstring.
        DM_MW_halo : float, optional
            Fixed MW halo DM contribution [pc cm⁻³] (default 50).
        rel_error_DM_MW : float, optional
            Fractional uncertainty on the MW ISM DM (default 0.2 = 20 %).
        N_int : int, optional
            Number of host-DM integration points (default 200).
        n_gh : int, optional
            Number of Gauss-Hermite nodes for the MW integral (default 10).

        Returns
        -------
        float
            Log-posterior value, or ``−1e200`` if out of bounds / non-finite.
        """
        if not self._valid_params(params):
            return -1e200
        cosmo = self._build_cosmo(params)

        mu_ln, sigma_ln           = self._lss_params(cosmo)
        mu_ln_host, sigma_ln_host = self._host_ln_params(params[0], params[1])

        X_DM_host = np.geomspace(1e-2, 5000, N_int)
        pdf_ln    = self._pdf_log_normal(X_DM_host, mu_ln_host, sigma_ln_host)
        pdf_ln    = pdf_ln / integrate.simpson(pdf_ln, x=X_DM_host)

        # GH quadrature:  ∫ f(X_MW) N(X_MW; μ, σ²) dX_MW ≈ (1/√π) Σ_k w_k f(μ + √2·σ·t_k)
        t_gh, w_gh = np.polynomial.hermite.hermgauss(n_gh)
        sigma_MW   = self.DM_MW * rel_error_DM_MW                              # (N,)
        X_MW_gh    = self.DM_MW[:, None] + np.sqrt(2.0) * sigma_MW[:, None] * t_gh  # (N, n_gh)

        arg   = (self.DM_obs[:, None, None]
                 - X_MW_gh[:, None, :]
                 - DM_MW_halo
                 - X_DM_host[None, :, None] / (1.0 + self.z[:, None, None]))
        f_lss = self._pdf_log_normal(arg, mu=mu_ln, sigma=sigma_ln, diag=True)

        lss_mw = (f_lss * w_gh[None, None, :]).sum(axis=-1) / np.sqrt(np.pi)  # (N, N_int)
        result = integrate.simpson(pdf_ln[None, :] * lss_mw, x=X_DM_host, axis=-1)
        lnl    = np.sum(np.log(result)) + self._planck_prior(params[3:])
        return lnl if np.isfinite(lnl) else -1e200

    def log_likelihood_with_pdetect(self, params, DM_MW_halo=50,
                                     DMmax=1500, N_DM_int=500,
                                     dV=3.0, V_max=4000.0):
        """Log-posterior with a step-function detection selection (DM < DMmax).

        The marginal PDF P_i(DM) is computed via FFT convolution of the scaled
        host PDF with the LSS PDF (see :meth:`_host_lss_convolution`), then
        renormalised to account for the detection threshold:

            P_i^det(DM) ∝ P_i(DM) · 𝟙[DM < DMmax]

        The result is evaluated at the observed DM of each FRB.

        MW DM is treated as perfectly known (fixed at ``DM_MW_i + halo``).
        For MW marginalisation combined with detection efficiency use
        :meth:`log_likelihood_mw_marginalized_with_pdetect`.

        Parameters
        ----------
        params : array_like of shape (8,)
            Parameter vector — see class docstring.
        DM_MW_halo : float, optional
            Fixed MW halo DM contribution [pc cm⁻³] (default 50).
        DMmax : float, optional
            Detection threshold [pc cm⁻³] — FRBs above this DM are undetected
            (default 1500).
        N_DM_int : int, optional
            Number of output DM grid points used for normalisation (default 500).
        dV : float, optional
            Uniform convolution grid spacing [pc cm⁻³] (default 3).
            Finer spacing improves accuracy; ``dV=2`` gives ``|Δ ln L| < 2×10⁻⁵``.
        V_max : float, optional
            Upper limit of the convolution grid [pc cm⁻³] (default 4000).

        Returns
        -------
        float
            Log-posterior value, or ``−1e200`` if out of bounds / non-finite.
        """
        if not self._valid_params(params):
            return -1e200
        cosmo = self._build_cosmo(params)

        mu_ln, sigma_ln           = self._lss_params(cosmo, sorted_z=True)
        mu_ln_host, sigma_ln_host = self._host_ln_params(params[0], params[1])

        P_conv, V_grid = self._host_lss_convolution(
            mu_ln, sigma_ln, mu_ln_host, sigma_ln_host, dV, V_max)
        N_V = len(V_grid)

        DM_int_max = V_max + float(self._DM_MW_sorted.min()) + DM_MW_halo
        DM_int = np.geomspace(50, DM_int_max, N_DM_int)
        Y      = DM_int[:, None] - self._DM_MW_sorted[None, :] - DM_MW_halo  # (N_DM_int, N)

        y_idx  = Y / dV
        j      = np.clip(np.floor(y_idx).astype(int), 0, N_V - 2)
        t      = np.clip(y_idx - j, 0.0, 1.0)
        fidx   = np.arange(len(self._z_sorted))
        result = np.where(Y >= 0,
                          (1.0 - t) * P_conv[j, fidx] + t * P_conv[j + 1, fidx],
                          0.0)

        norm0 = integrate.simpson(result, x=DM_int, axis=0)
        # norm0 < 0.5 means the convolution grid (V_max) is too small for
        # these parameters: the PDF tail extends beyond V_max, and dividing
        # by the truncated integral would artificially inflate the PDF.
        if np.any(norm0 < 0.5):
            return -1e200
        result /= norm0
        result  = np.where(DM_int[:, None] < DMmax, result, 0.0)
        norm1 = integrate.simpson(result, x=DM_int, axis=0)
        if np.any(norm1 <= self._TINY):
            return -1e200
        result /= norm1

        real_result = self._interp_pdf_at_obs(self._DM_obs_sorted, DM_int, result)
        lnl = np.sum(np.log(real_result)) + self._planck_prior(params[3:])
        return lnl if np.isfinite(lnl) else -1e200

    def log_likelihood_mw_marginalized_with_pdetect(
        self, params, DM_MW_halo=50, rel_error_DM_MW=0.2, DMmax=1500,
        N_DM_int=100, n_gh=10, dV=3.0, V_max=4000.0,
    ):
        """Log-posterior marginalising over MW DM with a detection selection function.

        Combines two numerical strategies to avoid constructing the full
        ``(N_DM_int, N, N_host, N_MW)`` array that the naive implementation
        would require:

        * **Host DM** — FFT convolution via :meth:`_host_lss_convolution`,
          computed once for all FRBs.  This gives ``P_conv[k, i]`` on a
          uniform V grid.
        * **MW DM** — Gauss-Hermite quadrature with ``n_gh`` nodes per FRB.
          The ``n_gh`` offsets per FRB select evaluation points in the
          precomputed ``P_conv`` table without rebuilding it.

        The combined integral is

            P_i(DM) ≈ (1/√π) Σ_k w_k · P_conv_i(DM − X_MW_gh[i,k] − halo)

        where ``X_MW_gh[i,k] = DM_MW[i] + √2·σ_MW[i]·t_k`` are the GH nodes.

        Parameters
        ----------
        params : array_like of shape (8,)
            Parameter vector — see class docstring.
        DM_MW_halo : float, optional
            Fixed MW halo DM contribution [pc cm⁻³] (default 50).
        rel_error_DM_MW : float, optional
            Fractional uncertainty on the MW ISM DM (default 0.2).
        DMmax : float, optional
            Detection threshold [pc cm⁻³] (default 1500).
        N_DM_int : int, optional
            Number of output DM grid points for normalisation (default 100).
        n_gh : int, optional
            Number of Gauss-Hermite nodes for the MW integral (default 10).
        dV : float, optional
            Convolution grid spacing [pc cm⁻³] (default 3).
        V_max : float, optional
            Upper limit of the convolution grid [pc cm⁻³] (default 4000).

        Returns
        -------
        float
            Log-posterior value, or ``−1e200`` if out of bounds / non-finite.
        """
        if not self._valid_params(params):
            return -1e200
        cosmo = self._build_cosmo(params)

        mu_ln, sigma_ln           = self._lss_params(cosmo, sorted_z=True)
        mu_ln_host, sigma_ln_host = self._host_ln_params(params[0], params[1])

        P_conv, V_grid = self._host_lss_convolution(
            mu_ln, sigma_ln, mu_ln_host, sigma_ln_host, dV, V_max)
        N_V = len(V_grid)

        # GH nodes for the MW integral
        t_gh, w_gh = np.polynomial.hermite.hermgauss(n_gh)
        sigma_MW   = self._DM_MW_sorted * rel_error_DM_MW
        X_MW_gh    = self._DM_MW_sorted[:, None] + np.sqrt(2.0) * sigma_MW[:, None] * t_gh
        # X_MW_gh shape: (N, n_gh)

        # Evaluate P_conv at Y[d, i, k] = DM_int[d] − X_MW_gh[i,k] − halo
        DM_int_max = V_max + float(X_MW_gh.min()) + DM_MW_halo
        DM_int = np.geomspace(50, DM_int_max, N_DM_int)
        Y      = DM_int[:, None, None] - X_MW_gh[None, :, :] - DM_MW_halo  # (N_DM_int, N, n_gh)

        y_idx    = Y / dV
        j        = np.clip(np.floor(y_idx).astype(int), 0, N_V - 2)
        t_interp = np.clip(y_idx - j, 0.0, 1.0)
        fidx     = np.arange(len(self._z_sorted))
        P_at_Y   = np.where(
            Y >= 0,
            ((1.0 - t_interp) * P_conv[j, fidx[None, :, None]]
             + t_interp        * P_conv[j + 1, fidx[None, :, None]]),
            0.0,
        )  # (N_DM_int, N, n_gh)

        result = (P_at_Y * w_gh[None, None, :]).sum(axis=-1) / np.sqrt(np.pi)
        norm0 = integrate.simpson(result, x=DM_int, axis=0)
        # norm0 < 0.5: V_max too small; see log_likelihood_with_pdetect for details.
        if np.any(norm0 < 0.5):
            return -1e200
        result /= norm0
        result  = np.where(DM_int[:, None] < DMmax, result, 0.0)
        norm1 = integrate.simpson(result, x=DM_int, axis=0)
        if np.any(norm1 <= self._TINY):
            return -1e200
        result /= norm1

        real_result = self._interp_pdf_at_obs(self._DM_obs_sorted, DM_int, result)
        lnl = np.sum(np.log(real_result)) + self._planck_prior(params[3:])
        return lnl if np.isfinite(lnl) else -1e200

    def log_likelihood_figm_marginalized(self, params, DM_MW_halo=50, N_int=2000):
        """Log-posterior treating the IGM baryon fraction f_IGM as a free parameter.

        The f_IGM parameter (``params[8]``) scales the LSS mean DM and variance:

            DM_LSS   →  f_IGM · DM_LSS
            Var_LSS  →  f_IGM² · Var_LSS

        A Gaussian prior ``f_IGM ~ N(1, 0.2²)`` is applied.

        Parameters
        ----------
        params : array_like of shape (9,)
            Parameter vector with ``params[8] = f_IGM`` — see class docstring.
        DM_MW_halo : float, optional
            Fixed MW halo DM contribution [pc cm⁻³] (default 50).
        N_int : int, optional
            Number of host-DM integration points on
            ``geomspace(1, 1000)`` (default 2000).

        Returns
        -------
        float
            Log-posterior value, or ``−1e200`` if out of bounds / non-finite.
        """
        if not self._valid_params(params):
            return -1e200
        figm  = params[8]
        cosmo = self._build_cosmo(params)

        mu_ln, sigma_ln           = self._lss_params(cosmo, figm=figm)
        mu_ln_host, sigma_ln_host = self._host_ln_params(params[0], params[1])

        X_DM_host = np.geomspace(1e-2, 5000, N_int)
        pdf_ln    = self._pdf_log_normal(X_DM_host, mu_ln_host, sigma_ln_host)
        pdf_ln    = pdf_ln / integrate.simpson(pdf_ln, x=X_DM_host)
        integrand = pdf_ln[None, :] * self._pdf_log_normal(
            self.DM_obs[:, None] - self.DM_MW[:, None] - DM_MW_halo
            - X_DM_host[None, :] / (1.0 + self.z[:, None]),
            mu=mu_ln, sigma=sigma_ln, diag=True,
        )
        result     = integrate.simpson(integrand, x=X_DM_host, axis=-1)
        figm_prior = np.log(self._gaussian_pdf(figm, 1.0, 0.2))
        lnl        = np.sum(np.log(result)) + self._planck_prior(params[3:-1]) + figm_prior
        return lnl if np.isfinite(lnl) else -1e200
