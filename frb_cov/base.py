import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RegularGridInterpolator
import astropy.cosmology
from astropy import units as u
from astropy.io import ascii
from scipy.special import sph_harm
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import integrate
from astropy import constants
from scipy.special import jv

class covariance_frb_background():
    def __init__(self,
                 cosmo_dict,
                 bias_emu,
                 power_spec_emu,
                 zet,
                 delta_theta,
                 N_interp_chi):
        self.flat_sky = True
        self.delta_theta = delta_theta
        self.dispersion_constant_times_c = 4.03068e-17 # in cm3 GHz2 Mpc / pc
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
            500)
        self.spline_z_of_chi = UnivariateSpline(self.cosmology.comoving_distance(
            self.los_integration_z).value*self.cosmology.h, self.los_integration_z, k=2, s=0, ext=0)
        self.chi = self.cosmology.comoving_distance(
            zet).value*self.cosmology.h
        self.zet = zet
        self.chi_interp = np.geomspace(10, self.chi[-1], N_interp_chi)
        self.zet_interp = np.linspace(0, np.amax(zet), N_interp_chi)
        
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
        self.spline_Pee = RegularGridInterpolator((self.chi_interp,np.log(power_spec_emu.modes)), bias_emu.predictions_np(params) + power_spec_emu.predictions_np(params) ,bounds_error= False, fill_value = None)
        self.spline_potential = RegularGridInterpolator((self.chi_interp,np.log(power_spec_emu.modes)), power_spec_emu.predictions_np(params)/power_spec_emu.modes**4 ,bounds_error= False, fill_value = None)
        self.DM = np.zeros_like(self.zet)
        for z_idx, z_val in enumerate(self.zet):
            self.DM[z_idx] = (integrate.quad(self.weight, 0,
                                        z_val)[0])
        self.ell = np.geomspace(1,10000,100)
        self.limber_covariance(self.ell)
        if self.flat_sky:
            self.get_covariance_flat_sky()


    def weight(self, z):
        # prefactor:
        # 3*cosmo.h**2 * (H0/h)**2 * cosmo.Omega_b * 2997.9 / (8 * np.pi * G * m_p)

        # eq. 6 in https://arxiv.org/abs/2007.04054, background DM prefactor
        f_IGM = 0.9  # keep constant for now at redshifts < 1. Can be calculated from https://github.com/FRBs/FRB/blob/main/frb/dm/igm.py as f_diffuse
        f_He = 0.24  # Helium fraction
        f_H = 1. - f_He
        f_e = (f_H + 1/2. * f_He)  # electron fraction
        result = f_e * f_IGM * self.prefac*(1+z)/self.cosmology.efunc(z)
        return result

    def weight_wep(self):
        return 1.5*self.cosmology.Om0/(self.dispersion_constant_times_c*self.chi_H**2)

    def limber_covariance(self, ells):
        self.C_ell = np.zeros((len(ells), len(self.zet), len(self.zet)))
        flat_idx = 0
        chi_max = self.cosmology.comoving_distance(np.max(self.zet)).value*self.cosmology.h
        chi = np.geomspace(1, chi_max, 500)
        x_values = np.zeros((2,len(ells)*len(chi)))
        for i_chi in range(len(chi)):
            for i_ell in range(len(ells)):    
                ki = np.log((ells[i_ell] + 0.5)/chi[i_chi])
                x_values[0,flat_idx] = chi[i_chi]
                x_values[1,flat_idx] = ki
                flat_idx +=1
        self.exp = self.spline_Pee((x_values[0,:],x_values[1,:])).reshape((len(chi),len(ells)))
        chi_integrand = self.weight(self.spline_z_of_chi(chi))**2/chi**2*(self.cosmology.efunc(self.spline_z_of_chi(chi))/self.chi_H)**2
        integrand = 10**self.exp.T*chi_integrand[None, :]
        tomo_weighting = np.ones((len(self.zet), len(self.zet), len(chi)))
        for z_idx_i, z_val_i in enumerate(self.zet):
            for z_idx_j, z_val_j in enumerate(self.zet):
                index_min_z = z_idx_i
                if z_val_i > z_val_j:
                   index_min_z = z_idx_j
                indices = np.where(self.spline_z_of_chi(chi)> self.zet[index_min_z])[0]
                tomo_weighting[z_idx_i, z_idx_j, indices] = np.zeros(len(indices))
        self.C_ell = simpson(tomo_weighting[None, :, :, :]*integrand[:, None, None, :], chi, axis = -1)

    def getcovariance(self):
        N_ell0 = 10000
        MAX_ELL = 20000
        ell_fac = 100
        fac = int(MAX_ELL/N_ell0)
        N_ell = N_ell0
        interp_ells = self.gen_log_space(fac*N_ell, N_ell/100)
        result = np.zeros((len(self.zet), len(self.zet)))
        # cov = self.limber_covariance(interp_ells)
        self.get_limber_covariance_parallel(interp_ells)
        cov = self.limber_covariance_p
        for z_idx_i, z_val_i in enumerate(self.zet):
            for z_idx_j, z_val_j in enumerate(self.zet[z_idx_i:]):
                spline = interp1d(np.log(interp_ells[1:]), np.log(
                    cov[1:, z_idx_i, z_idx_j+z_idx_i]))
                if (z_idx_i != z_idx_j + z_idx_i):
                    N_ell = int(
                        ell_fac/self.delta_theta[z_idx_i, z_idx_i+z_idx_j])
                    if (N_ell >= MAX_ELL):
                        N_ell = MAX_ELL
                    ells = np.linspace(1, N_ell, N_ell)[1:N_ell-1]
                    cov_ell = np.exp(spline(np.log(ells)))
                    sph0 = np.sqrt((2.*ells + 1)/4/np.pi)
                    # sph_harm(0, ells, 0, 0, out=None)
                    sph_delta = sph_harm(
                        0, ells, 0, self.delta_theta[z_idx_i, z_idx_i+z_idx_j], out=None).real
                    product = sph0*sph_delta
                    result[z_idx_i, z_idx_j+z_idx_i] = np.sum(product*cov_ell)
                    result[z_idx_j+z_idx_i,
                           z_idx_i] = result[z_idx_i, z_idx_j+z_idx_i]
                    N_ell = N_ell0
                else:
                    ells_diag = np.linspace(
                        1, fac*N_ell, fac*N_ell)[1:fac*N_ell-1]
                    cov_ell = np.exp(spline(np.log(ells_diag)))
                    product = (2.*ells_diag + 1)/4/np.pi
                    result[z_idx_i, z_idx_j+z_idx_i] = np.sum(product*cov_ell)
        return result

    def FileCheck(self, fn):
        try:
            open(fn, "r")
            return 1
        except IOError:
            print("Error: File does not appear to exist.")
            return 0

    def get_covariance_flat_sky(self):
        self.covariance = np.zeros((len(self.zet), len(self.zet)))
        ell_integral = np.geomspace(self.ell[0], self.ell[-1], int(1e4))
        for z_idx_i in range (len(self.zet)):
            for z_idx_j in range (len(self.zet)):
                bessel = jv(0,ell_integral*self.delta_theta[z_idx_i,z_idx_j])
                integrand = np.interp(ell_integral, self.ell, self.C_ell[:,z_idx_i,z_idx_j])*ell_integral*bessel
                self.covariance[z_idx_i,z_idx_j] = simpson(integrand,ell_integral)/2.0/np.pi
                self.covariance[z_idx_j,z_idx_i] = self.covariance[z_idx_i,z_idx_j]
    
    def getcovariance_parallel(self, file_name):
        N_ell0 = 10000
        MAX_ELL = 20000
        ell_fac = 100
        fac = int(MAX_ELL/N_ell0)
        N_ell = N_ell0
        interp_ells = self.gen_log_space(fac*N_ell, N_ell/100)
        result = np.zeros((len(self.zet), len(self.zet)))
        print("Calculating covariance in ell space.")
        filename_npy = file_name+".npy"
        if (self.FileCheck(filename_npy) == 0):
            self.get_limber_covariance_parallel(interp_ells)
            np.save(file_name, self.limber_covariance_p)
            cov = np.load(filename_npy)
        else:
            cov = np.load(filename_npy)
        print("Covariance in ell space done.")
        print("Projecting to real space")
        global compute_covariance

        def compute_covariance(redshift_idx):
            print("redshift index: ", redshift_idx)
            aux = np.zeros(len(self.zet))
            z_idx_i = int(redshift_idx)
            for z_idx_j, z_val_j in enumerate(self.zet[z_idx_i:]):
                spline = interp1d(np.log(interp_ells[1:]), np.log(
                    cov[1:, z_idx_i, z_idx_j+z_idx_i]))
                if (z_idx_i != z_idx_j + z_idx_i):
                    N_ell = int(
                        ell_fac/self.delta_theta[z_idx_i, z_idx_i+z_idx_j])
                    if (N_ell >= MAX_ELL):
                        N_ell = MAX_ELL
                    ells = np.linspace(1, N_ell, N_ell)[1:N_ell-1]
                    cov_ell = np.exp(spline(np.log(ells)))
                    sph0 = np.sqrt((2.*ells + 1)/4/np.pi)
                    sph_delta = sph_harm(
                        0, ells, 0, self.delta_theta[z_idx_i, z_idx_i+z_idx_j], out=None).real
                    product = sph0*sph_delta
                    aux[z_idx_j + z_idx_i] = np.sum(product*cov_ell)
                    N_ell = N_ell0
                else:
                    N_ell = N_ell0
                    ells_diag = np.linspace(
                        1, fac*N_ell, fac*N_ell)[1:fac*N_ell-1]
                    cov_ell = np.exp(spline(np.log(ells_diag)))
                    product = (2.*ells_diag + 1)/4/np.pi
                    aux[z_idx_i+z_idx_j] = np.sum(product*cov_ell)
            return aux
        pool = mp.Pool(self.num_cores)
        result = np.array(
            pool.map(compute_covariance, np.linspace(
                0, len(self.zet)-1, len(self.zet))))
        pool.close()
        pool.terminate()
        for i in range(len(self.zet)):
            for j in range(len(self.zet)):
                result[j, i] = result[i, j]
        return result

    def getcovariance_parallel_for_spline(self):
        N_ell0 = 1000
        MAX_ELL = 3000
        ell_fac = 100
        fac = int(MAX_ELL/N_ell0)
        self.N_ell = N_ell0
        self.interp_ells = self.gen_log_space(fac*self.N_ell, self.N_ell/100)
        zet_spline = np.geomspace(self.spline_z_of_chi(10.0), 3.0, 50)
        filename = "cov_interpolator"
        filename_npy = filename+".npy"
        filename_ell = "cov_interpolator_ell"
        filename_ell_npy = filename_ell+".npy"
        if (self.FileCheck(filename_ell_npy) == 0):
            save_zet = np.copy(self.zet)
            self.zet = zet_spline
            self.get_limber_covariance_parallel(self.interp_ells)
            np.save(filename_ell, self.limber_covariance_p)
            self.zet = save_zet
        else:
            self.limber_covariance_p = np.load(filename_ell_npy)
        print("Calculating covariance in ell space.")
        print("Covariance in ell space done.")
        print("Projecting to real space")
        global compute_covariance

        def compute_covariance(theta):
            result = np.zeros((len(zet_spline), len(zet_spline)))
            if (theta != 0):
                self.N_ell = int(
                    ell_fac/theta)
            else:
                self.N_ell = MAX_ELL
            if (self.N_ell >= MAX_ELL):
                self.N_ell = MAX_ELL
            ells = np.linspace(1, self.N_ell, self.N_ell)[
                0:self.N_ell-1]
            ells_diag = np.linspace(
                1, self.N_ell, self.N_ell)[0:self.N_ell-1]
            sph0 = np.sqrt((2.*ells + 1)/4/np.pi)
            sph_delta = sph_harm(
                0, ells, 0, theta, out=None).real
            product = sph0*sph_delta
            product_diag = (2.*ells_diag + 1)/4/np.pi
            for z_idx_i, z_val_i in enumerate(zet_spline):
                for z_idx_j, z_val_j in enumerate(zet_spline[z_idx_i:]):
                    spline = interp1d(np.log(self.interp_ells[1:]), np.log(
                        self.limber_covariance_p[1:, z_idx_i, z_idx_j+z_idx_i]))
                    if (z_idx_i != z_idx_j + z_idx_i and theta != 0):
                        cov_ell = np.exp(spline(np.log(ells)))
                        result[z_idx_i, z_idx_j +
                               z_idx_i] = np.sum(product*cov_ell)
                        result[z_idx_j+z_idx_i,
                               z_idx_i] = result[z_idx_i, z_idx_j+z_idx_i]
                        self.N_ell = N_ell0
                    else:
                        cov_ell = np.exp(spline(np.log(ells_diag)))
                        result[z_idx_i, z_idx_j +
                               z_idx_i] = np.sum(product_diag*cov_ell)
                        result[z_idx_j+z_idx_i,
                               z_idx_i] = result[z_idx_i, z_idx_j+z_idx_i]
            return result
        self.theta = np.geomspace(1e-4, np.pi, 200)
        self.theta = np.insert(self.theta, 0, 0.0)
        result = np.zeros((len(self.theta), len(zet_spline), len(zet_spline)))
        filename = "cov_interpolator"
        filename_npy = filename+".npy"
        if (self.FileCheck(filename_npy) == 0):
            pool = mp.Pool(self.num_cores)
            result = np.array(
                pool.map(compute_covariance, self.theta))
            pool.close()
            pool.terminate()
            np.save(filename, result)
            result = np.load(filename_npy)
        else:
            result = np.load(filename_npy)
        self.covariance_interpolator = []
        for i in range(len(self.theta)):
            self.covariance_interpolator.append(
                interpolate.interp2d(zet_spline, zet_spline, result[i, :, :]))

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        if (value - array[idx] > 0):
            return int(idx)
        else:
            return int(idx - 1)

    def get_covariance_from_spline(self):
        result = np.zeros((len(self.zet), len(self.zet)))
        for z_idx_i, z_val_i in enumerate(self.zet):
            for z_idx_j, z_val_j in enumerate(self.zet[z_idx_i:]):
                theta_i = self.find_nearest(
                    self.theta, self.delta_theta[z_idx_i, z_idx_i+z_idx_j])
                C0 = (
                    self.covariance_interpolator[theta_i](z_val_i, z_val_j))
                slope = ((
                    self.covariance_interpolator[theta_i+1](z_val_i, z_val_j)) - C0)/((self.theta[theta_i+1]) - (self.theta[theta_i]))
                result[z_idx_i, z_idx_j+z_idx_i] = (slope*((
                    self.delta_theta[z_idx_i, z_idx_i+z_idx_j]) - (self.theta[theta_i])) + C0)
                result[z_idx_j+z_idx_i,
                       z_idx_i] = result[z_idx_i, z_idx_j+z_idx_i]
        return result

    def get_covariance_from_spline_EP(self):
        result_auto = np.zeros((len(self.zet), len(self.zet)))
        result_cross = np.zeros((len(self.zet), len(self.zet)))

        for z_idx_i, z_val_i in enumerate(self.zet):
            for z_idx_j, z_val_j in enumerate(self.zet[z_idx_i:]):
                theta_i = self.find_nearest(
                    self.theta, self.delta_theta[z_idx_i, z_idx_i+z_idx_j])
                C0_auto = (
                    self.covariance_interpolator_EP_auto[theta_i](z_val_i, z_val_j))
                C0_cross = (
                    self.covariance_interpolator_EP_cross[theta_i](z_val_i, z_val_j))
                slope_auto = ((
                    self.covariance_interpolator_EP_auto[theta_i+1](z_val_i, z_val_j)) - C0_auto)/((self.theta[theta_i+1]) - (self.theta[theta_i]))
                slope_cross = ((
                    self.covariance_interpolator_EP_cross[theta_i+1](z_val_i, z_val_j)) - C0_cross)/((self.theta[theta_i+1]) - (self.theta[theta_i]))
                result_auto[z_idx_i, z_idx_j+z_idx_i] = (slope_auto*((
                    self.delta_theta[z_idx_i, z_idx_i+z_idx_j]) - (self.theta[theta_i])) + C0_auto)
                result_cross[z_idx_i, z_idx_j+z_idx_i] = (slope_cross*((
                    self.delta_theta[z_idx_i, z_idx_i+z_idx_j]) - (self.theta[theta_i])) + C0_cross)
                result_auto[z_idx_j+z_idx_i,
                       z_idx_i] = result_auto[z_idx_i, z_idx_j+z_idx_i]
                result_cross[z_idx_j+z_idx_i,
                       z_idx_i] = result_cross[z_idx_i, z_idx_j+z_idx_i]
        return result_auto, result_cross

    def get_variance(self, zet):
        return self.covariance_interpolator[0](zet, zet)

    def gen_log_space(self, limit, n):
        result = [1]
        if n > 1:
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
        while len(result) < n:
            next_value = result[-1]*ratio
            if next_value - result[-1] >= 1:
                result.append(next_value)
            else:
                result.append(result[-1]+1)
                ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
        return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)

    def get_limber_covariance_parallel_EP(self, ells):
        self.limber_covariance_p = np.zeros(
            (len(ells), len(self.zet), len(self.zet)))

        global limber_covariance_parallel_EP

        def limber_covariance_parallel_EP(ell):
            aux_auto = np.zeros(
                (len(self.zet), len(self.zet)))
            aux_cross = np.zeros(
                (len(self.zet), len(self.zet)))
            for z_idx_i, z_val_i in enumerate(self.zet):
                for z_idx_j, z_val_j in enumerate(self.zet[z_idx_i:]):
                    chi_max = self.cosmology.comoving_distance(
                        np.min((z_val_i, z_val_j))).value*self.cosmology.h
                    chi = np.linspace(1, chi_max, 500)
                    z = self.spline_z_of_chi(chi)
                    exp_auto = np.diagonal(self.power_potential_interpolator(
                        np.log10((ell + 0.5) /
                                 chi), z)[:, ::-1])
                    exp_cross = np.diagonal(self.power_interpolator(
                        np.log10((ell + 0.5) /
                                 chi), z)[:, ::-1])
                    integrand_auto = 10**exp_auto*self.weight_wep()**2 / chi**2
                    integrand_cross = 2.0*10**(0.5*(exp_cross + exp_auto))*self.weight(z)*self.weight_wep()/chi**2*(self.cosmology.efunc(
                        z)/self.chi_H)
                    aux_auto[z_idx_i, z_idx_j+z_idx_i] = np.trapz(
                        integrand_auto, chi)
                    aux_auto[z_idx_j+z_idx_i,
                             z_idx_i] = aux_auto[z_idx_i, z_idx_j+z_idx_i]
                    aux_cross[z_idx_i, z_idx_j+z_idx_i] = np.trapz(
                        integrand_cross, chi)
                    aux_cross[z_idx_j+z_idx_i,
                              z_idx_i] = aux_cross[z_idx_i, z_idx_j+z_idx_i]

            return aux_auto, aux_cross
        ells_aux = np.zeros((len(ells), 2))
        for ell_idx, ell_val in enumerate(ells):
            ells_aux[ell_idx, 0] = int(ell_idx)
            ells_aux[ell_idx, 1] = int(ell_val)
        pool = mp.Pool(self.num_cores)
        self.limber_covariance_p_EP = np.array(
            pool.map(limber_covariance_parallel_EP, ells))
        pool.close()
        pool.terminate()

    def getcovariance_parallel_for_spline_with_EP(self):
        N_ell0 = 10000
        MAX_ELL = 40000
        ell_fac = 100
        fac = int(MAX_ELL/N_ell0)
        self.N_ell = N_ell0
        self.interp_ells = self.gen_log_space(fac*self.N_ell, self.N_ell/100)
        zet_spline = np.geomspace(self.spline_z_of_chi(10.0), 3.0, 50)
        filename = "cov_interpolator_EP"
        filename_npy = filename+".npy"
        filename_ell = "cov_interpolator_ell_EP"
        filename_ell_npy = filename_ell+".npy"
        if (self.FileCheck(filename_ell_npy) == 0):
            save_zet = np.copy(self.zet)
            self.zet = zet_spline
            self.get_limber_covariance_parallel_EP(self.interp_ells)
            np.save(filename_ell, self.limber_covariance_p_EP)
            self.zet = save_zet
        else:
            self.limber_covariance_p_EP = np.load(filename_ell_npy)
        print("Calculating covariance in ell space.")
        print("Covariance in ell space done.")
        print("Projecting to real space")
        global compute_covariance

        def compute_covariance(theta):
            result_auto = np.zeros((len(zet_spline), len(zet_spline)))
            result_cross = np.zeros((len(zet_spline), len(zet_spline)))
            if (theta != 0):
                self.N_ell = int(
                    ell_fac/theta)
            else:
                self.N_ell = MAX_ELL
            if (self.N_ell >= MAX_ELL):
                self.N_ell = MAX_ELL
            ells = np.linspace(1, self.N_ell, self.N_ell)[
                0:self.N_ell-1]
            ells_diag = np.linspace(
                1, self.N_ell, self.N_ell)[0:self.N_ell-1]
            sph0 = np.sqrt((2.*ells + 1)/4/np.pi)
            sph_delta = sph_harm(
                0, ells, 0, theta, out=None).real
            product = sph0*sph_delta
            product_diag = (2.*ells_diag + 1)/4/np.pi
            for z_idx_i, z_val_i in enumerate(zet_spline):
                for z_idx_j, z_val_j in enumerate(zet_spline[z_idx_i:]):
                    spline_auto = interp1d(np.log(self.interp_ells[1:]), np.log(
                        self.limber_covariance_p_EP[1:, 0, z_idx_i, z_idx_j+z_idx_i]))
                    spline_cross = interp1d(np.log(self.interp_ells[1:]), np.log(
                        self.limber_covariance_p_EP[1:, 1, z_idx_i, z_idx_j+z_idx_i]))
                    if (z_idx_i != z_idx_j + z_idx_i and theta != 0):
                        cov_ell_auto = np.exp(spline_auto(np.log(ells)))
                        result_auto[z_idx_i, z_idx_j +
                                    z_idx_i] = np.sum(product*cov_ell_auto)
                        result_auto[z_idx_j+z_idx_i,
                                    z_idx_i] = result_auto[z_idx_i, z_idx_j+z_idx_i]
                        cov_ell_cross = np.exp(spline_cross(np.log(ells)))
                        result_cross[z_idx_i, z_idx_j +
                                     z_idx_i] = np.sum(product*cov_ell_cross)
                        result_cross[z_idx_j+z_idx_i,
                                     z_idx_i] = result_cross[z_idx_i, z_idx_j+z_idx_i]
                        self.N_ell = N_ell0
                    else:
                        cov_ell_auto = np.exp(spline_auto(np.log(ells_diag)))
                        result_auto[z_idx_i, z_idx_j +
                                    z_idx_i] = np.sum(product_diag*cov_ell_auto)
                        result_auto[z_idx_j+z_idx_i,
                                    z_idx_i] = result_auto[z_idx_i, z_idx_j+z_idx_i]
                        cov_ell_cross = np.exp(spline_cross(np.log(ells_diag)))
                        result_cross[z_idx_i, z_idx_j +
                                     z_idx_i] = np.sum(product_diag*cov_ell_cross)
                        result_cross[z_idx_j+z_idx_i,
                                     z_idx_i] = result_cross[z_idx_i, z_idx_j+z_idx_i]
            return result_auto, result_cross
        self.theta = np.geomspace(1e-4, np.pi, 200)
        self.theta = np.insert(self.theta, 0, 0.0)
        result = np.zeros(
            (len(self.theta), 2,  len(zet_spline), len(zet_spline)))
        if (self.FileCheck(filename_npy) == 0):
            pool = mp.Pool(self.num_cores)
            result = np.array(
                pool.map(compute_covariance, self.theta))
            pool.close()
            pool.terminate()
            np.save(filename, result)
            result = np.load(filename_npy)
        else:
            result = np.load(filename_npy)
        self.covariance_interpolator_EP_auto = []
        self.covariance_interpolator_EP_cross = []
        for i in range(len(self.theta)):
            self.covariance_interpolator_EP_auto.append(
                interpolate.interp2d(zet_spline, zet_spline, result[i, 0, :, :]))
            self.covariance_interpolator_EP_cross.append(
                interpolate.interp2d(zet_spline, zet_spline, result[i, 1, :, :]))
