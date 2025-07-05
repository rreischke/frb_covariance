import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RegularGridInterpolator
import astropy.cosmology
from astropy import units as u
from scipy import integrate
from astropy import constants
from scipy.special import jv
from scipy.special import eval_legendre
from tqdm import tqdm
import time as time
import os

class covariance_frb_background():
    def __init__(self,
                 cosmo_dict,
                 bias_emu,
                 power_spec_emu,
                 zet,
                 delta_theta,
                 flat_sky = False,
                 frequency_width = None,
                 frequency_band = None):
        #print("Calculating FRB-Covariance")
        self.frequency_width = frequency_width
        self.frequency_bands = frequency_band
        if cosmo_dict['delta_gamma'] != 0 and (frequency_width == None or frequency_band == None):
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

        self.frequncies_high = np.ones(len(zet))
        self.frequncies_low = 1.1*np.ones(len(zet))
        if cosmo_dict['delta_gamma'] != 0:
            self.frequency_width = np.array(self.frequency_width)
            self.frequency_bands = np.array(self.frequency_bands)
            self.frequncies_high = (self.frequency_bands + self.frequency_width/2)
            self.frequncies_low = (self.frequency_bands - self.frequency_width/2)
        N_interp_chi = 500
        self.flat_sky = flat_sky
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
        self.power_at_z0 = power_spec_emu.predictions_np(params)
        self.spline_Pee = RegularGridInterpolator((self.chi_interp,np.log(power_spec_emu.modes)), bias_emu.predictions_np(params) + power_spec_emu.predictions_np(params) ,bounds_error= False, fill_value = None)
        self.spline_Pmm = RegularGridInterpolator((self.chi_interp,np.log(power_spec_emu.modes)), power_spec_emu.predictions_np(params) ,bounds_error= False, fill_value = None)
        self.spline_potential = RegularGridInterpolator((self.chi_interp,np.log(power_spec_emu.modes)), power_spec_emu.predictions_np(params) - 4*np.log(power_spec_emu.modes) ,bounds_error= False, fill_value = None)
        self.spline_potential_ee = RegularGridInterpolator((self.chi_interp,np.log(power_spec_emu.modes)), .5*bias_emu.predictions_np(params) + power_spec_emu.predictions_np(params)  - 2*np.log(power_spec_emu.modes) ,bounds_error= False, fill_value = None)
        self.DM = np.zeros_like(self.zet)
        for z_idx, z_val in enumerate(self.zet):
            self.DM[z_idx] = (integrate.quad(self.weight, 0,
                                        z_val)[0])
        self.ell = np.geomspace(1,50000,100)
        self.limber_cell(self.ell)
        if self.flat_sky or len(self.zet) >= 100:
            self.get_covariance_flat_sky()
        else:
            self.get_covariance()



    def weight(self, z):
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
        return 1.5*self.cosmology.Om0/(self.dispersion_constant_times_c*self.chi_H**2)*self.cosmo_dict_fiducial['delta_gamma']

    def limber_cell(self, ells):
        flat_idx = 0
        chi_max = self.cosmology.comoving_distance(np.max(self.zet)).value*self.cosmology.h
        chi = np.geomspace(1, chi_max, 100)
        x_values = np.zeros((2,len(ells)*len(chi)))
        for i_chi in range(len(chi)):
            for i_ell in range(len(ells)):    
                ki = np.log((ells[i_ell] + 0.5)/chi[i_chi])
                x_values[0,flat_idx] = chi[i_chi]
                x_values[1,flat_idx] = ki
                flat_idx +=1
        self.exp = self.spline_Pee((x_values[0,:],x_values[1,:])).reshape((len(chi),len(ells)))
        self.exp_ep = self.spline_potential((x_values[0,:],x_values[1,:])).reshape((len(chi),len(ells)))
        self.exp_ep_ee = self.spline_potential_ee((x_values[0,:],x_values[1,:])).reshape((len(chi),len(ells)))
        chi_integrand = self.weight(self.spline_z_of_chi(chi))**2/chi**2*(self.cosmology.efunc(self.spline_z_of_chi(chi))/self.chi_H)**2
        chi_integrand_ep = self.weight_wep()**2/chi**2/(1+self.spline_z_of_chi(chi))
        freq_weight = 1.0/(1/self.frequncies_low**2 - 1/self.frequncies_high**2)
        freq_weight_squared = freq_weight[:,None] * freq_weight[None, :]
        if self.diagonal:
            integrand = 10**self.exp.T[:, None, :]*chi_integrand[None, None, :] + (10**self.exp_ep.T[:, None, :]*np.diag(freq_weight_squared)[None, :, None])*chi_integrand_ep[None, None,:] + (10**self.exp_ep_ee.T[:, None, :]*(np.diag(freq_weight_squared)**.5)[None, :, None])*(2*(chi_integrand_ep*chi_integrand)**.5)[None,None,:]
            tomo_weighting = np.ones((len(self.zet), len(chi)))
            for z_idx_i, z_val_i in enumerate(self.zet):
                index_min_z = z_idx_i
                indices = np.where(self.spline_z_of_chi(chi)> self.zet[index_min_z])[0]
                tomo_weighting[z_idx_i, indices] = np.zeros(len(indices))
            self.C_ell = simpson(tomo_weighting[None, :, :]*integrand,x = chi, axis = -1)

        else:
            if len(self.zet) < 100:
                integrand = 10**self.exp.T[:, None, None, :]*chi_integrand[None, None, None, :] + (10**self.exp_ep.T[:, None, None, :]*freq_weight_squared[None, :, :, None])*chi_integrand_ep[None,None, None,:] + (10**self.exp_ep_ee.T[:, None, None, :]*(freq_weight_squared**.5)[None, :, :, None])*(2*(chi_integrand_ep*chi_integrand)**.5)[None,None,None,:]
                tomo_weighting = np.ones((len(self.zet), len(self.zet), len(chi)))
                for z_idx_i, z_val_i in enumerate(self.zet):
                    for z_idx_j, z_val_j in enumerate(self.zet):
                        index_min_z = z_idx_i
                        if z_val_i > z_val_j:
                            index_min_z = z_idx_j
                        indices = np.where(self.spline_z_of_chi(chi)> self.zet[index_min_z])[0]
                        tomo_weighting[z_idx_i, z_idx_j, indices] = np.zeros(len(indices))
                self.C_ell = simpson(tomo_weighting[None, :, :, :]*integrand,x = chi, axis = -1)
            else:
                integrand = 10**self.exp.T[:, None, None, :]*chi_integrand[None, None, None, :] + (10**self.exp_ep.T[:, None, None, :]*freq_weight_squared[None, :, :, None])*chi_integrand_ep[None,None, None,:] + (10**self.exp_ep_ee.T[:, None, None, :]*(freq_weight_squared**.5)[None, :, :, None])*(2*(chi_integrand_ep*chi_integrand)**.5)[None,None,None,:]
                zet_interpolation = np.linspace(self.zet[0], self.zet[-1], 100)
                tomo_weighting = np.ones((len(zet_interpolation), len(zet_interpolation), len(chi)))
                for z_idx_i, z_val_i in enumerate(zet_interpolation):
                    for z_idx_j, z_val_j in enumerate(zet_interpolation):
                        index_min_z = z_idx_i
                        if z_val_i > z_val_j:
                            index_min_z = z_idx_j
                        indices = np.where(self.spline_z_of_chi(chi)> zet_interpolation[index_min_z])[0]
                        tomo_weighting[z_idx_i, z_idx_j, indices] = np.zeros(len(indices))
                self.C_ell = simpson(tomo_weighting[None, :, :, :]*integrand, x = chi, axis = -1)
                self.spline_C_ell = RegularGridInterpolator((np.log(self.ell),zet_interpolation,zet_interpolation), np.log(self.C_ell) ,bounds_error= False, fill_value = None)

    def get_covariance(self):
        MAX_ELL = 20000
        ell_fac = 1000
        self.covariance = np.zeros((len(self.zet), len(self.zet)))
        for z_idx_i in tqdm(np.arange(len(self.zet))):
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
        self.covariance = np.zeros((len(self.zet), len(self.zet)))
        ell_integral = np.geomspace(self.ell[0], self.ell[-1], int(1e5))
        x_values = np.zeros((3,len(ell_integral)))
        x_values[0,:] = np.log(ell_integral)
        
        if self.diagonal:
            for z_idx_i in range(len(self.zet)):
                integrand = np.interp(ell_integral, self.ell, self.C_ell[:,z_idx_i])*ell_integral
                self.covariance[z_idx_i,z_idx_i] = simpson(integrand,x = ell_integral)/2.0/np.pi
        else:
            for z_idx_i in tqdm(np.arange(len(self.zet))):
                for z_idx_j in range (z_idx_i, len(self.zet)):    
                    if self.delta_theta[z_idx_i,z_idx_j] > 20./180*np.pi:
                        continue
                    else:
                        bessel = jv(0,ell_integral*self.delta_theta[z_idx_i,z_idx_j])
                        if len(self.zet) < 100:
                            integrand = np.interp(ell_integral, self.ell, self.C_ell[:,z_idx_i,z_idx_j])*ell_integral*bessel
                            self.covariance[z_idx_i,z_idx_j] = simpson(integrand,x = ell_integral)/2.0/np.pi
                            self.covariance[z_idx_j,z_idx_i] = self.covariance[z_idx_i,z_idx_j]
                        else:
                            x_values[1,:] = self.zet[z_idx_i]*np.ones_like(ell_integral)
                            x_values[2,:] = self.zet[z_idx_j]*np.ones_like(ell_integral)
                            integrand = np.exp(self.spline_C_ell((x_values[0,:],x_values[1,:], x_values[2,:]))).reshape(len(ell_integral))*ell_integral*bessel
                            self.covariance[z_idx_i,z_idx_j] = simpson(integrand,x =ell_integral)/2.0/np.pi
                            self.covariance[z_idx_j,z_idx_i] = self.covariance[z_idx_i,z_idx_j]

                        
