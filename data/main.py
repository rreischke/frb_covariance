from re import S
from base import covariance_frb_background
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker, cm
from scipy.special import sph_harm
from scipy import integrate
import csv
from astropy.io import ascii
import astropy.coordinates as coord
import astropy.units as units
from astropy.coordinates import Longitude, Latitude
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.linalg import cholesky
import time


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

'''
Defining cosmology dictionary here
'''
sigma8 = 0.834
h = 0.674
omegam = 0.3
omegab = 0.05
omegade = 1.0 - omegam
w0 = -1
wa = 0
ns = 0.963
neff = 3.046
mnu = 0
Tcmb0 = 2.725
keys = ['sigma8', 'h', 'omega_m', 'omega_b', 'omega_de', 'w0', 'wa',
        'ns', 'neff', 'm_nu', 'Tcmb0']
values = [sigma8, h, omegam, omegab, omegade,
          w0, wa,  ns, neff, mnu, Tcmb0]
cosmo = dict(zip(keys, values))


m_bins = 500
log10m_min = 9
log10m_max = 18
hmf_model = 'Tinker10'
mdef_model = 'SOMean'
mdef_params = ['overdensity', 200]
mdef_params = \
    dict(zip(mdef_params[::2], mdef_params[1::2]))
for key in mdef_params.keys():
    mdef_params[key] = float(mdef_params[key])
disable_mass_conversion = True
delta_c = 1.686
transfer_model = 'EH'
small_k_damping = 'damped'
log10k_bins = 200
log10k_min = -5
log10k_max = 7
keys = ['M_bins', 'log10M_min', 'log10M_max', 'hmf_model',
        'mdef_model', 'mdef_params', 'disable_mass_conversion',
        'delta_c', 'transfer_model', 'small_k_damping']
values = [1, log10m_min, log10m_max,
          hmf_model, mdef_model, mdef_params,
          disable_mass_conversion, delta_c,
          transfer_model, small_k_damping]
hm_prec = dict(zip(keys, values))

log10k_bins = 500
log10k_min = -5
log10k_max = 4
keys = ['log10k_bins', 'log10k_min', 'log10k_max']
values = [log10k_bins, log10k_min, log10k_max]
powspec_prec = dict(zip(keys, values))


f_sky = np.sqrt(1e-3)
sample_size = int(1000)


def n_z(z, alpha=5):
    norm = 2./alpha**3
    return z**2 * np.exp(-z*alpha) / norm


frbcat = False
mycat = True

if (mycat):
    frbcat = False
    frb_cat = ascii.read("catalogue_updated.csv", data_start=1)
    ra = coord.Angle(frb_cat["ra (deg)"], unit=units.degree)
    dec = coord.Angle(frb_cat["dec (deg)"], unit=units.degree)
    ra = ra.radian - np.pi
    dec = dec.radian
  
    counts = len(frb_cat["redshift"])
    n_with_hist = counts
    zet = frb_cat["redshift"][:n_with_hist]
    z = np.zeros(n_with_hist)
    DM_obs = np.zeros(n_with_hist)
    for i in range(len(zet)):
        z[i] = zet[i]
        sep = '&'
        DM_obs[i] = frb_cat["dm"][i]
    ra = ra[:n_with_hist]
    dec = dec[:n_with_hist]
    

if (frbcat):
    frb_cat = ascii.read("frbcat_20210302.csv", data_start=1)
    ra = coord.Angle(frb_cat["rop_raj"], unit=units.hourangle)
    dec = coord.Angle(frb_cat["rop_decj"], unit=units.degree)
    ra = ra.radian - np.pi
    dec = dec.radian

    counts = np.where(frb_cat["rmp_redshift_host"] != "null", 1, 0)
    n_with_hist = np.sum(counts)
    zet = frb_cat["rmp_redshift_host"][:n_with_hist]
    z = np.zeros(n_with_hist)
    DM_obs = np.zeros(n_with_hist)
    for i in range(len(zet)):
        z[i] = zet[i]
        sep = '&'
        stripped = frb_cat["rmp_dm"][i].split(sep, 1)[0]
        DM_obs[i] = float(stripped) - frb_cat["rop_mw_dm_limit"][i]
    ra = ra[:n_with_hist]
    dec = dec[:n_with_hist]
else:
    # Draw random points from distribution in z and draw random angles
    if mycat == False:
        n_with_hist = sample_size
        np.random.seed(4)
        nz_CDF = []
        z_dist = np.linspace(0, 3, 200)
        alpha = 5
        # calculate CDF
        for aux_z in z_dist:
            nz_CDF.append(integrate.quad(n_z, 0., aux_z, args=(alpha))[0])

        nz_CDF = np.array(nz_CDF) / nz_CDF[-1]

        # build inversion spline
        nz_CDF_spline = ius(nz_CDF, z_dist)

        # draw points
        points = np.random.uniform(size=sample_size)
        np.random.seed(42)
        ra = np.random.uniform(size=sample_size)*np.pi*f_sky
        dec = np.random.uniform(size=sample_size)*np.pi*f_sky*2.0*np.pi
        #plt.plot(ra, dec, ls="", marker="o")
        # plt.show()
        z = nz_CDF_spline(points)
        #plt.hist(z, bins=30, density=True)
        #plt.plot(z_dist, n_z(z_dist, alpha))
        np.save("n_z_samples", z)
        # plt.show()

'''
    DM_host_fid = 100.
    DM_scatter_fid = 50.

    sigma_LSS = (40. + z_points*140.)# * h_mcmc/cosmology.h # sigma_LSS should scale with h
    sigma_host = 50./(1.+z_points)
    sigma_MW = 30. # uncertainty on NE2001 model for milky way from https://arxiv.org/abs/astro-ph/0412641
    sigma = np.sqrt(sigma_LSS**2 + sigma_host**2 + sigma_MW**2)
    
    mean = DM_of_z(z_points) + DM_host_fid / (1.+z_points)
    # build mock data array
    mock_DM = np.random.normal(loc=mean, scale = sigma)
    mock_DM[mock_DM < 10.] = 10.

    mock_data = np.array((z_points, mock_DM))
    mock_data = mock_data.T
    return(mock_data)
'''


N_FRB = n_with_hist


delta_theta = np.zeros((N_FRB, N_FRB))
for i in range(N_FRB):
    for j in range(N_FRB):
        if (i != j):
            delta_theta[i, j] = np.arccos(np.sin(
                ra[i])*np.sin(ra[j]) + np.cos(ra[i])*np.cos(ra[j])*np.cos(dec[i] - dec[j]))

#plt.hist(delta_theta.flatten(), bins=30, density=True)
# plt.show()

np.savetxt("delta_n", delta_theta)

frb_ep = False
if (frb_ep):
    data = np.loadtxt("frb_ep_sample.txt")
    z = np.array(data[:, 4])
    delta_theta = np.zeros((len(z), len(z)))
    N_FRB = len(z)
    print(N_FRB)
    for i in range(N_FRB):
        for j in range(N_FRB):
            if (i != j):
                delta_theta[i, j] = np.arccos(np.sin(
                    ra[i])*np.sin(ra[j]) + np.cos(ra[i])*np.cos(ra[j])*np.cos(dec[i] - dec[j]))

cov = covariance_frb_background(
    cosmo, hm_prec, powspec_prec, z, delta_theta, 100)

#np.save("cov_DM_EP", cov.get_covariance_from_spline())
#np.save("cov_EP", np.array(cov.get_covariance_from_spline_EP()))
#np.save("fiducial_DM_EP", np.array(cov.DM_of_z(z)))

print("done")


# cov.getcovariance_parallel_for_spline()

if (frbcat):
    if (cov.FileCheck("covariance_frbcat.npy") == 0):
        filename_ell = "covariance_in_ell_frb_cat"
        np.save("covariance_frbcat", cov.getcovariance_parallel(filename_ell))
        covariance = np.load("covariance_frbcat.npy")

covariance = cov.get_covariance_from_spline()
file_name = "covariance_koustav"
np.savetxt(file_name, covariance)
#print(cov.get_variance(0.1), 1.387 * cov.DM_of_z(0.1)**2)


fitting = False

if fitting:

    def gauss(x, mean, sigma):
        return (1./np.sqrt(2.*np.pi*sigma**2) * np.exp(-(x-mean)**2/(2.*sigma**2)))


    def full_covariance_matrix(z_obs):
        sigma_host = 50./(1.+z_obs)
        sigma_mw = 30.
        sigma_rest = np.diag(sigma_host**2 + sigma_mw**2.0)
        cov_mat = covariance + sigma_rest
        return cov_mat


    def full_covariance_matrix_cut(zet):
        N_data = len(zet)
        sigma_host = 50./(1.+zet)
        sigma_mw = 30.
        sigma_rest = np.diag(sigma_host**2 + sigma_mw**2.0)
        cov_mat = covariance[:N_data, :N_data] + sigma_rest
        return cov_mat


    def draw_data(zet):
        N_data = len(zet)
        C = full_covariance_matrix_cut(zet)
        U = cholesky(C)  # Cholesky decomposition
        R = np.random.randn((N_data))  # Three uncorrelated sequences
        DM_obs_aux = R @ U + cov.DM_of_z(zet) + 100./(1+zet)
        for j in range(N_data):
            if (DM_obs_aux[j] <= 0.0):
                DM_obs_aux[j] = 10.0
        return DM_obs_aux


    # generate the data
    if (not frbcat):
        DM_obs = draw_data(z)

    # print(np.sqrt(np.diagonal(covariance))/DM_obs)
    # log_lkl for a single FRB with given DM_obs, z_obs


    def log_lkl(DM_obs, z_obs, h_mcmc, covariance):
        # gaussian scatter
        sigma_host = 50./(1.+z_obs)
        sigma_mw = 30.
        sigma_rest = np.diag(sigma_host**2 + sigma_mw**2.0)

        DM_mean = cov.DM_of_z(z_obs) * (h_mcmc/cov.cosmology.h) + 100./(1+z_obs)
        cov_mat = covariance * (h_mcmc/cov.cosmology.h)**2.0 + sigma_rest
        precision_mat = np.linalg.inv(cov_mat)
        log_det_cov = np.linalg.slogdet(cov_mat)[1]
        quadraticform = np.dot(
            DM_obs - DM_mean, np.dot(precision_mat, DM_obs - DM_mean))
        result = -.5*(log_det_cov) - .5*quadraticform
        return result


    def log_lkl_no_dep(DM_obs, z_obs, h_mcmc, covariance):
        # gaussian scatter
        sigma_host = 50./(1.+z_obs)
        sigma_mw = 30.
        sigma_rest = np.diag(sigma_host**2 + sigma_mw**2.0)

        DM_mean = cov.DM_of_z(z_obs) * (h_mcmc/cov.cosmology.h) + 100./(1+z_obs)
        cov_mat = covariance + sigma_rest
        precision_mat = np.linalg.inv(cov_mat)
        log_det_cov = np.linalg.slogdet(cov_mat)[1]
        quadraticform = np.dot(
            DM_obs - DM_mean, np.dot(precision_mat, DM_obs - DM_mean))
        result = -.5*(log_det_cov) - .5*quadraticform
        return result


    def old_error(z_obs):
        sigma_host = 50./(1.+z_obs)
        sigma_mw = 30.
        sigma_rest = (sigma_host**2 + sigma_mw**2.0)
        sigma_LSS = (40. + z_obs*140.)
        return np.sqrt(sigma_LSS*2 + sigma_rest)


    def log_lkl_steffen(DM_obs, z_obs, h_mcmc):
        # gaussian scatter
        sigma_host = 50./(1.+z_obs)
        sigma_mw = 30.
        sigma_LSS_vec = ((40. + z_obs*140.) * (h_mcmc/cov.cosmology.h))
        sigma_rest = np.diag(sigma_host**2 + sigma_mw**2.0)
        sigma_LSS = np.diag((40. + z_obs*140.) * (h_mcmc/cov.cosmology.h))
        cov_mat = sigma_rest + sigma_LSS**2
        DM_mean = cov.DM_of_z(z_obs) * (h_mcmc/cov.cosmology.h) + 100./(1+z_obs)
        precision_mat = np.linalg.inv(cov_mat)
        #log_det_cov = np.linalg.slogdet(cov_mat)[1]
        # quadraticform = np.dot(
        #    DM_obs - DM_mean, np.dot(precision_mat, DM_obs - DM_mean))
        quadraticform = 0.0
        log_det_cov = 0.0
        for i_z, v_z in enumerate(z_obs):
            quadraticform += (DM_mean[i_z] - DM_obs[i_z])**2 / \
                (sigma_LSS_vec[i_z]**2 + sigma_host[i_z]**2 + sigma_mw**2)
            log_det_cov += np.log(sigma_LSS_vec[i_z]
                                ** 2 + sigma_host[i_z]**2 + sigma_mw**2)

        result = - .5*(log_det_cov) - .5*quadraticform
        return result


    def get_mean_and_variance(x, px):
        norm = np.trapz(px, x)
        px /= norm
        mean = np.trapz(px*x, x)
        variance = np.trapz(px*x*x, x) - mean**2
        return mean, mean - 2.0*np.sqrt(variance), mean + 2.0*np.sqrt(variance)


    def iterate_over_mock_data():
        N = sample_size
        DN = 50
        index_i = 0
        h = np.linspace(.2, 1.2, 500)
        subtrac_N = N
        best_fit_h_real = []
        best_fit_h_diag = []
        best_fit_h_no_dep = []
        best_fit_h_real_p_var = []
        best_fit_h_diag_p_var = []
        best_fit_h_no_dep_p_var = []
        best_fit_h_real_m_var = []
        best_fit_h_diag_m_var = []
        best_fit_h_no_dep_m_var = []
        samples = []
        while subtrac_N > 60:
            subtrac_N = N - index_i*DN
            print(subtrac_N)
            samples.append(subtrac_N)
            aux_z = np.copy(z[:subtrac_N])
            aux_obs_dm = draw_data(aux_z)
            plt.plot(aux_z, aux_obs_dm, ls="", marker="o")

            loglike_new = np.zeros_like(h)
            loglike_new_diag = np.zeros_like(h)
            loglike_new_no_dep = np.zeros_like(h)
            aux_cov = covariance[:subtrac_N, :subtrac_N]
            aux_cov_diag = np.diag(np.diagonal(aux_cov))
            for i in range(len(h)):
                loglike_new[i] = log_lkl(aux_obs_dm, aux_z, h[i], aux_cov)
                loglike_new_diag[i] = log_lkl(
                    aux_obs_dm, aux_z, h[i], aux_cov_diag)
                # loglike_new_no_dep[i] = log_lkl_no_dep(
                #    aux_obs_dm, aux_z, h[i], aux_cov)
            index_i += 1
            real_dist = get_mean_and_variance(
                h, np.exp(loglike_new - np.max(loglike_new)))
            diag_dist = get_mean_and_variance(
                h, np.exp(loglike_new_diag - np.max(loglike_new_diag)))
            # no_dep_dist = get_mean_and_variance(
            #    h, np.exp(loglike_new_no_dep - np.max(loglike_new_no_dep)))
            best_fit_h_real.append(real_dist[0])
            best_fit_h_real_p_var.append(real_dist[1])
            best_fit_h_real_m_var.append(real_dist[2])
            best_fit_h_diag.append(diag_dist[0])
            best_fit_h_diag_p_var.append(diag_dist[1])
            best_fit_h_diag_m_var.append(diag_dist[2])
            # best_fit_h_no_dep.append(no_dep_dist[0])
            # best_fit_h_no_dep_p_var.append(no_dep_dist[1])
            # best_fit_h_no_dep_m_var.append(no_dep_dist[2])
        # plt.show()
        fontsi = 11
        fontsi2 = 22
        plt.tick_params(labelsize=fontsi)
        plt.rc('text', usetex=True)
        plt.rc('font', family='sans-serif')
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        h_fid = np.ones_like(np.array(samples))*cov.cosmology.h

        plt.plot(samples, (best_fit_h_diag - h_fid)/h_fid,
                color="orange", label=r"diagonal covariance", lw=3, ls="dashdot")
        plt.fill_between(samples, (best_fit_h_diag_p_var - h_fid)/h_fid,
                        (best_fit_h_diag_m_var - h_fid)/h_fid, color="orange", ls="-", alpha=.4)

        plt.plot(samples, (best_fit_h_real - h_fid)/h_fid,
                color="blue", label=r"accurate covariance", lw=3, ls="-")
        plt.fill_between(samples, (best_fit_h_real_p_var - h_fid)/h_fid,
                        (best_fit_h_real_m_var - h_fid)/h_fid, color="blue", ls="-", alpha=0.2)



        np.save("sample_size", samples)
        np.save("best_fit_tiny_sky_accurate_covariance",
                (best_fit_h_real - h_fid)/h_fid)
        np.save("bands_tiny_sky_accurate_covriance", np.array([(best_fit_h_real_p_var - h_fid)/h_fid,
                (best_fit_h_real_m_var - h_fid)/h_fid]))

        np.save("best_fit_tiny_sky_diagonal_covariance",
                (best_fit_h_diag - h_fid)/h_fid)
        np.save("bad_tiny_sky_diagonal_covriance", np.array([(best_fit_h_diag_p_var - h_fid)/h_fid,
                (best_fit_h_diag_m_var - h_fid)/h_fid]))
        #plt.plot(samples, best_fit_h_no_dep, color=CB_color_cycle[4], label=r"independent", lw = 3)
        # plt.fill_between(samples, best_fit_h_no_dep_p_var,
        #                 best_fit_h_no_dep_m_var, color=CB_color_cycle[4], ls="-", alpha=.25)

        plt.legend(loc='upper right', frameon=False, fontsize=fontsi)

        plt.plot(samples, np.zeros_like(samples), ls="--", color="black")
        plt.xlim(60, sample_size)
        ax1.set_ylabel(
            r'$\frac{\Delta h}{h}$', fontsize=fontsi2)
        ax1.set_xlabel(r'$N_\mathrm{FRB}$', fontsize=fontsi2)
        for t in ax1.get_yticklabels():
            t.set_fontsize(fontsi)
        for t in ax1.get_xticklabels():
            t.set_fontsize(fontsi)

        plt.tight_layout()
        plt.savefig("error_tiny_sky.pdf")


    if (frbcat):
        cov_diagonal = np.diag(np.diagonal(covariance))
        h = np.linspace(.3, 1.1, 500)
        posterior_ac = np.zeros_like(h)
        posterior_diag = np.zeros_like(h)
        posterior_no_dep_ac = np.zeros_like(h)
        posterior_no_dep_diag = np.zeros_like(h)
        posterior_steffen = np.zeros_like(h)

        for i in range(len(h)):
            posterior_ac[i] = log_lkl(DM_obs, z, h[i], covariance)
            posterior_diag[i] = log_lkl(DM_obs, z, h[i], cov_diagonal)
            posterior_no_dep_ac[i] = log_lkl_no_dep(DM_obs, z, h[i], covariance)
            posterior_no_dep_diag[i] = log_lkl_no_dep(
                DM_obs, z, h[i], cov_diagonal)
            posterior_steffen[i] = log_lkl_steffen(DM_obs, z, h[i])

        with open('frb_cat_posterior.txt', 'w') as f:
            for i in range(len(h)):
                outstr = str(h[i]) + ' ' + \
                    str(np.exp(posterior_ac[i] - np.max(posterior_ac)))
                outstr += ' ' + \
                    str(np.exp(posterior_diag[i] - np.max(posterior_diag)))
                outstr += ' ' + \
                    str(np.exp(posterior_no_dep_ac[i] -
                        np.max(posterior_no_dep_ac)))
                outstr += ' ' + \
                    str(np.exp(
                        posterior_no_dep_diag[i] - np.max(posterior_no_dep_diag)))
                outstr += ' ' + \
                    str(np.exp(posterior_steffen[i] - np.max(posterior_steffen)))
                f.write(outstr)
                f.write("\n")
    else:
        iterate_over_mock_data()
