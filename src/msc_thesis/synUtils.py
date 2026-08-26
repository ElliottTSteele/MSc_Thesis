# %% FILE SYSTEM AND DEPENDENCY SETUP

# forcing root location such that the notebook can access everything 
import sys
from pathlib import Path
print(sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import h5py
import gc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import chaosmagpy as cp
from tqdm import tqdm
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import os


# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# import simulation setup
from src.msc_thesis.synSetup import *

# ---------------------------------------------------------
# RESOLUTION MATRIX
# ---------------------------------------------------------

# code to read in non-R (i.e. P, Hc) parts
def R_Read_In(R_part, which):

    # safe guard against R
    if which == "R":
        return "R should not be read in naively! Use RAM safe alternative."

    f = h5py.File((f'{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5'),"r")

    R_sub_matrix = np.asarray(f[R_part])

    f.close()
    return R_sub_matrix

# reading in the 'cheap' matrices P and H (over all components)
derivative_keys = ["MF", "SV", "SA"]

# reading in the (smaller) covariance file information provided

file_path = Path(f"{CHAOS_COV_DIR}/CHAOS_Cov_1997_2026_0806_BSpl.h5")

with h5py.File(file_path, "r") as cov_file:

    knots = np.asarray(cov_file["knots"])
    n_spl =  np.asarray(cov_file["n_spl"])
    n_m =  np.asarray(cov_file["n_m"])
    nmax_cov =  np.asarray(cov_file["nmax"])
    order =  np.asarray(cov_file["order"])


    '''print(f"There are {len(knots)} knots")
    print(f"There are: {n_spl} splines")
    print(f"There are: {nmax_cov} degrees")'''

P = R_Read_In("P", "MF")

H_sv = 365.25 * cp.model_utils.colloc_matrix(x=times_used_mjd, knots=knots, order=order, deriv = 1)

# applies R in 2 halves, fine for RAM
def R_Apply_Two_Halves(gnm_spl, verbose=True):
    """
    Apply the CHAOS resolution matrix in two row halves.

    Equivalent to:

        gnm_spl_vec = gnm_spl.ravel(order="F")
        gnm_filt_spl_vec = R @ gnm_spl_vec
        gnm_filt_spl = gnm_filt_spl_vec.reshape(gnm_spl.shape, order="F")

    but only half of the rows of R are loaded into RAM at once.
    """

    gnm_spl_vec = gnm_spl.ravel(order="F")

    file_path = Path(
        f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_MF.h5"
    )

    with h5py.File(file_path, "r") as f:
        R = f["R"]

        n_rows, n_cols = R.shape

        if n_cols != gnm_spl_vec.size:
            raise ValueError(
                f"Shape mismatch: R has {n_cols} columns, "
                f"but gnm_spl_vec has length {gnm_spl_vec.size}"
            )

        out_dtype = np.result_type(R.dtype, gnm_spl_vec.dtype)
        gnm_filt_spl_vec = np.empty(n_rows, dtype=out_dtype)

        midpoint = n_rows // 2

        if verbose:
            print(f"Processing rows 0:{midpoint} / {n_rows}")

        R_first = R[:midpoint, :]
        gnm_filt_spl_vec[:midpoint] = R_first @ gnm_spl_vec
        del R_first

        if verbose:
            print(f"Processing rows {midpoint}:{n_rows} / {n_rows}")

        R_second = R[midpoint:, :]
        gnm_filt_spl_vec[midpoint:] = R_second @ gnm_spl_vec
        del R_second

    return gnm_filt_spl_vec.reshape(gnm_spl.shape, order="F")
# GAUSS COEFFICIENT HANDLING CODE:
from scipy.signal import periodogram


# ---------------------------------------------------------
# GAUSS COEFFICIENT HANDLING
# ---------------------------------------------------------

# number of gauss coefficients for SH degree lmax
def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)

# truncates any data with shape (Nx, Ngauss), with 
def Truncate_Gauss_Coeffs(gauss_data, tmax):

    gauss_data = np.asarray(gauss_data)

    # Exclusive endpoint immediately after degree tmax
    stop_index = (tmax + 1)**2 - 1

    return gauss_data[..., :stop_index]


# infer maximum spherical harmonic degree from number of coefficients
def Gauss_Lmax(gauss_data):

    n_coeffs = np.asarray(gauss_data).shape[-1]
    lmax = int(np.sqrt(n_coeffs + 1) - 1)

    if n_Gauss_Coeffs(lmax) != n_coeffs:
        raise ValueError(
            f"{n_coeffs} is not a complete set of Gauss coefficients."
        )

    return lmax


# return the coefficient slice corresponding to spherical harmonic degree n
def Gauss_Degree_Slice(n):

    if n < 1:
        raise ValueError("Spherical harmonic degree must be at least 1.")

    return slice(n**2 - 1, (n + 1)**2 - 1)


# calculate the Lowes weighting for spherical harmonic degree n
def Lowes_Degree_Weight(n, a=r_earth, r=r_cmb):

    return (n + 1) * (a / r)**(2*n + 4)


# code to calculate instantaneous Lowes power for a degree from g(t_step=k)
def Instantaneous_Lowes_Degree_Power(
    gauss_step,
    n,
    a=r_earth,
    r=r_cmb,
):

    gauss_step = np.asarray(gauss_step)

    if gauss_step.ndim != 1:
        raise ValueError("gauss_step must have shape (ng,).")

    degree_slice = Gauss_Degree_Slice(n)

    if degree_slice.stop > gauss_step.size:
        raise ValueError(
            f"Input does not contain coefficients up to degree n={n}."
        )

    degree_coeffs = gauss_step[degree_slice]
    degree_weight = Lowes_Degree_Weight(n, a=a, r=r)

    return degree_weight * np.sum(np.abs(degree_coeffs)**2)


# code to calculate instantaneous Lowes spectrum from g(t_step=k)
def Instantaneous_Lowes_Spectrum(
    gauss_step,
    a=r_earth,
    r=r_cmb,
):

    gauss_step = np.asarray(gauss_step)
    lmax = Gauss_Lmax(gauss_step)

    return np.array([
        Instantaneous_Lowes_Degree_Power(
            gauss_step,
            n,
            a=a,
            r=r,
        )
        for n in range(1, lmax + 1)
    ])


# code to calculate instantaneous total Lowes power from g(t_step=k)
def Instantaneous_Total_Lowes_Power(
    gauss_step,
    a=r_earth,
    r=r_cmb,
):

    return np.sum(
        Instantaneous_Lowes_Spectrum(
            gauss_step,
            a=a,
            r=r,
        )
    )


# intakes a time series g = [g(0), ..., g(t_step=k), ..., g(t_step=K)]^T of gauss
# coefficients (nt, ng) and returns mean instantaneous total Lowes power
def Mean_Instantaneous_Total_Lowes_Power(
    gauss_data,
    a=r_earth,
    r=r_cmb,
):

    gauss_data = np.asarray(gauss_data)

    if gauss_data.ndim != 2:
        raise ValueError("gauss_data must have shape (nt, ng).")

    total_power = np.array([
        Instantaneous_Total_Lowes_Power(
            gauss_step,
            a=a,
            r=r,
        )
        for gauss_step in gauss_data
    ])

    return np.mean(total_power)


# intakes g_i(t_step) time series and computes PSD s_i(f)
def Gauss_Coeff_PSD(
    gauss_coeff_series,
    fs,
    detrend=None,
    window="hann",
    scaling="spectrum",
):

    gauss_coeff_series = np.asarray(gauss_coeff_series)

    frequencies, psd = periodogram(
        gauss_coeff_series,
        fs=fs,
        detrend=detrend,
        window=window,
        scaling=scaling,
        axis=0,
    )

    return frequencies, psd


# computes Lowes-weighted PSD at degree n for g(t_step) (nt, ng)
def Lowes_Degree_PSD(
    gauss_data,
    n,
    fs,
    a=r_earth,
    r=r_cmb,
    detrend=None,
    window="hann",
    scaling="spectrum",
):

    gauss_data = np.asarray(gauss_data)

    if gauss_data.ndim != 2:
        raise ValueError("gauss_data must have shape (nt, ng).")

    degree_slice = Gauss_Degree_Slice(n)

    if degree_slice.stop > gauss_data.shape[1]:
        raise ValueError(
            f"Input does not contain coefficients up to degree n={n}."
        )

    frequencies, coeff_psds = Gauss_Coeff_PSD(
        gauss_data[:, degree_slice],
        fs=fs,
        detrend=detrend,
        window=window,
        scaling=scaling,
    )

    degree_weight = Lowes_Degree_Weight(n, a=a, r=r)
    degree_psd = degree_weight * np.sum(coeff_psds, axis=1)

    return frequencies, degree_psd


def Lowes_Degree_PSD_All_Degrees(gnm, a, r, f_sample, nmax):

    degrees = np.arange(1, nmax+1, 1)

    degree_psds = []

    if np.shape(gnm)[1] == 3720:
        degrees=np.arange(1, 61)

    for n in degrees:

        frequencies, degree_psd = Lowes_Degree_PSD(
            gnm,
            n=n,
            fs=f_sample,
            a=a,
            r=r,
            detrend=None,
            window="hann",
            scaling="spectrum",
        )

        degree_psds.append(degree_psd)

    # Shape: (n_degrees, n_frequencies)
    degree_psds = np.stack(
        degree_psds,
        axis=0,
    )

    # getting rid of everything beyond nyquist
    nyquist_mask = (frequencies <= 0.5)
    degree_psds = degree_psds[:, nyquist_mask]
    frequencies = frequencies[nyquist_mask]

    return degree_psds, frequencies

# ---------------------------------------------------------
# GAUSS COEFFICIENT -> GRID CONVERSION
# ---------------------------------------------------------


# for projecting gauss coefficients onto gridded data form
# these need to be in degrees for chaosmagpy
theta_v = np.rad2deg(theta_grid.copy().ravel())
phi_v = np.rad2deg(phi_grid.copy().ravel())
radius_v = np.full(len(phi_v), r_cmb)

# gauss -> physical forward operators
# using nmax = 15
A_15_r, A_15_t, A_15_p = cp.model_utils.design_gauss(
    radius_v, theta_v, phi_v, nmax=15, source="internal"
)

A_15_dict = {"r": A_15_r, "theta": A_15_t, "phi": A_15_p}

# ---------------------------------------------------------
# LOADING CHAOS DATA 
# ---------------------------------------------------------

def CHAOS_Full_SV_Record_Obtain(model_version="CHAOS-8.6.mat", nmax=15):
    # load the CHAOS model from the mat-file
    model = cp.load_CHAOS_matfile(f'{CHAOS_DIR}/{model_version}')

    # printing full unbounded extent of CHAOS8.6 model (decimal year)
    # print('Full CHAOS-8.6 timespan is:', cp.mjd_to_dyear(model.model_tdep.breaks[[0, -1]]))

    # gauss coeffs in natural order, i.e. g(n,m): g(1,0), g(1, 1), h(1, 1), ...
    gnm_chaos = model.synth_coeffs_tdep(times_used_mjd, nmax=nmax, deriv=1)  # shape: (10, 224)

    return(gnm_chaos)

# ---------------------------------------------------------
# LOADING SYNTHETIC DATA
# ---------------------------------------------------------

# basic mode i loading code - assumes SV wanted
def Component_Load_MF(mode_number, directory=FELIX_DIR, nmax = 15):
    # select a mode number and corresponding file
    file = h5py.File(f'{directory}/mode_surface_including_gnm_{mode_number}.h5',"r")

    # Converts V_alfven (arbitrary) to nT (arbitrary)
    va_to_nt_arbitrary = (v_alfven * np.sqrt(mu_0 * rho) * 1e9) 

    # load all components, transpose to lat, long

    # gauss coefficient (magnetic scalar potential) phasors
    gnm_60 = va_to_nt_arbitrary * (np.asarray(file["gnmr"]) +\
         1j*np.asarray(file["gnmi"])) # [nT, arbitrary]
    gnm = Truncate_Gauss_Coeffs(gnm_60, tmax=nmax)

    # Br
    #br = va_to_nt_arbitrary * (np.asarray(file["brr"]).T +\
    #     1j*np.asarray(file["bri"]).T) # [nT, arbitrary]

    # u_theta
    #utheta = np.asarray(file["uthetar"]).T +\
    #    1j*np.asarray(file["uthetai"]).T # [V_alfven, arbitrary]
    # u_phi
    #uphi = np.asarray(file["uphir"]).T +\
    #    1j*np.asarray(file["uphii"]).T # [V_alfven, arbitrary]

    # loading in true period and decay rate
    omega = file["omega"][()] # angular frequency (rad/year)
    #sigma = file["sigma"][()] # annual decay rate (fractional decay/year)

    #eigenvalue_decaying = sigma + 1j * omega # with decay

    eigenvalue = 0 + 1j * omega # storing as eigenvalue - NO DECAY

    file.close() # close file

    mode_i_info = {
        "mode_number": mode_number,
        "gnm": gnm,
        #"sv": eigenvalue * br,
        #"utheta": utheta,
        #"uphi": uphi,
        "eigenvalue": eigenvalue,
        #"eigenvalue_decaying": eigenvalue_decaying
    }

    # returns all basic components for the mode
    return mode_i_info

# basic mode i loading code - assumes SV wanted
def Component_Load_SV(mode_number, directory=FELIX_DIR, nmax = 15):
    # select a mode number and corresponding file
    file = h5py.File(f'{directory}/mode_surface_including_gnm_{mode_number}.h5',"r")

    # Converts V_alfven (arbitrary) to nT (arbitrary)
    va_to_nt_arbitrary = (v_alfven * np.sqrt(mu_0 * rho) * 1e9) 

    # load all components, transpose to lat, long

    # gauss coefficient (magnetic scalar potential) phasors
    gnm_60 = va_to_nt_arbitrary * (np.asarray(file["gnmr"]) +\
         1j*np.asarray(file["gnmi"])) # [nT, arbitrary]
    gnm = Truncate_Gauss_Coeffs(gnm_60, tmax=nmax)

    # Br
    #br = va_to_nt_arbitrary * (np.asarray(file["brr"]).T +\
    #     1j*np.asarray(file["bri"]).T) # [nT, arbitrary]

    # u_theta
    #utheta = np.asarray(file["uthetar"]).T +\
    #    1j*np.asarray(file["uthetai"]).T # [V_alfven, arbitrary]
    # u_phi
    #uphi = np.asarray(file["uphir"]).T +\
    #    1j*np.asarray(file["uphii"]).T # [V_alfven, arbitrary]

    # loading in true period and decay rate
    omega = file["omega"][()] # angular frequency (rad/year)
    #sigma = file["sigma"][()] # annual decay rate (fractional decay/year)

    #eigenvalue_decaying = sigma + 1j * omega # with decay

    eigenvalue = 0 + 1j * omega # storing as eigenvalue - NO DECAY

    file.close() # close file

    mode_i_info = {
        "mode_number": mode_number,
        "gnm": eigenvalue * gnm,
        #"sv": eigenvalue * br,
        #"utheta": utheta,
        #"uphi": uphi,
        "eigenvalue": eigenvalue,
        #"eigenvalue_decaying": eigenvalue_decaying
    }

    # returns all basic components for the mode
    return mode_i_info

# function to create time series from gauss coefficient phasor
def G_Time_Series_Eval(G_mode_i, eigenvalue, times=times_evaluate_ideal_phasors):

    mode_i_contribution_list = []
    
    for t in times:
        # computing at time t
        G_t_mode_i = np.real(np.exp(eigenvalue * t) * G_mode_i)
        G_t_mode_i = np.ravel(G_t_mode_i) # state vector
        mode_i_contribution_list.append(G_t_mode_i)
        
    mode_i_contribution_array = np.vstack(mode_i_contribution_list)

    return mode_i_contribution_array


def Guest_Wave_Scale(mode_number, h5_file, nmax, mode_amp_scalings=mode_amp_scalings):

    guest_num = mode_number

    guest_info = mode_number.split("_")
    g_num, g_type, g_period_raw = guest_info[-3:]
    candidate_num = g_num[1:] # getting rid of g flag
    candidate_amplitude_scaler = mode_amp_scalings[candidate_num]

    # amplitude scaling to have power = candidate target
    guest_spline_path = f"mode_{guest_num}/without_decay"
    candidate_spline_path = f"mode_{candidate_num}/without_decay"

    gnm_spline_guest = np.asarray(h5_file[guest_spline_path][()])
    gnm_spline_candidate = np.asarray(h5_file[candidate_spline_path][()])

    gnm_guest_resolved = H_sv @ (gnm_spline_guest)
    gnm_guest_resolved = Truncate_Gauss_Coeffs(gnm_guest_resolved, tmax=nmax)

    
    gnm_candidate_resolved = H_sv @ (
        candidate_amplitude_scaler * gnm_spline_candidate
    )
    gnm_candidate_resolved = Truncate_Gauss_Coeffs(gnm_candidate_resolved, tmax=nmax)

    power_candidate = np.sum(Lowes_Degree_PSD_All_Degrees(gnm_candidate_resolved, a=r_earth,
                                                           r=r_cmb, f_sample=f_sample, nmax=nmax)[0])
    power_guest = np.sum(Lowes_Degree_PSD_All_Degrees(gnm_guest_resolved, a=r_earth,
                                                           r=r_cmb, f_sample=f_sample, nmax=nmax)[0])

    guest_amplitude_scaler = np.sqrt(power_candidate/power_guest)
    mode_amp_scalings[guest_num] = guest_amplitude_scaler

    return(mode_amp_scalings)
    


def Load_A_Mode_SV(mode_number, nmax, times, scale_flag=True, mode_amp_scalings=mode_amp_scalings):

    spline_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"


    # with spline file open
    with h5py.File(spline_path, "r") as h5_file:
        # if dealing with guest mode
        if mode_number[0] == 'g':
            mode_amp_scalings = Guest_Wave_Scale(mode_number, h5_file, nmax, mode_amp_scalings)

        # getting amplitude scaler
        if scale_flag:
            amplitude_scaler = mode_amp_scalings[mode_number]
        else:
            amplitude_scaler = 1


        # 1) getting ideal mode gauss coefficient time series
        mode_data = Component_Load_SV(mode_number, nmax=nmax)
        eigenvalue = mode_data["eigenvalue"]
        ideal_gnm_phasor = amplitude_scaler * mode_data["gnm"]
        true_period = 2.0 * np.pi / np.abs(eigenvalue.imag)
        gnm_mode_ideal = G_Time_Series_Eval(
            ideal_gnm_phasor,
            eigenvalue,
            times=times
        )

        # 2) getting the resolved gauss coefficient time series
        dataset_name = f"mode_{mode_number}/without_decay"
        gnm_spline = np.asarray(h5_file[dataset_name][()])
        gnm_mode_resolved = H_sv @ (
            amplitude_scaler * gnm_spline
        )
        gnm_mode_resolved = Truncate_Gauss_Coeffs(gnm_mode_resolved, tmax=nmax)

        # store it all for future reference
        syn_mode_info = {
            "true_period": float(true_period),
            "true_eigenvalue": eigenvalue,
            "gnm_phasor": ideal_gnm_phasor,
            "gnm_ideal": gnm_mode_ideal,
            "gnm_resolved": gnm_mode_resolved,
        }

        return(syn_mode_info)


# define function - pulls out signals necessary + input mode info for comparison
def Synthetic_Full_SV_Record_Obtain(mode_numbers, times=times_evaluate_ideal_phasors, 
                                    scale_flag=True, nmax=15, mode_amp_scalings=mode_amp_scalings):
    # for each mode - forming the ideal input record and withdrawing precomputed resolved record

    synthetic_suite_info = {}
    gnm_total_ideal_list = []
    gnm_total_resolved_list = []

    for mode_number in mode_numbers:

        mode_info = Load_A_Mode_SV(mode_number, nmax, times, scale_flag=scale_flag)
        gnm_total_ideal_list.append(mode_info["gnm_ideal"])
        gnm_total_resolved_list.append(mode_info["gnm_resolved"])
        synthetic_suite_info[mode_number] = mode_info

    # stacking lists to get cumulative wave signals
    gnm_total_ideal = np.sum(np.asarray(gnm_total_ideal_list), axis=0)
    gnm_total_resolved = np.sum(
        np.asarray(gnm_total_resolved_list),
        axis=0,
    )

    # deleting deprecated objects
    del gnm_total_ideal_list
    del gnm_total_resolved_list

    return(gnm_total_ideal, gnm_total_resolved, synthetic_suite_info)


# ---------------------------------------------------------
# CODE TO ACCOUNT FOR NON WAVE ORIGIN SPECTRUM
# ---------------------------------------------------------

from scipy.signal import firwin2, lfilter

def Non_Wave_Spectral_Infill(gnm_chaos, gnm_syn, nmax=15, dt_sample=dt_sample, seed=42, n_taps=61):

    Nt, Ng = np.shape(gnm_chaos)

    f_sample = 1/dt_sample


    g_noise = np.zeros_like(gnm_chaos)

    rng = np.random.default_rng(seed)

    for g_idx in range(Ng):

        g_series_chaos = gnm_chaos[:, g_idx]
        g_series_mode = gnm_syn[:, g_idx]

        g_f_c, g_p_c = periodogram(
            g_series_chaos,
            fs=f_sample,
            detrend=None,
            window="hann"
        )

        _, g_p_m = periodogram(
            g_series_mode,
            fs=f_sample,
            detrend=None,
            window="hann"
        )

        g_p_diff = np.maximum(g_p_c - g_p_m, 0)

        frequencies = g_f_c.copy()

        mask = (frequencies>1/2)
        g_p_diff[mask] = 0


        f_nyquist = f_sample / 2
        
        amp_gains = np.sqrt(g_p_diff * f_sample / 2)
        amp_gains[0] = np.sqrt(g_p_diff[0] * f_sample)

        if np.isclose(frequencies[-1], f_nyquist):
            amp_gains[-1] = np.sqrt(g_p_diff[-1] * f_sample)


        if frequencies[-1] < f_nyquist:
            frequencies = np.append(frequencies, f_nyquist)
            amp_gains = np.append(amp_gains, amp_gains[-1])


        taps = firwin2(
            numtaps=n_taps,
            freq=frequencies,
            gain=amp_gains,
            fs=f_sample,
        )

        burn_in = n_taps - 1
        white = rng.normal(size=Nt + burn_in)
        noise = lfilter(taps, 1.0, white)[burn_in:]

        g_noise[:, g_idx] = noise

    # procedure produces amplified noise at low degree - scale by comparison

    '''# compute lowes psd at each degree
    for n in range(1, nmax, 1):
        n_f, n_ps_chaos = Lowes_Degree_PSD(gnm_chaos, n=n, fs=f_sample)
        _, n_ps_noise = Lowes_Degree_PSD(g_noise, n=n, fs=f_sample)

        reliable_mask = (n_f < 1/3)
        # sum to get total power in degree
        tot_p_n_chaos = np.sum(n_ps_chaos[reliable_mask])
        tot_p_n_noise = np.sum(n_ps_noise[reliable_mask])

        n_factor = np.sqrt(tot_p_n_chaos/tot_p_n_noise)

        n_slice = Gauss_Degree_Slice(n)

        g_noise[:, n_slice] *= n_factor
'''
    return(g_noise)

# function to calculate the total lowes power of a phasor over the record
def Total_Power_Over_Record(input, nmax, eigenvalue=None, ideal=False):

    has_complex_values = np.any(np.iscomplex(input))

    # in the case of complex values, is phasor:
    if has_complex_values:

        gnm_phasor = input

        # if it is an 'ideal' synthetic phasor - evaluate on shifted record
        if ideal:
            times = times_evaluate_ideal_phasors
        else:
            times = times_used_relative

        gnm_ts = G_Time_Series_Eval(gnm_phasor, eigenvalue, times=times)

    # if not, assume gauss time series
    else:

        gnm_ts = input

    p_arr, _ = Lowes_Degree_PSD_All_Degrees(gnm_ts, a=r_earth, r=r_cmb, f_sample=f_sample, nmax=nmax)

    power = np.sum(p_arr)

    return(power)


# ---------------------------------------------------------
# GENERIC VISUALISATION CODE
# ---------------------------------------------------------

# converts a state vector series array ((ntheta*nphi), nt) into a cube (nt, ntheta, nphi)
def Series_To_Cube(series, state_shape=state_shape):

    return series.T.reshape((series.shape[1], *state_shape))

# basic movie for data cube
def Cube_Movie(series, name="data_cube", fig_dir=FIG_DIR, fps=5, cmap="seismic", set_vlim = False):
    """
    data_cube shape: (nt, nlat, nlon)
    """
    if series.ndim == 2: # convet to cube if ravelled state series
        cube = Series_To_Cube(series)
    else:
        cube = series

    os.makedirs(fig_dir, exist_ok=True)

    nt = cube.shape[0]

    vmax = np.nanmax(np.abs(cube))
    vmin = -vmax

    if set_vlim:
        vmin = -set_vlim
        vmax = set_vlim

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    im = ax.imshow(
        cube[0],
        cmap=cmap,
        origin="upper",
        extent=[0, 360, 180, 0],
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Colatitude [°]")

    title = ax.set_title(f"{name} | frame 0/{nt-1}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(name)

    def update(frame):
        im.set_data(cube[frame])
        title.set_text(f"{name} | frame {frame}/{nt-1}")
        return im, title

    anim = FuncAnimation(
        fig,
        update,
        frames=nt,
        interval=1000 / fps,
        blit=True
    )

    filename = os.path.join(fig_dir, f"{name}.mp4")
    anim.save(filename, fps=fps, dpi=150)

    plt.close(fig)

    return filename