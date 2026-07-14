'''
syn_deriv_check.py

Purpose: check that the derivative operators (nested in H_c @ P) work as expected
notes:
- worked perfectly on synthetic wave data
'''

# %% SETTING UP AUTOUPDATES

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

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

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *



# preamble code that allows conversion from v_a to physical units
seconds_in_year = 365.25 * 24 * 60 * 60
seconds_in_day = 24 * 60 * 60

# known quantities
rho = 1e4 # [kg/m3] mass density of fluid outer core
mu_0 = 4 * np.pi * 1e-7 # [N/A2] magnetic permeability of free space
Le=2e-4 # [unitless] Lehnert number (ration of rotation time over Alfvén time) used in calculations
L=3845 # [km] radius of earth's core
t_omega=1/(2*np.pi) # [days] earth's 'rotation time' 

# function to compute value of Va
def Alfven_Velocity_Compute(
    Le, #Lehnert number (ration of rotation time over Alfvén time) used in calculations
    L, # radius of earth's core (km)
    t_omega # earth's 'rotation time' (days/ rad)
):
    # want output in [metres/seconds]
    # therefore must convert all units to SI
    L *= 1000 # km -> m
    t_omega *= seconds_in_day # days -> hours -> minutes -> seconds
    
    return((Le * L) / t_omega) # units of [m/s]

# compute Alfven Velocity
v_alfven = Alfven_Velocity_Compute(Le, L, t_omega) # [m/s]

# %% Defining the spatial parameters

radius = 3485 # km radius at CMB

state_shape = (181, 360) # snapshot shape used globally
n_points_globally = state_shape[0]*state_shape[1] # number of points in grid

# all data uses 1^o evenly spaced grid, incl poles
# latitude, colatitude, longitude (degrees):
colatitude = np.arange(0, 181, 1)
latitude = colatitude - 90
longitude = np.arange(0, 360, 1)
# theta, phi (radians):
theta = np.deg2rad(colatitude)
phi = np.deg2rad(longitude)

# latitude weighting variables
phi_grid, theta_grid = np.meshgrid(phi, theta)
W_theta = np.sin(theta)
W2D = np.sin(theta_grid)
W2D_norm = W2D / np.sum(W2D)

# max spherical harmonic degree provided by Felix - 60 as of 03/07/2026
lmax_felix_provided = 60
# %% defining time parameters

# note: times relative to start of synthetic time series (i.e. t=0)
t_start = 0.0 # start time
t_end = 29.5 # end time
t_record = t_end - t_start # record length
f_bin = 1 / t_record # associatied fourier resolution

dt_years = 0.2   # 6-month sampling (same as CHAOS independent information)
f_sample = 1 / dt_years    # samples per year

# getting decimal year and julian date formats
times_dyear = np.arange(t_start, t_end, dt_years)
n_times = int(len(times_dyear)) # number of total samples


# %% function to load in all components for given mode
def Component_Load(mode_number, directory=FELIX_DIR):
    # select a mode number and corresponding file
    file = h5py.File(f'{directory}/mode_surface_including_gnm_{mode_number}.h5',"r")

    # Converts V_alfven (arbitrary) to nT (arbitrary)
    va_to_nt_arbitrary = (v_alfven * np.sqrt(mu_0 * rho) * 1e9) 

    # load all components, transpose to lat, long

    # gauss coefficient (magnetic scalar potential) phasors
    gnm = va_to_nt_arbitrary * (np.asarray(file["gnmr"]) +\
         1j*np.asarray(file["gnmi"])) # [nT, arbitrary]

    # Br
    br = va_to_nt_arbitrary * (np.asarray(file["brr"]).T +\
         1j*np.asarray(file["bri"]).T) # [nT, arbitrary]

    # u_theta
    utheta = np.asarray(file["uthetar"]).T +\
        1j*np.asarray(file["uthetai"]).T # [V_alfven, arbitrary]
    # u_phi
    uphi = np.asarray(file["uphir"]).T +\
        1j*np.asarray(file["uphii"]).T # [V_alfven, arbitrary]

    # loading in true period and decay rate
    omega = file["omega"][()] # angular frequency (rad/year)
    sigma = file["sigma"][()] # annual decay rate (fractional decay/year)
    eigenval = sigma + 1j * omega # storing as eigenvalue

    file.close() # close file

    mode_i_info = {
        "mode_number": mode_number,
        "gnm": gnm,
        "br": br,
        "utheta": utheta,
        "uphi": uphi,
        "eigenvalue": eigenval
    }

    # returns all basic components for the mode
    return mode_i_info

# function to create time series from gauss coefficient phasor
def G_Time_Series_Eval(G_mode_i, eigenvalue):

    mode_i_contribution_list = []
    
    for t in times_dyear:
        # computing at time t
        G_t_mode_i = np.real(np.exp(eigenvalue * t) * G_mode_i)
        G_t_mode_i = np.ravel(G_t_mode_i) # state vector
        mode_i_contribution_list.append(G_t_mode_i)
        
    mode_i_contribution_array = np.vstack(mode_i_contribution_list)

    return mode_i_contribution_array

# these need to be in degrees for chaosmagpy
theta_v = np.rad2deg(theta_grid.copy().ravel())
phi_v = np.rad2deg(phi_grid.copy().ravel())
radius_v = np.full(len(phi_v), radius)

# gauss -> physical forward operators
# using nmax = 20
A_r, A_t, A_p = cp.model_utils.design_gauss(
    radius_v, theta_v, phi_v, nmax=20, source="internal"
)

A_20_dict = {"r": A_r, "theta": A_t, "phi": A_p}

# given a maximum spherical harmonic degree lmax
# this function returns the number of gauss coefficients
def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)

# %%

# provided an array containing gauss coefficient data
# where shape(array)[-1] = number of gauss coefficients
# this function truncates to degree tmax
def Truncate_Gauss_Coeffs(gauss_data, tmax):

    gauss_data = np.asarray(gauss_data)

    n_keep = n_Gauss_Coeffs(tmax)

    if gauss_data.shape[-1] < n_keep:
        raise ValueError(
            f"Input only has {gauss_data.shape[-1]} coefficients, "
            f"but tmax={tmax} requires {n_keep}."
        )

    return gauss_data[..., :n_keep]

def R_Read_In(R_part, which):
    f = h5py.File((f'{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5'),"r")

    R_sub_matrix = np.asarray(f[R_part])

    f.close()
    return R_sub_matrix


# R very big, use code to apply in chunks to prevent RAM overflow
def R_Apply_Blockwise(gnm_spl, which="MF", R_part="R", block_size=4000, verbose=True):
    """
    Apply large CHAOS resolution matrix R to gnm_spl without loading full R.

    Equivalent to:

        gnm_spl_vec = gnm_spl.ravel(order="F")
        gnm_filt_spl_vec = R @ gnm_spl_vec
        gnm_filt_spl = gnm_filt_spl_vec.reshape(gnm_spl.shape, order="F")

    but reads R in block_size x block_size chunks.
    """

    # Flatten using same convention as original code
    gnm_spl_vec = gnm_spl.ravel(order="F")

    file_path = Path(f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5")

    with h5py.File(file_path, "r") as f:
        R = f[R_part]

        n_rows, n_cols = R.shape

        if n_cols != gnm_spl_vec.size:
            raise ValueError(
                f"Shape mismatch: R has {n_cols} columns, "
                f"but gnm_spl_vec has length {gnm_spl_vec.size}"
            )

        # Output vector. Use safe dtype in case gnm_spl is complex.
        out_dtype = np.result_type(R.dtype, gnm_spl_vec.dtype)
        gnm_filt_spl_vec = np.zeros(n_rows, dtype=out_dtype)

        # Blocked matrix-vector multiply
        for i0 in range(0, n_rows, block_size):
            i1 = min(i0 + block_size, n_rows)

            if verbose:
                print(f"Processing output rows {i0}:{i1} / {n_rows}")

            y_block = np.zeros(i1 - i0, dtype=out_dtype)

            for j0 in tqdm(range(0, n_cols, block_size)):
                j1 = min(j0 + block_size, n_cols)

                R_block = R[i0:i1, j0:j1]
                x_block = gnm_spl_vec[j0:j1]

                y_block += R_block @ x_block

                del R_block, x_block
                gc.collect()

            gnm_filt_spl_vec[i0:i1] = y_block

            del y_block
            gc.collect()

    gnm_filt_spl = np.reshape(
        gnm_filt_spl_vec,
        gnm_spl.shape,
        order="F"
    )

    return gnm_filt_spl
# code to add degree markers on plot
degrees = np.arange(0, 21, 1)
degrees_60 = np.arange(0, 61, 1)

def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)

degrees_idx = n_Gauss_Coeffs(degrees)-1
degrees_idx_60 =  n_Gauss_Coeffs(degrees_60)-1
# developing a function to 'scale' waves
# by lowes power to better visualise the high degree signal
def lowes_weight_gnm(gnm, degrees_idx=degrees_idx):
    """
    Apply sqrt(n + 1) Lowes weighting to a Gauss-coefficient
    time series with shape (nt, ncoeff).

    degrees_idx contains the coefficient indices at degree boundaries.
    """
    gnm = np.asarray(gnm, dtype=float)
    weighted = np.zeros_like(gnm)

    n_coeff = gnm.shape[1]

    boundaries = np.asarray(degrees_idx, dtype=int)
    boundaries = boundaries[
        (boundaries > 0) & (boundaries < n_coeff)
    ]

    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, n_coeff]

    for degree, (start, end) in enumerate(
        zip(starts, ends),
        start=1,
    ):
        weighted[:, start:end] = (
            np.sqrt(degree + 1) * gnm[:, start:end]
        )

    return weighted

def degree_normalise_gnm(gnm, degrees_idx=degrees_idx, eps=1e-12):
    """
    Normalise each spherical-harmonic degree by its RMS amplitude.

    Parameters
    ----------
    gnm : ndarray, shape (nt, ncoeff)
        Gauss-coefficient time series.

    degrees_idx : array-like
        Coefficient indices marking the boundaries between degrees.

    eps : float
        Degrees with RMS below this value are left as zero.

    Returns
    -------
    gnm_normalised : ndarray
        Degree-normalised coefficient time series.

    degree_scales : ndarray
        RMS scale removed from each degree.
    """
    gnm = np.asarray(gnm, dtype=float)
    gnm_normalised = np.zeros_like(gnm)

    n_coeff = gnm.shape[1]

    boundaries = np.asarray(degrees_idx, dtype=int)
    boundaries = boundaries[
        (boundaries > 0) & (boundaries < n_coeff)
    ]

    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, n_coeff]

    degree_scales = []

    for degree, (start, end) in enumerate(
        zip(starts, ends),
        start=1,
    ):
        degree_block = gnm[:, start:end]

        degree_rms = np.sqrt(
            np.nanmean(degree_block**2)
        )

        degree_scales.append(degree_rms)

        if degree_rms > eps:
            gnm_normalised[:, start:end] = (
                degree_block / degree_rms
            )

    return gnm_normalised, np.asarray(degree_scales)

#%%
import numpy as np


def lowes_spectrum_phasor(
    gnm_phasor,
    a=6371.2,
    r=3485, # r_cmb
    mode="cycle_mean",
    phase=0.0,
):
    """
    Compute the Lowes-Mauersberger spectrum directly from a complex
    Gauss-coefficient phasor.

    Assumes standard Gauss-coefficient ordering:

        [g_1^0, g_1^1, h_1^1,
         g_2^0, g_2^1, h_2^1, g_2^2, h_2^2, ...]

    and a real time-dependent signal of the form

        g(t) = Re{gnm_phasor * exp(i * phase)}.

    Parameters
    ----------
    gnm_phasor : ndarray, shape (ncoeff,)
        Complex Gauss-coefficient phasor.

    a : float, default 6371.2
        Reference radius of the Gauss coefficients.

    r : float or None
        Radius at which the spectrum is evaluated.
        If None, r = a.

    mode : {"cycle_mean", "instantaneous"}
        "cycle_mean":
            Exact spectrum averaged over a complete oscillation cycle.

        "instantaneous":
            Spectrum at the supplied phase.

    phase : float, default 0
        Wave phase in radians, used only for mode="instantaneous".
        For a wave with angular frequency omega:

            phase = omega * t

    Returns
    -------
    degrees : ndarray, shape (nmax,)
        Spherical-harmonic degrees.

    spectrum : ndarray, shape (nmax,)
        Lowes spectrum by degree.
    """

    phasor = np.asarray(gnm_phasor, dtype=complex)

    if phasor.ndim != 1:
        raise ValueError("gnm_phasor must be a one-dimensional array.")

    if r is None:
        r = a

    ncoeff = phasor.size

    # Complete standard Gauss set has ncoeff = nmax(nmax + 2)
    nmax = int(np.sqrt(ncoeff + 1) - 1)

    if nmax * (nmax + 2) != ncoeff:
        raise ValueError(
            f"{ncoeff} coefficients do not form a complete standard "
            "Gauss-coefficient set."
        )

    degrees = np.arange(1, nmax + 1)
    spectrum = np.zeros(nmax, dtype=float)

    for n in degrees:

        # Degree n occupies n^2 - 1 : (n + 1)^2 - 1
        start = n**2 - 1
        end = (n + 1)**2 - 1

        degree_phasor = phasor[start:end]

        if mode == "cycle_mean":
            coefficient_power = (
                0.5 * np.sum(np.abs(degree_phasor)**2)
            )

        elif mode == "instantaneous":
            degree_coefficients = np.real(
                degree_phasor * np.exp(1j * phase)
            )

            coefficient_power = np.sum(
                degree_coefficients**2
            )

        else:
            raise ValueError(
                "mode must be 'cycle_mean' or 'instantaneous'."
            )

        spectrum[n - 1] = (
            (n + 1)
            * (a / r)**(2 * n + 4)
            * coefficient_power
        )

    return degrees, spectrum
# %% ACTUAL LIVE CODE, NOT FUNCTION DEFINTIONS:
# pulling in the small (i.e. non R) matrices from resolution data
P_res = R_Read_In("P", "MF")
H_res = R_Read_In("H", "MF")
mode_numbers = [15, 30, 45, 60]

lowes_max = []
periods = []

for mode_number in mode_numbers:
    mode_i_info = Component_Load(mode_number, FELIX_DIR)
    eigenvalue_i = mode_i_info["eigenvalue"]

    # testing physical coefficient vs truncated gauss

    gnm_full_phasor = mode_i_info["gnm"] # [nT, arbitrary]
    gnm_phasor = Truncate_Gauss_Coeffs(gnm_full_phasor, tmax=20)
    degrees, Rn_mean = lowes_spectrum_phasor(
    gnm_full_phasor,
    mode="cycle_mean",
)
    

    print(f"period is {2 * np.pi / eigenvalue_i.imag}")
    periods.append(2 * np.pi / eigenvalue_i.imag)
    max_idx = np.argmax(Rn_mean)
    lowes_max.append(degrees[max_idx])

    # lowes spectrum from phasor

    # A_r = A_20_dict["r"] # forward operator for nmax=20

    # testing truncated gauss vs truncated gauss after B-spline
    # have N=20 phasor for wave (gnm_t20), need time series
    # gnm = G_Time_Series_Eval(gnm_phasor, eigenvalue_i)


    # now projecting onto B-spline
    # gnm_spl = P_res @ gnm
    # now applying R blockwise for RAM
    # gnm_spl_res = R_Apply_Blockwise(gnm_spl)
    # now retreiving back
    # gnm_res = H_res @ gnm_spl_res

# %%

plt.plot(periods, lowes_max, 'x')

# %%
