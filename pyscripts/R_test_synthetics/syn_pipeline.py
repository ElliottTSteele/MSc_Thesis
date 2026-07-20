'''
syn_pipeline.py

purpose:
be a function repository inside R test synthetics to store static helper
code. This code will not be updated (to prevent breakage of code that
utilises it) inside this file once implemented, but it may be extracted and
places into sdm py after completion
'''

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

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# PREAMBLE FOR ARBITRARY ALFVEN TO ARBITRARY PHYSICAL UNITS
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

# max spherical harmonic degree provided by Felix - 60 as of 03/07/2026
lmax_felix_provided = 60

# CODE FOR SYNTHETIC WAVE SPATIAL PARAMETERS

r_cmb = 3485 # km radius at CMB
r_earth = 6371.2 # km radius at earth surface

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

# CODE FOR SYNTHETIC WAVE TEMPORAL PARAMETERS

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

# RESOLUTION MATRIX HANDLING CODE:

# important events and reliable times recorded
t_r_start = 1997.1
times_mjd2000 = cp.data_utils.dyear_to_mjd(times_dyear+t_r_start)
dt_years = 0.2

notable_times_raw = {
    "CHAMP Start (2000/08)": 2000 + 8/12,
    "CHAMP End (2010/09)": 2010 + 9/12,
    "Swarm Start (2013/11)": 2013 + 11/12,
    "(2026/01)": 2026 
}

notable_times_relative = {}
notable_times_aspline = {}

for event in notable_times_raw:
    notable_times_relative[event] = \
        (notable_times_raw[event] - t_r_start)
    
# 'good' record mask
# absolute decimal-year time associated with each Gauss time step
times_absolute = t_r_start + times_dyear

# reliable record limits
good_record_start = notable_times_raw["CHAMP Start (2000/08)"]
good_record_end = notable_times_raw["(2026/01)"]

# indices lying within the good record
good_record_idx = np.where(
    (times_absolute >= good_record_start)
    & (times_absolute <= good_record_end)
)[0]

# start/end indices and equivalent slice
good_record_start_idx = good_record_idx[0]
good_record_end_idx = good_record_idx[-1]

good_record_slice = slice(
    good_record_start_idx,
    good_record_end_idx + 1
)

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

P = R_Read_In("P", "MF")
H_mf = R_Read_In("H", "MF")
H_sv = R_Read_In("H", "SV")
H_sa = R_Read_In("H", "SA")

H_dict = {
    "MF":H_mf,
    "SV":H_sv,
    "SA":H_sa
}

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


# GAUSS COEFFICIENT HANDLING CODE:

# number of gauss coefficients for SH degree lmax
def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)


# code to truncate g time series (nt, ng) to spherical harmonic degree
# N_t
def Truncate_Gauss_Coeffs(gauss_data, tmax):

    gauss_data = np.asarray(gauss_data)

    n_keep = n_Gauss_Coeffs(tmax)

    if gauss_data.shape[-1] < n_keep:
        raise ValueError(
            f"Input only has {gauss_data.shape[-1]} coefficients, "
            f"but tmax={tmax} requires {n_keep}."
        )

    return gauss_data[..., :n_keep]


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
    detrend="linear",
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
    detrend="linear",
    window="hann",
    scaling="density",
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

degrees = np.arange(1, 21)
def Lowes_Degree_PSD_All_Degrees(gnm, a, r, f_sample=1/dt_years, degrees=degrees):

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
            detrend="constant",
            window="hann",
            scaling="spectrum",
        )

        degree_psds.append(degree_psd)

    # Shape: (n_degrees, n_frequencies)
    degree_psds = np.stack(
        degree_psds,
        axis=0,
    )

    return degree_psds, frequencies



# visualising psd in 3d space
def PSD_3D_Bar_Visualise(
    degree_psds,
    frequencies,
    degrees=None,
    title=None,
    cmap="rainbow",
    vmin=None,
    vmax=None,
):
    """
    Plot degree-wise PSD as contiguous 3D bars.

    Each bar fills its complete frequency-degree cell, with cell boundaries
    halfway between neighbouring frequency and degree coordinates.

    Parameters
    ----------
    degree_psds : (n_degree, n_freq) array
        PSD values.
    frequencies : (n_freq,) array
        Uniformly spaced frequency coordinates.
    degrees : (n_degree,) array-like, optional
        Degree coordinates. Defaults to 1, ..., n_degree.
    title : str, optional
        Figure title.
    cmap : str, optional
        Colormap name.
    vmin, vmax : float, optional
        Colour-normalisation limits.
    """

    degree_psds = np.asarray(degree_psds, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)

    if degree_psds.ndim != 2:
        raise ValueError("degree_psds must be a 2D array.")

    if frequencies.ndim != 1:
        raise ValueError("frequencies must be a 1D array.")

    if degree_psds.shape[1] != frequencies.size:
        raise ValueError(
            "frequencies length must match the second dimension of degree_psds."
        )

    if frequencies.size < 2:
        raise ValueError("At least two frequency points are required.")

    if degrees is None:
        degrees = np.arange(1, degree_psds.shape[0] + 1, dtype=float)
    else:
        degrees = np.asarray(degrees, dtype=float)

    if degrees.ndim != 1 or degrees.size != degree_psds.shape[0]:
        raise ValueError(
            "degrees length must match the first dimension of degree_psds."
        )

    if degrees.size < 2:
        raise ValueError("At least two spherical harmonic degrees are required.")

    # Constant cell widths
    df = frequencies[1] - frequencies[0]
    dn = degrees[1] - degrees[0]

    if df <= 0:
        raise ValueError("frequencies must be strictly increasing.")

    if dn <= 0:
        raise ValueError("degrees must be strictly increasing.")

    if not np.allclose(np.diff(frequencies), df):
        raise ValueError("frequencies must be uniformly spaced.")

    if not np.allclose(np.diff(degrees), dn):
        raise ValueError("degrees must be uniformly spaced.")

    # Lower-left corner of every cell
    frequency_left = frequencies - df / 2
    degree_left = degrees - dn / 2

    X, Y = np.meshgrid(
        frequency_left,
        degree_left,
        indexing="xy",
    )

    valid = np.isfinite(degree_psds) & (degree_psds > 0)

    positive_psd = degree_psds[valid]

    if positive_psd.size == 0:
        raise ValueError("No positive finite PSD values found.")

    if vmin is None:
        vmin = positive_psd.min()

    if vmax is None:
        vmax = positive_psd.max()

    if vmin <= 0 or vmax <= vmin:
        raise ValueError("Require 0 < vmin < vmax.")

    # Start bars at the lower positive limit so the logarithmic z-axis works
    z_base = vmin

    X = X[valid].ravel()
    Y = Y[valid].ravel()
    CVAL = degree_psds[valid].ravel()

    Z0 = np.full_like(X, z_base)
    DX = np.full_like(X, df)
    DY = np.full_like(X, dn)
    DZ = np.maximum(CVAL - z_base, np.finfo(float).eps * z_base)

    # Colour bars by PSD value
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(CVAL))

    # Plot
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(
        X,
        Y,
        Z0,
        DX,
        DY,
        DZ,
        color=colors,
        shade=True,
        zsort="average",
        linewidth=0,
        edgecolor="none",
    )

    ax.set_xlabel("Frequency [cycles yr$^{-1}$]", labelpad=12)
    ax.set_ylabel("Spherical harmonic degree $n$", labelpad=12)
    ax.set_zlabel(
        r"Lowes-weighted SV PSD "
        r"$[(\mathrm{nT\,yr^{-1}})^2/(\mathrm{cycles\,yr^{-1}})]$",
        labelpad=14,
    )

    ax.set_xlim(
        frequencies[0] - df / 2,
        frequencies[-1] + df / 2,
    )
    ax.set_ylim(
        degrees[0] - dn / 2,
        degrees[-1] + dn / 2,
    )

    ax.set_zscale("log")
    ax.set_yticks(degrees)

    if title is not None:
        ax.set_title(title)

    ax.view_init(elev=28, azim=45)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.10,
        shrink=0.65,
        aspect=22,
    )
    cbar.set_label("Lowes-weighted SV PSD")

    plt.tight_layout()
    plt.show()

# for projecting gauss coefficients onto gridded data form
# these need to be in degrees for chaosmagpy
theta_v = np.rad2deg(theta_grid.copy().ravel())
phi_v = np.rad2deg(phi_grid.copy().ravel())
radius_v = np.full(len(phi_v), r_cmb)

# gauss -> physical forward operators
# using nmax = 20
A_r, A_t, A_p = cp.model_utils.design_gauss(
    radius_v, theta_v, phi_v, nmax=20, source="internal"
)

A_20_dict = {"r": A_r, "theta": A_t, "phi": A_p}


# SYNTHETIC WAVE LOADING CODE

# basic mode i loading code
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
