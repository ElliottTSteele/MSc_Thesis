'''
R_simple_g_tests.py

purpose:
To test the effects of the provided resolution matrix dataset on
simple gauss coefficient tests, looking at the effects of degree,
frequency of gauss coefficient oscillation etc.

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
from matplotlib.animation import FuncAnimation
import h5py
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import chaosmagpy as cp
import gc
from tqdm import tqdm
import ctypes
import os

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# %% code to read in a synthetic data set

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
print(n_times) # needs to be 148 for resolution matrix compatibility 
print(times_dyear[-1])

# %% code for handling R parts read in

# for P and H - small can easily be read in full
def R_Read_In(R_part, which):
    f = h5py.File((f'{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5'),"r")

    R_sub_matrix = np.asarray(f[R_part])

    f.close()
    return R_sub_matrix

# Force python to garbage collect RAM
def Clear_Ram():

    gc.collect()

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

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
        for i0 in tqdm(range(0, n_rows, block_size)):
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

# define sparse R apply:
# by default make it apply the whole matrix by each gauss coeff
def Sparse_R_Apply(gnm_spl, s_list= np.arange(0, 440, 1), which="MF"):
    file_path = Path(f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5")

    v_tot = np.zeros(28160)

    for s in s_list:

        # pulling out the correct b_vector
        b_s = gnm_spl[:, s]

        with h5py.File(file_path, "r") as f:
            R = f["R"]

            R_s = R[:,64*(s):64*(s+1)] # getting used part of R

            v_s = R_s @ b_s

            v_tot += v_s
    
    gnm_filt_spl = np.reshape(
        v_tot,
        gnm_spl.shape,
        order="F"
    )
    return gnm_filt_spl

# define sparse R apply:
# by default make it apply the whole matrix by each gauss coeff
def Sparse_R_Apply_One(gnm_spl, s, which="MF"):
    file_path = Path(f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5")

    # pulling out the correct b_vector
    b_s = gnm_spl[:, s]

    with h5py.File(file_path, "r") as f:
        R = f["R"]

        R_s = R[:,64*(s):64*(s+1)] # getting used part of R

        v_tot = R_s @ b_s

    gnm_filt_spl = np.reshape(
        v_tot,
        gnm_spl.shape,
        order="F"
    )
    return gnm_filt_spl

# naive applying R whole at once
''''def R_apply(gnm_spl):

    gnm_spl_vec = gnm_spl.ravel(order="F")

    R = R_Read_In(R_part="R")
    gnm_filt_spl_vec = R @ gnm_spl_vec
    del R
    Clear_Ram()


    gnm_filt_spl = np.reshape(
        gnm_filt_spl_vec,
        gnm_spl.shape,
        order="F"
    )

    return(gnm_filt_spl)'''

# %% function to generate simple gauss coefficient data sets

def Simple_G_Generate(setup, times):
    gnm = np.zeros((148, 440)) # N = 20 gauss time series template

    for row in setup:
        i = row["index"]
        f = row["frequency"]

        g_i = np.cos(2 * np.pi * f * times)

        gnm[:, i] = g_i

    return(gnm)

# %% function to generate simple gauss coefficient data sets

def Simple_G_Generate_One(s, f, times):
    gnm = np.zeros((148, 440)) # N = 20 gauss time series template
    gsv_nm = np.zeros((148, 440))
    gsa_nm = np.zeros((148, 440))

    g_i = np.cos(2 * np.pi * f * times)
    gsv_i = -(2 * np.pi * f) * np.sin(2 * np.pi * f * times)
    gsa_i = -((2 * np.pi * f)**2) * np.cos(2 * np.pi * f * times)

    gnm[:, s] = g_i
    gsv_nm[:, s] = gsv_i
    gsa_nm[:, s] = gsa_i

    return(gnm, gsv_nm, gsa_nm)

# %% setting up the simple gauss time series parameters
# coefficient index, frequency

# doing for more than one s band injection

'''coeff_indices = np.arange(0,440,1)# which coeff


coeff_frequencies = np.full(len(coeff_indices), fill_value=0.1) # [/yr]

sim_gauss_setup = []

for index, frequency in zip(coeff_indices, coeff_frequencies):

    sim_gauss_row = {
        "index":index,
        "frequency":frequency
    }

    sim_gauss_setup.append(sim_gauss_row)


gnm = Simple_G_Generate(sim_gauss_setup, times_dyear)

plt.imshow(gnm)

P = R_Read_In("P", "MF")

gnm_spl = P @ gnm
gnm_filt_spl = Sparse_R_Apply(gnm_spl, coeff_indices)

print("done!")

H_MF = R_Read_In("H", "MF")
H_SV = R_Read_In("H", "SV")
H_SA = R_Read_In("H", "SA")

gnm_filt = H_MF @ gnm_filt_spl
gnm_sv_filt = H_SV @ gnm_filt_spl
gnm_sa_filt = H_SA @ gnm_filt_spl'''

# %% defining functions for performance evaluation

# takes in gnm (nt, ng) and finds total naive power
def Power_g(gnm, s=None, type='total'):

    # elementwise square to get gnm**2
    gnm_sq = gnm**2

    # sum
    gnm_sq_sum = np.sum(gnm_sq)

    if type=="total":

        return(gnm_sq_sum)
    
    elif type=="s_band":

        # get s_band power
        s_band_pow = np.sum(gnm_sq[:, s])

        # get rest if power
        rest_band_pow = gnm_sq_sum - s_band_pow

        return (s_band_pow, rest_band_pow)

# %% frequency analysis

import numpy as np
from scipy.optimize import minimize_scalar

def fit_sinusoid_frequency( y,t, f_min=0.001, f_max=1, n_grid=200, refine=True):
    """
    Fit y(t) = A cos(2πft) + B sin(2πft) + C.

    Returns best frequency, period, amplitude, phase, R2, relative residual.
    """

    t = np.asarray(t)
    y = np.asarray(y)

    # remove NaNs if needed
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    # centre time for numerical stability
    t0 = t.mean()
    tau = t - t0

    def fit_at_frequency(f):
        X = np.column_stack([
            np.cos(2*np.pi*f*tau),
            np.sin(2*np.pi*f*tau),
            np.ones_like(tau)
        ])

        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_fit = X @ coeffs

        residual = y - y_fit
        rss = np.sum(residual**2)
        tss = np.sum((y - y.mean())**2)

        rel_resid = np.sqrt(rss / np.sum(y**2))
        r2 = 1 - rss / tss if tss > 0 else np.nan

        A, B, C = coeffs
        amp = np.sqrt(A**2 + B**2)
        phase = np.arctan2(-B, A)

        return rss, rel_resid, r2, amp, phase, C, y_fit

    # coarse grid search
    f_grid = np.linspace(f_min, f_max, n_grid)
    rss_grid = np.array([fit_at_frequency(f)[0] for f in f_grid])
    f_best = f_grid[np.argmin(rss_grid)]

    # optional local refinement
    if refine:
        df = f_grid[1] - f_grid[0]
        lo = max(f_min, f_best - 2*df)
        hi = min(f_max, f_best + 2*df)

        res = minimize_scalar(
            lambda f: fit_at_frequency(f)[0],
            bounds=(lo, hi),
            method="bounded"
        )
        f_best = res.x

    rss, rel_resid, r2, amp, phase, C, y_fit = fit_at_frequency(f_best)

    return {
        "frequency": f_best,
        "period": 1 / f_best,
        "amplitude": amp,
        "phase": phase,
        "offset": C,
        "r2": r2,
        "rel_resid": rel_resid,
        "fit": y_fit,
    }
# %% FOR ONE AT A TIME
# coefficient index, frequency

coeff_indices = np.arange(0,440,1)
coeff_periods = np.array([2, 4, 8, 16, 32, 64, 128])
coeff_frequency = (1/coeff_periods)

summary_stats = {
    "total_retained_power_frac": [],
    "s_power_vs_total_power_frac": [],
    "sin_fit": [],
    "sin_fit_r2": []
}
# between satellite data first start and 2026/01
start_reliable_idx, end_reliable_idx = 9, 58

for f in tqdm(coeff_frequency):

    s_pow_frac_f = []
    ret_pow_frac_f = []
    sin_fit_f = []
    sin_fit_r2_f = []
    for s in tqdm(coeff_indices):

        gnm, gsv_nm, gsa_nm =\
              Simple_G_Generate_One(s, f, times_dyear)

        P = R_Read_In("P", "MF")

        gnm_spl = P @ gnm
        gnm_filt_spl = Sparse_R_Apply_One(gnm_spl, s)

  
        H_MF = R_Read_In("H", "MF")
        H_SV = R_Read_In("H", "SV")
        H_SA = R_Read_In("H", "SA")

        gnm_filt = H_MF @ gnm_filt_spl
        gnm_sv_filt = H_SV @ gnm_filt_spl
        gnm_sa_filt = H_SA @ gnm_filt_spl

        # clipping off start window
        gnm = gnm[start_reliable_idx: end_reliable_idx, :]
        gnm_filt = gnm_filt[start_reliable_idx: end_reliable_idx, :]

        # getting total power in whole video
        tot_power_original = Power_g(gnm)
        tot_power_filtered = Power_g(gnm_filt)

        # getting power in band and rest of gnm
        s_power_filtered, rest_power_filtered = Power_g(gnm_filt, s=s, type='s_band')

        # storing
        s_pow_frac_f.append(s_power_filtered / tot_power_filtered)
        ret_pow_frac_f.append(tot_power_filtered/ tot_power_original)

        # examining frequency
        sin_fit_result_s = fit_sinusoid_frequency(gnm_filt[:, s], t=times_dyear[start_delay_idx:])
        sin_fit_f.append(np.abs((sin_fit_result_s["frequency"]-f)/f)) # normalised percent diff
        sin_fit_r2_f.append(sin_fit_result_s["r2"])
    
    # storing
    summary_stats["s_power_vs_total_power_frac"].append(s_pow_frac_f)
    summary_stats["total_retained_power_frac"].append(ret_pow_frac_f)
    summary_stats["sin_fit"].append(sin_fit_f)
    summary_stats["sin_fit_r2"].append(sin_fit_r2_f)

# %%

# %% RUN DIAGNOSTICS FOR MF, SV, SA

coeff_indices = np.arange(0, 440, 40)
coeff_periods = np.array([2, 4, 8, 16, 32, 64, 128])
coeff_frequency = 1 / coeff_periods

start_reliable_idx, end_reliable_idx = 9, 58
times_reliable = times_dyear[start_reliable_idx:end_reliable_idx]

# Read these once, not inside every loop
P = R_Read_In("P", "MF")

H = {
    "MF": R_Read_In("H", "MF"),
    "SV": R_Read_In("H", "SV"),
    "SA": R_Read_In("H", "SA"),
}

summary_stats = {
    field: {
        "total_retained_power_frac": [],
        "s_power_vs_total_power_frac": [],
        "sin_fit": [],
        "sin_fit_r2": [],
    }
    for field in ["MF", "SV", "SA"]
}

for f in tqdm(coeff_frequency, desc="Frequencies"):

    # temporary storage for this frequency
    f_stats = {
        field: {
            "total_retained_power_frac": [],
            "s_power_vs_total_power_frac": [],
            "sin_fit": [],
            "sin_fit_r2": [],
        }
        for field in ["MF", "SV", "SA"]
    }

    for s in tqdm(coeff_indices, desc=f"coefficients, f={f:.4f}", leave=False):

        # Generate simple input signal in MF, SV, SA
        gnm, gsv_nm, gsa_nm = Simple_G_Generate_One(s, f, times_dyear)

        original = {
            "MF": gnm,
            "SV": gsv_nm,
            "SA": gsa_nm,
        }

        # Apply resolution in spline space
        gnm_spl = P @ gnm
        gnm_filt_spl = Sparse_R_Apply_One(gnm_spl, s)

        # Reconstruct filtered MF, SV, SA from filtered spline coefficients
        filtered = {
            field: H[field] @ gnm_filt_spl
            for field in ["MF", "SV", "SA"]
        }

        for field in ["MF", "SV", "SA"]:

            original_clip = original[field][start_reliable_idx:end_reliable_idx, :]
            filtered_clip = filtered[field][start_reliable_idx:end_reliable_idx, :]

            # Total power before and after resolution
            tot_power_original = Power_g(original_clip)
            tot_power_filtered = Power_g(filtered_clip)

            # Power retained in injected coefficient s vs leaked elsewhere
            s_power_filtered, rest_power_filtered = Power_g(
                filtered_clip,
                s=s,
                type="s_band"
            )

            f_stats[field]["s_power_vs_total_power_frac"].append(
                s_power_filtered / tot_power_filtered
            )

            f_stats[field]["total_retained_power_frac"].append(
                tot_power_filtered / tot_power_original
            )

            # Frequency fitted to the filtered version of the injected coefficient
            sin_fit_result_s = fit_sinusoid_frequency(
                filtered_clip[:, s],
                t=times_reliable
            )

            f_stats[field]["sin_fit"].append(
                np.abs((sin_fit_result_s["frequency"] - f) / f)
            )

            f_stats[field]["sin_fit_r2"].append(
                sin_fit_result_s["r2"]
            )

    # store one list per frequency
    for field in ["MF", "SV", "SA"]:
        for key in summary_stats[field]:
            summary_stats[field][key].append(f_stats[field][key])
# %%

s = 300
gnm = Simple_G_Generate_One(s, 0.1, times_dyear)

P = R_Read_In("P", "MF")

gnm_spl = P @ gnm
gnm_filt_spl = Sparse_R_Apply_One(gnm_spl, s)

H_MF = R_Read_In("H", "MF")
#H_SV = R_Read_In("H", "SV")
#H_SA = R_Read_In("H", "SA")

gnm_filt = H_MF @ gnm_filt_spl
#gnm_sv_filt = H_SV @ gnm_filt_spl
#gnm_sa_filt = H_SA @ gnm_filt_spl

# clipping off start window
gnm = gnm[start_delay_idx:, :]
gnm_filt = gnm_filt[start_delay_idx:, :]

# %%

plt.imshow(gnm)
# %% code to add degree markers on plot
degrees = np.arange(1, 21, 1)

def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)

degrees_idx = n_Gauss_Coeffs(degrees)

# %% post processing summary stats into numpy array
s_pow_frac_arr = np.asarray(summary_stats["s_power_vs_total_power_frac"])
ret_pow_frac_arr = np.asarray(summary_stats["total_retained_power_frac"])
sin_fit_arr = np.asarray(summary_stats["sin_fit"])
sin_fit_r2_arr = np.asarray(summary_stats["sin_fit_r2"])

# %% examining frequency
fig, ax = plt.subplots(1, 3, figsize=(6, 6))

pcm = ax[0].pcolormesh(
    coeff_indices,
    coeff_frequency,
    sin_fit_arr,
    shading="auto"
)

ax[0].set_box_aspect(1)
fig.colorbar(pcm, ax=ax[0])

ax[0].set_ylabel("frequency [yr$^{-1}$]")
ax[0].set_xlabel("degree")

pcm = ax[1].pcolormesh(
    coeff_indices,
    coeff_frequency,
    sin_fit_r2_arr,
    shading="auto"
)

ax[1].set_box_aspect(1)
fig.colorbar(pcm, ax=ax[1])

ax[1].set_ylabel("frequency [yr$^{-1}$]")
ax[1].set_xlabel("degree")

pcm = ax[2].pcolormesh(
    coeff_indices,
    coeff_frequency,
    sin_fit_r2_arr*sin_fit_arr,
    shading="auto"
)

ax[2].set_box_aspect(1)
fig.colorbar(pcm, ax=ax[2])

ax[2].set_ylabel("frequency [yr$^{-1}$]")
ax[2].set_xlabel("degree")


# %% making a pc colourmesh of this:
fig, ax = plt.subplots(1, 2, figsize=(6, 6))

pcm = ax[0].pcolormesh(
    coeff_indices,
    coeff_frequency,
    s_pow_frac_arr,
    shading="auto"
)

ax[0].set_box_aspect(1)
fig.colorbar(pcm, ax=ax[0])

ax[0].set_ylabel("frequency [yr$^{-1}$]")
ax[0].set_xlabel("degree")

pcm = ax[1].pcolormesh(
    coeff_indices,
    coeff_frequency,
    ret_pow_frac_arr,
    shading="auto"
)

ax[1].set_box_aspect(1)
fig.colorbar(pcm, ax=ax[1])

ax[1].set_ylabel("frequency [yr$^{-1}$]")
ax[1].set_xlabel("degree")

# %% making a pc colourmesh of this (period):
coeff_frequency = (1/coeff_periods)
coeff_frequency[-1] = 0
period_mask = coeff_frequency > 0


fig, ax = plt.subplots(1, 2, figsize=(6, 6))

pcm = ax[0].pcolormesh(
    coeff_indices,
    coeff_periods[period_mask],
    s_pow_frac_arr[period_mask, :],
    shading="auto"
)

ax[0].set_box_aspect(1)
fig.colorbar(pcm, ax=ax[0])

ax[0].set_ylabel("frequency [yr$^{-1}$]")
ax[0].set_xlabel("degree")

pcm = ax[1].pcolormesh(
    coeff_indices,
    coeff_periods[period_mask],
    ret_pow_frac_arr[period_mask, :],
    shading="auto"
)

ax[1].set_box_aspect(1)
fig.colorbar(pcm, ax=ax[1])

ax[1].set_ylabel("frequency [yr$^{-1}$]")
ax[1].set_xlabel("degree")

# %%
coeff_frequency = (1/coeff_periods)
coeff_frequency[-1] = 0
extent_summary = [coeff_indices.min(), coeff_indices.max(), coeff_frequency.min(), coeff_frequency.max()]

plt.imshow(s_pow_frac_arr, extent = extent_summary, aspect='auto')

# %%

plt.imshow(ret_pow_frac_arr, extent = extent_summary, aspect='auto')
# %%

plt.plot(coeff_indices, summary_stats["s_power_vs_total_power_frac"])
for marker in degrees_idx:
    plt.axvline(marker)

# %%

plt.plot(coeff_indices, summary_stats["total_retained_power_frac"])
for marker in degrees_idx:
    plt.axvline(marker)
# %%
plt.imshow(gnm_filt)
# %%
plt.imshow(gnm_sv_filt)

#%%
plt.imshow(gnm_sa_filt)
# %% Defining functions to transform between physical and gauss

# -------------------- BELOW IS GOOD CODE, BUT NOT CURRENTLY WANTED TO USE --------------------------

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

# %% need a way to convert gauss back to physical grid for video

# gets gauss coefficient time series, retrieves physical grid time series of radial component
def Gauss_To_Grid_Time_Series(G, A_r=A_20_dict["r"], state_shape=(181, 360)):

    n_times, n_coeffs = G.shape # G ~ nt, ncoeffs
    n_grid = np.prod(state_shape) # n_grid = 360 x 181

    if A_r.shape != (n_grid, n_coeffs):
        raise ValueError(
            f"Expected A_r shape {(n_grid, n_coeffs)}, got {A_r.shape}"
        )

    grid_v = A_r @ G.T                     # (n_grid, n_times)
    grid = grid_v.T.reshape(n_times, *state_shape) # (nt, ntheta, nphi)

    return grid

# %% converting to gridded physical data

pre_resolution_br = Gauss_To_Grid_Time_Series(gnm)
post_resolution_br = Gauss_To_Grid_Time_Series(gnm_filt)

# %% making movie
# converts a state vector series array ((ntheta*nphi), nt) into a cube (nt, ntheta, nphi)
# not used here, but required for below to compile
def Series_To_Cube(series, state_shape=state_shape):

    return series.T.reshape((series.shape[1], *state_shape))

def Cube_Movie(series, name="data_cube", fig_dir=FIG_DIR, fps=10, cmap="seismic"):
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

Cube_Movie(pre_resolution_br, name=f"{test_name}_br_pre_res")
Cube_Movie(post_resolution_br, name=f"{test_name}_br_post_res")
# %%

def Plot_3D_Surface(M, title="3D matrix surface", xlabel="Gauss coefficient index",
                    ylabel="Spline coefficient index", zlabel="Amplitude",
                    max_abs=None, stride=20):

    M_plot = np.asarray(M)

    # Optional downsampling for large matrices
    M_plot = M_plot[::, ::stride]

    y = np.arange(M_plot.shape[0])   # spline index
    x = np.arange(M_plot.shape[1])   # Gauss index
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, M_plot,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, label=zlabel)

    if max_abs is not None:
        ax.set_zlim(-max_abs, max_abs)

    plt.tight_layout()
    plt.show()

Plot_3D_Surface(gnm_filt)
# %%
Plot_3D_Surface(gnm)
# %%
