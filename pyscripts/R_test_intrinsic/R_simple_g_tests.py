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
# code to add degree markers on plot
degrees = np.arange(1, 21, 1)

def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)

degrees_idx = n_Gauss_Coeffs(degrees)

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

# function to generate simple gauss coefficient data sets

def Simple_G_Generate(setup, times):
    gnm = np.zeros((148, 440)) # N = 20 gauss time series template

    for row in setup:
        i = row["index"]
        f = row["frequency"]

        g_i = np.cos(2 * np.pi * f * times)

        gnm[:, i] = g_i

    return(gnm)

# function to generate simple gauss coefficient data sets

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

# defining functions for performance evaluation

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

# %% run a simple example for one input s gauss

# %% RECORDING IMPORTANT TIME POINTS HERE FOR FUTURE USE:
t_r_start = 1997.1
dt_years = 0.2

notable_times_raw = {
    "CHAMP Start (2000/08)": 2000 + 8/12,
    "CHAMP End (2010/09)": 2010 + 9/12,
    "SWARM Start (2013/11)": 2013 + 11/12,
    "(2026/01)": 2026
}

notable_times_relative = {}
notable_times_aspline = {}

for event in notable_times_raw:
    notable_times_relative[event] = \
        (notable_times_raw[event] - t_r_start)
    
# %% building reliable time indices

reliable_t_start = notable_times_relative["CHAMP Start (2000/08)"]
reliable_t_end = notable_times_relative["(2026/01)"]


reliable_mask_t = (times_dyear > reliable_t_start) & (times_dyear < reliable_t_end)

# %%
coeff_indices = degrees_idx -1

coeff_periods = np.array([
    2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35, 40, 50, 60, 70
])

coeff_frequency = 1 / coeff_periods

times_reliable = times_dyear[reliable_mask_t]

fields = ("MF", "SV", "SA")

# Read matrices once
P = R_Read_In("P", "MF")

H = {
    field: R_Read_In("H", field)
    for field in fields
}

summary_stats = {
    field: {
        "signal_power_retained_frac": [],
        "frequency_fit_error": [],
    }
    for field in fields
}


for f in tqdm(coeff_frequency, desc="Frequencies"):

    f_stats = {
        field: {
            "signal_power_retained_frac": [],
            "frequency_fit_error": [],
        }
        for field in fields
    }

    for s in tqdm(
        coeff_indices,
        desc=f"coefficients, f={f:.4f}",
        leave=False,
    ):

        # Generate input signal in MF, SV and SA
        gnm, gsv_nm, gsa_nm = Simple_G_Generate_One(
            s,
            f,
            times_dyear,
        )

        original = {
            "MF": gnm,
            "SV": gsv_nm,
            "SA": gsa_nm,
        }

        # Apply resolution in spline-coefficient space
        gnm_spl = P @ gnm
        gnm_filt_spl = Sparse_R_Apply_One(gnm_spl, s)

        # Reconstruct filtered MF, SV and SA
        filtered = {
            field: H[field] @ gnm_filt_spl
            for field in fields
        }

        for field in fields:

            original_clip = original[field][
                reliable_mask_t,
                :
            ]

            filtered_clip = filtered[field][
                reliable_mask_t,
                :
            ]

            # Total input power
            input_power = Power_g(original_clip)

            # Filtered power remaining in injected coefficient s
            s_power_filtered, _ = Power_g(
                filtered_clip,
                s=s,
                type="s_band",
            )

            signal_power_retained_frac = (
                s_power_filtered / input_power
                if input_power > 0
                else np.nan
            )

            f_stats[field]["signal_power_retained_frac"].append(
                signal_power_retained_frac
            )

            # Fit frequency to filtered injected coefficient
            sin_fit_result = fit_sinusoid_frequency(
                filtered_clip[:, s],
                t=times_reliable,
            )

            fitted_frequency = sin_fit_result["frequency"]
            fit_r2 = sin_fit_result["r2"]

            signed_frequency_error = (
                fitted_frequency - f
            ) / f

            # Only retain frequency error for reliable coherent signals
            valid_frequency_fit = (
                fit_r2 > 0.99
                and signal_power_retained_frac > 0.1
            )

            if valid_frequency_fit:
                f_stats[field]["frequency_fit_error"].append(
                    signed_frequency_error
                )
            else:
                f_stats[field]["frequency_fit_error"].append(
                    np.nan
                )

    # Store one row per frequency
    for field in fields:
        for key in summary_stats[field]:
            summary_stats[field][key].append(
                f_stats[field][key]
            )

# %% USEFUL PLOTTING METRICS

def get_global_diagnostic_scales(
    summary_stats,
    fields=("MF", "SV", "SA"),
):
    """
    Compute common colour limits across MF, SV and SA.
    """

    scales = {
        "signal_power_retained_frac": (0, 1),
    }

    all_frequency_errors = []

    for field in fields:
        Z = np.asarray(
            summary_stats[field]["frequency_fit_error"],
            dtype=float,
        )

        finite_values = Z[np.isfinite(Z)]

        if finite_values.size > 0:
            all_frequency_errors.append(finite_values)

    if all_frequency_errors:
        all_frequency_errors = np.concatenate(
            all_frequency_errors
        )

        frequency_absmax = np.nanmax(
            np.abs(all_frequency_errors)
        )

        # Prevent a zero-width colour scale
        frequency_absmax = max(
            frequency_absmax,
            1e-12,
        )

        scales["frequency_fit_error"] = (
            -0.5,
            0.5,
        )

    else:
        scales["frequency_fit_error"] = (-1, 1)

    return scales

def plot_field_diagnostics(
    summary_stats,
    field,
    coeff_indices,
    coeff_periods,
    degrees_idx=None,
    global_scales=None,
    figsize=(text_width, 0.5*text_width),
):
    """
    Plot two resolution diagnostics:

    1. Power remaining in the injected coefficient relative to
       the original input power.

    2. Signed relative frequency error, shown only where:
           R² > 0.99
       and P_s / P_input > 0.1.
    """

    diagnostics = {
        "signal_power_retained_frac": {
            "title": (
                "Power retained in injected coefficient"
            ),
            "cbar": (
                r"$P_s^\mathrm{filtered}"
                r" / P_\mathrm{input}$"
            ),
            "cmap": "viridis",
            "default_vmin": 0,
            "default_vmax": 1,
        },
        "frequency_fit_error": {
            "title": (
                "Reliable signed frequency error"
                "\n"
                r"$R^2>0.99$ and "
                r"$P_s/P_\mathrm{input}>0.1$"
            ),
            "cbar": r"$(\hat{f}-f)/f$",
            "cmap": "RdBu_r",
            "default_vmin": -1,
            "default_vmax": 1,
        },
    }

    x_vals = np.asarray(coeff_indices)
    y_vals = np.asarray(coeff_periods)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    for ax, (key, meta) in zip(
        axes,
        diagnostics.items(),
    ):

        Z = np.asarray(
            summary_stats[field][key],
            dtype=float,
        )

        if global_scales is not None:
            vmin, vmax = global_scales[key]
        else:
            vmin = meta["default_vmin"]
            vmax = meta["default_vmax"]

        cmap = plt.get_cmap(meta["cmap"]).copy()

        if key == "frequency_fit_error":
            Z = np.ma.masked_invalid(Z)
            cmap.set_bad("black")

        pcm = ax.pcolormesh(
            x_vals,
            y_vals,
            Z,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if degrees_idx is not None:
            for idx in degrees_idx:
                if x_vals[0] <= idx <= x_vals[-1]:
                    ax.axvline(
                        idx,
                        color="red",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.8,
                    )

        ax.set_title(meta["title"])
        ax.set_xlabel(
            r"input Gauss coefficient index $s$"
        )
        ax.set_ylabel("input period / yr")
        ax.set_ylim(2, 70)
        ax.grid(True, alpha=0.25)

        cbar = fig.colorbar(
            pcm,
            ax=ax,
        )

        cbar.set_label(
            meta["cbar"]
        )

    fig.suptitle(
        f"{field} resolution diagnostics",
        fontsize=16,
    )

    plt.savefig(
        f"{FIG_DIR}/final/{field}_R_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return fig, axes

global_scales = get_global_diagnostic_scales(
    summary_stats,
    fields=fields,
)

for field in fields:
    plot_field_diagnostics(
        summary_stats,
        field=field,
        coeff_indices=coeff_indices,
        coeff_periods=coeff_periods,
        degrees_idx=degrees_idx,
        global_scales=global_scales,
    )
# %%
