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

# %% run a simple example for one input s gauss

# %% RECORDING IMPORTANT TIME POINTS HERE FOR FUTURE USE:
t_r_start = 1997.1
dt_years = 0.2

notable_times_raw = {
    "CHAMP Start (2000/08)": 2000 + 8/12,
    "CHAMP Start (2010/09)": 2010 + 9/12,
    "SWARM End (2013/11)": 2013 + 11/12,
    "(2026/01)": 2025
}

notable_times_relative = {}
notable_times_aspline = {}

for event in notable_times_raw:
    notable_times_relative[event] = \
        (notable_times_raw[event] - t_r_start)
    
# %%
# %% Multi-coefficient resolution diagnostic: 6 x 4 imshow panel

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User choices
# -----------------------------
inject_indices = [100, 200, 300, 400]
inject_period = 10
inject_frequency = 1 / inject_period

use_reliable_window = False
start_reliable_idx, end_reliable_idx = 1, 1

# -----------------------------
# Plot style
# -----------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# -----------------------------
# Read matrices once
# -----------------------------
P = R_Read_In("P", "MF")

H = {
    "MF": R_Read_In("H", "MF"),
    "SV": R_Read_In("H", "SV"),
    "SA": R_Read_In("H", "SA"),
}

fields = ["MF", "SV", "SA"]

# -----------------------------
# Choose plotting window
# -----------------------------
if use_reliable_window:
    time_slice = slice(start_reliable_idx, end_reliable_idx)
else:
    time_slice = slice(None)

times_plot = times_dyear[time_slice]

# -----------------------------
# Convert relative notable times
# to decimal years
# -----------------------------
record_start_time = times_dyear[0]

notable_times_abs = {
    key: record_start_time + value
    for key, value in notable_times_relative.items()
}

# -----------------------------
# Precompute data for each injected coefficient
# -----------------------------
all_results = {}

for inject_idx in inject_indices:

    # Generate exact input signal
    gnm, gsv_nm, gsa_nm = Simple_G_Generate_One(
        inject_idx,
        inject_frequency,
        times_dyear
    )

    original = {
        "MF": gnm,
        "SV": gsv_nm,
        "SA": gsa_nm,
    }

    # Apply resolution in spline space
    gnm_spl = P @ gnm
    gnm_filt_spl = Sparse_R_Apply_One(gnm_spl, inject_idx)

    # Reconstruct recovered MF, SV and SA
    recovered = {
        field: H[field] @ gnm_filt_spl
        for field in fields
    }

    original_plot = {
        field: original[field][time_slice, :]
        for field in fields
    }

    recovered_plot = {
        field: recovered[field][time_slice, :]
        for field in fields
    }

    all_results[inject_idx] = {
        "original": original_plot,
        "recovered": recovered_plot,
    }

# -----------------------------
# Plot
# 6 rows:
#   0 MF true
#   1 MF resolved
#   2 SV true
#   3 SV resolved
#   4 SA true
#   5 SA resolved
# 4 columns = input indices
# -----------------------------
fig, axes = plt.subplots(
    6,
    4,
    figsize=(8.27, 11.69),   # A4 portrait in inches
    sharex=True,
    sharey=True,
    constrained_layout=True
)

n_coeff = all_results[inject_indices[0]]["original"]["MF"].shape[1]

x_min = -0.5
x_max = n_coeff - 0.5

y_min = times_plot[0]
y_max = times_plot[-1]

extent = [x_min, x_max, y_min, y_max]

row_map = [
    ("MF", "original",  "MF true / exact",        "RdBu_r"),
    ("MF", "recovered", "MF recovered post $R$",  "PuOr_r"),
    ("SV", "original",  "SV true / exact",        "RdBu_r"),
    ("SV", "recovered", "SV recovered post $R$",  "PuOr_r"),
    ("SA", "original",  "SA true / exact",        "RdBu_r"),
    ("SA", "recovered", "SA recovered post $R$",  "PuOr_r"),
]

for col, inject_idx in enumerate(inject_indices):

    for row, (field, data_key, row_label, cmap_name) in enumerate(row_map):

        ax = axes[row, col]
        Z = all_results[inject_idx][data_key][field]

        vmax = np.nanmax(np.abs(Z))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        im = ax.imshow(
            Z,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap_name,
            vmin=-vmax,
            vmax=vmax,
            interpolation="none"
        )

        # Degree boundaries
        for idx in degrees_idx:
            if x_min <= idx <= x_max:
                ax.axvline(
                    idx,
                    color="deepskyblue",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.2
                )

        # Notable times
        for t_abs in notable_times_abs.values():
            if y_min <= t_abs <= y_max:
                ax.axhline(
                    t_abs,
                    color="white",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.9
                )

        # Column titles only on top row
        if row == 0:
            ax.set_title(
                f"input $s={inject_idx}$",
                fontsize=12
            )

        # Row labels only on first column
        if col == 0:
            ax.set_ylabel(
                f"{row_label}\n\ntime / decimal year",
                fontsize=11
            )

        # Bottom x-labels
        if row == 5:
            ax.set_xlabel(
                "Gauss coefficient index",
                fontsize=11
            )

        # One colourbar per panel
        cbar = fig.colorbar(
            im,
            ax=ax,
            shrink=0.82,
            pad=0.01
        )
        cbar.ax.tick_params(labelsize=9)

fig.suptitle(
    f"Resolution diagnostics for multiple injected coefficients "
    f"(period = {inject_period:g} yr)",
    fontsize=14
)

plt.savefig(
    FIG_DIR / f"multi_coeff_resolution_imshow_T{inject_period:g}.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
# %% Single-coefficient resolution diagnostic: 3 x 2 imshow panel

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User choices
# -----------------------------
inject_idx = 300
inject_period = 10
inject_frequency = 1 / inject_period

use_reliable_window = False
start_reliable_idx, end_reliable_idx = 1, 1

# -----------------------------
# Read matrices once
# -----------------------------
P = R_Read_In("P", "MF")

H = {
    "MF": R_Read_In("H", "MF"),
    "SV": R_Read_In("H", "SV"),
    "SA": R_Read_In("H", "SA"),
}

# -----------------------------
# Generate exact input signal
# -----------------------------
gnm, gsv_nm, gsa_nm = Simple_G_Generate_One(
    inject_idx,
    inject_frequency,
    times_dyear
)

original = {
    "MF": gnm,
    "SV": gsv_nm,
    "SA": gsa_nm,
}

# -----------------------------
# Apply resolution in spline space
# -----------------------------
gnm_spl = P @ gnm
gnm_filt_spl = Sparse_R_Apply_One(gnm_spl, inject_idx)

# Reconstruct recovered MF, SV and SA
recovered = {
    field: H[field] @ gnm_filt_spl
    for field in ["MF", "SV", "SA"]
}

# -----------------------------
# Choose plotting window
# -----------------------------
if use_reliable_window:
    time_slice = slice(start_reliable_idx, end_reliable_idx)
else:
    time_slice = slice(None)

times_plot = times_dyear[time_slice]

original_plot = {
    field: original[field][time_slice, :]
    for field in ["MF", "SV", "SA"]
}

recovered_plot = {
    field: recovered[field][time_slice, :]
    for field in ["MF", "SV", "SA"]
}

# -----------------------------
# Convert relative notable times
# to decimal years
# -----------------------------
record_start_time = times_dyear[0]

notable_times_abs = {
    key: record_start_time + value
    for key, value in notable_times_relative.items()
}

# -----------------------------
# Plot
# -----------------------------
fields = ["MF", "SV", "SA"]

fig, axes = plt.subplots(
    3,
    2,
    figsize=(17, 14),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

n_coeff = original_plot["MF"].shape[1]

x_min = -0.5
x_max = n_coeff - 0.5

y_min = times_plot[0]
y_max = times_plot[-1]

extent = [
    x_min,
    x_max,
    y_min,
    y_max
]

for row, field in enumerate(fields):

    exact_data = original_plot[field]
    recovered_data = recovered_plot[field]

    # Independent symmetric colour scales
    vmax_exact = np.nanmax(np.abs(exact_data))
    vmax_recovered = np.nanmax(np.abs(recovered_data))

    if not np.isfinite(vmax_exact) or vmax_exact == 0:
        vmax_exact = 1.0

    if not np.isfinite(vmax_recovered) or vmax_recovered == 0:
        vmax_recovered = 1.0

    row_data = [
        {
            "data": exact_data,
            "title": f"{field} true / exact",
            "cmap": "RdBu_r",
            "vmax": vmax_exact,
            "cbar_label": f"Exact {field} amplitude",
        },
        {
            "data": recovered_data,
            "title": f"{field} recovered post $R$",
            "cmap": "PuOr_r",
            "vmax": vmax_recovered,
            "cbar_label": f"Recovered {field} amplitude",
        },
    ]

    for col, plot_info in enumerate(row_data):

        ax = axes[row, col]

        im = ax.imshow(
            plot_info["data"],
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=plot_info["cmap"],
            vmin=-plot_info["vmax"],
            vmax=plot_info["vmax"],
            interpolation="none"
        )

        # Gauss-coefficient degree boundaries
        for idx in degrees_idx:
            if x_min <= idx <= x_max:
                ax.axvline(
                    idx,
                    color="deepskyblue",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.5
                )

        # Notable times
        for t_abs in notable_times_abs.values():
            if y_min <= t_abs <= y_max:
                ax.axhline(
                    t_abs,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.9
                )

        ax.set_title(plot_info["title"])

        if col == 0:
            ax.set_ylabel("time / decimal year")

        if row == 2:
            ax.set_xlabel("Gauss coefficient index")

        # Independent colourbar for every panel
        cbar = fig.colorbar(
            im,
            ax=ax,
            shrink=0.9,
            pad=0.02
        )

        cbar.set_label(
            plot_info["cbar_label"]
        )

fig.suptitle(
    f"Single-coefficient resolution diagnostic: "
    f"input index $s={inject_idx}$, "
    f"period = {inject_period:g} yr",
    fontsize=16
)

plt.savefig(
     FIG_DIR
     / f"single_coeff_resolution_imshow_s{inject_idx}_T{inject_period:g}.png",
     dpi=600,
     bbox_inches="tight"
 )

plt.show()
# %% RUN DIAGNOSTICS FOR MF, SV, SA

coeff_indices = degrees_idx - 1
coeff_periods = np.array([2, 4, 8, 12, 16, 20, 24, 28, 36, 52, 68])
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
                (sin_fit_result_s["frequency"] - f) / f
            )

            f_stats[field]["sin_fit_r2"].append(
                sin_fit_result_s["r2"]
            )

    # store one list per frequency
    for field in ["MF", "SV", "SA"]:
        for key in summary_stats[field]:
            summary_stats[field][key].append(f_stats[field][key])

# %%

from matplotlib.colors import LogNorm

import numpy as np
import matplotlib.pyplot as plt


def get_global_diagnostic_scales(summary_stats, fields=("MF", "SV", "SA")):
    """
    Compute global colour limits across MF, SV, SA for each diagnostic.
    """

    scales = {}

    # Fractions and R2 are naturally 0 to 1
    scales["total_retained_power_frac"] = (0, 1)
    scales["s_power_vs_total_power_frac"] = (0, 1)
    scales["sin_fit_r2"] = (0, 1)

    # Signed frequency error: use symmetric colour scale around zero
    all_freq_err = []

    for field in fields:
        Z = np.asarray(summary_stats[field]["sin_fit"], dtype=float)
        all_freq_err.append(Z[np.isfinite(Z)])

    all_freq_err = np.concatenate(all_freq_err)

    freq_err_absmax = np.nanmax(np.abs(all_freq_err))

    scales["sin_fit"] = (-freq_err_absmax, freq_err_absmax)

    return scales

def plot_field_diagnostics(
    summary_stats,
    field,
    coeff_indices,
    coeff_periods,
    degrees_idx=None,
    global_scales=None,
    figsize=(5, 16),
):
    """
    Plot pcolormesh diagnostic suite for one field: MF, SV, or SA.

    summary_stats[field][diagnostic] should have shape:
        (n_periods, n_coefficients)

    Period axis is plotted in real years, linearly from 0 to 100 yr.
    """

    diagnostics = {
        "total_retained_power_frac": {
            "title": "Total retained power fraction",
            "cbar": r"$P_\mathrm{filtered} / P_\mathrm{input}$",
            "default_vmin": 0,
            "default_vmax": 1,
            "cmap": "viridis",
        },
        "s_power_vs_total_power_frac": {
            "title": "Injected coefficient power fraction",
            "cbar": r"$P_s / P_\mathrm{filtered}$",
            "default_vmin": 0,
            "default_vmax": 1,
            "cmap": "viridis",
        },
        "sin_fit": {
            "title": "Signed normalised frequency error",
            "cbar": r"$(\hat{f} - f) / f$",
            "default_vmin": -1,
            "default_vmax": 1,
            "cmap": "RdBu_r",
        },
        "sin_fit_r2": {
            "title": "Sinusoid fit $R^2$",
            "cbar": r"$R^2$",
            "default_vmin": 0.9,
            "default_vmax": 1,
            "cmap": "viridis",
        },
    }

    x_vals = np.asarray(coeff_indices)
    y_vals = np.asarray(coeff_periods)

    fig, ax = plt.subplots(
        4, 1,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    ax = ax.ravel()

    for a, (key, meta) in zip(ax, diagnostics.items()):

        Z = np.asarray(summary_stats[field][key], dtype=float)

        if global_scales is not None:
            vmin, vmax = global_scales[key]
        else:
            vmin = meta["default_vmin"]
            vmax = meta["default_vmax"]

        if key == "sin_fit":
            absmax = np.nanmax(np.abs(Z))
            vmin, vmax = -0.5, 0.5

        if key == "sin_fit_r2":
            absmax = np.nanmax(np.abs(Z))
            vmin, vmax = 0.9, 1



        pcm = a.pcolormesh(
            x_vals,
            y_vals,
            Z,
            shading="auto",
            cmap=meta["cmap"],
            vmin=vmin,
            vmax=vmax,
        )

        if degrees_idx is not None:
            for idx in degrees_idx:
                if x_vals[0] <= idx <= x_vals[-1]:
                    a.axvline(
                        idx,
                        color="red",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.8,
                    )

        a.set_title(meta["title"])
        a.set_xlabel("input Gauss coefficient index $s$")
        a.set_ylabel("input period / yr")

        a.set_ylim(0, 80)
        a.grid(True, alpha=0.25)

        cbar = fig.colorbar(pcm, ax=a)
        cbar.set_label(meta["cbar"])

    fig.suptitle(f"{field} resolution diagnostics", fontsize=16)

    plt.savefig(f"{FIG_DIR}/{field}_R_diagnostics.png", dpi=300, bbox_inches="tight")


    plt.show()

    return fig, ax
# %%
global_scales = get_global_diagnostic_scales(
    summary_stats,
    fields=("MF", "SV", "SA")
)

for field in ["MF", "SV", "SA"]:
    plot_field_diagnostics(
        summary_stats,
        field=field,
        coeff_indices=coeff_indices,
        coeff_periods=coeff_periods,
        degrees_idx=degrees_idx,
        global_scales=global_scales,
    )
# %%
print(notable_times_relative)
# %%
coeff_indices = degrees_idx - 1

coeff_periods = np.array([
    2, 4, 8, 12, 16, 20, 24, 28, 36, 52, 68
])

coeff_frequency = 1 / coeff_periods

start_reliable_idx, end_reliable_idx = 9, 58 # NEEDS TO BE THE RELIABLE TIME PERIODS!!!
times_reliable = times_dyear[start_reliable_idx:end_reliable_idx]

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
                start_reliable_idx:end_reliable_idx,
                :
            ]

            filtered_clip = filtered[field][
                start_reliable_idx:end_reliable_idx,
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
        ax.set_ylim(0, 80)
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
        f"{FIG_DIR}/{field}_R_diagnostics.png",
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
