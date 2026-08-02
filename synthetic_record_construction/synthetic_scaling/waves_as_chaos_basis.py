'''
waves_making_chaos.py

file to investigate to what extent the wave modes
provided can recreate the signal character and power
spectral behaviour of CHAOS-8.6
'''

# %% PREAMBLE

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

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
from src.msc_thesis.synSetup import (
    good_record_slice,
    r_cmb,
    r_earth,
    times_dyear,
    times_mjd2000,
)
from src.msc_thesis.synUtils import (
    H_sv,
    Lowes_Degree_PSD_All_Degrees,
    n_Gauss_Coeffs,
)
from src.msc_thesis.synConstruct import Component_Load

# defining radius choice for power to be referenced to
r_choice = r_cmb

# used time interval
used_times_dyear = times_dyear[good_record_slice]

mode_numbers = np.arange(1, 63, 1)
M = len(mode_numbers) # number of modes
Nt = len(used_times_dyear) # number of time points
Ng = n_Gauss_Coeffs(20)

# setting gauss coefficient degree fitting over
N_min = 1
N_max = 11
N_slice = slice(Nt*(N_min**2 - 1), Nt*(n_Gauss_Coeffs(N_max)))

# %% WEIGHTING FOR LOWES POWER

# correct (checked) Lowes weight construction
def Lowes_Coefficient_Weights(lmax, a, r):
    weights = []

    for n in range(1, lmax + 1):
        degree_weight = (
            (n + 1)
            * (a / r) ** (2 * n + 4)
        )

        # There are 2n + 1 Gauss coefficients at degree n
        weights.extend(
            [degree_weight] * (2 * n + 1)
        )

    return np.asarray(weights)

# %% LOADING IN CHAOS DATA

chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=20,
    deriv=1,
    extrapolate="off",
)

gnm_chaos = chaos_sv_gnm[good_record_slice]

# %% LOADING IN SYNTHETIC DATA 

def Synthetic_G_Extract(mode_number):

    file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

    with h5py.File(file_path, "r") as h5_file:

        gnm_spl_re = np.asarray(
        h5_file[f"mode_{mode_number}/without_decay"][()]
    )   
        gnm_spl_im = np.asarray(
        h5_file[f"mode_{mode_number}/without_decay_imaginary"][()]
    )
        
    # converting these into SV G

    gnm_re = H_sv @ gnm_spl_re
    gnm_im = H_sv @ gnm_spl_im

    # return Gnm of SV, clipped to high quality part of record
    return(gnm_re[good_record_slice], gnm_im[good_record_slice])


# APPLYING LOWES WEIGHTING FOR DATA SET

# creating appropriate weighting vector for lmax=20
# can adjust as necessary
weights = Lowes_Coefficient_Weights(
    lmax=20,
    a=r_earth,
    r=r_cmb,
)

# square rooting weights for integrating with workflow
sqrt_weights = np.sqrt(weights)[None, :]

def Weighted_Ravel(x, w=sqrt_weights):

    # applying to CHAOS
    # x = G ~ (Nt, Ng), rt(w) ~ (Ng, 1)
    x_unravelled = (x * w) # ~ (Nt, Ng)
    x_ravelled = x_unravelled.ravel(order="F") # ~ (Nt*(Ng))

    return(x_ravelled)

# CONSTRUCTING y

y = Weighted_Ravel(gnm_chaos)

# %% CONSTRUCTING W

def W_Construct(mode_numbers):

    # initialise W for storage
    W = np.zeros(((Nt*Ng), 2*M))
    X = np.zeros(((Nt*Ng), 2*M))

    for i, mode_number in enumerate(mode_numbers):
        # extract real and imaginary part
        gnm_mode_re, gnm_mode_im = Synthetic_G_Extract(mode_number)

        # weight, ravel both
        x_re = gnm_mode_re.ravel(order="F")
        x_im = gnm_mode_im.ravel(order="F")
        x_re_weighted = Weighted_Ravel(gnm_mode_re)
        x_im_weighted = Weighted_Ravel(gnm_mode_im)

        W[:, 2*i] = x_re_weighted
        W[:, 2*i+1] = x_im_weighted
        X[:, 2*i] = x_re
        X[:, 2*i+1] = x_im
    
    return(W, X)


# %% LEAST SQUARES PROJECTION

# constructed W, matrix where each column
# is a lowes weighted mode basis vector
W, X = W_Construct(mode_numbers)

# perform least squares to get coefficients
coeffs, residuals, rank, singular_values = np.linalg.lstsq(
    W[N_slice, :],
    y[N_slice],
    rcond=None,
)

# %% MATRIX RESHAPE + UNWEIGHT CODE
def Reshaper(x_ravel):

    x = x_ravel.reshape\
    (np.shape(gnm_chaos), order="F") # ~ (Nt*(Ng))

    return(x)

def Unweighter(x_weighted, w=sqrt_weights):

    x = (x_weighted * (1/w)) # ~ (Nt, Ng)

    return(x)

def Unweighted_Reshape(x_weighted_ravel, w=sqrt_weights):

    x = Unweighter(Reshaper(x_weighted_ravel))
    
    return(x)
# %% COMPUTING y_projected

# getting y projected
y_projected = X @ coeffs

# obtaining corresponding Gnm time series
gnm_chaos_bf = Reshaper(y_projected)


# confirms weighted ravel is correctly undone
print(np.allclose(gnm_chaos, Unweighted_Reshape(Weighted_Ravel(gnm_chaos))))
# %%

c_psd, c_f = Lowes_Degree_PSD_All_Degrees\
    (gnm_chaos, a=r_earth, r=r_choice)
proj_psd, proj_f = Lowes_Degree_PSD_All_Degrees\
    (gnm_chaos_bf, a=r_earth, r=r_choice)


# %%
vmin = np.min(c_psd)
vmax = np.max(c_psd)

# %%
plt.imshow(c_psd, vmin=vmin)
plt.plot

# %%
plt.imshow(proj_psd, vmax=vmax)
plt.plot
# %%
# Projected result in weighted space
y_weighted_projected = W @ coeffs

# Projected result directly in unweighted space
y_projected = X @ coeffs
gnm_chaos_bf = Reshaper(y_projected)

# Independently unweight the weighted projection
gnm_chaos_bf_check = Unweighted_Reshape(
    y_weighted_projected
)

print(
    "Weight/unweight identity:",
    np.allclose(
        gnm_chaos,
        Unweighted_Reshape(Weighted_Ravel(gnm_chaos)),
    ),
)

print(
    "Projected reconstruction consistent:",
    np.allclose(
        gnm_chaos_bf,
        gnm_chaos_bf_check,
    ),
)
# %%
weighted_residual = y[N_slice] - y_weighted_projected[N_slice]

captured_fraction = (
    1
    - np.linalg.norm(weighted_residual)**2
    / np.linalg.norm(y[N_slice])**2
)

relative_error = (
    np.linalg.norm(weighted_residual)
    / np.linalg.norm(y[N_slice])
)

print(f"Rank: {rank}/{W.shape[1]}")
print(f"Weighted energy captured: {captured_fraction:.4%}")
print(f"Relative weighted error: {relative_error:.4%}")
print(f"Condition number: {np.linalg.cond(W[N_slice, :]):.3e}")
# %% REPEATING FOR PSD

# %% PSD PROJECTION SETUP

from scipy.optimize import nnls

r_choice = r_cmb

used_times_dyear = times_dyear[good_record_slice]

mode_numbers = np.arange(1, 63)
M = len(mode_numbers)

N_min = 1
N_max = 20

f_min = 0
f_max = 0.33

# PSD row 0 corresponds to degree 1.
degree_slice = slice(N_min - 1, N_max)

Nn = N_max - N_min + 1

print(f"Number of fitted degrees: {Nn}")

# %% LOAD CHAOS DATA

chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"

chaos_model = cp.load_CHAOS_matfile(
    str(chaos_file)
)

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=20,
    deriv=1,
    extrapolate="off",
)

gnm_chaos = chaos_sv_gnm[good_record_slice]

Nt, Ng = gnm_chaos.shape

chaos_psd, chaos_f = Lowes_Degree_PSD_All_Degrees(
    gnm_chaos,
    a=r_earth,
    r=r_choice,
)

# Exclude zero frequency unless you specifically want to fit
# the record-mean SV contribution.
f_mask = (
    (chaos_f > f_min)
    & (chaos_f <= f_max)
)

f_mask[1] = False

print(f"CHAOS PSD shape: {chaos_psd.shape}")
print(f"Number of fitted frequencies: {np.sum(f_mask)}")

# %% LOAD SYNTHETIC DATA

def Synthetic_G_Extract(mode_number):

    file_path = (
        Path(FELIX_DIR)
        / "R_splines_arbitrary.h5"
    )

    with h5py.File(file_path, "r") as h5_file:

        gnm_spl_re = np.asarray(
            h5_file[
                f"mode_{mode_number}/without_decay"
            ][()]
        )

        gnm_spl_im = np.asarray(
            h5_file[
                f"mode_{mode_number}/without_decay_imaginary"
            ][()]
        )

    gnm_re = H_sv @ gnm_spl_re
    gnm_im = H_sv @ gnm_spl_im

    return (
        gnm_re[good_record_slice],
        gnm_im[good_record_slice],
    )


# %% CONSTRUCT PSD BASIS

def PSD_Basis_Construct(mode_numbers):

    # Full PSD for every mode:
    # shape = (M, Ndegree, Nfrequency)
    mode_psds = np.zeros(
        (len(mode_numbers),) + chaos_psd.shape
    )

    # Real-quadrature time series, retained only if you later
    # want to construct a particular coherent realization.
    mode_gnm = np.zeros(
        (len(mode_numbers), Nt, Ng)
    )

    for i, mode_number in enumerate(mode_numbers):

        gnm_mode_re, _ = Synthetic_G_Extract(
            mode_number
        )

        mode_psd, mode_f = (
            Lowes_Degree_PSD_All_Degrees(
                gnm_mode_re,
                a=r_earth,
                r=r_choice,
            )
        )

        if mode_psd.shape != chaos_psd.shape:
            raise ValueError(
                f"Mode {mode_number}: PSD shape "
                f"{mode_psd.shape} does not match CHAOS "
                f"shape {chaos_psd.shape}."
            )

        if not np.allclose(mode_f, chaos_f):
            raise ValueError(
                f"Mode {mode_number} has a different "
                "frequency grid from CHAOS."
            )

        mode_psds[i] = mode_psd
        mode_gnm[i] = gnm_mode_re

    return mode_psds, mode_gnm


mode_psds, mode_gnm = PSD_Basis_Construct(
    mode_numbers
)
# %% CONSTRUCT PSD LEAST-SQUARES SYSTEM

# Target PSD region: shape (Nn, Nf_fit)
chaos_psd_fit = chaos_psd[
    degree_slice,
    :
][:, f_mask]

# Mode PSD region:
# shape (M, Nn, Nf_fit)
mode_psds_fit = mode_psds[
    :,
    degree_slice,
    :
][:, :, f_mask]

# Target vector: shape (Ncells,)
y_psd = chaos_psd_fit.ravel()

# Design matrix: shape (Ncells, M)
A_psd = mode_psds_fit.reshape(
    M,
    -1,
).T

print(f"PSD target shape: {y_psd.shape}")
print(f"PSD design matrix shape: {A_psd.shape}")

# %% NONNEGATIVE LOG-SPACE PSD FIT

from scipy.optimize import least_squares, nnls

# A small positive floor is required because log10(0) is undefined.
#
# Setting it relative to the largest target PSD prevents extremely
# small or zero cells from dominating the optimisation.
floor_fraction = 1e-6
psd_floor = floor_fraction * np.max(y_psd)

def Log_PSD_Residual(coeffs):
    """
    Log10 difference between the additive wave-PSD model and CHAOS.

    coeffs[k] is the nonnegative PSD/power multiplier for mode k.
    """

    predicted_psd = A_psd @ coeffs

    residual = (
        np.log10(predicted_psd + psd_floor)
        - np.log10(y_psd + psd_floor)
    )

    return residual


# Obtain a stable nonnegative initial solution using ordinary NNLS.
initial_coeffs, _ = nnls(
    A_psd,
    y_psd,
)

# Give modes assigned exactly zero by NNLS a very small positive
# starting value. The bounds still allow them to return to zero.
positive_initial_scale = max(
    np.max(initial_coeffs),
    1.0,
)

initial_coeffs = np.maximum(
    initial_coeffs,
    1e-12 * positive_initial_scale,
)

log_fit_result = least_squares(
    Log_PSD_Residual,
    x0=initial_coeffs,
    bounds=(0.0, np.inf),
    method="trf",
    x_scale="jac",
    max_nfev=20000,
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-12,
)

psd_coeffs = log_fit_result.x
residual_norm = np.linalg.norm(
    log_fit_result.fun
)

print(
    "Log-space optimisation successful:",
    log_fit_result.success,
)

print(
    "Optimiser message:",
    log_fit_result.message,
)

print(
    "Number of function evaluations:",
    log_fit_result.nfev,
)

print(
    "Number of active modes:",
    np.count_nonzero(
        psd_coeffs
        > 1e-10 * np.max(psd_coeffs)
    ),
)

print(
    "Log-space residual norm:",
    residual_norm,
)

print(
    "Log-space RMSE:",
    np.sqrt(
        np.mean(log_fit_result.fun**2)
    ),
)
print(
    "Number of active modes:",
    np.count_nonzero(psd_coeffs > 1e-12),
)

print(
    "NNLS residual norm:",
    residual_norm,
)

print(
    "Minimum fitted coefficient:",
    psd_coeffs.min(),
)
# %% CONSTRUCT ACTUAL AMPLITUDE-SCALED WAVE SUM

# NNLS coefficients multiply PSD, so the corresponding
# signal-amplitude multipliers are their square roots.
mode_amplitudes = np.sqrt(
    np.maximum(psd_coeffs, 0.0)
)

# mode_gnm has shape:
#     (M, Nt, Ng)
#
# Contract over the mode axis to form:
#     sum_k amplitude_k * gnm_k(t)
#
# This uses the stored real quadrature/phase of every mode.
gnm_nnls_sum = np.tensordot(
    mode_amplitudes,
    mode_gnm,
    axes=(0, 0),
)

# Compute the PSD of the actual coherent time-series sum.
chaos_psd_actual, actual_f = (
    Lowes_Degree_PSD_All_Degrees(
        gnm_nnls_sum,
        a=r_earth,
        r=r_choice,
    )
)

if not np.allclose(actual_f, chaos_f):
    raise ValueError(
        "The actual summed-signal PSD has a different "
        "frequency grid from CHAOS."
    )

# %% RECONSTRUCT PROJECTED PSD

# Full degree-frequency PSD predicted by the additive model
chaos_psd_projected = np.tensordot(
    psd_coeffs,
    mode_psds,
    axes=(0, 0),
)

# Selected-region reconstruction
y_psd_projected = A_psd @ psd_coeffs

chaos_psd_projected_fit = (
    y_psd_projected.reshape(
        chaos_psd_fit.shape
    )
)

psd_residual = (
    y_psd - y_psd_projected
)

captured_fraction = (
    1
    - np.linalg.norm(psd_residual)**2
    / np.linalg.norm(y_psd)**2
)

relative_error = (
    np.linalg.norm(psd_residual)
    / np.linalg.norm(y_psd)
)

print(
    f"PSD squared-norm fraction captured: "
    f"{captured_fraction:.4%}"
)

print(
    f"Relative PSD error: "
    f"{relative_error:.4%}"
)

print(
    "Metric identity satisfied:",
    np.allclose(
        captured_fraction,
        1 - relative_error**2,
    ),
)

# %% PLOT PSD PROJECTION

positive = np.concatenate([
    chaos_psd_fit[chaos_psd_fit > 0],
    chaos_psd_projected_fit[
        chaos_psd_projected_fit > 0
    ],
])

floor = positive.min()

chaos_log = np.log10(
    np.maximum(chaos_psd_fit, floor)
)

projected_log = np.log10(
    np.maximum(
        chaos_psd_projected_fit,
        floor,
    )
)

log_residual = (
    projected_log - chaos_log
)

vmin = 0
vmax = 8

error_limit = np.max(
    np.abs(log_residual)
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

im = axes[0].imshow(
    chaos_log,
    origin="lower",
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
    cmap="viridis",
)

axes[1].imshow(
    projected_log,
    origin="lower",
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
    cmap="viridis",
)

err = axes[2].imshow(
    (log_residual),
    origin="lower",
    aspect="auto",
    vmin=-1.5,
    vmax=1.5,
    cmap="RdBu_r", #RdBu_r
)

axes[0].set_title("CHAOS PSD")
axes[1].set_title("NNLS wave PSD")
axes[2].set_title("Log-space difference")

axes[0].set_ylabel("Degree index")

fig.colorbar(
    im,
    ax=axes[:2],
    label=r"$\log_{10}$ Lowes PSD",
)

fig.colorbar(
    err,
    ax=axes[2],
    label=r"$\log_{10}P_{\rm fit}-\log_{10}P_{\rm CHAOS}$",
)

plt.show()
# %%
# %% COMPARE LINEAR AND ACTUAL PSD ERRORS

linear_psd_fit = chaos_psd_projected[
    degree_slice,
    :
][:, f_mask]

actual_psd_fit = chaos_psd_actual[
    degree_slice,
    :
][:, f_mask]

linear_residual = (
    chaos_psd_fit - linear_psd_fit
)

actual_residual = (
    chaos_psd_fit - actual_psd_fit
)

linear_relative_error = (
    np.linalg.norm(linear_residual)
    / np.linalg.norm(chaos_psd_fit)
)

actual_relative_error = (
    np.linalg.norm(actual_residual)
    / np.linalg.norm(chaos_psd_fit)
)

linear_fit_fraction = (
    1 - linear_relative_error**2
)

actual_fit_fraction = (
    1 - actual_relative_error**2
)

print(
    "Linear additive PSD fit:"
    f"\n  Relative error = {linear_relative_error:.4%}"
    f"\n  Squared-norm fit fraction = {linear_fit_fraction:.4%}"
)

print(
    "Actual summed-signal PSD:"
    f"\n  Relative error = {actual_relative_error:.4%}"
    f"\n  Squared-norm fit fraction = {actual_fit_fraction:.4%}"
)
# %% FULL-RANGE THREE-PANEL PSD COMPARISON

from matplotlib.patches import Rectangle

psds_to_plot = [
    chaos_psd,
    chaos_psd_projected,
    chaos_psd_actual,
]

titles = [
    "CHAOS-8.6 SV PSD",
    "NNLS additive PSD",
    "PSD of actual scaled wave sum",
]

# Establish one positive plotting floor across all three PSDs.
positive_values = np.concatenate([
    psd[psd > 0]
    for psd in psds_to_plot
])

if positive_values.size == 0:
    raise ValueError("No positive PSD values are available to plot.")

floor = positive_values.min()

log_psds = [
    np.log10(np.maximum(psd, floor))
    for psd in psds_to_plot
]

# Use the full CHAOS range as the common colour scale.
vmin = 1
vmax = 8

degrees = np.arange(
    1,
    chaos_psd.shape[0] + 1,
)

# Actual lowest and highest included frequencies.
f_fit_values = chaos_f[f_mask]

if f_fit_values.size == 0:
    raise ValueError(
        "The frequency mask contains no frequency bins."
    )

f_box_min = f_fit_values.min()
f_box_max = f_fit_values.max()

# Rectangle edges in degree coordinates.
degree_box_min = N_min - 0.5
degree_box_max = N_max + 0.5

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5.5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ax, log_psd, title in zip(
    axes,
    log_psds,
    titles,
):
    im = ax.pcolormesh(
        chaos_f,
        degrees,
        log_psd,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    fit_box = Rectangle(
        (f_box_min, degree_box_min),
        f_box_max - f_box_min,
        degree_box_max - degree_box_min,
        fill=False,
        edgecolor="black",
        linewidth=2,
        linestyle=":",
        zorder=10,
    )

    ax.add_patch(fit_box)

    ax.set_title(title)
    ax.set_xlabel(
        r"Frequency (cycles yr$^{-1}$)"
    )

axes[0].set_ylabel(
    "Spherical harmonic degree"
)

fig.colorbar(
    im,
    ax=axes,
    label=r"$\log_{10}$ Lowes PSD",
)
for ax in axes:
    ax.set_xlim(0, 0.5)
plt.show()

# %% MODE PERIODS AND SCALED MEAN LOWES POWER

lowes_weights = Lowes_Coefficient_Weights(
    lmax=20,
    a=r_earth,
    r=r_choice,
)

if lowes_weights.size != Ng:
    raise ValueError(
        f"Lowes weight vector has length "
        f"{lowes_weights.size}, expected {Ng}."
    )

# Instantaneous power:
# shape (M, Nt)
mode_instantaneous_power = np.sum(
    mode_gnm**2
    * lowes_weights[None, None, :],
    axis=2,
)

# Original arbitrary-amplitude mean power:
# shape (M,)
mode_mean_power = np.mean(
    mode_instantaneous_power,
    axis=1,
)

# Power scales with amplitude squared, and therefore directly
# with the NNLS PSD coefficients.
mode_scaled_mean_power = (
    psd_coeffs * mode_mean_power
)

mode_periods = np.zeros(M)

for i, mode_number in enumerate(mode_numbers):

    mode_data = Component_Load(mode_number)

    omega = np.abs(
        mode_data["eigenvalue"].imag
    )

    if np.isclose(omega, 0.0):
        mode_periods[i] = np.inf
    else:
        mode_periods[i] = (
            2 * np.pi / omega
        )

# %% SCALED POWER VERSUS MODE PERIOD

# NNLS solutions are often sparse. Exclude exactly zero-power
# modes from a logarithmic plot.
active_mask = (
    psd_coeffs > 1e-12
) & np.isfinite(mode_periods) \
  & (mode_scaled_mean_power > 0)

fig, ax = plt.subplots(
    figsize=(8, 5.5),
    constrained_layout=True,
)

scatter = ax.scatter(
    mode_periods[active_mask],
    mode_scaled_mean_power[active_mask],
    c=psd_coeffs[active_mask],
    cmap="viridis",
    s=60,
    edgecolor="black",
    linewidth=0.5,
)

ax.set_xscale("linear")
ax.set_yscale("log")

ax.set_xlabel("Mode period (years)")
ax.set_ylabel(
    "NNLS-scaled mean instantaneous Lowes power"
)

ax.set_title(
    "Mode power selected by PSD-space NNLS"
)

fig.colorbar(
    scatter,
    ax=ax,
    label=r"NNLS PSD coefficient $q_k$",
)

plt.show()

print(
    f"Active modes plotted: "
    f"{np.sum(active_mask)}/{M}"
)

for i in np.where(active_mask)[0]:
    ax.annotate(
        str(mode_numbers[i]),
        (
            mode_periods[i],
            mode_scaled_mean_power[i],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )
# %%
water_level = 1e-3 * np.max(y_psd)

row_scale = np.maximum(
    y_psd,
    water_level,
)

A_relative = A_psd / row_scale[:, None]
y_relative = y_psd / row_scale

relative_coeffs, relative_residual = nnls(
    A_relative,
    y_relative,
)

unrestricted_coeffs, *_ = np.linalg.lstsq(
    A_psd,
    y_psd,
    rcond=None,
)

unrestricted_fit = (
    A_psd @ unrestricted_coeffs
)

unrestricted_relative_error = (
    np.linalg.norm(y_psd - unrestricted_fit)
    / np.linalg.norm(y_psd)
)

print(
    "Negative coefficients:",
    np.sum(unrestricted_coeffs < 0),
)

print(
    "Unrestricted relative error:",
    unrestricted_relative_error,
)
# %%
negative_mask = unrestricted_coeffs < 0

negative_l1_fraction = (
    np.sum(np.abs(unrestricted_coeffs[negative_mask]))
    / np.sum(np.abs(unrestricted_coeffs))
)

negative_l2_fraction = (
    np.linalg.norm(unrestricted_coeffs[negative_mask])
    / np.linalg.norm(unrestricted_coeffs)
)

print(
    f"Negative coefficient count: "
    f"{negative_mask.sum()}/{M}"
)

print(
    f"Negative L1 fraction: "
    f"{negative_l1_fraction:.4%}"
)

print(
    f"Negative L2 fraction: "
    f"{negative_l2_fraction:.4%}"
)

print(
    "Coefficient range:",
    unrestricted_coeffs.min(),
    unrestricted_coeffs.max(),
)
# %%
# %% UNRESTRICTED LS: PERIOD VERSUS SCALED POWER

# unrestricted_coeffs were obtained from:
#
# unrestricted_coeffs, *_ = np.linalg.lstsq(
#     A_psd,
#     y_psd,
#     rcond=None,
# )

# Original mean instantaneous Lowes power of each arbitrary mode.
# mode_gnm has shape (M, Nt, Ng).
lowes_weights = Lowes_Coefficient_Weights(
    lmax=20,
    a=r_earth,
    r=r_choice,
)

mode_instantaneous_power = np.sum(
    mode_gnm**2
    * lowes_weights[None, None, :],
    axis=2,
)

mode_mean_power = np.mean(
    mode_instantaneous_power,
    axis=1,
)

# The unrestricted coefficients multiply PSD/power directly.
# Retain the sign separately, but plot the magnitude on a log axis.
unrestricted_scaled_power_signed = (
    unrestricted_coeffs * mode_mean_power
)

unrestricted_scaled_power_magnitude = np.abs(
    unrestricted_scaled_power_signed
)

# Retrieve periods if this has not already been done.
mode_periods = np.zeros(M)

for i, mode_number in enumerate(mode_numbers):

    mode_data = Component_Load(mode_number)

    omega = np.abs(
        mode_data["eigenvalue"].imag
    )

    if np.isclose(omega, 0.0):
        mode_periods[i] = np.inf
    else:
        mode_periods[i] = 2 * np.pi / omega


# Treat extremely small numerical coefficients as inactive.
coefficient_tolerance = (
    1e-8 * np.max(np.abs(unrestricted_coeffs))
)

active_mask = (
    np.abs(unrestricted_coeffs)
    > coefficient_tolerance
)

positive_mask = (
    active_mask
    & (unrestricted_coeffs > 0)
    & np.isfinite(mode_periods)
    & (unrestricted_scaled_power_magnitude > 0)
)

negative_mask = (
    active_mask
    & (unrestricted_coeffs < 0)
    & np.isfinite(mode_periods)
    & (unrestricted_scaled_power_magnitude > 0)
)


# %% PLOT

fig, ax = plt.subplots(
    figsize=(8, 5.5),
    constrained_layout=True,
)

# Positive coefficients: red circles
ax.scatter(
    mode_periods[positive_mask],
    unrestricted_scaled_power_magnitude[positive_mask],
    color="red",
    marker="o",
    s=65,
    edgecolor="black",
    linewidth=0.5,
    label="Positive coefficient: added power",
    zorder=3,
)

# Negative coefficients: blue crosses
ax.scatter(
    mode_periods[negative_mask],
    unrestricted_scaled_power_magnitude[negative_mask],
    color="blue",
    marker="x",
    s=75,
    linewidth=1.8,
    label="Negative coefficient: subtracted power",
    zorder=4,
)

ax.set_xscale("linear")
ax.set_yscale("log")

ax.set_xlabel("Mode period (years)")
ax.set_ylabel(
    "Magnitude of LS-scaled mean instantaneous Lowes power"
)

ax.set_title(
    "Unrestricted PSD least-squares mode contributions"
)

ax.grid(
    True,
    which="both",
    linestyle=":",
    alpha=0.35,
)

ax.legend()

plt.show()

print(
    f"Positive active modes: {np.sum(positive_mask)}"
)

print(
    f"Negative active modes: {np.sum(negative_mask)}"
)
# %%
