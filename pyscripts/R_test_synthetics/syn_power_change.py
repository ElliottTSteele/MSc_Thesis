'''
syn_power_change.py

purpose:
this code is designed to see how the characters of the gauss
power for the synthetic waves change with each processing step
'''


# PREAMBLE
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
# import library made for these synthetic tests
from pyscripts.R_test_synthetics.syn_pipeline import *


# %% 


# LOAD IN THE DATA FOR EACH WAVE

# get full gauss phasor no decay

# compute time series of full gauss

# truncate this to N=20 and store separately

# withdraw the no decay resolved gauss time series

# compute total power of each, store metrics for plotting


# PREAMBLE
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
# import library made for these synthetic tests
from pyscripts.R_test_synthetics.syn_pipeline import *

# %% SYNTHETIC MODE: DEGREE-WISE LOWES-WEIGHTED SV PSD

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------
# Mode and sampling setup
# ---------------------------------------------------------------------

mode_number = "mode_5"
NMAX = 20

t_start = 0.0
t_end = 29.5
dt_years = 0.2

t_r_start = 1997.1

# Full time axis required by H_sv
times_relative_full = np.arange(t_start, t_end, dt_years)
times_absolute_full = t_r_start + times_relative_full

n_times_full = len(times_relative_full)
f_sample = 1.0 / dt_years


# Good sampling interval: CHAMP start to 2026/01
good_record_start = notable_times_raw["CHAMP Start (2000/08)"]
good_record_end = 2026 + 1 / 12

good_record_mask = (
    (times_absolute_full >= good_record_start)
    & (times_absolute_full <= good_record_end)
)

if not np.any(good_record_mask):
    raise ValueError("No samples lie inside the good sampling interval.")

times_relative = times_relative_full[good_record_mask]
times_absolute = times_absolute_full[good_record_mask]

n_times = len(times_relative)

# ---------------------------------------------------------------------
# Load resolved spline coefficients for one synthetic mode
# ---------------------------------------------------------------------

file_path = Path(f"{FELIX_DIR}/R_splines_arbitrary.h5")

with h5py.File(file_path, "r") as h5_file:

    if mode_number not in h5_file:
        raise KeyError(
            f"Mode {mode_number} was not found. "
            f"Available top-level keys: {list(h5_file.keys())}"
        )

    mode_group = h5_file[mode_number]

    gnm_spl = mode_group["without_decay"][()]

gnm_spl = np.asarray(gnm_spl)


# ---------------------------------------------------------------------
# Apply H_sv to reconstruct the SV Gauss-coefficient time series
# ---------------------------------------------------------------------

def Apply_Hsv_To_Gnm_Splines(
    gnm_spl,
    H_sv,
    n_times,
):
    """
    Apply the SV spline evaluation matrix to stored Gauss spline
    coefficients.

    Supports:

    1. Ordinary basis matrix:
           H_sv.shape = (nt, nspl)
           gnm_spl.shape = (nspl, ng)

       giving:
           gnm_sv.shape = (nt, ng)

    2. Full block matrix:
           H_sv.shape = (nt * ng, nspl * ng)

       with gnm_spl vectorised in Fortran ordering.
    """

    gnm_spl = np.asarray(gnm_spl)

    if len(H_sv.shape) != 2:
        raise ValueError("H_sv must be a two-dimensional matrix.")

    h_rows, h_cols = H_sv.shape

    # Full block-matrix form:
    # vec(G_sv) = H_sv @ vec(G_spl)
    if h_cols == gnm_spl.size and h_rows != n_times:

        gnm_sv_vector = np.asarray(
            H_sv @ gnm_spl.ravel(order="F")
        ).reshape(-1)

        if gnm_sv_vector.size % n_times != 0:
            raise ValueError(
                "The H_sv output size is not divisible by the "
                f"number of time samples ({n_times})."
            )

        return gnm_sv_vector.reshape(
            n_times,
            -1,
            order="F",
        )

    # If stored as an F-ordered flattened spline coefficient vector,
    # reconstruct the (nspl, ng) matrix.
    if gnm_spl.ndim == 1:

        n_splines = h_cols

        if gnm_spl.size % n_splines != 0:
            raise ValueError(
                f"Cannot reshape gnm_spl of size {gnm_spl.size} "
                f"into a spline matrix with {n_splines} rows."
            )

        gnm_spl = gnm_spl.reshape(
            n_splines,
            -1,
            order="F",
        )

    if gnm_spl.ndim != 2:
        raise ValueError(
            "gnm_spl must be a 2D spline-coefficient matrix or "
            "an F-ordered flattened vector."
        )

    # Expected orientation: (nspl, ng)
    if h_cols == gnm_spl.shape[0]:
        gnm_sv = H_sv @ gnm_spl

    # Handle storage as (ng, nspl)
    elif h_cols == gnm_spl.shape[1]:
        gnm_sv = H_sv @ gnm_spl.T

    # Full block matrix with a 2D stored gnm_spl array
    elif h_cols == gnm_spl.size:

        gnm_sv_vector = np.asarray(
            H_sv @ gnm_spl.ravel(order="F")
        ).reshape(-1)

        if gnm_sv_vector.size % n_times != 0:
            raise ValueError(
                "The H_sv output size is not divisible by the "
                f"number of time samples ({n_times})."
            )

        gnm_sv = gnm_sv_vector.reshape(
            n_times,
            -1,
            order="F",
        )

    else:
        raise ValueError(
            "H_sv and gnm_spl have incompatible shapes:\n"
            f"    H_sv:    {H_sv.shape}\n"
            f"    gnm_spl: {gnm_spl.shape}"
        )

    gnm_sv = np.asarray(gnm_sv)

    if gnm_sv.shape[0] != n_times:
        raise ValueError(
            f"H_sv produced {gnm_sv.shape[0]} time samples, "
            f"but {n_times} were expected."
        )

    return gnm_sv


synthetic_sv_gnm_full = Apply_Hsv_To_Gnm_Splines(
    gnm_spl=gnm_spl,
    H_sv=H_sv,
    n_times=n_times_full,
)
# Remove negligible numerical imaginary components if present
synthetic_sv_gnm_full = np.real_if_close(
    synthetic_sv_gnm_full,
    tol=1000,
)

if np.iscomplexobj(synthetic_sv_gnm_full):
    raise ValueError(
        "The reconstructed SV coefficients remain significantly complex. "
        "Use the physical real-valued spline coefficients or explicitly "
        "construct the real mode time series before computing the PSD."
    )

# Retain only the reliable CHAMP-era sampling interval
synthetic_sv_gnm = synthetic_sv_gnm_full[good_record_mask]

# Retain degrees 1,...,20
synthetic_sv_gnm = Truncate_Gauss_Coeffs(
    synthetic_sv_gnm,
    NMAX,
)

# ---------------------------------------------------------------------
# Degree-wise Lowes-weighted PSD
# ---------------------------------------------------------------------

degrees = np.arange(1, NMAX + 1)

degree_psds = []

for n in degrees:

    frequencies, degree_psd = Lowes_Degree_PSD(
        synthetic_sv_gnm,
        n=n,
        fs=f_sample,
        a=r_earth,
        r=r_cmb,
        detrend="linear",
        window="hann",
        scaling="density",
    )

    degree_psds.append(degree_psd)

# Shape: (n_degrees, n_frequencies)
degree_psds = np.stack(
    degree_psds,
    axis=0,
)


# ---------------------------------------------------------------------
# Remove zero frequency for logarithmic PSD display
# ---------------------------------------------------------------------

positive_frequency = (frequencies > 0) & (frequencies < 0.5)

frequencies_plot = frequencies[positive_frequency]
degree_psds_plot = degree_psds[:, positive_frequency]

frequency_grid, degree_grid = np.meshgrid(
    frequencies_plot,
    degrees,
)

positive_psd = degree_psds_plot[
    np.isfinite(degree_psds_plot)
    & (degree_psds_plot > 0)
]

if positive_psd.size == 0:
    raise ValueError("No positive finite PSD values were obtained.")

z_min = positive_psd.min()
z_max = positive_psd.max()


# ---------------------------------------------------------------------
# 3D Lowes-weighted PSD surface
# ---------------------------------------------------------------------

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(
    frequency_grid,
    degree_grid,
    degree_psds_plot,
    cmap="viridis",
    norm=LogNorm(
        vmin=z_min,
        vmax=z_max,
    ),
    linewidth=0,
    antialiased=True,
)

ax.set_xlabel(
    "Frequency [cycles yr$^{-1}$]",
    labelpad=12,
)

ax.set_ylabel(
    "Spherical harmonic degree $n$",
    labelpad=12,
)

ax.set_zlabel(
    r"Lowes-weighted SV PSD "
    r"$[(\mathrm{nT\,yr^{-1}})^2/"
    r"(\mathrm{cycles\,yr^{-1}})]$",
    labelpad=14,
)

ax.set_zscale("log")
ax.set_yticks(degrees)

ax.set_title(
    "Synthetic mode degree-wise Lowes-weighted SV PSD\n"
    f"Mode {mode_number}, "
    f"$n_{{max}}={NMAX}$, "
    f"$\\Delta t={dt_years}$ yr"
)

ax.view_init(
    elev=28,
    azim=45,
)

cbar = fig.colorbar(
    surface,
    ax=ax,
    pad=0.10,
    shrink=0.65,
    aspect=22,
)

cbar.set_label("Lowes-weighted SV PSD")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

print(f"Mode number:                  {mode_number}")
print(f"Stored gnm_spl shape:         {gnm_spl.shape}")
print(f"H_sv shape:                   {H_sv.shape}")
print(f"Reconstructed SV shape:       {synthetic_sv_gnm.shape}")
print(f"Sampling frequency:           {f_sample:.3f} samples/year")
print(
    "Periodogram frequency step:  "
    f"{frequencies[1] - frequencies[0]:.5f} cycles/year"
)
print(f"Degree-PSD array shape:        {degree_psds.shape}")
# %%
