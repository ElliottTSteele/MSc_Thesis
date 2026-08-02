# %% CHAOS-8.6 DEGREE-WISE LOWES-WEIGHTED SV PSD

from pathlib import Path

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import periodogram
import chaosmagpy as cp


# ---------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------

t_start = 0.0
t_end = 29.5
dt_years = 0.2

t_r_start = 1997.1
NMAX = 20

times_relative = np.arange(t_start, t_end, dt_years)
times_dyear = t_r_start + times_relative
times_mjd2000 = cp.data_utils.dyear_to_mjd(times_dyear)

n_times = len(times_dyear)
f_sample = 1.0 / dt_years


# ---------------------------------------------------------------------
# Load CHAOS-8.6 and retrieve SV Gauss coefficients
# ---------------------------------------------------------------------

chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"

if not chaos_file.exists():
    raise FileNotFoundError(f"CHAOS model not found: {chaos_file}")

chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=NMAX,
    deriv=1,
    extrapolate="off",
)

chaos_sv_gnm = np.asarray(chaos_sv_gnm)
chaos_sv_gnm = Truncate_Gauss_Coeffs(chaos_sv_gnm, NMAX)

expected_shape = (n_times, n_Gauss_Coeffs(NMAX))

if chaos_sv_gnm.shape != expected_shape:
    raise ValueError(
        f"Unexpected coefficient shape {chaos_sv_gnm.shape}; "
        f"expected {expected_shape}."
    )


# ---------------------------------------------------------------------
# Compute Lowes-weighted SV PSD separately for each degree
# ---------------------------------------------------------------------

degrees = np.arange(1, NMAX + 1)

degree_psds = []

for n in degrees:

    frequencies, degree_psd = Lowes_Degree_PSD(
        chaos_sv_gnm,
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
degree_psds = np.stack(degree_psds, axis=0)

# Remove the zero-frequency bin for logarithmic plotting
positive_frequency = (frequencies > 0) & (frequencies < 0.5)

frequencies_plot = frequencies[positive_frequency]
degree_psds_plot = degree_psds[:, positive_frequency]

frequency_grid, degree_grid = np.meshgrid(
    frequencies_plot,
    degrees,
)


# ---------------------------------------------------------------------
# 3D surface plot
# ---------------------------------------------------------------------

positive_psd = degree_psds_plot[degree_psds_plot > 0]

if positive_psd.size == 0:
    raise ValueError("No positive PSD values were obtained.")

z_min = positive_psd.min()
z_max = positive_psd.max()

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(
    frequency_grid,
    degree_grid,
    degree_psds_plot,
    cmap="viridis",
    norm=LogNorm(vmin=z_min, vmax=z_max),
    linewidth=0,
    antialiased=True,
)

ax.set_xlabel("Frequency [cycles yr$^{-1}$]", labelpad=12)
ax.set_ylabel("Spherical harmonic degree $n$", labelpad=12)
ax.set_zlabel(
    r"Lowes-weighted SV PSD "
    r"$[(\mathrm{nT\,yr^{-1}})^2/(\mathrm{cycles\,yr^{-1}})]$",
    labelpad=14,
)

ax.set_yscale("linear")
ax.set_zscale("log")

ax.set_yticks(np.arange(1, NMAX + 1))
ax.set_title(
    "CHAOS-8.6 degree-wise Lowes-weighted secular-variation PSD\n"
    f"{times_dyear[0]:.1f}–{times_dyear[-1]:.1f}, "
    f"$n_{{max}}={NMAX}$"
)

ax.view_init(elev=28, azim=0)

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
# Basic diagnostics
# ---------------------------------------------------------------------

print(f"CHAOS SV coefficient shape: {chaos_sv_gnm.shape}")
print(f"Decimal-year range: {times_dyear[0]:.2f} to {times_dyear[-1]:.2f}")
print(f"Number of samples: {n_times}")
print(f"Sampling frequency: {f_sample:.3f} samples/year")
print(f"Periodogram frequency spacing: {frequencies[1] - frequencies[0]:.5f} cycles/year")
print(f"PSD array shape: {degree_psds.shape}")
# %%
