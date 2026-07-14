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
from matplotlib.colors import LogNorm
from scipy.signal import periodogram
import chaosmagpy as cp

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *
# import library made for these synthetic tests
from pyscripts.R_test_synthetics.syn_pipeline import *

# %% DEGREE-SUPPORT UTILITIES

import numpy as np
import matplotlib.pyplot as plt


def Infer_Nmax_From_Ngauss(n_gauss):
    """
    Infer nmax for standard packed internal-field Gauss coefficients:

        [g_1^0, g_1^1, h_1^1,
         g_2^0, g_2^1, h_2^1, g_2^2, h_2^2, ...]

    The total number of coefficients is

        n_gauss = nmax * (nmax + 2).
    """
    nmax = int(np.sqrt(n_gauss + 1) - 1)

    if nmax * (nmax + 2) != n_gauss:
        raise ValueError(
            f"{n_gauss} coefficients do not match "
            "nmax * (nmax + 2) for standard Gauss ordering."
        )

    return nmax


def Gauss_Degree_Indices(nmax):
    """
    Return the packed-array indices belonging to each spherical-harmonic
    degree under standard Gauss-coefficient ordering.
    """
    degree_indices = {}

    start = 0

    for n in range(1, nmax + 1):
        n_coeffs = 2 * n + 1
        stop = start + n_coeffs

        degree_indices[n] = np.arange(start, stop)
        start = stop

    return degree_indices


def Complex_Gauss_Degree_Power(
    gnm,
    a=1.0,
    r=1.0,
    lowes_weight=True,
):
    """
    Compute degree-wise power directly from a complex Gauss-coefficient
    phasor.

    Parameters
    ----------
    gnm : (n_gauss,) array
        Packed complex Gauss coefficients.

    a, r : float
        Reference and evaluation radii.

    lowes_weight : bool
        If True, apply the Lowes-Mauersberger degree factor

            (n + 1) * (a/r)^(2n + 4).

        If False, return only the coefficient squared norm per degree.

    Returns
    -------
    degrees : (nmax,) array
    degree_power : (nmax,) array
    """
    gnm = np.asarray(gnm).squeeze()

    if gnm.ndim != 1:
        raise ValueError(
            f"Expected one packed phasor, but received shape {gnm.shape}."
        )

    nmax = Infer_Nmax_From_Ngauss(gnm.size)
    degree_indices = Gauss_Degree_Indices(nmax)

    degrees = np.arange(1, nmax + 1)
    degree_power = np.zeros(nmax, dtype=float)

    for i, n in enumerate(degrees):

        coeff_power = np.sum(np.abs(gnm[degree_indices[n]]) ** 2)

        if lowes_weight:
            degree_factor = (n + 1) * (a / r) ** (2 * n + 4)
        else:
            degree_factor = 1.0

        degree_power[i] = degree_factor * coeff_power

    return degrees, degree_power


def Time_Series_Degree_Mean_Power(
    gnm_time_series,
    a=1.0,
    r=1.0,
    lowes_weight=True,
):
    """
    Compute mean instantaneous degree power from a real Gauss-coefficient
    time series with shape (n_time, n_gauss).
    """
    gnm_time_series = np.asarray(gnm_time_series)

    if gnm_time_series.ndim != 2:
        raise ValueError(
            "gnm_time_series must have shape (n_time, n_gauss)."
        )

    n_time, n_gauss = gnm_time_series.shape
    nmax = Infer_Nmax_From_Ngauss(n_gauss)
    degree_indices = Gauss_Degree_Indices(nmax)

    degrees = np.arange(1, nmax + 1)
    degree_power = np.zeros(nmax, dtype=float)

    for i, n in enumerate(degrees):

        coeffs_n = gnm_time_series[:, degree_indices[n]]

        instantaneous_power = np.sum(coeffs_n**2, axis=1)

        if lowes_weight:
            degree_factor = (n + 1) * (a / r) ** (2 * n + 4)
        else:
            degree_factor = 1.0

        degree_power[i] = degree_factor * np.mean(
            instantaneous_power
        )

    return degrees, degree_power

# %% RAW PHASOR DEGREE SUPPORT ACROSS THE COMPLETE DATA SET

mode_numbers = np.arange(1, 63)

raw_degree_powers = []
periods = []
decay_rates = []

for mode_number in mode_numbers:

    mode_info = Component_Load(mode_number)

    gnm = np.asarray(mode_info["gnm"]).squeeze()
    eigenvalue = mode_info["eigenvalue"]

    degrees, degree_power = Complex_Gauss_Degree_Power(
        gnm,
        a=r_earth,
        r=r_earth,
    )

    raw_degree_powers.append(degree_power)

    periods.append(
        2 * np.pi / np.abs(np.imag(eigenvalue))
    )

    decay_rates.append(np.real(eigenvalue))


raw_degree_powers = np.asarray(raw_degree_powers)
periods = np.asarray(periods)
decay_rates = np.asarray(decay_rates)

# Normalise each mode by its own total degree power.
normalised_degree_powers = (
    raw_degree_powers
    / raw_degree_powers.sum(axis=1, keepdims=True)
)

print("Raw degree-power array shape:", raw_degree_powers.shape)
print("Degrees:", degrees)


# %% MODE-BY-DEGREE SUPPORT HEATMAP

power_floor = 1e-16

fig, ax = plt.subplots(figsize=(11, 8))

im = ax.imshow(
    np.log10(normalised_degree_powers + power_floor),
    origin="lower",
    aspect="auto",
    extent=[
        degrees[0] - 0.5,
        degrees[-1] + 0.5,
        mode_numbers[0] - 0.5,
        mode_numbers[-1] + 0.5,
    ],
    vmin=-12,
    vmax=0,
)

ax.set_xlabel("Spherical-harmonic degree $n$")
ax.set_ylabel("Mode number")
ax.set_xticks(degrees)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(
    r"$\log_{10}(P_n / \sum_n P_n)$"
)

plt.tight_layout()
plt.show()