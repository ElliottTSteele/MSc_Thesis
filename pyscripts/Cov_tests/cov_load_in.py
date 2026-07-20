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
from matplotlib.colors import SymLogNorm, LogNorm


# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *
# import library made for these synthetic tests
from pyscripts.Cov_tests.cov_tests import *

# %%
# PERTURBATION PLOT
# -----------------------------------------------------
def Perturb_Plot(perturbations):
    perturbations_plot = perturbations[good_record_slice]

    abs_nonzero = np.abs(
        perturbations_plot[perturbations_plot != 0]
    )

    linthresh = np.percentile(abs_nonzero, 5)
    vmax = np.max(abs_nonzero)

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(
        perturbations_plot,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        norm=SymLogNorm(
            linthresh=linthresh,
            vmin=-vmax,
            vmax=vmax,
        ),
    )

    ax.set_xlabel("Gauss coefficient index")
    ax.set_ylabel("Time index")
    ax.set_title("Covariance-informed SV perturbation")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"SV Gauss coefficient perturbation (nT yr$^{-1}$)")

    plt.tight_layout()
    plt.show()

# %%
# PSD PLOT
# -----------------------------------------------------
def PSD_Perturb_Plot(perturbations):
    psd, f = Lowes_Degree_PSD_All_Degrees(
        perturbations,
        a=r_earth,
        r=r_cmb,
    )

    f_clip = (f > 0) & (f < 1)
    psd_plot = psd[:, f_clip]
    f_plot = f[f_clip]

    # LogNorm requires strictly positive values
    positive_psd = psd_plot[psd_plot > 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        psd_plot,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        norm=LogNorm(
            vmin=np.min(positive_psd),
            vmax=np.max(positive_psd),
        ),
        extent=[
            f_plot.min(),
            f_plot.max(),
            0.5,
            psd_plot.shape[0] + 0.5,
        ],
    )

    ax.set_xlabel(r"Frequency (yr$^{-1}$)")
    ax.set_ylabel("Spherical harmonic degree")
    ax.set_title("Lowes–Mauersberger PSD of SV perturbation")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        r"SV power spectral density"
        "\n"
        r"($\mathrm{nT^2\,yr^{-2}}$ per frequency bin)"
    )

    plt.tight_layout()
    plt.show()

# %%
# LOADING IN CHAOS DATA
# -----------------------------------------------------

file_path = Path(f"{CHAOS_COV_DIR}/CHAOS_Cov_1997_2026_0806_SV.h5")

with h5py.File(file_path, "r") as cov_file:

    Cov_full = np.asarray(cov_file["Cnm"])
    print(f"C has shape dimensions: {np.shape(Cov_full)}")


# %%
# SAME SEED NOISE GENERATION
# -----------------------------------------------------

# getting random vector ~N(0, 1)
z = np.random.default_rng().standard_normal(440)

k_list = np.arange(0, 148, 1)
perturbations = np.zeros((148, 440))

for k in k_list:

    C = Cov_full[k, :, :]
    # get cholesky:
    C = 0.5 * (C + C.T)
    L = np.linalg.cholesky(C)

    dg_k = L @ z

    perturbations[k, :] = dg_k

# showing
plt.imshow(perturbations[good_record_slice, :])

# %%
# DIFFERENT SEED NOISE REALISATION
# -----------------------------------------------------
k_list = np.arange(0, 148, 1)
perturbations = np.zeros((148, 440))

for k in k_list:

    C = Cov_full[k, :, :]
    # get cholesky:
    C = 0.5 * (C + C.T)
    L = np.linalg.cholesky(C)

    # getting random vector ~N(0, 1)
    z = np.random.default_rng().standard_normal(440)


    dg_k = L @ z

    perturbations[k, :] = dg_k


PSD_Perturb_Plot(perturbations[good_record_slice])

# %%
# simple time correlated
def Temporal_Corr_z(
        k,
        z_t_0=np.random.default_rng().standard_normal(440),
        dt=dt_years,
        Tau=0.2,
        Cov=Cov_full
        ):
    
    eta = np.random.default_rng().standard_normal(440)

    rho = np.exp(-dt / Tau)

    z_t_1 = rho * z_t_0 + np.sqrt(1 - rho**2) * eta

    C = Cov[int(k), :, :]

    C = 0.5 * (C + C.T)
    L = np.linalg.cholesky(C)

    epsilon = L @ z_t_1

    return epsilon, z_t_1

k_list = np.arange(0, 148, 1)
perturbations = np.zeros((148, 440))

z_k = np.random.default_rng().standard_normal(440)

for k_step in k_list:

    epsilon_k, z_k = Temporal_Corr_z(
        k=k_step,
        z_t_0=z_k
    )

    perturbations[k_step, :] = epsilon_k
    
PSD_Perturb_Plot(perturbations[good_record_slice])

