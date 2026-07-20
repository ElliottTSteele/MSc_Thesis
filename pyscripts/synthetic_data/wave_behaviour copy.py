'''
Plan:

investigating simple optimised dmd
on a single wave case
'''

# PREAMBLE
# %% SETTING UP AUTOUPDATES

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %% FILE SYSTEM AND DEPENDENCY SETUP

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


import numpy as np
import matplotlib.pyplot as plt
import chaosmagpy as cp

import os
import sys
import pydmd
import cmath
import copy
import h5py
import math
import pickle
import gc
import pydmd

from pathlib import Path
from scipy.signal import periodogram
from matplotlib.animation import FuncAnimation
from IPython.display import Video
from tqdm import tqdm
from scipy.interpolate import make_interp_spline
from chaosmagpy.chaos import BaseModel

# importing from pyDMD
from pydmd import BOPDMD

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# import simulation setup
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *

import numpy as np


# %% ------------------------------------------------------
# SIMULATION SETUP
# ---------------------------------------------------------

# put list of mode numbers used
mode_numbers = np.arange(1, 63, 1)
mode_numbers = [str(num) for num in mode_numbers]

# - if windowing to 'high quality' record
high_q_flag = True
#   - how often to sub-sample time series (1 = full sampling)
n_skip = 1

# deciding on spherical harmonic truncation degree
Nmax = 20
A_r_20 = A_20_dict["r"]
A_r = Truncate_Gauss_Coeffs(A_r_20, tmax=Nmax)

A_t = A_20_dict["phi"]
A_comp = A_t
# ---------------------------------------------------------
# DECIDING HOW TO DIVIDE EARTH REGIONALLY
# ---------------------------------------------------------

region_names = [
    "Southern",
    "Equatorial",
    "Northern"
]

def regional_wave_power(
    wave,
    latitudes,
    longitudes=None,
    remove_temporal_mean=False,
):

    wave = np.asarray(wave)
    latitudes = np.asarray(latitudes)

    # Add a time dimension for a static spatial mode
    if wave.ndim == 2:
        wave = wave[None, :, :]

    if wave.ndim != 3:
        raise ValueError(
            "wave must have shape (nt, nlat, nlon) or (nlat, nlon)"
        )

    nt, nlat, nlon = wave.shape

    if len(latitudes) != nlat:
        raise ValueError("latitudes does not match the wave latitude dimension")

    if longitudes is not None and len(longitudes) != nlon:
        raise ValueError("longitudes does not match the wave longitude dimension")

    if remove_temporal_mean and nt > 1:
        wave = wave - np.mean(wave, axis=0, keepdims=True)

    # Correct spherical-area weighting for a regular lat-lon grid
    latitude_weights = np.cos(np.deg2rad(latitudes))

    # Expand across longitude
    area_weights = np.broadcast_to(
        latitude_weights[:, None],
        (nlat, nlon),
    )

    # Works for both real time series and complex mode phasors
    point_power = np.mean(np.abs(wave) ** 2, axis=0)

    region_masks = {
        "Southern": latitudes < -30,
        "Equatorial": (latitudes >= -30) & (latitudes <= 30),
        "Northern": latitudes > 30,
    }

    integrated_power = {}
    represented_area = {}

    for name, latitude_mask in region_masks.items():

        weights_region = area_weights[latitude_mask, :]
        power_region = point_power[latitude_mask, :]

        integrated_power[name] = np.sum(
            weights_region * power_region
        )

        represented_area[name] = np.sum(weights_region)

    global_power = sum(integrated_power.values())
    global_area = sum(represented_area.values())

    global_mean_square = global_power / global_area

    results = {}

    for name in region_masks:

        mean_square = (
            integrated_power[name] / represented_area[name]
        )

        results[name] = {
            "mean_square": mean_square,
            "rms": np.sqrt(mean_square),
            "total_power": integrated_power[name],
            "power_fraction": (
                integrated_power[name] / global_power
                if global_power > 0 else np.nan
            ),
            "area_fraction": represented_area[name] / global_area,

            # >1 means preferentially concentrated in this region
            "concentration": (
                mean_square / global_mean_square
                if global_mean_square > 0 else np.nan
            ),
        }
    return results

def regional_signal_distortion(
    signal_input,
    signal_resolved,
    latitudes,
    longitudes=None,
    remove_temporal_mean=False,
    include_global=True,
):
    """
    Calculate area-weighted signal distortion ratio between an input
    wave and its resolved version in different latitude regions.

    SDR_R = weighted input signal power / weighted error power

    Parameters
    ----------
    signal_input : ndarray, shape (nt, nlat, nlon)
        Original input SV grid time series.

    signal_resolved : ndarray, shape (nt, nlat, nlon)
        Resolved SV grid time series.

    latitudes : ndarray, shape (nlat,)
        Grid latitudes in degrees.

    longitudes : ndarray, optional
        Grid longitudes, used for shape checking.

    remove_temporal_mean : bool, default=False
        If True, remove the temporal mean from both signals before
        calculating SDR.

    include_global : bool, default=True
        Include a whole-Earth result for comparison.

    Returns
    -------
    results : dict
        For each region:

            signal_power
                Area-weighted integrated input power.

            resolved_power
                Area-weighted integrated resolved power.

            error_power
                Area-weighted integrated squared error.

            sdr
                Linear signal distortion ratio.

            sdr_db
                SDR in decibels: 10 log10(SDR).

            relative_rmse
                sqrt(error_power / signal_power).

            power_retention
                resolved_power / signal_power.
    """

    signal_input = np.asarray(signal_input)
    signal_resolved = np.asarray(signal_resolved)
    latitudes = np.asarray(latitudes)

    if signal_input.shape != signal_resolved.shape:
        raise ValueError(
            "Input and resolved signals must have the same shape. "
            f"Received {signal_input.shape} and {signal_resolved.shape}."
        )

    if signal_input.ndim != 3:
        raise ValueError(
            "Signals must have shape (nt, nlat, nlon)."
        )

    nt, nlat, nlon = signal_input.shape

    if len(latitudes) != nlat:
        raise ValueError(
            "latitudes does not match the signal latitude dimension"
        )

    if longitudes is not None and len(longitudes) != nlon:
        raise ValueError(
            "longitudes does not match the signal longitude dimension"
        )

    if remove_temporal_mean:
        signal_input = (
            signal_input
            - np.mean(signal_input, axis=0, keepdims=True)
        )

        signal_resolved = (
            signal_resolved
            - np.mean(signal_resolved, axis=0, keepdims=True)
        )

    # Resolution-induced error
    error = signal_input - signal_resolved

    # Spherical area weights for a regular latitude-longitude grid
    latitude_weights = np.cos(np.deg2rad(latitudes))

    area_weights = np.broadcast_to(
        latitude_weights[:, None],
        (nlat, nlon),
    )

    region_masks = {
        "Southern": latitudes < -30,
        "Equatorial": (latitudes >= -30) & (latitudes <= 30),
        "Northern": latitudes > 30,
    }

    if include_global:
        region_masks["Global"] = np.ones(nlat, dtype=bool)

    results = {}

    for region, latitude_mask in region_masks.items():

        weights_region = area_weights[latitude_mask, :]

        input_region = signal_input[:, latitude_mask, :]
        resolved_region = signal_resolved[:, latitude_mask, :]
        error_region = error[:, latitude_mask, :]

        # Add time dimension to the spatial area weights
        weights_3d = weights_region[None, :, :]

        signal_power = np.sum(
            weights_3d * np.abs(input_region) ** 2
        )

        resolved_power = np.sum(
            weights_3d * np.abs(resolved_region) ** 2
        )

        error_power = np.sum(
            weights_3d * np.abs(error_region) ** 2
        )

        if signal_power > 0:
            relative_rmse = np.sqrt(error_power / signal_power)
            power_retention = resolved_power / signal_power
        else:
            relative_rmse = np.nan
            power_retention = np.nan

        if error_power > 0:
            sdr = signal_power / error_power
            sdr_db = 10 * np.log10(sdr)
        elif signal_power > 0:
            sdr = np.inf
            sdr_db = np.inf
        else:
            sdr = np.nan
            sdr_db = np.nan

        results[region] = {
            "signal_power": signal_power,
            "resolved_power": resolved_power,
            "error_power": error_power,
            "sdr": sdr,
            "sdr_db": sdr_db,
            "relative_rmse": relative_rmse,
            "power_retention": power_retention,
        }

    return results

from matplotlib.colors import LogNorm, TwoSlopeNorm


def spatial_uncertainty_ensemble(
    chaos_gauss,
    noise_gauss,
    synthesis_matrix,
    latitudes,
    longitudes,
    spatial_stride=2,
    relative_floor_fraction=1e-8,
):
    """
    Project Gauss-coefficient uncertainty realisations onto a spatial grid
    and calculate spatial uncertainty statistics.

    Parameters
    ----------
    chaos_gauss : ndarray, shape (nt, ng)
        Mean CHAOS SV Gauss-coefficient time series.

    noise_gauss : ndarray, shape (n_realisations, nt, ng)
        Noise-only Gauss-coefficient realisations.

    synthesis_matrix : ndarray, shape (nlat*nlon, ng)
        Radial-field synthesis matrix, e.g. A_r_20.

    latitudes : ndarray, shape (nlat,)
        Grid latitudes in degrees.

    longitudes : ndarray, shape (nlon,)
        Grid longitudes in degrees.

    spatial_stride : int, default=2
        Spatial subsampling factor. A value of 2 uses every second
        latitude and longitude point.

    relative_floor_fraction : float
        Floor applied to CHAOS RMS before division.

    Returns
    -------
    results : dict
        Contains spatial grids and uncertainty statistics.
    """

    chaos_gauss = np.asarray(chaos_gauss)
    noise_gauss = np.asarray(noise_gauss)
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    if chaos_gauss.ndim != 2:
        raise ValueError(
            "chaos_gauss must have shape (nt, ng)"
        )

    if noise_gauss.ndim != 3:
        raise ValueError(
            "noise_gauss must have shape "
            "(n_realisations, nt, ng)"
        )

    n_realisations, nt, ng = noise_gauss.shape
    nt_chaos, ng_chaos = chaos_gauss.shape

    if (nt, ng) != (nt_chaos, ng_chaos):
        raise ValueError(
            "CHAOS and noise coefficient dimensions do not match: "
            f"{chaos_gauss.shape} versus {noise_gauss.shape}"
        )

    nlat = len(latitudes)
    nlon = len(longitudes)

    if synthesis_matrix.shape != (nlat * nlon, ng):
        raise ValueError(
            "synthesis_matrix should have shape "
            f"{(nlat * nlon, ng)}, but has "
            f"{synthesis_matrix.shape}"
        )

    # Select a lower-resolution subset of spatial points.
    latitude_indices = np.arange(
        0, nlat, spatial_stride
    )

    longitude_indices = np.arange(
        0, nlon, spatial_stride
    )

    # Assumes the flattened grid follows the same ordering used by
    # Series_To_Cube: (latitude, longitude).
    full_grid_indices = np.arange(
        nlat * nlon
    ).reshape(nlat, nlon)

    selected_grid_indices = full_grid_indices[
        np.ix_(latitude_indices, longitude_indices)
    ].ravel()

    latitudes_used = latitudes[latitude_indices]
    longitudes_used = longitudes[longitude_indices]

    nlat_used = len(latitudes_used)
    nlon_used = len(longitudes_used)
    npoints_used = nlat_used * nlon_used

    A_used = synthesis_matrix[selected_grid_indices, :]

    # -----------------------------------------------------
    # Mean CHAOS spatial RMS
    # -----------------------------------------------------

    chaos_grid_flat = A_used @ chaos_gauss.T

    chaos_rms_flat = np.sqrt(
        np.mean(chaos_grid_flat**2, axis=1)
    )

    chaos_rms = chaos_rms_flat.reshape(
        nlat_used,
        nlon_used,
    )

    # The full projected ensemble is not retained. Only one
    # time-averaged map is stored for each realisation.
    noise_rms_ensemble_flat = np.empty(
        (n_realisations, npoints_used),
        dtype=np.float64,
    )

    # Additional ensemble-time accumulators
    noise_sum_flat = np.zeros(
        npoints_used,
        dtype=np.float64,
    )

    noise_squared_sum_flat = np.zeros(
        npoints_used,
        dtype=np.float64,
    )

    total_noise_samples = n_realisations * nt

    # -----------------------------------------------------
    # Project each noise realisation
    # -----------------------------------------------------

    for realisation in tqdm(
        range(n_realisations),
        desc="Projecting uncertainty realisations",
    ):

        noise_grid_flat = (
            A_used @ noise_gauss[realisation].T
        )

        noise_rms_ensemble_flat[realisation] = np.sqrt(
            np.mean(noise_grid_flat**2, axis=1)
        )

        noise_sum_flat += np.sum(
            noise_grid_flat,
            axis=1,
        )

        noise_squared_sum_flat += np.sum(
            noise_grid_flat**2,
            axis=1,
        )

    # RMS over both time and ensemble
    uncertainty_rms_flat = np.sqrt(
        noise_squared_sum_flat / total_noise_samples
    )

    # Check for non-zero ensemble bias
    uncertainty_mean_flat = (
        noise_sum_flat / total_noise_samples
    )

    # Distribution of time-RMS uncertainty between realisations
    uncertainty_median_flat = np.median(
        noise_rms_ensemble_flat,
        axis=0,
    )

    uncertainty_q05_flat = np.quantile(
        noise_rms_ensemble_flat,
        0.05,
        axis=0,
    )

    uncertainty_q95_flat = np.quantile(
        noise_rms_ensemble_flat,
        0.95,
        axis=0,
    )

    # Avoid unstable division where CHAOS RMS is essentially zero
    signal_floor = (
        relative_floor_fraction
        * np.nanmax(chaos_rms_flat)
    )

    relative_uncertainty_flat = (
        uncertainty_rms_flat
        / np.maximum(chaos_rms_flat, signal_floor)
    )

    log10_relative_uncertainty_flat = np.log10(
        np.maximum(relative_uncertainty_flat, 1e-15)
    )

    return {
        "latitudes": latitudes_used,
        "longitudes": longitudes_used,

        "chaos_rms": chaos_rms,

        "uncertainty_rms": uncertainty_rms_flat.reshape(
            nlat_used, nlon_used
        ),

        "uncertainty_mean": uncertainty_mean_flat.reshape(
            nlat_used, nlon_used
        ),

        "uncertainty_median": uncertainty_median_flat.reshape(
            nlat_used, nlon_used
        ),

        "uncertainty_q05": uncertainty_q05_flat.reshape(
            nlat_used, nlon_used
        ),

        "uncertainty_q95": uncertainty_q95_flat.reshape(
            nlat_used, nlon_used
        ),

        "relative_uncertainty": relative_uncertainty_flat.reshape(
            nlat_used, nlon_used
        ),

        "log10_relative_uncertainty": (
            log10_relative_uncertainty_flat.reshape(
                nlat_used, nlon_used
            )
        ),

        "noise_rms_ensemble": (
            noise_rms_ensemble_flat.reshape(
                n_realisations,
                nlat_used,
                nlon_used,
            )
        ),
    }
#  %% -----------------------------------------------------
# OBTAIN INPUT DATA
# ---------------------------------------------------------


chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gauss = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=20,
    deriv=1,
    extrapolate="off",
)
if high_q_flag:
    chaos_sv_gauss = chaos_sv_gauss[good_record_slice]
n_realisations = 100

noise_gauss = Perturbation_Generate(
    gnm_input=chaos_sv_gauss,
    dt=dt_years,
    n_realisations=n_realisations,
    seed=42,
    covariance_used=Cov_full,
    temporal_z="independent",
    just_noise=True,
)

print("Noise ensemble shape:", noise_gauss.shape)

spatial_uncertainty = spatial_uncertainty_ensemble(
    chaos_gauss=chaos_sv_gauss,
    noise_gauss=noise_gauss,
    synthesis_matrix=A_r_20,
    latitudes=latitude,
    longitudes=longitude,
    spatial_stride=2,
)

lats_plot = spatial_uncertainty["latitudes"]
lons_plot = spatial_uncertainty["longitudes"]

lon_grid, lat_grid = np.meshgrid(
    lons_plot,
    lats_plot,
)

chaos_rms = spatial_uncertainty["chaos_rms"]
uncertainty_rms = spatial_uncertainty["uncertainty_rms"]
uncertainty_q95 = spatial_uncertainty["uncertainty_q95"]
log_relative = spatial_uncertainty["log10_relative_uncertainty"]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 7),
    subplot_kw={
        "projection": ccrs.Robinson()
    },
)

axes = axes.ravel()

positive_values = np.concatenate([
    chaos_rms[chaos_rms > 0],
    uncertainty_rms[uncertainty_rms > 0],
    uncertainty_q95[uncertainty_q95 > 0],
])

absolute_norm = LogNorm(
    vmin=np.nanpercentile(positive_values, 2),
    vmax=np.nanpercentile(positive_values, 98),
)

absolute_maps = [
    (
        chaos_rms,
        "CHAOS SV temporal RMS",
    ),
    (
        uncertainty_rms,
        "Ensemble RMS uncertainty",
    ),
    (
        uncertainty_q95,
        "95th percentile RMS uncertainty",
    ),
]

absolute_meshes = []

for ax, (data, title) in zip(
    axes[:3],
    absolute_maps,
):
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        data,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        norm=absolute_norm,
        shading="auto",
    )

    ax.coastlines(linewidth=0.5)
    ax.set_global()
    ax.set_title(title)

    absolute_meshes.append(mesh)

relative_limit = np.nanpercentile(
    np.abs(log_relative),
    98,
)

relative_limit = max(relative_limit, 0.1)

relative_mesh = axes[3].pcolormesh(
    lon_grid,
    lat_grid,
    log_relative,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    norm=TwoSlopeNorm(
        vmin=-relative_limit,
        vcenter=0,
        vmax=relative_limit,
    ),
    shading="auto",
)

axes[3].coastlines(linewidth=0.5)
axes[3].set_global()
axes[3].set_title(
    r"$\log_{10}(\mathrm{uncertainty\ RMS}/"
    r"\mathrm{CHAOS\ RMS})$"
)

absolute_colorbar = fig.colorbar(
    absolute_meshes[0],
    ax=axes[:3],
    orientation="horizontal",
    fraction=0.07,
    pad=0.08,
)

absolute_colorbar.set_label(
    "Radial SV RMS [nT/yr]"
)

relative_colorbar = fig.colorbar(
    relative_mesh,
    ax=axes[3],
    orientation="horizontal",
    fraction=0.07,
    pad=0.08,
)

relative_colorbar.set_label(
    "Log₁₀ relative uncertainty"
)

fig.tight_layout()

relative_uncertainty = spatial_uncertainty[
    "relative_uncertainty"
]

uncertainty_rms = spatial_uncertainty[
    "uncertainty_rms"
]

# Worst relative uncertainty
relative_index = np.unravel_index(
    np.nanargmax(relative_uncertainty),
    relative_uncertainty.shape,
)

print(
    "Maximum relative uncertainty:",
    relative_uncertainty[relative_index],
)

print(
    "Location:",
    f"latitude={lats_plot[relative_index[0]]:.1f},",
    f"longitude={lons_plot[relative_index[1]]:.1f}",
)

# Worst absolute uncertainty
absolute_index = np.unravel_index(
    np.nanargmax(uncertainty_rms),
    uncertainty_rms.shape,
)

print(
    "Maximum absolute RMS uncertainty:",
    uncertainty_rms[absolute_index],
    "nT/yr",
)

print(
    "Location:",
    f"latitude={lats_plot[absolute_index[0]]:.1f},",
    f"longitude={lons_plot[absolute_index[1]]:.1f}",
)

# %%

def radial_sv_posterior_std(
    covariance,
    synthesis_matrix,
    latitudes,
    longitudes,
    time_indices=None,
    chunk_size=4000,
):
    """
    Calculate the exact posterior standard deviation of radial SV
    at every latitude-longitude grid point.

    Parameters
    ----------
    covariance : ndarray
        Either:

            (nt, ng, ng) time-dependent Gauss covariance, or
            (ng, ng) covariance at one epoch.

    synthesis_matrix : ndarray, shape (nlat*nlon, ng)
        Matrix mapping SV Gauss coefficients to radial SV on the grid.

    latitudes : ndarray, shape (nlat,)
        Grid latitudes in degrees.

    longitudes : ndarray, shape (nlon,)
        Grid longitudes in degrees.

    time_indices : array-like or slice, optional
        Select which covariance epochs to average. If None, use all
        epochs.

    chunk_size : int
        Number of spatial points processed at once. This limits memory
        usage without changing the result.

    Returns
    -------
    posterior_std : ndarray, shape (nlat, nlon)
        Posterior standard deviation of radial SV.

        For time-dependent covariance, this is:

            sqrt(mean_t(posterior variance))

    posterior_variance : ndarray, shape (nlat, nlon)
        Corresponding posterior variance.
    """

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    synthesis_matrix = np.asarray(
        synthesis_matrix,
        dtype=np.float64,
    )

    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    nlat = len(latitudes)
    nlon = len(longitudes)
    npoints = nlat * nlon

    if synthesis_matrix.shape[0] != npoints:
        raise ValueError(
            "The number of synthesis-matrix rows must equal "
            f"nlat*nlon={npoints}. Received "
            f"{synthesis_matrix.shape[0]} rows."
        )

    # Select and average covariance through time
    if covariance.ndim == 3:

        if time_indices is not None:
            covariance_used = covariance[time_indices]
        else:
            covariance_used = covariance

        # Gives the exact mean posterior variance through time
        covariance_mean = np.mean(
            covariance_used,
            axis=0,
        )

    elif covariance.ndim == 2:
        # Covariance from one epoch
        covariance_mean = covariance

    else:
        raise ValueError(
            "covariance must have shape (nt, ng, ng) or (ng, ng)"
        )

    ng = synthesis_matrix.shape[1]

    if covariance_mean.shape != (ng, ng):
        raise ValueError(
            "Coefficient dimensions do not agree: "
            f"A has {ng} coefficients, while covariance has "
            f"shape {covariance_mean.shape}."
        )

    # Remove tiny numerical asymmetry
    covariance_mean = 0.5 * (
        covariance_mean + covariance_mean.T
    )

    posterior_variance_flat = np.empty(
        npoints,
        dtype=np.float64,
    )

    # Calculate diag(A C A.T) without constructing the enormous
    # full spatial covariance matrix A C A.T.
    for start in tqdm(
        range(0, npoints, chunk_size),
        desc="Calculating spatial posterior variance",
    ):

        stop = min(start + chunk_size, npoints)

        A_chunk = synthesis_matrix[start:stop]

        AC_chunk = A_chunk @ covariance_mean

        posterior_variance_flat[start:stop] = np.einsum(
            "ij,ij->i",
            AC_chunk,
            A_chunk,
        )

    # Roundoff can occasionally produce extremely small negative values
    posterior_variance_flat = np.maximum(
        posterior_variance_flat,
        0.0,
    )

    posterior_std_flat = np.sqrt(
        posterior_variance_flat
    )

    posterior_variance = posterior_variance_flat.reshape(
        nlat,
        nlon,
    )

    posterior_std = posterior_std_flat.reshape(
        nlat,
        nlon,
    )

    return posterior_std, posterior_variance



posterior_std_good, posterior_variance_good = (
    radial_sv_posterior_std(
        covariance=Cov_full,
        synthesis_matrix=A_r_20,
        latitudes=latitude,
        longitudes=longitude,
        time_indices=good_record_slice,
        chunk_size=4000,
    )
)

lon_grid, lat_grid = np.meshgrid(
    longitude,
    latitude,
)

fig, ax = plt.subplots(
    figsize=(10, 5),
    subplot_kw={
        "projection": ccrs.Robinson()
    },
)

positive_std = posterior_std_good[
    posterior_std_good > 0
]

mesh = ax.pcolormesh(
    lon_grid,
    lat_grid,
    posterior_std_good,
    transform=ccrs.PlateCarree(),
    cmap="magma",
    vmin=0,
    vmax=np.nanpercentile(posterior_std_good, 98),
    shading="auto",
)

ax.coastlines(
    linewidth=0.5,
    color="white",
)

ax.set_global()

ax.set_title(
    "Posterior standard deviation of radial secular variation"
)

colorbar = fig.colorbar(
    mesh,
    ax=ax,
    orientation="horizontal",
    pad=0.06,
    fraction=0.06,
)

colorbar.set_label(
    r"Posterior standard deviation [nT yr$^{-1}$]"
)

fig.tight_layout()
# %%
