
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
# import library made for these synthetic tests
from pyscripts.R_test_synthetics.syn_pipeline import *

# defining radius choice for power to be referenced to
r_choice = r_cmb

# used time interval
used_times_dyear = times_dyear[good_record_slice]

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

chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=20,
    deriv=1,
    extrapolate="off",
)

gnm_chaos = chaos_sv_gnm[good_record_slice]

file_path = Path(f"{CHAOS_COV_DIR}/CHAOS_Cov_1997_2026_0806_SV.h5")

with h5py.File(file_path, "r") as cov_file:

    Cov_full = np.asarray(cov_file["Cnm"])
    print(f"C has shape dimensions: {np.shape(Cov_full)}")

import numpy as np
from scipy.fft import rfft, rfftfreq


def factor_time_covariance(covariance, rtol=1e-12):
    """
    Factor time-dependent covariance matrices.

    Parameters
    ----------
    covariance : (nt, ng, ng) ndarray
        Gauss-coefficient covariance at each time step.
    rtol : float
        Tolerance for small negative eigenvalues caused by roundoff.

    Returns
    -------
    L : (nt, ng, ng) ndarray
        Factors satisfying L[t] @ L[t].T ≈ covariance[t].

    Notes
    -----
    A batched Cholesky factorisation is attempted first because it is
    substantially faster than an eigendecomposition. If any covariance
    matrix is only positive semi-definite, a batched eigendecomposition
    is used instead.
    """
    covariance = np.asarray(covariance, dtype=np.float64)

    if covariance.ndim != 3:
        raise ValueError("covariance must have shape (nt, ng, ng)")

    nt, ng1, ng2 = covariance.shape

    if ng1 != ng2:
        raise ValueError("Covariance matrices must be square")

    # Ensure exact symmetry once, rather than once per realisation.
    C = 0.5 * (covariance + covariance.swapaxes(-1, -2))

    try:
        # NumPy performs this over all time steps.
        L = np.linalg.cholesky(C)

    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        scale = np.maximum(
            np.max(np.abs(eigenvalues), axis=1),
            1.0,
        )
        tolerance = rtol * scale

        bad = np.min(eigenvalues, axis=1) < -tolerance

        if np.any(bad):
            bad_times = np.flatnonzero(bad)
            raise ValueError(
                "Covariance is not positive semi-definite at time "
                f"indices {bad_times.tolist()}"
            )

        eigenvalues = np.clip(eigenvalues, 0.0, None)

        # Scales each eigenvector column by sqrt(eigenvalue).
        L = eigenvectors * np.sqrt(eigenvalues)[:, None, :]

    return L

import numpy as np
from scipy.fft import rfft, rfftfreq


def chaos_psd_ensemble_fast(
    chaos_gauss,
    covariance,
    dt,
    nmax,
    n_realisations=500,
    r=r_cmb,
    a=r_earth,
    seed=42,
    fft_workers=-1,
    return_perturbed=False,
    temporal_z="independent",
):
    """
    Generate all covariance-perturbed CHAOS realisations simultaneously
    and calculate their degree-frequency Lowes PSDs.

    Parameters
    ----------
    chaos_gauss : (nt, ng) ndarray
        Mean CHAOS Gauss-coefficient time series.
    covariance : (nt, ng, ng) ndarray
        Gauss-coefficient covariance at each time step.
    dt : float
        Sampling interval in years.
    nmax : int
        Maximum spherical-harmonic degree included in the PSD.
    n_realisations : int
        Number of Monte Carlo realisations.
    r, a : float
        Evaluation radius and reference radius.
    seed : int
        Random seed.
    fft_workers : int
        Number of FFT workers. Use -1 for all available cores.
    return_perturbed : bool
        If True, also return the full perturbed Gauss ensemble.

    Returns
    -------
    frequencies : (nf,) ndarray
    ensemble_psds : (n_realisations, nmax, nf) ndarray
    summary : dict
    perturbed_gauss : (n_realisations, nt, ng_used) ndarray, optional
        Returned only when return_perturbed=True.
    """
    chaos_gauss = np.asarray(chaos_gauss, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)

    if chaos_gauss.ndim != 2:
        raise ValueError("chaos_gauss must have shape (nt, ng)")

    nt, ng_total = chaos_gauss.shape

    if covariance.shape != (nt, ng_total, ng_total):
        raise ValueError(
            f"Expected covariance shape {(nt, ng_total, ng_total)}, "
            f"got {covariance.shape}"
        )

    ng_used = nmax * (nmax + 2)

    if ng_used > ng_total:
        raise ValueError(
            f"nmax={nmax} requires {ng_used} Gauss coefficients, "
            f"but only {ng_total} were supplied"
        )

    # Only these coefficients contribute to degrees 1,...,nmax.
    chaos_used = chaos_gauss[:, :ng_used]
    covariance_used = covariance[:, :ng_used, :ng_used]

    # Factor each time-dependent covariance exactly once.
    L = factor_time_covariance(covariance_used)

    rng = np.random.default_rng(seed)

    if temporal_z not in {"independent", "constant"}:
        raise ValueError(
            "temporal_z must be either 'independent' or 'constant'"
        )

    # Store the covariance-correlated perturbation realisations.
    perturbed_gauss = np.empty(
        (n_realisations, nt, ng_used),
        dtype=np.float64,
    )

    if temporal_z == "independent":
        # New independent z for every realisation and every time step.
        z = rng.standard_normal(
            (n_realisations, nt, ng_used)
        )

        for t in range(nt):
            perturbed_gauss[:, t, :] = z[:, t, :] @ L[t].T

    elif temporal_z == "constant":
        # One latent z per realisation, reused at every time step.
        z_constant = rng.standard_normal(
            (n_realisations, ng_used)
        )

        for t in range(nt):
            perturbed_gauss[:, t, :] = z_constant @ L[t].T

    # Add the mean CHAOS model.
    perturbed_gauss += chaos_used[None, :, :]

    # Reproduce periodogram(..., detrend="constant").
    temporal_means = np.mean(
        perturbed_gauss,
        axis=1,
        keepdims=True,
    )
    perturbed_gauss -= temporal_means

    # Transform every realisation and coefficient simultaneously.
    spectrum = rfft(
        perturbed_gauss,
        axis=1,
        workers=fft_workers,
    )

    # Restore the original perturbed Gauss arrays if they will be returned.
    if return_perturbed:
        perturbed_gauss += temporal_means
    else:
        # The demeaned Gauss array is no longer needed.
        del perturbed_gauss

    del temporal_means
    del L

    frequencies = rfftfreq(nt, d=dt)

    # Equivalent to a boxcar, density-scaled scipy periodogram.
    coefficient_psd = np.abs(spectrum)**2
    coefficient_psd *= dt / nt

    del spectrum

    # Convert to a one-sided PSD.
    if nt % 2 == 0:
        coefficient_psd[:, 1:-1, :] *= 2.0
    else:
        coefficient_psd[:, 1:, :] *= 2.0

    nf = frequencies.size

    ensemble_psds = np.empty(
        (n_realisations, nmax, nf),
        dtype=np.float64,
    )

    # Sum the coefficient PSDs within each spherical-harmonic degree.
    for n in range(1, nmax + 1):
        coefficient_slice = slice(
            n**2 - 1,
            (n + 1)**2 - 1,
        )

        lowes_weight = (
            (n + 1)
            * (a / r)**(2 * n + 4)
        )

        ensemble_psds[:, n - 1, :] = (
            lowes_weight
            * np.sum(
                coefficient_psd[:, :, coefficient_slice],
                axis=2,
            )
        )

    del coefficient_psd

    quantiles = np.quantile(
        ensemble_psds,
        [0.025, 0.05, 0.50, 0.95, 0.975],
        axis=0,
    )

    summary = {
        "minimum": np.min(ensemble_psds, axis=0),
        "maximum": np.max(ensemble_psds, axis=0),
        "mean": np.mean(ensemble_psds, axis=0),
        "std": np.std(ensemble_psds, axis=0, ddof=1),
        "q025": quantiles[0],
        "q05": quantiles[1],
        "median": quantiles[2],
        "q95": quantiles[3],
        "q975": quantiles[4],
    }

    if return_perturbed:
        return (
            frequencies,
            ensemble_psds,
            summary,
            perturbed_gauss,
        )

    return frequencies, ensemble_psds, summary
# %%

# New independent draw at every time step: temporally uncorrelated case.
frequencies, psd_realisations, psd_summary_ind = (
    chaos_psd_ensemble_fast(
        chaos_gauss=gnm_chaos,
        covariance=Cov_full[good_record_slice,:,:],
        dt=0.2,
        nmax=20,
        temporal_z="independent",
    )
)

# Same latent z used throughout time: maximally correlated end member.
frequencies, psd_realisations, psd_summary_con = (
    chaos_psd_ensemble_fast(
        chaos_gauss=gnm_chaos,
        covariance=Cov_full[good_record_slice,:,:],
        dt=0.2,
        nmax=20,
        temporal_z="constant",
    )
)

# %% getting chaos true psd

chaos_psd, _ = Lowes_Degree_PSD_All_Degrees(gnm_chaos, a=r_earth, r=r_cmb)

# %% visualising results

from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch


# Control the thickness of the diagonal hatch lines.
plt.rcParams["hatch.linewidth"] = 0.35


def centres_to_edges(values):
    """Convert one-dimensional cell-centre coordinates to cell edges."""
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if values.size == 0:
        raise ValueError("values cannot be empty")

    if values.size == 1:
        return np.array([
            values[0] - 0.5,
            values[0] + 0.5,
        ])

    midpoints = 0.5 * (values[:-1] + values[1:])

    first_edge = (
        values[0]
        - 0.5 * (values[1] - values[0])
    )
    last_edge = (
        values[-1]
        + 0.5 * (values[-1] - values[-2])
    )

    return np.concatenate([
        [first_edge],
        midpoints,
        [last_edge],
    ])


def add_unreliable_hatching(
    ax,
    x,
    y,
    unreliable_mask,
    hatch="///",
    color='red',
):
    """
    Add diagonal hatching to individual unreliable PSD cells.

    Parameters
    ----------
    ax : matplotlib axis
    x, y : one-dimensional arrays
        Cell-centre coordinates.
    unreliable_mask : (len(y), len(x)) boolean array
        True for cells that should be hatched.
    hatch : str
        Matplotlib hatch pattern.
    color : matplotlib colour
        Hatch-line colour.
    """
    x_edges = centres_to_edges(x)
    y_edges = centres_to_edges(y)

    if unreliable_mask.shape != (y.size, x.size):
        raise ValueError(
            "unreliable_mask must have shape "
            f"{(y.size, x.size)}, got {unreliable_mask.shape}"
        )

    for i, j in np.argwhere(unreliable_mask):
        rectangle = Rectangle(
            xy=(x_edges[j], y_edges[i]),
            width=x_edges[j + 1] - x_edges[j],
            height=y_edges[i + 1] - y_edges[i],
            facecolor="none",
            edgecolor=color,
            linewidth=0.5,
            hatch=hatch,
            zorder=10,
        )
        ax.add_patch(rectangle)


def plot_psd_reliability(
    frequencies,
    chaos_psd,
    summary,
    degrees=None,
    frequency_range=(0.0, 0.2),
    degree_range=(1, 20),
    relative_error_threshold=0.5,
    floor_fraction=1e-12,
    hatch="xx",
    figsize=(text_width, 0.5*text_width),
):
    """
    Visualise where covariance uncertainty makes the CHAOS PSD unreliable.

    Parameters
    ----------
    frequencies : (nf,) ndarray
        Frequency coordinates in cycles/year.
    chaos_psd : (n_degree, nf) ndarray
        PSD calculated from the unperturbed CHAOS record.
    summary : dict
        Ensemble summary containing:
            "q025", "median", "q975"
        Each array must have shape (n_degree, nf).
    degrees : array-like, optional
        Degree coordinates. Defaults to 1, ..., n_degree.
    frequency_range : tuple
        Minimum and maximum plotted frequencies.
    degree_range : tuple
        Minimum and maximum plotted degrees.
    relative_error_threshold : float
        A cell is marked unreliable when the maximum relative departure
        of the 95% interval exceeds this value.

        Examples
        --------
        0.5 : greater than 50% departure
        1.0 : greater than 100% departure
        2.0 : greater than 200% departure
    floor_fraction : float
        Numerical PSD floor relative to the maximum CHAOS PSD.
    hatch : str
        Matplotlib hatch pattern used for unreliable cells.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes : matplotlib objects
    diagnostics : dict
        Full diagnostic arrays before plotting masks are applied.
    """
    chaos_psd = np.asarray(chaos_psd)
    frequencies = np.asarray(frequencies)

    lower = np.asarray(summary["q025"])
    median = np.asarray(summary["median"])
    upper = np.asarray(summary["q975"])

    if not (
        chaos_psd.shape
        == lower.shape
        == median.shape
        == upper.shape
    ):
        raise ValueError(
            "chaos_psd and ensemble summary arrays must have "
            "the same shape"
        )

    n_degrees, n_frequencies = chaos_psd.shape

    if frequencies.size != n_frequencies:
        raise ValueError(
            "frequencies must match the frequency dimension "
            "of chaos_psd"
        )

    if degrees is None:
        degrees = np.arange(1, n_degrees + 1)
    else:
        degrees = np.asarray(degrees)

    if degrees.size != n_degrees:
        raise ValueError(
            "degrees must match the degree dimension of chaos_psd"
        )

    # --------------------------------------------------------------
    # Calculate normalized covariance diagnostics
    # --------------------------------------------------------------
    positive_chaos = chaos_psd[
        np.isfinite(chaos_psd)
        & (chaos_psd > 0)
    ]

    if positive_chaos.size == 0:
        raise ValueError(
            "chaos_psd contains no positive finite values"
        )

    psd_floor = (
        floor_fraction
        * np.max(positive_chaos)
    )

    denominator = np.maximum(
        chaos_psd,
        psd_floor,
    )

    # Width of the central 95% covariance interval relative to CHAOS.
    relative_width = (
        upper - lower
    ) / denominator

    # Shift in the ensemble median relative to unperturbed CHAOS.
    relative_bias = (
        np.abs(median - chaos_psd)
        / denominator
    )

    # Largest departure of either 95% bound from unperturbed CHAOS.
    relative_envelope_error = (
        np.maximum(
            np.abs(lower - chaos_psd),
            np.abs(upper - chaos_psd),
        )
        / denominator
    )

    unreliable = (
        ~np.isfinite(relative_envelope_error)
        | (
            relative_envelope_error
            > relative_error_threshold
        )
    )

    # --------------------------------------------------------------
    # Select plotted frequency and degree ranges
    # --------------------------------------------------------------
    frequency_mask = (
        (frequencies > frequency_range[0])
        & (frequencies <= frequency_range[1])
    )

    degree_mask = (
        (degrees >= degree_range[0])
        & (degrees <= degree_range[1])
    )

    if not np.any(frequency_mask):
        raise ValueError(
            "No frequencies lie within frequency_range"
        )

    if not np.any(degree_mask):
        raise ValueError(
            "No degrees lie within degree_range"
        )

    f_plot = frequencies[frequency_mask]
    n_plot = degrees[degree_mask]

    selection = np.ix_(
        degree_mask,
        frequency_mask,
    )

    chaos_plot = chaos_psd[selection]
    width_plot = relative_width[selection]
    error_plot = relative_envelope_error[selection]
    unreliable_plot = unreliable[selection]

    # --------------------------------------------------------------
    # Set logarithmic colour limits
    # --------------------------------------------------------------
    chaos_positive = chaos_plot[
        np.isfinite(chaos_plot)
        & (chaos_plot > 0)
    ]

    chaos_vmin = np.percentile(
        chaos_positive,
        2,
    )
    chaos_vmax = np.percentile(
        chaos_positive,
        98,
    )

    if chaos_vmax <= chaos_vmin:
        chaos_vmax = chaos_vmin * 10.0

    uncertainty_values = np.concatenate([
        width_plot[
            np.isfinite(width_plot)
            & (width_plot > 0)
        ],
        error_plot[
            np.isfinite(error_plot)
            & (error_plot > 0)
        ],
    ])

    if uncertainty_values.size == 0:
        uncertainty_vmin = 1e-3
        uncertainty_vmax = max(
            1.0,
            relative_error_threshold * 2,
        )
    else:
        uncertainty_vmin = max(
            np.percentile(
                uncertainty_values,
                2,
            ),
            1e-3,
        )

        uncertainty_vmax = max(
            np.percentile(
                uncertainty_values,
                98,
            ),
            relative_error_threshold * 2,
        )

    if uncertainty_vmax <= uncertainty_vmin:
        uncertainty_vmax = uncertainty_vmin * 10.0

    # --------------------------------------------------------------
    # Create figure
    # --------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # --------------------------------------------------------------
    # 1. CHAOS PSD
    # --------------------------------------------------------------
    image = axes[0].pcolormesh(
        f_plot,
        n_plot,
        chaos_plot,
        shading="auto",
        cmap="viridis",
        norm=LogNorm(
            vmin=chaos_vmin,
            vmax=chaos_vmax,
        ),
    )

    fig.colorbar(
        image,
        ax=axes[0],
        label="CHAOS PSD",
    )

    axes[0].set_title(
        "CHAOS PSD and reliability mask"
    )

    # --------------------------------------------------------------
    # 2. Relative width of covariance interval
    # --------------------------------------------------------------
    image = axes[1].pcolormesh(
        f_plot,
        n_plot,
        width_plot,
        shading="auto",
        cmap="magma",
        norm=LogNorm(
            vmin=uncertainty_vmin,
            vmax=uncertainty_vmax,
        ),
    )

    fig.colorbar(
        image,
        ax=axes[1],
        label=(
            r"Relative 95% width "
            r"$(Q_{97.5}-Q_{2.5})/"
            r"P_{\mathrm{CHAOS}}$"
        ),
    )

    axes[1].set_title(
        "Covariance-induced PSD spread"
    )

    # --------------------------------------------------------------
    # 3. Maximum relative envelope departure
    # --------------------------------------------------------------
    image = axes[2].pcolormesh(
        f_plot,
        n_plot,
        error_plot,
        shading="auto",
        cmap="inferno",
        norm=LogNorm(
            vmin=uncertainty_vmin,
            vmax=uncertainty_vmax,
        ),
    )

    fig.colorbar(
        image,
        ax=axes[2],
        label=(
            "Maximum relative 95% "
            "envelope departure"
        ),
    )

    axes[2].set_title(
        "Total covariance effect"
    )

    # --------------------------------------------------------------
    # Hatch unreliable cells on all panels
    # --------------------------------------------------------------
    for ax in axes:
        add_unreliable_hatching(
            ax=ax,
            x=f_plot,
            y=n_plot,
            unreliable_mask=unreliable_plot,
            hatch=hatch,
            color="red",
        )

    # --------------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------------
    frequency_edges = centres_to_edges(f_plot)
    degree_edges = centres_to_edges(n_plot)

    for ax in axes:
        ax.set_xlabel(
            r"Frequency (yr$^{-1}$)"
        )
        ax.set_xlim(
            frequency_edges[0],
            frequency_edges[-1],
        )
        ax.set_ylim(
            degree_edges[0],
            degree_edges[-1],
        )
        ax.set_yticks(n_plot)

    axes[0].set_ylabel(
        "Spherical harmonic degree"
    )

    '''# Legend showing the meaning of hatching.
    unreliable_legend = Patch(
        facecolor="white",
        edgecolor="0.35",
        linewidth=0.01,
        hatch=hatch,
        label=(
            "95% envelope departure "
            f"> {relative_error_threshold:.0%}"
        ),
    )

    axes[0].legend(
        handles=[unreliable_legend],
        loc="upper right",
        framealpha=0.9,
    )'''

    diagnostics = {
        "relative_width": relative_width,
        "relative_bias": relative_bias,
        "relative_envelope_error": relative_envelope_error,
        "unreliable": unreliable,
        "threshold": relative_error_threshold,
        "psd_floor": psd_floor,
    }

    return fig, axes, diagnostics
# %%

fig, axes, reliability = plot_psd_reliability(
    frequencies=frequencies,
    chaos_psd=chaos_psd,
    summary=psd_summary_ind,
    frequency_range=(0.0, 0.5),
    degree_range=(1, 20),
    relative_error_threshold=10,
    hatch="xxx",
)

plt.show()
# %%
fig, axes, reliability = plot_psd_reliability(
    frequencies=frequencies,
    chaos_psd=chaos_psd,
    summary=psd_summary_con,
    frequency_range=(0.0, 0.5),
    degree_range=(1, 20),
    relative_error_threshold=1,
    hatch="xxx",
)

plt.show()