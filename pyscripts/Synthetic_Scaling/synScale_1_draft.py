
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
from pyscripts.Synthetic_Scaling.synScale_utils import *

# defining radius choice for power to be referenced to
r_choice = r_cmb

# used time interval
used_times_dyear = times_dyear[good_record_slice]

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

# %%
# function to convert C -> L (cholesky) at all time steps
# input: C ~ (nt, ng1, ng2) -> output: L ~ (nt, ng1, ng2)

def factor_time_covariance(covariance):

    covariance = np.asarray(covariance, dtype=np.float64)

    # NumPy performs this over all time steps.
    L = np.linalg.cholesky(covariance)

    # checks if needed:
    '''print(np.shape(L))
    print(np.allclose(Cov_full, L @ L.swapaxes(-1, -2)))'''

    return L

# %%
import numpy as np
from scipy.fft import rfft, rfftfreq

# function to realise many perturbations of the CHAOS-8.6 model
# and return summary statistics on their PSDs
# ensure covariance and chaos gnm have same number of gauss coefficients
# i.e. nmax=20, and span same part of time record (i.e. [good_record_slice])
def Chaos_PSD_Ensemble_Cov(
    chaos_gauss,
    covariance,
    tau=10,
    nmax=20,
    n_realisations=1000,
    r=r_cmb,
    a=r_earth,
    seed=42,
    temporal_z="independent",
    window='hann',
    dt=dt_years
):
    
    nt, _ = chaos_gauss.shape

    ng_used = nmax * (nmax + 2)

    # Only these coefficients contribute to degrees 1,...,nmax.
    chaos_used = chaos_gauss[:, :ng_used]
    covariance_used = covariance[:, :ng_used, :ng_used]

    # Factor each time-dependent covariance exactly once.
    L = factor_time_covariance(covariance_used)

    rng = np.random.default_rng(seed)

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

    elif temporal_z == "ar1":
        if tau is None:
            raise ValueError(
                "tau must be provided when temporal_z='ar1'"
            )

        if tau <= 0:
            raise ValueError("tau must be positive")

        # AR(1) coefficient corresponding to the specified
        # e-folding correlation time tau.
        rho = np.exp(-dt / tau)

        # Innovation scaling ensures that every z[t] has stationary
        # marginal variance equal to one.
        innovation_scale = np.sqrt(1.0 - rho**2)

        z_ar1 = np.empty(
            (n_realisations, nt, ng_used),
            dtype=np.float64,
        )

        # Draw the initial state from the stationary N(0, I)
        # distribution.
        z_ar1[:, 0, :] = rng.standard_normal(
            (n_realisations, ng_used)
        )

        # Generate temporally correlated latent vectors.
        for t in range(1, nt):
            innovation = rng.standard_normal(
                (n_realisations, ng_used)
            )

            z_ar1[:, t, :] = (
                rho * z_ar1[:, t - 1, :]
                + innovation_scale * innovation
            )

        # Apply the time-dependent Gauss covariance factors.
        for t in range(nt):
            perturbed_gauss[:, t, :] = (
                z_ar1[:, t, :] @ L[t].T
            )

    else:
        raise ValueError(
            "temporal_z must be one of "
            "'independent', 'constant', or 'ar1'"
        )

    noise_gauss = perturbed_gauss.copy()

    # Add the mean CHAOS model.
    perturbed_gauss += chaos_used[None, :, :]

    # now applying PSD at each realisation
    # storing in an ensemble psd array
    ensemble_psds_list = []
    noise_psds_list = []

    for realisation, noise_realisation in tqdm(
        zip(perturbed_gauss, noise_gauss),
        total=n_realisations,
    ):
        psd_r, frequencies = Lowes_Degree_PSD_All_Degrees(
            realisation,
            a=a,
            r=r,
            window=window,
        )

        noise_psd_r, frequencies_noise = Lowes_Degree_PSD_All_Degrees(
            noise_realisation,
            a=a,
            r=r,
            window=window,
        )

        ensemble_psds_list.append(psd_r)
        noise_psds_list.append(noise_psd_r)

    ensemble_psds = np.stack(ensemble_psds_list, axis=0)
    noise_psds = np.stack(noise_psds_list, axis=0)

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

    return frequencies, ensemble_psds, summary, noise_psds

def Chaos_Lowes_Ensemble_Cov(
    covariance, 
    chaos_gauss,
    k,
    nmax=20,
    n_realisations=10000,
    r=r_cmb,
    a=r_earth,
    seed=42,
    temporal_z="independent",
    window='hann',
):

    ng_used = nmax * (nmax + 2)

    # Only these coefficients contribute to degrees 1,...,nmax.
    chaos_used = chaos_gauss[k,:]
    covariance_used = covariance[k,:,:]

    # Factor each time-dependent covariance exactly once.
    L = factor_time_covariance(covariance_used)

    rng = np.random.default_rng(seed)

    # Store the covariance-correlated perturbation realisations.
    perturbed_gauss = np.empty(
        (n_realisations, ng_used),
        dtype=np.float64,
    )

    if temporal_z == "independent":
        # New independent z for every realisation and every time step.
        z = rng.standard_normal(
            (n_realisations, ng_used)
        )


        perturbed_gauss = z @ L.T


    # Add the mean CHAOS model.
    perturbed_gauss += chaos_used[None, :]

    # now applying PSD at each realisation
    # storing in an ensemble psd array
    
    ensemble_lowes_list = []

    for realisation in tqdm(perturbed_gauss):
        lowes_spectrum = Instantaneous_Lowes_Spectrum(realisation,
                                                  a=a,
                                                  r=r)
        ensemble_lowes_list.append(lowes_spectrum)

    ensemble_lowes = np.stack(ensemble_lowes_list, axis=0)

    quantiles = np.quantile(
        ensemble_lowes,
        [0.025, 0.05, 0.50, 0.95, 0.975],
        axis=0,
    )

    summary = {
        "minimum": np.min(ensemble_lowes, axis=0),
        "maximum": np.max(ensemble_lowes, axis=0),
        "mean": np.mean(ensemble_lowes, axis=0),
        "std": np.std(ensemble_lowes, axis=0, ddof=1),
        "q025": quantiles[0],
        "q05": quantiles[1],
        "median": quantiles[2],
        "q95": quantiles[3],
        "q975": quantiles[4],
    }

    return ensemble_lowes, summary

k = 20
a=r_earth
r=r_cmb
spectrum = Instantaneous_Lowes_Spectrum(gnm_chaos[k,:], a=a, r=r)
_, summary = Chaos_Lowes_Ensemble_Cov(covariance=Cov_full[good_record_slice,:,:], chaos_gauss=gnm_chaos, k=k, a=a, r=r)
#%%
plt.figure(figsize=(12,12))
plt.scatter(degrees, summary["mean"], marker='^')
plt.scatter(degrees, summary["q975"],marker='x')
plt.scatter(degrees, summary["q025"],marker='x')
plt.scatter(degrees, spectrum)
plt.yscale("log")
plt.show()


# %%

def Expected_Instantaneous_Lowes_Spectrum(
    chaos_gauss,
    covariance,
    k,
    nmax=20,
    a=r_earth,
    r=r_cmb,
):
    covariance = covariance[k]
    expected = np.empty(nmax)

    for n in range(1, nmax + 1):
        i0 = n**2 - 1
        i1 = (n + 1)**2 - 1

        g_n = chaos_gauss[i0:i1]
        C_nn = covariance[i0:i1, i0:i1]

        weight = (n + 1) * (a / r)**(2 * n + 4)

        expected[n - 1] = weight * (
            g_n @ g_n + np.trace(C_nn)
        )

    return expected


expected_mean = Expected_Instantaneous_Lowes_Spectrum(
    chaos_gauss=gnm_chaos[k, :],
    covariance=Cov_full[good_record_slice,:,:],
    k=k,
    nmax=20,
    a=r_earth,
    r=r_choice,
)

plt.semilogy(degrees, summary["mean"], "^", label="Monte Carlo mean")
plt.semilogy(degrees, expected_mean, "k+", label="Analytical mean")
plt.legend()

# %%
def Degree_Mahalanobis_SNR(
    chaos_gauss,
    covariance,
    nmax=20,
    rtol=1e-10,
):
    """
    Compute degree-wise Mahalanobis signal-to-uncertainty measures.

    Parameters
    ----------
    chaos_gauss : (ng,) array
        CHAOS Gauss coefficient vector at one time.

    covariance : (ng, ng) array
        Gauss-coefficient covariance matrix at the same time.

    nmax : int
        Maximum spherical-harmonic degree.

    rtol : float
        Relative eigenvalue threshold used for the pseudoinverse.

    Returns
    -------
    mahalanobis_squared : (nmax,) array
        g_n.T @ pinv(C_nn) @ g_n.

    mahalanobis_distance : (nmax,) array
        sqrt(mahalanobis_squared).

    reduced_mahalanobis : (nmax,) array
        mahalanobis_squared divided by the effective covariance rank.

    effective_rank : (nmax,) integer array
        Number of retained covariance eigenvalues.
    """
    chaos_gauss = np.asarray(chaos_gauss, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)

    mahalanobis_squared = np.full(nmax, np.nan)
    effective_rank = np.zeros(nmax, dtype=int)

    for n in range(1, nmax + 1):
        i0 = n**2 - 1
        i1 = (n + 1)**2 - 1

        g_n = chaos_gauss[i0:i1]
        C_nn = covariance[i0:i1, i0:i1]

        # Remove small numerical asymmetry.
        C_nn = 0.5 * (C_nn + C_nn.T)

        eigenvalues, eigenvectors = np.linalg.eigh(C_nn)

        threshold = rtol * np.max(eigenvalues)
        keep = eigenvalues > threshold

        effective_rank[n - 1] = np.count_nonzero(keep)

        if effective_rank[n - 1] == 0:
            continue

        # Express the signal in the covariance eigenvector basis.
        projected_signal = eigenvectors[:, keep].T @ g_n

        mahalanobis_squared[n - 1] = np.sum(
            projected_signal**2 / eigenvalues[keep]
        )

    mahalanobis_distance = np.sqrt(mahalanobis_squared)

    reduced_mahalanobis = np.divide(
        mahalanobis_squared,
        effective_rank,
        out=np.full(nmax, np.nan),
        where=effective_rank > 0,
    )

    return {
        "squared": mahalanobis_squared,
        "distance": mahalanobis_distance,
        "reduced": reduced_mahalanobis,
        "effective_rank": effective_rank,
    }
#%%
k = 70

mahal = Degree_Mahalanobis_SNR(
    chaos_gauss=gnm_chaos[k, :],
    covariance=(Cov_full[good_record_slice])[k, :, :],
    nmax=20,
)

degrees = np.arange(1, 21)

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.5),
    constrained_layout=True,
)

axes[0].plot(
    degrees,
    mahal["squared"],
    "o-",
)
axes[0].set_yscale("log")
axes[0].set_xlabel("Spherical-harmonic degree")
axes[0].set_ylabel(r"$g_n^\mathsf{T}C_{nn}^{+}g_n$")
axes[0].set_title("Squared Mahalanobis distance")
axes[0].grid(alpha=0.25)

axes[1].plot(
    degrees,
    mahal["reduced"],
    "o-",
)
axes[1].axhline(
    1.0,
    color="k",
    linestyle="--",
    label="Unit reference",
)
axes[1].set_yscale("log")
axes[1].set_xlabel("Spherical-harmonic degree")
axes[1].set_ylabel(
    r"$g_n^\mathsf{T}C_{nn}^{+}g_n/"
    r"\mathrm{rank}(C_{nn})$"
)
axes[1].set_title("Dimension-normalized measure")
axes[1].grid(alpha=0.25)
axes[1].legend()

plt.show()

# %%

nt = gnm_chaos.shape[0]
nmax = 20

reduced_all = np.empty((nt, nmax))
squared_all = np.empty((nt, nmax))

for k in range(nt):
    result = Degree_Mahalanobis_SNR(
        chaos_gauss=gnm_chaos[k, :],
        covariance=(Cov_full[good_record_slice, :, :])[k, :, :],
        nmax=nmax,
    )

    reduced_all[k] = result["reduced"]
    squared_all[k] = result["squared"]

degrees = np.arange(1, nmax + 1)

q025, median, q975 = np.quantile(
    reduced_all,
    [0.025, 0.50, 0.975],
    axis=0,
)

plt.figure(figsize=(7, 5))

plt.fill_between(
    degrees,
    q025,
    q975,
    alpha=0.25,
    label="Central 95% across time",
)

plt.plot(
    degrees,
    median,
    "o-",
    label="Temporal median",
)

plt.plot(
    degrees,
    np.min(reduced_all, axis=0),
    "x--",
    label="Minimum across time",
)

plt.axhline(1, color="k", linestyle="--", label="Unit reference")

plt.yscale("log")
plt.xlabel("Spherical-harmonic degree")
plt.ylabel("Dimension-normalized Mahalanobis measure")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# %% doing static tests

frequencies, psd_realisations, psd_summary_ar1, psd_perturbations = (
    Chaos_PSD_Ensemble_Cov(
        chaos_gauss=gnm_chaos,
        covariance=Cov_full[good_record_slice,:,:],
        temporal_z="ar1",
        tau=10,
        window='hann'
    )
)

# %% getting chaos true psd

chaos_psd, _ = Lowes_Degree_PSD_All_Degrees(gnm_chaos, a=r_earth, r=r_cmb,
                                            window='hann')

# %%
ensemble_mean = np.mean(psd_realisations, axis=0)
noise_mean = np.mean(psd_perturbations, axis=0)

predicted_ensemble_mean = chaos_psd + noise_mean
bias_corrected_mean = ensemble_mean - noise_mean

np.allclose(
    ensemble_mean,
    predicted_ensemble_mean,
    rtol=0.1,
    atol=0,
)

# %% pulling important arrays

# THIS IS A MESS GO BACK AND THINK ABOUT IT

S_c = chaos_psd
S_u = psd_summary_ar1["q975"]
S_l = psd_summary_ar1["q025"]
S_med = psd_summary_ar1["median"]
S_mean = psd_summary_ar1["mean"]

f_mask = (frequencies < 0.33) & (frequencies > 0)

plt.imshow(((S_u - S_med)/S_med)[:, f_mask], cmap='seismic', vmin=-1, vmax=1)
plt.show()
plt.imshow((noise_mean/S_c)[:, f_mask], cmap='gray', vmin=0.5, vmax=1)
plt.show()
# %% visualising results
plt.scatter(np.arange(0,148,1),H_sv[:, 32])
# %%


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
