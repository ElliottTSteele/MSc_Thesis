
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

# LOAD IN CHAOS DATA

chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=20,
    deriv=1,
    extrapolate="off",
)

gnm_chaos = chaos_sv_gnm[good_record_slice]

# COMPUTE PSD OF CHAOS

chaos_spectrum, chaos_f = Lowes_Degree_PSD_All_Degrees(gnm_chaos, a=r_earth, r=r_cmb,
                                            window='hann')

# LOAD IN COVARIANCE
def factor_time_covariance(covariance):

    covariance = np.asarray(covariance, dtype=np.float64)

    # NumPy performs this over all time steps.
    L = np.linalg.cholesky(covariance)

    # checks if needed:
    '''print(np.shape(L))
    print(np.allclose(Cov_full, L @ L.swapaxes(-1, -2)))'''

    return L

file_path = Path(f"{CHAOS_COV_DIR}/CHAOS_Cov_1997_2026_0806_SV.h5")

with h5py.File(file_path, "r") as cov_file:

    Cov_full = np.asarray(cov_file["Cnm"])
    chaos_from_cfile = np.asarray(cov_file["gnm"])
    print(f"C has shape dimensions: {np.shape(Cov_full)}")

print(np.allclose(chaos_from_cfile, chaos_sv_gnm))
#%%

# REALISE AN ENSEMBLE SET OF PERTURBATIONS, GET PSD FOR EACH
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

# FROM THIS GET THE MEAN - ON AVERAGE HOW MUCH DOES NOISE CONTRIBUTE TO CHAOS POWER

# FIND NOISE / CHAOS - IF > 1, THEN NOISE DOMINATES OVER CHAOS SIGNAL

# IF NOISE DOMINATES, CONSIDER POINT UNRELIABLE FOR PSD METRICS

# EXCLUDE FROM FITTING REGION

# THEN, LOAD IN SYNTHETIC DATA, COMPUTE PSD OF EACH, THEN DO WATER LEVEL SCALING TO CHAOS PSD


# THEN DO SOME KIND OF PUZZLE PIECE ALGORITHM TO FIT A COMBINATION TOGETHER THAT
# FITS PSD REASONABLY WELL OVER 'RELIABLE REGION'

# STRICT FULL-SPECTRUM WATER-LEVEL SCALING
# %%
mode_periods = []
water_level_power_scales = []
scaled_mode_mean_powers = []
scaled_mode_series = []
scaled_mode_spectra = []

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

mode_numbers = np.arange(1, 63, 1)

with h5py.File(file_path, "r") as h5_file:

    for mode_number in tqdm(mode_numbers, desc="Scaling modes"):

        mode_data = Component_Load(mode_number)
        eigenvalue = mode_data["eigenvalue"]

        # Period implied by the provided eigenvalue
        period = 2 * np.pi / np.abs(eigenvalue.imag)

        gnm_spl = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )

        gnm_mode = H_sv @ gnm_spl
        gnm_mode_good = gnm_mode[good_record_slice]

        mode_spectrum, mode_f = Lowes_Degree_PSD_All_Degrees(
            gnm_mode_good,
            a=r_earth,
            r=r_choice,
        )

        if not np.allclose(mode_f, chaos_f):
            raise ValueError(
                f"Frequency mismatch for mode {mode_number}"
            )
        
        # cropping spectrum match zone
        f_mask = chaos_f < 0.33

        # Strict water level over the complete degree-frequency spectrum.
        valid = (
            np.isfinite(chaos_spectrum)
            & np.isfinite(mode_spectrum)
            & (chaos_spectrum > 0)
            & (mode_spectrum > 0)
        )

        power_scale = np.min(
            chaos_spectrum[:, f_mask]
            / mode_spectrum[:, f_mask]
        )

        power_scale_location = np.argmin(
            chaos_spectrum[:, f_mask]
            / mode_spectrum[:, f_mask]
        )

        power_peak_location = np.argmax(
            chaos_spectrum[:, f_mask]
            / mode_spectrum[:, f_mask]
        )
        pi, pj = np.unravel_index(power_scale_location, chaos_spectrum[:, f_mask].shape)
        mi, mj = np.unravel_index(power_peak_location, chaos_spectrum[:, f_mask].shape)

        plt.scatter(pj, pi, marker='x')
        plt.scatter(mi, mj, marker='o')

        scaled_series = (
            gnm_mode_good
            * np.sqrt(power_scale)
        )

        scaled_mean_power = Mean_Instantaneous_Total_Lowes_Power(
            scaled_series,
            a=r_earth,
            r=r_choice,
        )

        mode_periods.append(period)
        water_level_power_scales.append(power_scale)
        scaled_mode_mean_powers.append(scaled_mean_power)
        scaled_mode_series.append(scaled_series)
        scaled_mode_spectra.append(
            mode_spectrum * power_scale
        )

mode_periods = np.asarray(mode_periods)
water_level_power_scales = np.asarray(water_level_power_scales)
scaled_mode_mean_powers = np.asarray(scaled_mode_mean_powers)
scaled_mode_series = np.asarray(scaled_mode_series)
scaled_mode_spectra = np.asarray(scaled_mode_spectra)

water_level_scale_by_mode = dict(
    zip(mode_numbers, water_level_power_scales)
)

#  SCALED MODE POWER AGAINST PROVIDED PERIOD

period_order = np.argsort(mode_periods)

with plt.rc_context({"font.size": 12}):

    fig, ax = plt.subplots(
        figsize=(text_width, 0.55 * text_width),
        constrained_layout=True,
    )

    ax.plot(
        mode_periods[period_order],
        scaled_mode_mean_powers[period_order],
        "x",
        markersize=6,
    )
    '''
    ax.axhline(
        chaos_mean_power,
        linestyle="--",
        linewidth=1,
        label="CHAOS-8.6 total mean power",
    )'''

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("Wave period / yr")
    ax.set_ylabel(
        r"Mean instantaneous Lowes SV power "
        r"/ $(\mathrm{nT\,yr^{-1}})^2$"
    )
    ax.set_title("Water-level-scaled wave power at the CMB")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.savefig(
        f"{FIG_DIR}/water_level_scaled_mode_power.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

# %% OPTIMISATION PATCH: n = 1--8, 0 < f < 0.2 yr^-1

degree_fit_mask = (
    (chaos_degrees >= degree_min)
    & (chaos_degrees <= degree_max)
)

frequency_fit_mask = (
    (chaos_f > frequency_min)
    & (chaos_f < frequency_max)
)

# getting rid of first bin
frequency_fit_mask[1] = False

chaos_patch = chaos_spectrum[
    np.ix_(degree_fit_mask, frequency_fit_mask)
]

mode_patches = scaled_mode_spectra[
    :,
    degree_fit_mask,
    :,
][
    :,
    :,
    frequency_fit_mask,
]

fit_valid = (
    np.isfinite(chaos_patch)
    & (chaos_patch > 0)
)

psd_epsilon = (
    1e-15
    * np.nanmax(chaos_patch[fit_valid])
)


def rms_log_misfit(candidate_patch):
    error = np.log10(
        (candidate_patch[fit_valid] + psd_epsilon)
        / (chaos_patch[fit_valid] + psd_epsilon)
    )
    return np.sqrt(np.mean(error**2))


# GREEDY LINEAR-SPECTRUM INITIAL SET

initial_indices = []
remaining_indices = list(range(len(mode_numbers)))
current_linear_patch = np.zeros_like(chaos_patch)

for _ in range(n_selected_modes):

    trial_losses = [
        rms_log_misfit(
            current_linear_patch + mode_patches[idx]
        )
        for idx in remaining_indices
    ]

    best_position = int(np.argmin(trial_losses))
    best_index = remaining_indices.pop(best_position)

    initial_indices.append(best_index)
    current_linear_patch += mode_patches[best_index]

initial_set = set(initial_indices)

# TRUE COHERENT POWER-SPECTRUM OBJECTIVE

loss_cache = {}


def evaluate_mode_set(mode_set, gnm_total=None):

    key = tuple(sorted(mode_set))

    if key in loss_cache:
        return loss_cache[key]

    if gnm_total is None:
        gnm_total = np.sum(
            scaled_mode_series[list(key)],
            axis=0,
        )

    spectrum, _ = Lowes_Degree_PSD_All_Degrees(
        gnm_total,
        a=r_earth,
        r=r_choice,
    )

    spectrum_patch = spectrum[
        np.ix_(degree_fit_mask, frequency_fit_mask)
    ]

    loss = rms_log_misfit(spectrum_patch)
    loss_cache[key] = loss

    return loss


initial_gnm = np.sum(
    scaled_mode_series[list(initial_set)],
    axis=0,
)

initial_loss = evaluate_mode_set(
    initial_set,
    initial_gnm,
)

# SIMULATED ANNEALING USING THE TRUE COHERENT SPECTRUM

rng = np.random.default_rng(random_seed)
all_indices = np.arange(len(mode_numbers))

best_set = initial_set.copy()
best_gnm = initial_gnm.copy()
best_loss = initial_loss

for restart in range(n_restarts):

    current_set = initial_set.copy()

    if restart > 0:
        n_perturb = max(
            1,
            int(np.ceil(0.25 * n_selected_modes)),
        )

        for _ in range(n_perturb):
            selected = np.asarray(list(current_set))
            unselected = np.setdiff1d(
                all_indices,
                selected,
            )

            mode_out = int(rng.choice(selected))
            mode_in = int(rng.choice(unselected))

            current_set.remove(mode_out)
            current_set.add(mode_in)

    current_gnm = np.sum(
        scaled_mode_series[list(current_set)],
        axis=0,
    )

    current_loss = evaluate_mode_set(
        current_set,
        current_gnm,
    )

    for step in range(annealing_steps):

        fraction = step / max(annealing_steps - 1, 1)

        temperature = (
            temperature_start
            * (temperature_end / temperature_start) ** fraction
        )

        selected = np.asarray(list(current_set))
        unselected = np.setdiff1d(
            all_indices,
            selected,
        )

        mode_out = int(rng.choice(selected))
        mode_in = int(rng.choice(unselected))

        candidate_set = current_set.copy()
        candidate_set.remove(mode_out)
        candidate_set.add(mode_in)

        candidate_gnm = (
            current_gnm
            - scaled_mode_series[mode_out]
            + scaled_mode_series[mode_in]
        )

        candidate_loss = evaluate_mode_set(
            candidate_set,
            candidate_gnm,
        )

        loss_change = candidate_loss - current_loss

        if (
            loss_change < 0
            or rng.random()
            < np.exp(-loss_change / temperature)
        ):
            current_set = candidate_set
            current_gnm = candidate_gnm
            current_loss = candidate_loss

        if current_loss < best_loss:
            best_set = current_set.copy()
            best_gnm = current_gnm.copy()
            best_loss = current_loss

    print(
        f"Restart {restart + 1}: "
        f"best RMS log error = {best_loss:.4f}"
    )

# DETERMINISTIC ONE-FOR-ONE SWAP REFINEMENT

for swap_pass in range(max_swap_passes):

    selected = np.asarray(sorted(best_set))
    unselected = np.setdiff1d(
        all_indices,
        selected,
    )

    best_swap = None
    best_swap_loss = best_loss
    best_swap_gnm = None

    for mode_out in selected:
        for mode_in in unselected:

            candidate_set = best_set.copy()
            candidate_set.remove(int(mode_out))
            candidate_set.add(int(mode_in))

            candidate_gnm = (
                best_gnm
                - scaled_mode_series[mode_out]
                + scaled_mode_series[mode_in]
            )

            candidate_loss = evaluate_mode_set(
                candidate_set,
                candidate_gnm,
            )

            if candidate_loss < best_swap_loss - 1e-6:
                best_swap = candidate_set
                best_swap_loss = candidate_loss
                best_swap_gnm = candidate_gnm

    if best_swap is None:
        break

    best_set = best_swap
    best_loss = best_swap_loss
    best_gnm = best_swap_gnm

    print(
        f"Swap pass {swap_pass + 1}: "
        f"RMS log error = {best_loss:.4f}"
    )

# FINAL OPTIMAL COHERENT COMBINATION

optimal_indices = np.asarray(
    sorted(best_set),
    dtype=int,
)

optimal_mode_numbers = mode_numbers[optimal_indices]
optimal_mode_periods = mode_periods[optimal_indices]
optimal_gnm_total = best_gnm

optimal_spectrum, optimal_f = Lowes_Degree_PSD_All_Degrees(
    optimal_gnm_total,
    a=r_earth,
    r=r_choice,
)

optimal_patch = optimal_spectrum[
    np.ix_(degree_fit_mask, frequency_fit_mask)
]

optimal_log_error = np.full_like(
    chaos_patch,
    np.nan,
    dtype=float,
)

error_valid = (
    fit_valid
    & np.isfinite(optimal_patch)
    & (optimal_patch >= 0)
)

optimal_log_error[error_valid] = np.log10(
    (optimal_patch[error_valid] + psd_epsilon)
    / (chaos_patch[error_valid] + psd_epsilon)
)

error_values = optimal_log_error[error_valid]

print("\nSelected modes:", optimal_mode_numbers)
print("Selected periods:", optimal_mode_periods)
print(f"Final RMS log10 error: {best_loss:.4f}")
print(
    "Cells within factor 2:",
    f"{np.mean(np.abs(error_values) <= np.log10(2)):.1%}",
)
print(
    "Cells within factor 3:",
    f"{np.mean(np.abs(error_values) <= np.log10(3)):.1%}",
)

# %% FINAL THREE-PANEL CMB POWER-SPECTRUM FIGURE

degree_patch = chaos_degrees[degree_fit_mask]
frequency_patch = chaos_f[frequency_fit_mask]

chaos_patch_masked = np.ma.masked_where(
    ~fit_valid,
    chaos_patch,
)

optimal_patch_masked = np.ma.masked_where(
    ~error_valid,
    optimal_patch,
)

positive_power = np.concatenate([
    chaos_patch_masked.compressed(),
    optimal_patch_masked.compressed(),
])

power_norm = LogNorm(
    vmin=np.min(positive_power),
    vmax=np.max(positive_power),
)

power_cmap = plt.get_cmap("viridis").copy()
power_cmap.set_bad("lightgrey")

factor_2 = np.log10(2)
factor_3 = np.log10(3)

max_error = np.nanmax(np.abs(optimal_log_error))
error_cap = max(1.01, max_error + 1e-6)

error_bounds = [
    -error_cap,
    -1,
    -factor_3,
    -factor_2,
    factor_2,
    factor_3,
    1,
    error_cap,
]

error_colors = [
    "#08306B",
    "#2171B5",
    "#9ECAE1",
    "#FFFFFF",
    "#FCAE91",
    "#FB6A4A",
    "#CB181D",
]

error_cmap = ListedColormap(error_colors)
error_cmap.set_bad("lightgrey")

error_norm = BoundaryNorm(
    error_bounds,
    error_cmap.N,
    clip=True,
)

df = np.mean(np.diff(frequency_patch))

extent = [
    frequency_patch[0] - df / 2,
    frequency_patch[-1] + df / 2,
    degree_patch[0] - 0.5,
    degree_patch[-1] + 0.5,
]
# 
with plt.rc_context({"font.size": 12}):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(text_width, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    chaos_im = axes[0].imshow(
        chaos_patch_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=power_cmap,
        norm=power_norm,
    )

    axes[1].imshow(
        optimal_patch_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=power_cmap,
        norm=power_norm,
    )

    error_im = axes[2].imshow(
        np.ma.masked_invalid(optimal_log_error),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=error_cmap,
        norm=error_norm,
    )

    axes[0].set_title("CHAOS-8.6")
    axes[1].set_title("Optimal mode\n combination")
    axes[2].set_title(
        r"$\log_{10}(P_{\mathrm{wave}}/P_{\mathrm{CHAOS}})$"
    )

    for ax in axes:
        ax.set_xlabel(
            r"Frequency / $\mathrm{yr}^{-1}$"
        )
        ax.set_yticks(degree_patch)

    axes[0].set_ylabel(
        r"Spherical harmonic degree $n$"
    )

    power_cbar = fig.colorbar(
        chaos_im,
        ax=axes[:2],
        location="bottom",
        fraction=0.08,
        pad=0.13,
    )

    power_cbar.set_label(
        r"Lowes SV power spectrum at the CMB "
        r"/ $(\mathrm{nT\,yr^{-1}})^2$"
    )

    error_cbar = fig.colorbar(
        error_im,
        ax=axes[2],
        boundaries=error_bounds,
        ticks=[
            -1,
            -factor_3,
            -factor_2,
            factor_2,
            factor_3,
            1,
        ],
    )

    error_cbar.set_ticklabels([
        r"$-1$",
        r"$-\log_{10}(3)$",
        r"$-\log_{10}(2)$",
        r"$\log_{10}(2)$",
        r"$\log_{10}(3)$",
        r"$1$",
    ])

    error_cbar.set_label(
        r"$\log_{10}(P_{\mathrm{wave}}/P_{\mathrm{CHAOS}})$"
    )

    fig.savefig(
        f"{FIG_DIR}/final/optimal_wave_chaos_power_spectrum.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
