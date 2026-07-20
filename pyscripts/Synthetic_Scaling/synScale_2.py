# %% IMPORTS AND SETTINGS
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

# %% rest of code
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

r_choice = r_cmb
mode_numbers = np.arange(1, 63)

n_selected_modes = 12
random_seed = 2
n_restarts = 3
annealing_steps = 1200
temperature_start = 0.05
temperature_end = 1e-4
max_swap_passes = 4

degree_min, degree_max = 1, 11
frequency_min, frequency_max = 0.0, 0.2

# CHAOS SV POWER SPECTRUM AT THE CMB

chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=20,
    deriv=1,
    extrapolate="off",
)

chaos_sv_gnm = Truncate_Gauss_Coeffs(
    np.asarray(chaos_sv_gnm),
    20,
)

chaos_good = chaos_sv_gnm[good_record_slice]

chaos_spectrum, chaos_f = Lowes_Degree_PSD_All_Degrees(
    chaos_good,
    a=r_earth,
    r=r_choice,
)

chaos_degrees = np.arange(1, chaos_spectrum.shape[0] + 1)

chaos_mean_power = Mean_Instantaneous_Total_Lowes_Power(
    chaos_good,
    a=r_earth,
    r=r_choice,
)

# STRICT FULL-SPECTRUM WATER-LEVEL SCALING

mode_periods = []
water_level_power_scales = []
scaled_mode_mean_powers = []
scaled_mode_series = []
scaled_mode_spectra = []

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

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

        # Strict water level over the complete degree-frequency spectrum.
        valid = (
            np.isfinite(chaos_spectrum)
            & np.isfinite(mode_spectrum)
            & (chaos_spectrum > 0)
            & (mode_spectrum > 0)
        )

        power_scale = np.min(
            chaos_spectrum[valid]
            / mode_spectrum[valid]
        )

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

    ax.axhline(
        chaos_mean_power,
        linestyle="--",
        linewidth=1,
        label="CHAOS-8.6 total mean power",
    )

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

# OPTIMISATION PATCH: n = 1--8, 0 < f < 0.2 yr^-1

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
