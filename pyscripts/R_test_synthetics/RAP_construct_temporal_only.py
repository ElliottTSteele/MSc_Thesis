# %% IMPORTS AND SETTINGS
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import sys
import pickle
from pathlib import Path

import chaosmagpy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.msc_thesis.paths import *
from pyscripts.R_test_synthetics.syn_pipeline import *


# %% EXPERIMENT SETTINGS
r_choice = r_cmb
mode_numbers = np.arange(1, 63)

n_selected_modes = 8
random_seed = 2
n_restarts = 3
annealing_steps = 1200
temperature_start = 0.05
temperature_end = 1e-4
max_swap_passes = 4

# Temporal-frequency interval used by the optimisation.
frequency_min, frequency_max = 0.0, 0.3

# Temporal-frequency interval used for each individual mode's strict
# water-level scaling. The zero-frequency bin is excluded below.
water_level_frequency_max = 0.33

# All degrees returned by Lowes_Degree_PSD_All_Degrees are summed. Because the
# inputs are truncated to nmax=20, this is total Lowes power over n=1,...,20.
nmax_power = 20


def total_temporal_lowes_spectrum(gnm, a, r):
    """Return total Lowes power versus frequency, summed over degree."""
    degree_spectrum, frequencies = Lowes_Degree_PSD_All_Degrees(
        gnm,
        a=a,
        r=r,
    )
    return np.sum(degree_spectrum, axis=0), frequencies


def log_error_statistics(candidate, target, valid, epsilon):
    """Return log10 ratio, RMS log error, and factor-2/3 fractions."""
    log_error = np.full_like(target, np.nan, dtype=float)
    comparison_valid = (
        valid
        & np.isfinite(candidate)
        & (candidate >= 0)
    )
    log_error[comparison_valid] = np.log10(
        (candidate[comparison_valid] + epsilon)
        / (target[comparison_valid] + epsilon)
    )
    values = log_error[comparison_valid]
    return {
        "log_error": log_error,
        "valid": comparison_valid,
        "rms": np.sqrt(np.mean(values**2)),
        "within_factor_2": np.mean(
            np.abs(values) <= np.log10(2)
        ),
        "within_factor_3": np.mean(
            np.abs(values) <= np.log10(3)
        ),
    }


def plot_temporal_spectrum_comparison(
    frequencies,
    chaos_spectrum,
    wave_spectrum,
    valid,
    title_suffix,
    output_path,
    fit_interval=None,
):
    """Three panels: CHAOS PSD, wave PSD, and their log10 ratio."""
    epsilon = 1e-15 * np.nanmax(chaos_spectrum[valid])
    stats = log_error_statistics(
        wave_spectrum,
        chaos_spectrum,
        valid,
        epsilon,
    )

    chaos_plot = np.where(valid, chaos_spectrum, np.nan)
    wave_plot = np.where(stats["valid"], wave_spectrum, np.nan)

    positive_power = np.concatenate([
        chaos_plot[np.isfinite(chaos_plot) & (chaos_plot > 0)],
        wave_plot[np.isfinite(wave_plot) & (wave_plot > 0)],
    ])
    power_limits = (
        np.min(positive_power),
        np.max(positive_power),
    )

    with plt.rc_context({"font.size": 12}):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(text_width, 4.5),
            sharex=True,
            constrained_layout=True,
        )

        axes[0].plot(frequencies, chaos_plot, color="black")
        axes[1].plot(frequencies, wave_plot, color="tab:blue")
        axes[2].plot(
            frequencies,
            stats["log_error"],
            color="tab:red",
        )

        axes[0].set_title("CHAOS-8.6")
        axes[1].set_title("Optimal mode\ncombination")
        axes[2].set_title(
            r"$\log_{10}(P_{\mathrm{wave}}/P_{\mathrm{CHAOS}})$"
        )

        for ax in axes[:2]:
            ax.set_yscale("log")
            ax.set_ylim(power_limits)
            ax.set_ylabel(
                r"Total Lowes SV power spectrum at the CMB "
                r"/ $(\mathrm{nT\,yr^{-1}})^2\,\mathrm{yr}$"
            )

        axes[2].axhline(0, color="black", linewidth=1)
        axes[2].axhline(
            np.log10(2),
            color="0.5",
            linestyle=":",
            linewidth=1,
        )
        axes[2].axhline(
            -np.log10(2),
            color="0.5",
            linestyle=":",
            linewidth=1,
        )
        axes[2].set_ylabel(
            r"$\log_{10}(P_{\mathrm{wave}}/P_{\mathrm{CHAOS}})$"
        )

        for ax in axes:
            ax.set_xlabel(r"Frequency / $\mathrm{yr}^{-1}$")
            ax.grid(True, which="both", alpha=0.25)
            if fit_interval is not None:
                ax.axvspan(
                    fit_interval[0],
                    fit_interval[1],
                    facecolor="none",
                    edgecolor="black",
                    linestyle=":",
                    linewidth=1.5,
                    label="Fit region",
                )

        if fit_interval is not None:
            axes[2].legend(loc="best", frameon=True)

        fig.suptitle(title_suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

    return stats


# %% CHAOS TOTAL TEMPORAL LOWES POWER SPECTRUM AT THE CMB
chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=nmax_power,
    deriv=1,
    extrapolate="off",
)
chaos_sv_gnm = Truncate_Gauss_Coeffs(
    np.asarray(chaos_sv_gnm),
    nmax_power,
)
chaos_good = chaos_sv_gnm[good_record_slice]

chaos_spectrum, chaos_f = total_temporal_lowes_spectrum(
    chaos_good,
    a=r_earth,
    r=r_choice,
)

chaos_mean_power = Mean_Instantaneous_Total_Lowes_Power(
    chaos_good,
    a=r_earth,
    r=r_choice,
)


# %% STRICT 1D TEMPORAL-SPECTRUM WATER-LEVEL SCALING
mode_periods = []
water_level_power_scales = []
scaled_mode_mean_powers = []
scaled_mode_series = []
scaled_mode_spectra = []
scalings_dict = {}

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

with h5py.File(file_path, "r") as h5_file:
    for mode_number in tqdm(mode_numbers, desc="Scaling modes"):
        mode_data = Component_Load(mode_number)
        eigenvalue = mode_data["eigenvalue"]
        period = 2 * np.pi / np.abs(eigenvalue.imag)

        gnm_spl = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )
        gnm_mode = H_sv @ gnm_spl
        gnm_mode_good = gnm_mode[good_record_slice]

        mode_spectrum, mode_f = total_temporal_lowes_spectrum(
            gnm_mode_good,
            a=r_earth,
            r=r_choice,
        )

        if not np.allclose(mode_f, chaos_f):
            raise ValueError(
                f"Frequency mismatch for mode {mode_number}"
            )

        water_level_valid = (
            np.isfinite(chaos_spectrum)
            & np.isfinite(mode_spectrum)
            & (chaos_spectrum > 0)
            & (mode_spectrum > 0)
            & (chaos_f > 0) # PUT HERE IF YOU WANT TO INCLUDE 0 FREQUENCY BIN OR NOT
            & (chaos_f < water_level_frequency_max)
        )
        if not np.any(water_level_valid):
            raise ValueError(
                f"No valid water-level cells for mode {mode_number}"
            )

        power_scale = np.min(
            chaos_spectrum[water_level_valid]
            / mode_spectrum[water_level_valid]
        )
        amplitude_scale = np.sqrt(power_scale)
        scaled_series = gnm_mode_good * amplitude_scale

        scaled_mean_power = Mean_Instantaneous_Total_Lowes_Power(
            scaled_series,
            a=r_earth,
            r=r_choice,
        )

        mode_periods.append(period)
        water_level_power_scales.append(power_scale)
        scaled_mode_mean_powers.append(scaled_mean_power)
        scaled_mode_series.append(scaled_series)
        scaled_mode_spectra.append(mode_spectrum * power_scale)
        scalings_dict[str(mode_number)] = amplitude_scale

mode_periods = np.asarray(mode_periods)
water_level_power_scales = np.asarray(water_level_power_scales)
scaled_mode_mean_powers = np.asarray(scaled_mode_mean_powers)
scaled_mode_series = np.asarray(scaled_mode_series)
scaled_mode_spectra = np.asarray(scaled_mode_spectra)

water_level_scale_by_mode = dict(
    zip(mode_numbers, water_level_power_scales)
)

scale_file = (
    Path(FELIX_DIR)
    / "mode_amplitude_scalings_total_temporal_spectrum.pkl"
)
with open(scale_file, "wb") as file:
    pickle.dump(scalings_dict, file)


# %% WATER-LEVEL-SCALED MODE POWER AGAINST PERIOD
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
    ax.set_yscale("log")
    ax.set_xlabel("Wave period / yr")
    ax.set_ylabel(
        r"Mean instantaneous Lowes SV power "
        r"/ $(\mathrm{nT\,yr^{-1}})^2$"
    )
    ax.set_title(
        "Temporal-spectrum-water-level-scaled wave power at the CMB"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(
        f"{FIG_DIR}/water_level_scaled_mode_power_total_temporal_spectrum.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# %% TEMPORAL-FREQUENCY OPTIMISATION INTERVAL
frequency_fit_mask = (
    (chaos_f > max(frequency_min, 0))
    & (chaos_f < frequency_max)
)

chaos_patch = chaos_spectrum[frequency_fit_mask]
mode_patches = scaled_mode_spectra[:, frequency_fit_mask]

fit_valid = np.isfinite(chaos_patch) & (chaos_patch > 0)
if not np.any(fit_valid):
    raise ValueError("No valid CHAOS cells in the fit interval")

psd_epsilon = 1e-15 * np.nanmax(chaos_patch[fit_valid])


def rms_log_misfit(candidate_patch):
    error = np.log10(
        (candidate_patch[fit_valid] + psd_epsilon)
        / (chaos_patch[fit_valid] + psd_epsilon)
    )
    return np.sqrt(np.mean(error**2))


# %% GREEDY LINEAR-SPECTRUM INITIAL SET
initial_indices = []
remaining_indices = list(range(len(mode_numbers)))
current_linear_patch = np.zeros_like(chaos_patch)

for _ in range(n_selected_modes):
    trial_losses = [
        rms_log_misfit(current_linear_patch + mode_patches[idx])
        for idx in remaining_indices
    ]
    best_position = int(np.argmin(trial_losses))
    best_index = remaining_indices.pop(best_position)
    initial_indices.append(best_index)
    current_linear_patch += mode_patches[best_index]

initial_set = set(initial_indices)


# %% TRUE COHERENT TEMPORAL-SPECTRUM OBJECTIVE
loss_cache = {}


def evaluate_mode_set(mode_set, gnm_total=None):
    key = tuple(sorted(mode_set))
    if key in loss_cache:
        return loss_cache[key]

    if gnm_total is None:
        gnm_total = np.sum(scaled_mode_series[list(key)], axis=0)

    spectrum, _ = total_temporal_lowes_spectrum(
        gnm_total,
        a=r_earth,
        r=r_choice,
    )
    loss = rms_log_misfit(spectrum[frequency_fit_mask])
    loss_cache[key] = loss
    return loss


initial_gnm = np.sum(
    scaled_mode_series[list(initial_set)],
    axis=0,
)
initial_loss = evaluate_mode_set(initial_set, initial_gnm)


# %% SIMULATED ANNEALING
rng = np.random.default_rng(random_seed)
all_indices = np.arange(len(mode_numbers))

best_set = initial_set.copy()
best_gnm = initial_gnm.copy()
best_loss = initial_loss

for restart in range(n_restarts):
    current_set = initial_set.copy()

    if restart > 0:
        n_perturb = max(1, int(np.ceil(0.25 * n_selected_modes)))
        for _ in range(n_perturb):
            selected = np.asarray(list(current_set))
            unselected = np.setdiff1d(all_indices, selected)
            mode_out = int(rng.choice(selected))
            mode_in = int(rng.choice(unselected))
            current_set.remove(mode_out)
            current_set.add(mode_in)

    current_gnm = np.sum(
        scaled_mode_series[list(current_set)],
        axis=0,
    )
    current_loss = evaluate_mode_set(current_set, current_gnm)

    for step in range(annealing_steps):
        fraction = step / max(annealing_steps - 1, 1)
        temperature = temperature_start * (
            temperature_end / temperature_start
        ) ** fraction

        selected = np.asarray(list(current_set))
        unselected = np.setdiff1d(all_indices, selected)
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
            or rng.random() < np.exp(-loss_change / temperature)
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


# %% DETERMINISTIC ONE-FOR-ONE SWAP REFINEMENT
for swap_pass in range(max_swap_passes):
    selected = np.asarray(sorted(best_set))
    unselected = np.setdiff1d(all_indices, selected)
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


# %% FINAL OPTIMAL COHERENT COMBINATION AND METRICS
optimal_indices = np.asarray(sorted(best_set), dtype=int)
optimal_mode_numbers = mode_numbers[optimal_indices]
optimal_mode_periods = mode_periods[optimal_indices]
optimal_gnm_total = best_gnm

optimal_spectrum, optimal_f = total_temporal_lowes_spectrum(
    optimal_gnm_total,
    a=r_earth,
    r=r_choice,
)
if not np.allclose(optimal_f, chaos_f):
    raise ValueError("Optimal and CHAOS frequency arrays do not match")

optimal_patch = optimal_spectrum[frequency_fit_mask]
fit_stats = log_error_statistics(
    optimal_patch,
    chaos_patch,
    fit_valid,
    psd_epsilon,
)

print("\nSelected modes:", optimal_mode_numbers)
print("Selected periods:", optimal_mode_periods)
print(f"Final RMS log10 error: {fit_stats['rms']:.4f}")
print(
    "Frequency bins within factor 2:",
    f"{fit_stats['within_factor_2']:.1%}",
)
print(
    "Frequency bins within factor 3:",
    f"{fit_stats['within_factor_3']:.1%}",
)


# %% THREE-PANEL COMPARISON OVER THE FIT INTERVAL
frequency_patch = chaos_f[frequency_fit_mask]

plot_temporal_spectrum_comparison(
    frequencies=frequency_patch,
    chaos_spectrum=chaos_patch,
    wave_spectrum=optimal_patch,
    valid=fit_valid,
    title_suffix="Total temporal spectrum over the optimisation interval",
    output_path=(
        f"{FIG_DIR}/final/"
        "optimal_wave_chaos_total_temporal_spectrum_fit.png"
    ),
)


# %% THREE-PANEL COMPARISON OVER THE FULL DISPLAY INTERVAL
frequency_plot_mask = (chaos_f > 0) & (chaos_f < 0.5)
frequency_plot = chaos_f[frequency_plot_mask]
chaos_plot = chaos_spectrum[frequency_plot_mask]
optimal_plot = optimal_spectrum[frequency_plot_mask]
plot_valid = (
    np.isfinite(chaos_plot)
    & np.isfinite(optimal_plot)
    & (chaos_plot > 0)
    & (optimal_plot > 0)
)

full_stats = plot_temporal_spectrum_comparison(
    frequencies=frequency_plot,
    chaos_spectrum=chaos_plot,
    wave_spectrum=optimal_plot,
    valid=plot_valid,
    title_suffix="Total temporal spectrum over the full display interval",
    output_path=(
        f"{FIG_DIR}/final/"
        "optimal_wave_chaos_total_temporal_spectrum_full.png"
    ),
    fit_interval=(
        max(frequency_min, frequency_plot[0]),
        min(frequency_max, frequency_plot[-1]),
    ),
)

print(f"Full-display RMS log10 error: {full_stats['rms']:.4f}")
print(
    "Full-display frequency bins within factor 2:",
    f"{full_stats['within_factor_2']:.1%}",
)
print(
    "Full-display frequency bins within factor 3:",
    f"{full_stats['within_factor_3']:.1%}",
)

# %% DEGREE-FREQUENCY DIAGNOSTIC FOR TEMPORAL-ONLY OPTIMISATION

from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

# -----------------------------------------------------
# RECOMPUTE FULL DEGREE-FREQUENCY SPECTRA
# -----------------------------------------------------

chaos_degree_spectrum, chaos_degree_f = (
    Lowes_Degree_PSD_All_Degrees(
        chaos_good,
        a=r_earth,
        r=r_choice,
    )
)

optimal_degree_spectrum, optimal_degree_f = (
    Lowes_Degree_PSD_All_Degrees(
        optimal_gnm_total,
        a=r_earth,
        r=r_choice,
    )
)

if not np.allclose(chaos_degree_f, optimal_degree_f):
    raise ValueError(
        "CHAOS and optimal-wave frequency arrays do not match"
    )

degrees = np.arange(
    1,
    chaos_degree_spectrum.shape[0] + 1,
)

# -----------------------------------------------------
# FULL DISPLAY REGION
# -----------------------------------------------------

degree_plot_mask = (
    (degrees >= 1)
    & (degrees <= nmax_power)
)

frequency_plot_mask = (
    (chaos_degree_f > 0)
    & (chaos_degree_f < 0.5)
)

degree_plot = degrees[degree_plot_mask]
frequency_plot = chaos_degree_f[frequency_plot_mask]

chaos_degree_plot = chaos_degree_spectrum[
    np.ix_(degree_plot_mask, frequency_plot_mask)
]

optimal_degree_plot = optimal_degree_spectrum[
    np.ix_(degree_plot_mask, frequency_plot_mask)
]

plot_valid = (
    np.isfinite(chaos_degree_plot)
    & np.isfinite(optimal_degree_plot)
    & (chaos_degree_plot > 0)
    & (optimal_degree_plot > 0)
)

chaos_degree_plot_masked = np.ma.masked_where(
    ~plot_valid,
    chaos_degree_plot,
)

optimal_degree_plot_masked = np.ma.masked_where(
    ~plot_valid,
    optimal_degree_plot,
)

# -----------------------------------------------------
# LOG10 WAVE/CHAOS ERROR
# -----------------------------------------------------

degree_log_error = np.full_like(
    chaos_degree_plot,
    np.nan,
    dtype=float,
)

degree_log_error[plot_valid] = np.log10(
    optimal_degree_plot[plot_valid]
    / chaos_degree_plot[plot_valid]
)

error_values = degree_log_error[plot_valid]

print(
    "Full degree-frequency RMS log10 error:",
    f"{np.sqrt(np.mean(error_values**2)):.4f}",
)

print(
    "Degree-frequency cells within factor 2:",
    f"{np.mean(np.abs(error_values) <= np.log10(2)):.1%}",
)

print(
    "Degree-frequency cells within factor 3:",
    f"{np.mean(np.abs(error_values) <= np.log10(3)):.1%}",
)

# -----------------------------------------------------
# SHARED POWER COLOUR SCALE
# -----------------------------------------------------

positive_power = np.concatenate([
    chaos_degree_plot_masked.compressed(),
    optimal_degree_plot_masked.compressed(),
])

power_norm = LogNorm(
    vmin=np.min(positive_power),
    vmax=np.max(positive_power),
)

power_cmap = plt.get_cmap("viridis").copy()
power_cmap.set_bad("lightgrey")

# -----------------------------------------------------
# DISCRETE ERROR COLOUR SCALE
# -----------------------------------------------------

factor_2 = np.log10(2)
factor_3 = np.log10(3)

max_error = np.nanmax(np.abs(degree_log_error))
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
    "#08306B",  # underprediction by >10
    "#2171B5",  # underprediction by 3--10
    "#9ECAE1",  # underprediction by 2--3
    "#FFFFFF",  # within factor 2
    "#FCAE91",  # overprediction by 2--3
    "#FB6A4A",  # overprediction by 3--10
    "#CB181D",  # overprediction by >10
]

error_cmap = ListedColormap(error_colors)
error_cmap.set_bad("lightgrey")

error_norm = BoundaryNorm(
    error_bounds,
    error_cmap.N,
    clip=True,
)

# -----------------------------------------------------
# IMAGE EXTENT
# -----------------------------------------------------

df = np.mean(np.diff(frequency_plot))

extent = [
    frequency_plot[0] - df / 2,
    frequency_plot[-1] + df / 2,
    degree_plot[0] - 0.5,
    degree_plot[-1] + 0.5,
]

# Temporal-only optimisation interval spans every degree.
fit_frequencies = chaos_degree_f[
    (chaos_degree_f > max(frequency_min, 0))
    & (chaos_degree_f < frequency_max)
]

show_fit_rectangle = fit_frequencies.size > 0

if show_fit_rectangle:
    fit_x0 = fit_frequencies[0] - df / 2
    fit_x1 = fit_frequencies[-1] + df / 2
    fit_y0 = degree_plot[0] - 0.5
    fit_y1 = degree_plot[-1] + 0.5

# -----------------------------------------------------
# THREE-PANEL FIGURE
# -----------------------------------------------------

with plt.rc_context({"font.size": 12}):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(text_width, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    chaos_im = axes[0].imshow(
        chaos_degree_plot_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=power_cmap,
        norm=power_norm,
    )

    axes[1].imshow(
        optimal_degree_plot_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=power_cmap,
        norm=power_norm,
    )

    error_im = axes[2].imshow(
        np.ma.masked_invalid(degree_log_error),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=error_cmap,
        norm=error_norm,
    )

    axes[0].set_title("CHAOS-8.6")

    axes[1].set_title(
        "Temporal-only optimal\nmode combination"
    )

    axes[2].set_title(
        r"$\log_{10}(P_{\mathrm{wave}}/"
        r"P_{\mathrm{CHAOS}})$"
    )

    if show_fit_rectangle:
        for ax in axes:
            rectangle = Rectangle(
                (fit_x0, fit_y0),
                fit_x1 - fit_x0,
                fit_y1 - fit_y0,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                linestyle=":",
                label="Temporal fit interval",
            )
            ax.add_patch(rectangle)

    for ax in axes:
        ax.set_xlabel(
            r"Frequency / $\mathrm{yr}^{-1}$"
        )

        ax.set_xlim(
            frequency_plot[0] - df / 2,
            frequency_plot[-1] + df / 2,
        )

        ax.set_ylim(
            degree_plot[0] - 0.5,
            degree_plot[-1] + 0.5,
        )

        ax.set_yticks(
            np.arange(2, nmax_power + 1, 2)
        )

    axes[0].set_ylabel(
        r"Spherical harmonic degree $n$"
    )

    if show_fit_rectangle:
        axes[2].legend(
            loc="upper right",
            frameon=True,
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
        r"/ $(\mathrm{nT\,yr^{-1}})^2"
        r"\,\mathrm{yr}$"
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
        r"$\log_{10}(P_{\mathrm{wave}}/"
        r"P_{\mathrm{CHAOS}})$"
    )

    fig.savefig(
        f"{FIG_DIR}/final/"
        "temporal_only_optimal_degree_frequency_diagnostic.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()