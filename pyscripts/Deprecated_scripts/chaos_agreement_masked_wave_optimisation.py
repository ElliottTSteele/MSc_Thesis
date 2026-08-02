# %% CHAOS-AGREEMENT-MASKED WAVE OPTIMISATION
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import pickle
import sys
from pathlib import Path

import chaosmagpy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.msc_thesis.paths import *
from pyscripts.R_test_synthetics.syn_pipeline import *


# %% SETTINGS
chaos_7_file = Path(CHAOS_DIR) / "CHAOS-7.18.mat"
chaos_8_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
wave_file = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

r_choice = r_cmb
nmax = 20
mode_numbers = np.arange(1, 63)

# Resolved synthetic Gauss series are assumed to begin at 1997.1 and to be
# sampled every 0.2 yr. Only synthetic epochs inside this common interval are
# retained; CHAOS-7.18 and CHAOS-8.6 are evaluated on those exact epochs.
synthetic_start_year = 1997.1
sample_dt = 0.2
common_start_year = 2000.0 + 7.0 / 12.0
common_end_year = 2024.0

# Optimisation domain before applying the CHAOS-7/CHAOS-8 agreement mask.
degree_min, degree_max = 1, 20
frequency_min, frequency_max = 0.0, 0.3

# A cell is admitted to the fit when the two CHAOS spectra agree within this
# absolute log10 power ratio. Default: agreement within a factor of two.
max_abs_chaos_log_difference = np.log10(2)

n_selected_modes = 10
random_seed = 2
n_restarts = 3
annealing_steps = 1200
temperature_start = 0.05
temperature_end = 1e-4
max_swap_passes = 4

comparison_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos_7_18_vs_8_6_agreement_mask.png"
)
fit_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos_agreement_masked_optimal_wave_fit.png"
)
full_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos_agreement_masked_optimal_wave_full_diagnostic.png"
)
scale_file = (
    Path(FELIX_DIR)
    / "mode_amplitude_scalings_chaos8_full_nf_water_level.pkl"
)


def decimal_year_to_mjd2000(decimal_years):
    """Convert decimal years to MJD2000 using each calendar year's length."""
    decimal_years = np.asarray(decimal_years, dtype=float)
    output = np.empty_like(decimal_years)
    epoch = np.datetime64("2000-01-01T00:00:00", "ns")

    for index, decimal_year in np.ndenumerate(decimal_years):
        year = int(np.floor(decimal_year))
        fraction = decimal_year - year
        year_start = np.datetime64(
            f"{year:04d}-01-01T00:00:00",
            "ns",
        )
        next_year = np.datetime64(
            f"{year + 1:04d}-01-01T00:00:00",
            "ns",
        )
        year_length_ns = (
            (next_year - year_start) / np.timedelta64(1, "ns")
        )
        timestamp = year_start + np.timedelta64(
            int(np.rint(fraction * year_length_ns)),
            "ns",
        )
        output[index] = (
            (timestamp - epoch) / np.timedelta64(1, "D")
        )

    return output


def load_sv_gauss(model_path, times, nmax):
    """Evaluate SV Gauss coefficients from one CHAOS model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path}")
    model = cp.load_CHAOS_matfile(str(model_path))
    coefficients = model.synth_coeffs_tdep(
        times,
        nmax=nmax,
        deriv=1,
        extrapolate="off",
    )
    return Truncate_Gauss_Coeffs(
        np.asarray(coefficients),
        nmax,
    )


def degree_frequency_spectrum(gnm):
    """Lowes degree-frequency SV spectrum at the selected radius."""
    return Lowes_Degree_PSD_All_Degrees(
        gnm,
        a=r_earth,
        r=r_choice,
    )


def log_error_statistics(candidate, target, valid, epsilon):
    """Compute log-ratio field and summary statistics on a supplied mask."""
    comparison_valid = (
        valid
        & np.isfinite(candidate)
        & np.isfinite(target)
        & (candidate >= 0)
        & (target > 0)
    )
    log_error = np.full_like(target, np.nan, dtype=float)
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


def discrete_log_ratio_colours(log_ratio):
    """Return the discrete factor-2/factor-3/factor-10 colour mapping."""
    factor_2 = np.log10(2)
    factor_3 = np.log10(3)
    max_error = np.nanmax(np.abs(log_ratio))
    error_cap = max(1.01, max_error + 1e-6)
    bounds = [
        -error_cap,
        -1,
        -factor_3,
        -factor_2,
        factor_2,
        factor_3,
        1,
        error_cap,
    ]
    colours = [
        "#08306B",
        "#2171B5",
        "#9ECAE1",
        "#FFFFFF",
        "#FCAE91",
        "#FB6A4A",
        "#CB181D",
    ]
    cmap = ListedColormap(colours)
    cmap.set_bad("lightgrey")
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    ticks = [-1, -factor_3, -factor_2, factor_2, factor_3, 1]
    ticklabels = [
        r"$-1$",
        r"$-\log_{10}(3)$",
        r"$-\log_{10}(2)$",
        r"$\log_{10}(2)$",
        r"$\log_{10}(3)$",
        r"$1$",
    ]
    return cmap, norm, bounds, ticks, ticklabels


def plot_three_panel_power_comparison(
    reference,
    candidate,
    log_ratio,
    display_valid,
    degrees,
    frequencies,
    reference_title,
    candidate_title,
    ratio_title,
    output_path,
    outline_mask=None,
):
    """Plot two power spectra and their log10 ratio."""
    reference_masked = np.ma.masked_where(~display_valid, reference)
    candidate_masked = np.ma.masked_where(~display_valid, candidate)
    positive_power = np.concatenate([
        reference_masked.compressed(),
        candidate_masked.compressed(),
    ])
    power_norm = LogNorm(
        vmin=np.min(positive_power),
        vmax=np.max(positive_power),
    )
    power_cmap = plt.get_cmap("viridis").copy()
    power_cmap.set_bad("lightgrey")

    (
        error_cmap,
        error_norm,
        error_bounds,
        error_ticks,
        error_ticklabels,
    ) = discrete_log_ratio_colours(log_ratio)

    df = np.mean(np.diff(frequencies))
    extent = [
        frequencies[0] - df / 2,
        frequencies[-1] + df / 2,
        degrees[0] - 0.5,
        degrees[-1] + 0.5,
    ]

    with plt.rc_context({"font.size": 12}):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(text_width, 4.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        reference_im = axes[0].imshow(
            reference_masked,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=power_cmap,
            norm=power_norm,
        )
        axes[1].imshow(
            candidate_masked,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=power_cmap,
            norm=power_norm,
        )
        error_im = axes[2].imshow(
            np.ma.masked_where(~display_valid, log_ratio),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=error_cmap,
            norm=error_norm,
        )

        axes[0].set_title(reference_title)
        axes[1].set_title(candidate_title)
        axes[2].set_title(ratio_title)

        if (
            outline_mask is not None
            and np.any(outline_mask)
            and np.any(~outline_mask)
        ):
            for ax in axes:
                ax.contour(
                    frequencies,
                    degrees,
                    outline_mask.astype(float),
                    levels=[0.5],
                    colors="black",
                    linewidths=1.2,
                    linestyles=":",
                )

        for ax in axes:
            ax.set_xlabel(r"Frequency / $\mathrm{yr}^{-1}$")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_yticks(np.arange(2, degree_max + 1, 2))
        axes[0].set_ylabel(r"Spherical harmonic degree $n$")

        power_cbar = fig.colorbar(
            reference_im,
            ax=axes[:2],
            location="bottom",
            fraction=0.08,
            pad=0.13,
        )
        power_cbar.set_label(
            r"Lowes SV power spectrum at the CMB "
            r"/ $(\mathrm{nT\,yr^{-1}})^2\,\mathrm{yr}$"
        )
        error_cbar = fig.colorbar(
            error_im,
            ax=axes[2],
            boundaries=error_bounds,
            ticks=error_ticks,
        )
        error_cbar.set_ticklabels(error_ticklabels)
        error_cbar.set_label(ratio_title)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()


# %% LOAD ONE MODE TO DETERMINE THE SYNTHETIC TIME AXIS
if not wave_file.exists():
    raise FileNotFoundError(f"Could not find {wave_file}")

with h5py.File(wave_file, "r") as h5_file:
    example_spline = np.asarray(
        h5_file[f"mode_{mode_numbers[0]}/without_decay"][()]
    )

example_gauss = H_sv @ example_spline
n_synthetic_times = example_gauss.shape[0]
synthetic_decimal_years = (
    synthetic_start_year
    + sample_dt * np.arange(n_synthetic_times)
)
shared_time_mask = (
    (synthetic_decimal_years >= common_start_year)
    & (synthetic_decimal_years <= common_end_year)
)
shared_decimal_years = synthetic_decimal_years[shared_time_mask]

if shared_decimal_years.size < 3:
    raise ValueError("Too few shared synthetic epochs were selected")

shared_times_mjd2000 = decimal_year_to_mjd2000(
    shared_decimal_years
)

print(
    f"Using {shared_decimal_years.size} shared samples from "
    f"{shared_decimal_years[0]:.1f} to "
    f"{shared_decimal_years[-1]:.1f} at dt={sample_dt} yr"
)


# %% CHAOS-7.18 AND CHAOS-8.6 ON THE WAVE-ALIGNED EPOCHS
chaos_7_sv = load_sv_gauss(
    chaos_7_file,
    shared_times_mjd2000,
    nmax,
)
chaos_8_sv = load_sv_gauss(
    chaos_8_file,
    shared_times_mjd2000,
    nmax,
)

chaos_7_spectrum, chaos_7_f = degree_frequency_spectrum(
    chaos_7_sv
)
chaos_8_spectrum, chaos_8_f = degree_frequency_spectrum(
    chaos_8_sv
)

if not np.allclose(chaos_7_f, chaos_8_f):
    raise ValueError("CHAOS-7.18 and CHAOS-8.6 frequency grids differ")

frequencies = chaos_8_f
degrees = np.arange(1, chaos_8_spectrum.shape[0] + 1)

chaos_comparison_valid = (
    np.isfinite(chaos_7_spectrum)
    & np.isfinite(chaos_8_spectrum)
    & (chaos_7_spectrum > 0)
    & (chaos_8_spectrum > 0)
)
chaos_log_ratio = np.full_like(
    chaos_8_spectrum,
    np.nan,
    dtype=float,
)
chaos_log_ratio[chaos_comparison_valid] = np.log10(
    chaos_8_spectrum[chaos_comparison_valid]
    / chaos_7_spectrum[chaos_comparison_valid]
)

domain_mask = (
    (degrees[:, None] >= degree_min)
    & (degrees[:, None] <= degree_max)
    & (frequencies[None, :] >= max(frequency_min, 0.0))
    & (frequencies[None, :] < frequency_max)
)
fit_valid = (
    chaos_comparison_valid
    & domain_mask
    & (
        np.abs(chaos_log_ratio)
        <= max_abs_chaos_log_difference
    )
)

if not np.any(fit_valid):
    raise ValueError("The CHAOS-agreement fit mask contains no cells")

print(
    "Fit cells retained after CHAOS agreement threshold:",
    f"{np.sum(fit_valid)}/{np.sum(domain_mask)} "
    f"({np.sum(fit_valid) / np.sum(domain_mask):.1%})",
)


# %% CHAOS-7.18 VS CHAOS-8.6 AND THE ADMITTED FIT REGION
degree_display_mask = (
    (degrees >= degree_min) & (degrees <= degree_max)
)
frequency_display_mask = (
    (frequencies >= max(frequency_min, 0.0))
    & (frequencies < frequency_max)
)
degree_plot = degrees[degree_display_mask]
frequency_plot = frequencies[frequency_display_mask]

display_ix = np.ix_(degree_display_mask, frequency_display_mask)
chaos_7_plot = chaos_7_spectrum[display_ix]
chaos_8_plot = chaos_8_spectrum[display_ix]
chaos_log_ratio_plot = chaos_log_ratio[display_ix]
comparison_valid_plot = chaos_comparison_valid[display_ix]
fit_valid_plot = fit_valid[display_ix]

plot_three_panel_power_comparison(
    reference=chaos_7_plot,
    candidate=chaos_8_plot,
    log_ratio=chaos_log_ratio_plot,
    display_valid=comparison_valid_plot,
    degrees=degree_plot,
    frequencies=frequency_plot,
    reference_title="CHAOS-7.18",
    candidate_title="CHAOS-8.6",
    ratio_title=r"$\log_{10}(P_{8.6}/P_{7.18})$",
    output_path=comparison_figure,
    outline_mask=fit_valid_plot,
)


# %% STRICT WATER-LEVEL SCALING AGAINST ALL POSITIVE-FREQUENCY CHAOS-8 CELLS
mode_periods = []
water_level_power_scales = []
scaled_mode_mean_powers = []
scaled_mode_series = []
scaled_mode_spectra = []
scalings_dict = {}

with h5py.File(wave_file, "r") as h5_file:
    for mode_number in tqdm(mode_numbers, desc="Scaling modes"):
        mode_data = Component_Load(mode_number)
        eigenvalue = mode_data["eigenvalue"]
        period = 2 * np.pi / np.abs(eigenvalue.imag)

        gnm_spline = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )
        gnm_mode_full = H_sv @ gnm_spline

        if gnm_mode_full.shape[0] != n_synthetic_times:
            raise ValueError(
                f"Mode {mode_number} has {gnm_mode_full.shape[0]} "
                f"epochs; expected {n_synthetic_times}"
            )

        gnm_mode = gnm_mode_full[shared_time_mask]
        mode_spectrum, mode_f = degree_frequency_spectrum(gnm_mode)

        if not np.allclose(mode_f, frequencies):
            raise ValueError(
                f"Frequency mismatch for mode {mode_number}"
            )

        water_level_valid = (
            np.isfinite(chaos_8_spectrum)
            & np.isfinite(mode_spectrum)
            & (chaos_8_spectrum > 0)
            & (mode_spectrum > 0)
            & (frequencies[None, :] < 0.333)
        )
        if not np.any(water_level_valid):
            raise ValueError(
                f"No valid water-level cells for mode {mode_number}"
            )

        power_scale = np.min(
            chaos_8_spectrum[water_level_valid]
            / mode_spectrum[water_level_valid]
        )
        amplitude_scale = np.sqrt(power_scale)
        scaled_series = gnm_mode * amplitude_scale

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

with open(scale_file, "wb") as file:
    pickle.dump(scalings_dict, file)


# %% WATER-LEVEL-SCALED POWER AGAINST MODE PERIOD
chaos_8_mean_power = Mean_Instantaneous_Total_Lowes_Power(
    chaos_8_sv,
    a=r_earth,
    r=r_choice,
)
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
        chaos_8_mean_power,
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
    ax.set_title("Full-n-f-water-level-scaled wave power at the CMB")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    water_level_figure = (
        Path(FIG_DIR)
        / "water_level_scaled_mode_power_chaos8_full_nf.png"
    )
    fig.savefig(water_level_figure, dpi=300, bbox_inches="tight")
    plt.show()


# %% LOG-MISFIT OBJECTIVE ON CHAOS-AGREEMENT CELLS ONLY
psd_epsilon = 1e-15 * np.nanmax(chaos_8_spectrum[fit_valid])


def rms_log_misfit(candidate_spectrum):
    error = np.log10(
        (candidate_spectrum[fit_valid] + psd_epsilon)
        / (chaos_8_spectrum[fit_valid] + psd_epsilon)
    )
    return np.sqrt(np.mean(error**2))


# %% GREEDY LINEAR-SPECTRUM INITIAL SET
initial_indices = []
remaining_indices = list(range(len(mode_numbers)))
current_linear_spectrum = np.zeros_like(chaos_8_spectrum)

for _ in range(n_selected_modes):
    trial_losses = [
        rms_log_misfit(
            current_linear_spectrum + scaled_mode_spectra[index]
        )
        for index in remaining_indices
    ]
    best_position = int(np.argmin(trial_losses))
    best_index = remaining_indices.pop(best_position)
    initial_indices.append(best_index)
    current_linear_spectrum += scaled_mode_spectra[best_index]

initial_set = set(initial_indices)


# %% TRUE COHERENT DEGREE-FREQUENCY OBJECTIVE
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

    spectrum, _ = degree_frequency_spectrum(gnm_total)
    loss = rms_log_misfit(spectrum)
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


# %% FINAL OPTIMAL COHERENT COMBINATION
optimal_indices = np.asarray(sorted(best_set), dtype=int)
optimal_mode_numbers = mode_numbers[optimal_indices]
optimal_mode_periods = mode_periods[optimal_indices]
optimal_gnm_total = best_gnm

optimal_spectrum, optimal_f = degree_frequency_spectrum(
    optimal_gnm_total
)
if not np.allclose(optimal_f, frequencies):
    raise ValueError("Optimal and CHAOS frequency grids differ")

fit_stats = log_error_statistics(
    optimal_spectrum,
    chaos_8_spectrum,
    fit_valid,
    psd_epsilon,
)

print("\nSelected modes:", optimal_mode_numbers)
print("Selected periods:", optimal_mode_periods)
print(f"Final masked RMS log10 error: {fit_stats['rms']:.4f}")
print(
    "Admitted cells within factor 2:",
    f"{fit_stats['within_factor_2']:.1%}",
)
print(
    "Admitted cells within factor 3:",
    f"{fit_stats['within_factor_3']:.1%}",
)


# %% MASKED-FIT DIAGNOSTIC: ALWAYS FINAL OUTPUT VS CHAOS-8.6
optimal_plot = optimal_spectrum[display_ix]
fit_wave_stats = log_error_statistics(
    optimal_plot,
    chaos_8_plot,
    fit_valid_plot,
    psd_epsilon,
)

plot_three_panel_power_comparison(
    reference=chaos_8_plot,
    candidate=optimal_plot,
    log_ratio=fit_wave_stats["log_error"],
    display_valid=fit_wave_stats["valid"],
    degrees=degree_plot,
    frequencies=frequency_plot,
    reference_title="CHAOS-8.6\nadmitted cells",
    candidate_title="Optimal combination\nadmitted cells",
    ratio_title=r"$\log_{10}(P_{\mathrm{wave}}/P_{8.6})$",
    output_path=fit_figure,
)


# %% FULL-DOMAIN DIAGNOSTIC WITH THE ADMITTED REGION OUTLINED
full_display_valid = (
    np.isfinite(chaos_8_plot)
    & np.isfinite(optimal_plot)
    & (chaos_8_plot > 0)
    & (optimal_plot > 0)
)
full_wave_stats = log_error_statistics(
    optimal_plot,
    chaos_8_plot,
    full_display_valid,
    psd_epsilon,
)

plot_three_panel_power_comparison(
    reference=chaos_8_plot,
    candidate=optimal_plot,
    log_ratio=full_wave_stats["log_error"],
    display_valid=full_wave_stats["valid"],
    degrees=degree_plot,
    frequencies=frequency_plot,
    reference_title="CHAOS-8.6",
    candidate_title="Optimal mode\ncombination",
    ratio_title=r"$\log_{10}(P_{\mathrm{wave}}/P_{8.6})$",
    output_path=full_figure,
    outline_mask=fit_valid_plot,
)

print(f"Full-domain RMS log10 error: {full_wave_stats['rms']:.4f}")
print(
    "Full-domain cells within factor 2:",
    f"{full_wave_stats['within_factor_2']:.1%}",
)
print(
    "Full-domain cells within factor 3:",
    f"{full_wave_stats['within_factor_3']:.1%}",
)
print(f"Saved scalings to {scale_file}")
