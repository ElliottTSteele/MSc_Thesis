# %% FULL-DOMAIN CHAOS-8.6 WAVE OPTIMISATION
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

# ------------------------------------------------------------------
# MODE-SELECTION SETTINGS
# ------------------------------------------------------------------
# Choose how the optimiser handles the number of selected modes:
#
#   "fixed" -> exactly n_selected_modes modes are retained.
#   "auto"  -> the optimiser may add/remove modes and determines the
#              final number itself by minimising the same RMS log-misfit.
#
mode_count_strategy = "auto"

# Used only when mode_count_strategy == "fixed".
n_selected_modes = 10

# Modes that MUST appear in the final solution.
# These are physical mode numbers, not zero-based array indices.
# Examples:
#   seeded_mode_numbers = []
#   seeded_mode_numbers = [13, 18]
seeded_mode_numbers = []

# Bounds used only for automatic mode-count selection.
# The minimum is automatically raised if necessary so that all seeded
# modes can be included.
auto_min_modes = 1
auto_max_modes = len(mode_numbers)

random_seed = 2
n_restarts = 3
annealing_steps = 1200
temperature_start = 0.05
temperature_end = 1e-4
max_swap_passes = 4

# In automatic mode, the final deterministic refinement repeatedly tries
# all single-mode additions, removals, and swaps until none improves the
# RMS log-misfit by more than this tolerance.
refinement_tolerance = 1e-6

comparison_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos_7_18_vs_8_6_agreement_mask.png"
)
fit_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos_full_domain_optimal_wave_fit.png"
)
full_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos_full_domain_optimal_wave_full_diagnostic.png"
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
# FULL-DOMAIN FIT MASK
#
# The optimiser now fits every valid CHAOS-8.6 PSD cell inside the requested
# degree-frequency domain. CHAOS-7.18 is retained only as a diagnostic
# comparison and does NOT determine which cells enter the optimisation.
fit_valid = (
    domain_mask
    & np.isfinite(chaos_8_spectrum)
    & (chaos_8_spectrum > 0)
)

if not np.any(fit_valid):
    raise ValueError("The full-domain CHAOS-8.6 fit mask contains no cells")

print(
    "Full-domain CHAOS-8.6 fit cells:",
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

# Diagnostic-only mask showing where CHAOS-7.18 and CHAOS-8.6 agree within
# the chosen factor threshold. This is NOT used by the optimiser.
chaos_agreement_mask = (
    chaos_comparison_valid
    & domain_mask
    & (
        np.abs(chaos_log_ratio)
        <= max_abs_chaos_log_difference
    )
)
chaos_agreement_mask_plot = chaos_agreement_mask[display_ix]

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
    outline_mask=chaos_agreement_mask_plot,
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


# %% LOG-MISFIT OBJECTIVE ON THE FULL REQUESTED CHAOS-8.6 DOMAIN
psd_epsilon = 1e-15 * np.nanmax(chaos_8_spectrum[fit_valid])


def rms_log_misfit(candidate_spectrum):
    error = np.log10(
        (candidate_spectrum[fit_valid] + psd_epsilon)
        / (chaos_8_spectrum[fit_valid] + psd_epsilon)
    )
    return np.sqrt(np.mean(error**2))


# %% MODE-SELECTION SETUP
if mode_count_strategy not in {"fixed", "auto"}:
    raise ValueError(
        "mode_count_strategy must be either 'fixed' or 'auto'"
    )

mode_number_to_index = {
    int(mode_number): index
    for index, mode_number in enumerate(mode_numbers)
}

unknown_seeded_modes = [
    int(mode_number)
    for mode_number in seeded_mode_numbers
    if int(mode_number) not in mode_number_to_index
]
if unknown_seeded_modes:
    raise ValueError(
        "The following seeded modes are not present in mode_numbers: "
        f"{unknown_seeded_modes}"
    )

seeded_indices = {
    mode_number_to_index[int(mode_number)]
    for mode_number in seeded_mode_numbers
}

if mode_count_strategy == "fixed":
    if n_selected_modes < len(seeded_indices):
        raise ValueError(
            "n_selected_modes cannot be smaller than the number of "
            "seeded modes"
        )
    if n_selected_modes > len(mode_numbers):
        raise ValueError(
            "n_selected_modes cannot exceed the number of available modes"
        )

    min_modes = n_selected_modes
    max_modes = n_selected_modes

else:
    min_modes = max(int(auto_min_modes), len(seeded_indices))
    max_modes = min(int(auto_max_modes), len(mode_numbers))

    if min_modes < 0:
        raise ValueError("auto_min_modes must be non-negative")
    if max_modes < min_modes:
        raise ValueError(
            "auto_max_modes must be >= auto_min_modes and large enough "
            "to contain all seeded modes"
        )

print(
    "\nMode-selection strategy:",
    mode_count_strategy,
)
print(
    "Seeded mandatory modes:",
    np.asarray(sorted(seeded_mode_numbers), dtype=int),
)
if mode_count_strategy == "fixed":
    print("Required number of modes:", n_selected_modes)
else:
    print(
        "Allowed automatic mode-count range:",
        f"{min_modes} to {max_modes}",
    )


# %% GREEDY LINEAR-SPECTRUM INITIAL SET
#
# Start with all mandatory seeded modes, then greedily add modes until:
#   fixed mode -> n_selected_modes is reached
#   auto mode  -> min_modes is reached
#
# The coherent objective below subsequently performs the true optimisation.

initial_indices = list(sorted(seeded_indices))
remaining_indices = [
    index
    for index in range(len(mode_numbers))
    if index not in seeded_indices
]

if initial_indices:
    current_linear_spectrum = np.sum(
        scaled_mode_spectra[initial_indices],
        axis=0,
    )
else:
    current_linear_spectrum = np.zeros_like(chaos_8_spectrum)

initial_target_count = (
    n_selected_modes
    if mode_count_strategy == "fixed"
    else min_modes
)

while len(initial_indices) < initial_target_count:
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


def sum_mode_series(mode_set):
    """Return the coherent Gauss-series sum for a set of mode indices."""
    if not mode_set:
        return np.zeros_like(scaled_mode_series[0])

    return np.sum(
        scaled_mode_series[list(sorted(mode_set))],
        axis=0,
    )


def evaluate_mode_set(mode_set, gnm_total=None):
    """Evaluate coherent RMS log-misfit for one selected mode set."""
    key = tuple(sorted(mode_set))

    if key in loss_cache:
        return loss_cache[key]

    if gnm_total is None:
        gnm_total = sum_mode_series(mode_set)

    spectrum, _ = degree_frequency_spectrum(gnm_total)
    loss = rms_log_misfit(spectrum)

    loss_cache[key] = loss
    return loss


initial_gnm = sum_mode_series(initial_set)
initial_loss = evaluate_mode_set(initial_set, initial_gnm)


# %% RANDOM PROPOSAL HELPERS
rng = np.random.default_rng(random_seed)
all_indices = np.arange(len(mode_numbers), dtype=int)


def removable_indices(mode_set):
    """Selected indices that are not locked by seeded_mode_numbers."""
    return np.asarray(
        sorted(set(mode_set) - seeded_indices),
        dtype=int,
    )


def unselected_indices(mode_set):
    """All currently unselected mode indices."""
    return np.setdiff1d(
        all_indices,
        np.asarray(sorted(mode_set), dtype=int),
        assume_unique=False,
    )


def propose_fixed_count_move(current_set, current_gnm):
    """One-for-one swap that preserves seeded modes and set size."""
    removable = removable_indices(current_set)
    unselected = unselected_indices(current_set)

    if removable.size == 0 or unselected.size == 0:
        return None

    mode_out = int(rng.choice(removable))
    mode_in = int(rng.choice(unselected))

    candidate_set = current_set.copy()
    candidate_set.remove(mode_out)
    candidate_set.add(mode_in)

    candidate_gnm = (
        current_gnm
        - scaled_mode_series[mode_out]
        + scaled_mode_series[mode_in]
    )

    return candidate_set, candidate_gnm


def propose_auto_count_move(current_set, current_gnm):
    """
    Randomly propose an add, remove, or swap move.

    Seeded modes are never removable. The total number of selected modes
    always remains between min_modes and max_modes.
    """
    removable = removable_indices(current_set)
    unselected = unselected_indices(current_set)

    allowed_moves = []

    if len(current_set) < max_modes and unselected.size > 0:
        allowed_moves.append("add")

    if len(current_set) > min_modes and removable.size > 0:
        allowed_moves.append("remove")

    if removable.size > 0 and unselected.size > 0:
        allowed_moves.append("swap")

    if not allowed_moves:
        return None

    move = str(rng.choice(allowed_moves))

    if move == "add":
        mode_in = int(rng.choice(unselected))

        candidate_set = current_set.copy()
        candidate_set.add(mode_in)

        candidate_gnm = (
            current_gnm
            + scaled_mode_series[mode_in]
        )

    elif move == "remove":
        mode_out = int(rng.choice(removable))

        candidate_set = current_set.copy()
        candidate_set.remove(mode_out)

        candidate_gnm = (
            current_gnm
            - scaled_mode_series[mode_out]
        )

    else:
        mode_out = int(rng.choice(removable))
        mode_in = int(rng.choice(unselected))

        candidate_set = current_set.copy()
        candidate_set.remove(mode_out)
        candidate_set.add(mode_in)

        candidate_gnm = (
            current_gnm
            - scaled_mode_series[mode_out]
            + scaled_mode_series[mode_in]
        )

    return candidate_set, candidate_gnm


def perturb_starting_set(mode_set):
    """
    Randomly perturb a restart while respecting seeded modes and mode-count
    constraints.
    """
    current_set = mode_set.copy()
    current_gnm = sum_mode_series(current_set)

    reference_count = (
        n_selected_modes
        if mode_count_strategy == "fixed"
        else max(len(current_set), min_modes)
    )
    n_perturb = max(1, int(np.ceil(0.25 * reference_count)))

    for _ in range(n_perturb):
        if mode_count_strategy == "fixed":
            proposal = propose_fixed_count_move(
                current_set,
                current_gnm,
            )
        else:
            proposal = propose_auto_count_move(
                current_set,
                current_gnm,
            )

        if proposal is None:
            break

        current_set, current_gnm = proposal

    return current_set, current_gnm


# %% SIMULATED ANNEALING
best_set = initial_set.copy()
best_gnm = initial_gnm.copy()
best_loss = initial_loss

for restart in range(n_restarts):

    if restart == 0:
        current_set = initial_set.copy()
        current_gnm = initial_gnm.copy()
    else:
        current_set, current_gnm = perturb_starting_set(
            initial_set
        )

    current_loss = evaluate_mode_set(
        current_set,
        current_gnm,
    )

    if current_loss < best_loss:
        best_set = current_set.copy()
        best_gnm = current_gnm.copy()
        best_loss = current_loss

    for step in range(annealing_steps):
        fraction = step / max(annealing_steps - 1, 1)
        temperature = temperature_start * (
            temperature_end / temperature_start
        ) ** fraction

        if mode_count_strategy == "fixed":
            proposal = propose_fixed_count_move(
                current_set,
                current_gnm,
            )
        else:
            proposal = propose_auto_count_move(
                current_set,
                current_gnm,
            )

        if proposal is None:
            break

        candidate_set, candidate_gnm = proposal

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
        f"best RMS log error = {best_loss:.4f}, "
        f"n_modes = {len(best_set)}"
    )


# %% DETERMINISTIC LOCAL REFINEMENT
#
# Fixed-count mode:
#   exhaustively tests every allowed one-for-one swap.
#
# Automatic-count mode:
#   exhaustively tests every allowed single addition, removal, and swap.
#   The best improving move is accepted, then the search repeats.
#
# Seeded modes are never considered for removal.

for refinement_pass in range(max_swap_passes):

    best_candidate_set = None
    best_candidate_loss = best_loss
    best_candidate_gnm = None
    best_move_description = None

    selected = np.asarray(sorted(best_set), dtype=int)
    removable = removable_indices(best_set)
    unselected = unselected_indices(best_set)

    # ----------------------------------------------------------
    # ADDITIONS: automatic-count mode only
    # ----------------------------------------------------------
    if (
        mode_count_strategy == "auto"
        and len(best_set) < max_modes
    ):
        for mode_in in unselected:
            mode_in = int(mode_in)

            candidate_set = best_set.copy()
            candidate_set.add(mode_in)

            candidate_gnm = (
                best_gnm
                + scaled_mode_series[mode_in]
            )

            candidate_loss = evaluate_mode_set(
                candidate_set,
                candidate_gnm,
            )

            if (
                candidate_loss
                < best_candidate_loss - refinement_tolerance
            ):
                best_candidate_set = candidate_set
                best_candidate_loss = candidate_loss
                best_candidate_gnm = candidate_gnm
                best_move_description = (
                    f"add mode {mode_numbers[mode_in]}"
                )

    # ----------------------------------------------------------
    # REMOVALS: automatic-count mode only
    # ----------------------------------------------------------
    if (
        mode_count_strategy == "auto"
        and len(best_set) > min_modes
    ):
        for mode_out in removable:
            mode_out = int(mode_out)

            candidate_set = best_set.copy()
            candidate_set.remove(mode_out)

            candidate_gnm = (
                best_gnm
                - scaled_mode_series[mode_out]
            )

            candidate_loss = evaluate_mode_set(
                candidate_set,
                candidate_gnm,
            )

            if (
                candidate_loss
                < best_candidate_loss - refinement_tolerance
            ):
                best_candidate_set = candidate_set
                best_candidate_loss = candidate_loss
                best_candidate_gnm = candidate_gnm
                best_move_description = (
                    f"remove mode {mode_numbers[mode_out]}"
                )

    # ----------------------------------------------------------
    # SWAPS: both fixed-count and automatic-count modes
    # ----------------------------------------------------------
    for mode_out in removable:
        mode_out = int(mode_out)

        for mode_in in unselected:
            mode_in = int(mode_in)

            candidate_set = best_set.copy()
            candidate_set.remove(mode_out)
            candidate_set.add(mode_in)

            candidate_gnm = (
                best_gnm
                - scaled_mode_series[mode_out]
                + scaled_mode_series[mode_in]
            )

            candidate_loss = evaluate_mode_set(
                candidate_set,
                candidate_gnm,
            )

            if (
                candidate_loss
                < best_candidate_loss - refinement_tolerance
            ):
                best_candidate_set = candidate_set
                best_candidate_loss = candidate_loss
                best_candidate_gnm = candidate_gnm
                best_move_description = (
                    f"swap mode {mode_numbers[mode_out]} "
                    f"-> {mode_numbers[mode_in]}"
                )

    if best_candidate_set is None:
        break

    best_set = best_candidate_set
    best_loss = best_candidate_loss
    best_gnm = best_candidate_gnm

    print(
        f"Refinement pass {refinement_pass + 1}: "
        f"{best_move_description}; "
        f"RMS log error = {best_loss:.4f}, "
        f"n_modes = {len(best_set)}"
    )


# %% FINAL MODE-SELECTION VALIDATION
if not seeded_indices.issubset(best_set):
    raise RuntimeError(
        "Internal error: one or more seeded modes were lost "
        "during optimisation"
    )

if mode_count_strategy == "fixed":
    if len(best_set) != n_selected_modes:
        raise RuntimeError(
            "Internal error: fixed-count optimisation returned the "
            "wrong number of modes"
        )
else:
    if not (min_modes <= len(best_set) <= max_modes):
        raise RuntimeError(
            "Internal error: automatic optimisation returned a mode "
            "count outside the permitted range"
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
print("Number of selected modes:", len(optimal_mode_numbers))
print(
    "Mandatory seeded modes:",
    np.asarray(sorted(seeded_mode_numbers), dtype=int),
)
print(f"Final full-domain RMS log10 error: {fit_stats['rms']:.4f}")
print(
    "Full-domain fit cells within factor 2:",
    f"{fit_stats['within_factor_2']:.1%}",
)
print(
    "Full-domain fit cells within factor 3:",
    f"{fit_stats['within_factor_3']:.1%}",
)


# %% FULL-DOMAIN FIT DIAGNOSTIC: FINAL OUTPUT VS CHAOS-8.6
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
    reference_title="CHAOS-8.6\nfull fit domain",
    candidate_title="Optimal combination\nfull fit domain",
    ratio_title=r"$\log_{10}(P_{\mathrm{wave}}/P_{8.6})$",
    output_path=fit_figure,
)


# %% FULL-DOMAIN DIAGNOSTIC WITH CHAOS-7/8 AGREEMENT REGION OUTLINED
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
    outline_mask=chaos_agreement_mask_plot,
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
