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

from matplotlib.colors import ListedColormap, BoundaryNorm
r_choice = r_cmb
factor_2 = np.log10(2)
factor_3 = np.log10(3)

# Bin edges, from most negative to most positive
bounds = [
    -10,         # effectively -infinity
    -1,
    -factor_3,
    -factor_2,
     factor_2,
     factor_3,
     1,
     10,         # effectively +infinity
]

# Colours for each interval:
# [-10,-1], [-1,-log10(3)], [-log10(3),-log10(2)],
# [-log10(2),log10(2)],
# [log10(2),log10(3)], [log10(3),1], [1,10]
colors = [
    "#08306B",   # dark blue
    "#2171B5",   # medium blue
    "#9ECAE1",   # light blue
    "#FFFFFF",   # white
    "#FCAE91",   # light red
    "#FB6A4A",   # medium red
    "#CB181D",   # dark red
]

disc_cmap = ListedColormap(colors)
disc_cmap.set_bad("lightgrey")   # for NaNs / masked cells

disc_norm = BoundaryNorm(bounds, disc_cmap.N)

# Chaos data handling and psd creation

chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"

if not chaos_file.exists():
    raise FileNotFoundError(f"CHAOS model not found: {chaos_file}")

chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=20,
    deriv=1,
    extrapolate="off",
)

chaos_sv_gnm = np.asarray(chaos_sv_gnm)
chaos_sv_gnm = Truncate_Gauss_Coeffs(chaos_sv_gnm, 20)

chaos_good = chaos_sv_gnm[good_record_slice]

chaos_psd, chaos_f = Lowes_Degree_PSD_All_Degrees(chaos_good, a=r_earth, r=r_choice)

# getting masks (will be applied to both PSDs)
f_mask = (chaos_f > 0) & (chaos_f < 0.5)
g_mask = (degrees > 0) & (degrees <= 21)

# getting windowed chaos psd
chaos_psd_plot = chaos_psd[:, f_mask]
chaos_psd_plot = chaos_psd_plot[g_mask, :]

# getting windowed axes
frequencies_plot = chaos_f[f_mask]
degrees_plot = degrees[g_mask]

mean_chaos_p_total = Mean_Instantaneous_Total_Lowes_Power(chaos_good, a=r_earth, r=r_choice)

# %%mode_numbers = np.arange(1, 63)
# %% BUILD WATER-LEVEL-SCALED MODE DATABASE

mode_numbers = np.arange(1, 63)

scaled_mode_series = []
scaled_mode_psds = []
water_level_power_scales = []
mode_periods = []

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

with h5py.File(file_path, "r") as h5_file:

    for mode_number in mode_numbers:

        mode_data = Component_Load(mode_number)
        eigenvalue = mode_data["eigenvalue"]

        period = 2 * np.pi / np.abs(eigenvalue.imag)

        gnm_spl = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )

        gnm_mode = H_sv @ gnm_spl
        gnm_mode_good = gnm_mode[good_record_slice]

        mode_psd, mode_f = Lowes_Degree_PSD_All_Degrees(
            gnm_mode_good,
            a=r_earth,
            r=r_choice,
        )

        if not np.allclose(mode_f, chaos_f):
            raise ValueError(
                f"Frequency mismatch for mode {mode_number}"
            )

        mode_psd_plot = mode_psd[g_mask][:, f_mask]

        # Ignore cells where this particular mode has only numerical noise
        mode_floor = 1e-12 * np.nanmax(mode_psd_plot)

        water_level_valid = (
            np.isfinite(chaos_psd_plot)
            & np.isfinite(mode_psd_plot)
            & (chaos_psd_plot > 0)
            & (mode_psd_plot > mode_floor)
        )

        if not np.any(water_level_valid):
            raise ValueError(
                f"No valid water-level cells for mode {mode_number}"
            )

        contacts = np.full_like(
            mode_psd_plot,
            np.inf,
            dtype=float,
        )

        contacts[water_level_valid] = (
            chaos_psd_plot[water_level_valid]
            / mode_psd_plot[water_level_valid]
        )

        power_scale = np.min(contacts)
        amplitude_scale = np.sqrt(power_scale)

        scaled_mode_series.append(
            gnm_mode_good * amplitude_scale
        )

        scaled_mode_psds.append(
            mode_psd_plot * power_scale
        )

        water_level_power_scales.append(power_scale)
        mode_periods.append(period)


scaled_mode_series = np.asarray(scaled_mode_series)
scaled_mode_psds = np.asarray(scaled_mode_psds)
water_level_power_scales = np.asarray(
    water_level_power_scales
)
mode_periods = np.asarray(mode_periods)

print("Scaled mode-series shape:", scaled_mode_series.shape)
print("Scaled mode-PSD shape:", scaled_mode_psds.shape)


# %% DEFINE DEGREE-FREQUENCY PATCH USED BY THE MISFIT

# Edit these limits to define the optimisation region.
degree_min = 1
degree_max = 7

frequency_min = 0   # yr^-1
frequency_max = 0.2  # yr^-1

degree_fit_mask = (
    (degrees_plot >= degree_min)
    & (degrees_plot <= degree_max)
)

frequency_fit_mask = (
    (frequencies_plot >= frequency_min)
    & (frequencies_plot <= frequency_max)
)

# Convert the separate 1D masks into one 2D degree-frequency mask
fit_patch_mask = (
    degree_fit_mask[:, None]
    & frequency_fit_mask[None, :]
)

# Exclude invalid or non-positive CHAOS cells inside the patch
reliable_mask = (
    fit_patch_mask
    & np.isfinite(chaos_psd_plot)
    & (chaos_psd_plot > 0)
)

if not np.any(reliable_mask):
    raise ValueError(
        "The selected degree-frequency patch contains no valid cells."
    )

print(
    f"Misfit patch: n={degree_min}–{degree_max}, "
    f"f={frequency_min:.4f}–{frequency_max:.4f} yr^-1"
)

print(
    "Number of misfit cells:",
    np.count_nonzero(reliable_mask),
)


# Optional visual check of the selected fitting region
fig, ax = plt.subplots(figsize=(9, 4))

mesh = ax.pcolormesh(
    frequencies_plot,
    degrees_plot,
    reliable_mask.astype(int),
    shading="auto",
)

ax.set_xlabel("Frequency [yr$^{-1}$]")
ax.set_ylabel("Spherical-harmonic degree $n$")
ax.set_title("Degree-frequency cells included in PSD misfit")

fig.colorbar(
    mesh,
    ax=ax,
    label="Included in misfit",
)

plt.tight_layout()
plt.show()


# %% LOG-SPACE PSD MISFIT FUNCTION

def Log_PSD_Misfit(
    synthetic_psd,
    target_psd,
    valid_mask,
    eps=1e-20,
):
    if not np.any(valid_mask):
        raise ValueError("valid_mask contains no cells.")

    error = np.log10(
        (synthetic_psd[valid_mask] + eps)
        / (target_psd[valid_mask] + eps)
    )

    return np.sqrt(np.mean(error**2))


# %% GREEDY FIXED-SCALE MODE SELECTION

selected_indices = []
remaining_indices = list(range(len(mode_numbers)))

current_psd = np.zeros_like(
    chaos_psd_plot,
    dtype=float,
)

max_modes = 40
target_rms_log_error = 0.1

misfit_history = []

for step in range(min(max_modes, len(mode_numbers))):

    current_misfit = Log_PSD_Misfit(
        current_psd,
        chaos_psd_plot,
        reliable_mask,
    )

    best_mode_idx = None
    best_trial_psd = None
    best_misfit = np.inf

    for mode_idx in remaining_indices:

        trial_psd = (
            current_psd
            + scaled_mode_psds[mode_idx]
        )

        trial_misfit = Log_PSD_Misfit(
            trial_psd,
            chaos_psd_plot,
            reliable_mask,
        )

        if trial_misfit < best_misfit:
            best_misfit = trial_misfit
            best_mode_idx = mode_idx
            best_trial_psd = trial_psd

    if best_mode_idx is None:
        print("No remaining candidate modes.")
        break

    # Stop when no unused mode improves the selected-patch misfit
    if best_misfit >= current_misfit:
        print(
            "Stopped: no remaining mode improves the "
            "degree-frequency patch misfit."
        )
        break

    selected_indices.append(best_mode_idx)
    remaining_indices.remove(best_mode_idx)

    current_psd = best_trial_psd
    misfit_history.append(best_misfit)

    print(
        f"Step {step + 1:2d}: "
        f"added mode {mode_numbers[best_mode_idx]:2d}, "
        f"period={mode_periods[best_mode_idx]:.2f} yr, "
        f"RMS log error={best_misfit:.3f}"
    )

    if best_misfit <= target_rms_log_error:
        print(
            "Stopped: target RMS log error reached."
        )
        break


selected_indices = np.asarray(
    selected_indices,
    dtype=int,
)

selected_mode_numbers = mode_numbers[selected_indices]
selected_periods = mode_periods[selected_indices]

print("\nSelected modes:", selected_mode_numbers)
print("Selected periods:", selected_periods)


# %% ERROR METRICS OVER THE SELECTED PATCH ONLY

eps = 1e-20

log_error = np.full_like(
    chaos_psd_plot,
    np.nan,
    dtype=float,
)

log_error[reliable_mask] = np.log10(
    (current_psd[reliable_mask] + eps)
    / (chaos_psd_plot[reliable_mask] + eps)
)

factor_2_limit = np.log10(2)
factor_3_limit = np.log10(3)

fraction_factor_2 = np.mean(
    np.abs(log_error[reliable_mask])
    <= factor_2_limit
)

fraction_factor_3 = np.mean(
    np.abs(log_error[reliable_mask])
    <= factor_3_limit
)

final_rms_log_error = np.sqrt(
    np.mean(log_error[reliable_mask] ** 2)
)

median_abs_log_error = np.median(
    np.abs(log_error[reliable_mask])
)

mean_log_bias = np.mean(
    log_error[reliable_mask]
)

print(
    f"\nFinal RMS log error:          "
    f"{final_rms_log_error:.3f}"
)
print(
    f"Median absolute log error:   "
    f"{median_abs_log_error:.3f}"
)
print(
    f"Mean signed log bias:        "
    f"{mean_log_bias:.3f}"
)
print(
    f"Cells within factor 2:       "
    f"{fraction_factor_2:.1%}"
)
print(
    f"Cells within factor 3:       "
    f"{fraction_factor_3:.1%}"
)


# %% CONSTRUCT THE SELECTED COHERENT TIME SERIES

if selected_indices.size > 0:
    selected_gnm_total = np.sum(
        scaled_mode_series[selected_indices],
        axis=0,
    )
else:
    selected_gnm_total = np.zeros_like(
        scaled_mode_series[0]
    )

print(
    "Selected combined time-series shape:",
    selected_gnm_total.shape,
)


# %% PLOT GREEDY MISFIT HISTORY

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    np.arange(1, len(misfit_history) + 1),
    misfit_history,
    "x-",
)

ax.axhline(
    target_rms_log_error,
    linestyle="--",
    label="Target RMS error",
)

ax.set_xlabel("Number of selected modes")
ax.set_ylabel("RMS log$_{10}$ PSD error")
ax.set_title("Fixed-scale greedy mode selection")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

# %% PLOT FINAL LINEAR PSD-SUM ERROR RELATIVE TO CHAOS

eps = 1e-20

# current_psd is the linear sum of the individually scaled PSD templates
predicted_linear_psd = current_psd

# Only calculate error inside the selected reliable patch
log10_psd_error = np.full_like(
    chaos_psd_plot,
    np.nan,
    dtype=float,
)

valid_error_mask = (
    reliable_mask
    & np.isfinite(predicted_linear_psd)
    & np.isfinite(chaos_psd_plot)
    & (predicted_linear_psd >= 0)
    & (chaos_psd_plot > 0)
)

log10_psd_error[valid_error_mask] = np.log10(
    (predicted_linear_psd[valid_error_mask] + eps)
    / (chaos_psd_plot[valid_error_mask] + eps)
)

# Symmetric plotting range:
# ±0.301 = factor 2
# ±0.477 = factor 3
# ±1.000 = factor 10
error_limit = 1

masked_error = np.ma.masked_invalid(log10_psd_error)

cmap = plt.get_cmap("coolwarm").copy()
cmap.set_bad("lightgrey")

fig, ax = plt.subplots(figsize=(11, 6))

im = ax.imshow(
    masked_error,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=[
        frequencies_plot[0],
        frequencies_plot[-1],
        degrees_plot[0] - 0.5,
        degrees_plot[-1] + 0.5,
    ],
    cmap=cmap,
    vmin=-error_limit,
    vmax=error_limit,
)

ax.set_xlabel("Frequency [yr$^{-1}$]")
ax.set_ylabel("Spherical-harmonic degree $n$")
ax.set_title(
    "Predicted linear PSD sum relative to CHAOS-8.6\n"
    r"$\log_{10}(S_{\mathrm{predicted}}/S_{\mathrm{CHAOS}})$"
)

ax.set_yticks(degrees_plot)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(
    r"$\log_{10}(S_{\mathrm{predicted}}/S_{\mathrm{CHAOS}})$"
)

plt.tight_layout()
plt.show()

# %% ACTUAL PSD ERROR FROM THE COHERENT SELECTED-MODE SUM

actual_psd, actual_f = Lowes_Degree_PSD_All_Degrees(
    selected_gnm_total,
    a=r_earth,
    r=r_choice,
)

if not np.allclose(actual_f, chaos_f):
    raise ValueError("Actual and CHAOS frequency axes do not match.")

actual_psd_plot = actual_psd[g_mask][:, f_mask]

actual_log_error = np.full_like(
    chaos_psd_plot,
    np.nan,
    dtype=float,
)

valid = (
    reliable_mask
    & np.isfinite(actual_psd_plot)
    & (actual_psd_plot >= 0)
)

actual_log_error[valid] = np.log10(
    (actual_psd_plot[valid] + eps)
    / (chaos_psd_plot[valid] + eps)
)

fig, ax = plt.subplots(figsize=(11, 6))

im = ax.imshow(
    np.ma.masked_invalid(actual_log_error),
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=[
        frequencies_plot[0],
        frequencies_plot[-1],
        degrees_plot[0] - 0.5,
        degrees_plot[-1] + 0.5,
    ],
    cmap="coolwarm",
    vmin=-error_limit,
    vmax=error_limit,
)

ax.set_xlabel("Frequency [yr$^{-1}$]")
ax.set_ylabel("Spherical-harmonic degree $n$")
ax.set_yticks(degrees_plot)
ax.set_title(
    "Actual coherent selected-mode PSD relative to CHAOS-8.6\n"
    r"$\log_{10}(S_{\mathrm{actual}}/S_{\mathrm{CHAOS}})$"
)

fig.colorbar(
    im,
    ax=ax,
    label=r"$\log_{10}(S_{\mathrm{actual}}/S_{\mathrm{CHAOS}})$",
)

plt.tight_layout()
plt.show()

actual_error_values = actual_log_error[reliable_mask]

print(
    "Actual coherent RMS log error:",
    np.sqrt(np.nanmean(actual_error_values**2)),
)
print(
    "Actual cells within factor 2:",
    f"{np.nanmean(np.abs(actual_error_values) <= np.log10(2)):.1%}",
)
print(
    "Actual cells within factor 3:",
    f"{np.nanmean(np.abs(actual_error_values) <= np.log10(3)):.1%}",
)
# %%
# %% MIXED-INTEGER MODE-SELECTION OPTIMISATION

import numpy as np

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix, eye, hstack, vstack


def Select_Mode_Set_MILP(
    scaled_mode_psds,
    chaos_psd,
    reliable_mask,
    factor_limit=2.0,
    min_modes=1,
    max_modes=20,
    mode_penalty=1e-3,
    slack_penalty=1.0,
    time_limit=120,
):
    """
    Select a subset of fixed-scale mode PSD templates.

    The optimiser attempts to keep the linear PSD sum within
    `factor_limit` of CHAOS over `reliable_mask`.

    Parameters
    ----------
    scaled_mode_psds : (n_modes, n_degree, n_frequency) array
        Individually water-level-scaled PSD templates.

    chaos_psd : (n_degree, n_frequency) array
        Target CHAOS PSD.

    reliable_mask : (n_degree, n_frequency) boolean array
        Cells included in the optimisation.

    factor_limit : float
        Desired multiplicative agreement, e.g. 2 means within
        a factor of two.

    min_modes, max_modes : int
        Bounds on the number of selected modes. Set both equal
        to request exactly a fixed number of modes.

    mode_penalty : float
        Penalty per selected mode. Increase to favour smaller sets.

    slack_penalty : float
        Penalty for violations of the factor bounds.

    time_limit : float
        Solver time limit in seconds.
    """

    scaled_mode_psds = np.asarray(scaled_mode_psds, dtype=float)
    chaos_psd = np.asarray(chaos_psd, dtype=float)
    reliable_mask = np.asarray(reliable_mask, dtype=bool)

    # Shape: (n_cells, n_modes)
    templates = scaled_mode_psds[:, reliable_mask].T

    # Shape: (n_cells,)
    target = chaos_psd[reliable_mask]

    valid_cells = (
        np.isfinite(target)
        & (target > 0)
        & np.all(np.isfinite(templates), axis=1)
        & np.all(templates >= 0, axis=1)
    )

    templates = templates[valid_cells]
    target = target[valid_cells]

    if target.size == 0:
        raise ValueError("No valid PSD cells remain for optimisation.")

    n_cells, n_modes = templates.shape

    # Normalize each row by CHAOS so the target value is one
    normalized_templates = templates / target[:, None]

    lower_ratio = 1.0 / factor_limit
    upper_ratio = factor_limit

    # Variables:
    #
    # x = [z_1, ..., z_M,
    #      lower_slack_1, ..., lower_slack_J,
    #      upper_slack_1, ..., upper_slack_J]
    #
    # z_i are binary mode-selection variables.

    identity_cells = eye(n_cells, format="csc")
    zero_cells = csc_matrix((n_cells, n_cells))
    template_matrix = csc_matrix(normalized_templates)

    # Lower constraint:
    #
    # normalized_templates @ z + lower_slack >= lower_ratio
    A_lower = hstack([
        template_matrix,
        identity_cells,
        zero_cells,
    ])

    # Upper constraint:
    #
    # normalized_templates @ z - upper_slack <= upper_ratio
    A_upper = hstack([
        template_matrix,
        zero_cells,
        -identity_cells,
    ])

    # Number-of-modes constraint
    A_count = hstack([
        csc_matrix(np.ones((1, n_modes))),
        csc_matrix((1, 2 * n_cells)),
    ])

    A = vstack([
        A_lower,
        A_upper,
        A_count,
    ]).tocsc()

    constraint_lower = np.concatenate([
        np.full(n_cells, lower_ratio),
        np.full(n_cells, -np.inf),
        [min_modes],
    ])

    constraint_upper = np.concatenate([
        np.full(n_cells, np.inf),
        np.full(n_cells, upper_ratio),
        [max_modes],
    ])

    constraints = LinearConstraint(
        A,
        constraint_lower,
        constraint_upper,
    )

    # Penalize factor-bound violations much more strongly than
    # simply including another mode.
    objective = np.concatenate([
        np.full(n_modes, mode_penalty),
        np.full(n_cells, slack_penalty / n_cells),
        np.full(n_cells, slack_penalty / n_cells),
    ])

    integrality = np.concatenate([
        np.ones(n_modes, dtype=int),
        np.zeros(2 * n_cells, dtype=int),
    ])

    variable_lower = np.zeros(
        n_modes + 2 * n_cells,
        dtype=float,
    )

    variable_upper = np.concatenate([
        np.ones(n_modes),
        np.full(2 * n_cells, np.inf),
    ])

    bounds = Bounds(
        variable_lower,
        variable_upper,
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "disp": True,
            "time_limit": time_limit,
            "mip_rel_gap": 1e-4,
        },
    )

    if result.x is None:
        raise RuntimeError(
            f"MILP returned no solution: {result.message}"
        )

    mode_variables = result.x[:n_modes]

    selected_indices = np.flatnonzero(
        mode_variables > 0.5
    )

    lower_slack = result.x[
        n_modes:n_modes + n_cells
    ]

    upper_slack = result.x[
        n_modes + n_cells:
    ]

    predicted_linear_psd = np.sum(
        scaled_mode_psds[selected_indices],
        axis=0,
    )

    return {
        "result": result,
        "selected_indices": selected_indices,
        "predicted_linear_psd": predicted_linear_psd,
        "lower_slack": lower_slack,
        "upper_slack": upper_slack,
        "valid_cell_mask_1d": valid_cells,
    }

milp_solution = Select_Mode_Set_MILP(
    scaled_mode_psds=scaled_mode_psds,
    chaos_psd=chaos_psd_plot,
    reliable_mask=reliable_mask,
    factor_limit=2.0,
    min_modes=5,
    max_modes=20,
    mode_penalty=1e-3,
    slack_penalty=1.0,
    time_limit=120,
)

milp_selected_indices = milp_solution[
    "selected_indices"
]

milp_selected_modes = mode_numbers[
    milp_selected_indices
]

milp_selected_periods = mode_periods[
    milp_selected_indices
]

milp_linear_psd = milp_solution[
    "predicted_linear_psd"
]

print("Solver message:", milp_solution["result"].message)
print("Selected modes:", milp_selected_modes)
print("Selected periods:", milp_selected_periods)
print("Number selected:", len(milp_selected_indices))
# %%

milp_log_error = np.full_like(
    chaos_psd_plot,
    np.nan,
    dtype=float,
)

milp_log_error[reliable_mask] = np.log10(
    (
        milp_linear_psd[reliable_mask] + eps
    )
    / (
        chaos_psd_plot[reliable_mask] + eps
    )
)

milp_error_values = milp_log_error[reliable_mask]

print(
    "MILP linear RMS log error:",
    np.sqrt(np.mean(milp_error_values**2)),
)

print(
    "Cells within factor 2:",
    f"{np.mean(np.abs(milp_error_values) <= np.log10(2)):.1%}",
)

print(
    "Cells within factor 3:",
    f"{np.mean(np.abs(milp_error_values) <= np.log10(3)):.1%}",
)

milp_gnm_total = np.sum(
    scaled_mode_series[milp_selected_indices],
    axis=0,
)

milp_actual_psd, milp_actual_f = (
    Lowes_Degree_PSD_All_Degrees(
        milp_gnm_total,
        a=r_earth,
        r=r_choice,
    )
)

if not np.allclose(milp_actual_f, chaos_f):
    raise ValueError("MILP and CHAOS frequency axes differ.")

milp_actual_psd_plot = (
    milp_actual_psd[g_mask][:, f_mask]
)

milp_actual_error = np.log10(
    (
        milp_actual_psd_plot[reliable_mask] + eps
    )
    / (
        chaos_psd_plot[reliable_mask] + eps
    )
)

print(
    "MILP coherent RMS log error:",
    np.sqrt(np.mean(milp_actual_error**2)),
)
# %%
# %% IMSHOW OF ACTUAL COHERENT PSD ERROR FROM THE MILP SOLUTION

# This assumes you already computed:
# - milp_selected_indices
# - scaled_mode_series
# - chaos_psd_plot
# - chaos_f
# - g_mask, f_mask
# - reliable_mask
# - frequencies_plot, degrees_plot

eps = 1e-20
error_limit = 1  # 1.0 = factor 10, np.log10(3) = factor 3, np.log10(2) = factor 2

# Build coherent summed time series from the MILP-selected modes
milp_gnm_total = np.sum(
    scaled_mode_series[milp_selected_indices],
    axis=0,
)

# Compute actual PSD of that coherent sum
milp_actual_psd, milp_actual_f = Lowes_Degree_PSD_All_Degrees(
    milp_gnm_total,
    a=r_earth,
    r=r_choice,
)

if not np.allclose(milp_actual_f, chaos_f):
    raise ValueError("MILP actual PSD frequency axis does not match CHAOS.")

milp_actual_psd_plot = milp_actual_psd[g_mask][:, f_mask]

# Compute signed log10 error only over the reliable patch
milp_actual_log_error = np.full_like(
    chaos_psd_plot,
    np.nan,
    dtype=float,
)

valid = (
    reliable_mask
    & np.isfinite(milp_actual_psd_plot)
    & np.isfinite(chaos_psd_plot)
    & (milp_actual_psd_plot >= 0)
    & (chaos_psd_plot > 0)
)

milp_actual_log_error[valid] = np.log10(
    (milp_actual_psd_plot[valid] + eps)
    / (chaos_psd_plot[valid] + eps)
)

# Plot
masked_error = np.ma.masked_invalid(milp_actual_log_error)


fig, ax = plt.subplots(figsize=(11, 6))

im = ax.imshow(
    masked_error,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=[
        frequencies_plot[0],
        frequencies_plot[-1],
        degrees_plot[0] - 0.5,
        degrees_plot[-1] + 0.5,
    ],
    cmap=disc_cmap,
    norm=disc_norm,
)

ax.set_xlabel("Frequency [yr$^{-1}$]")
ax.set_ylabel("Spherical-harmonic degree $n$")
ax.set_yticks(degrees_plot)
ax.set_title(
    "Actual coherent MILP solution PSD relative to CHAOS-8.6\n"
    r"$\log_{10}(S_{\mathrm{MILP,actual}}/S_{\mathrm{CHAOS}})$"
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(
    r"$\log_{10}(S_{\mathrm{MILP,actual}}/S_{\mathrm{CHAOS}})$"
)

plt.tight_layout()
plt.show()
# %%
# %% HYBRID TRUE-PSD MODE-SELECTION OPTIMISATION
#
# MILP/greedy initialisation
#       -> simulated annealing using actual coherent PSD
#       -> deterministic one-for-one swap refinement
#
# Required existing variables:
#   scaled_mode_series, scaled_mode_psds
#   mode_numbers, mode_periods
#   chaos_psd_plot, chaos_f
#   g_mask, f_mask, reliable_mask
#   frequencies_plot, degrees_plot
#   Lowes_Degree_PSD_All_Degrees
#   r_earth

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm


# ---------------------------------------------------------------------
# SEARCH SETTINGS
# ---------------------------------------------------------------------

random_seed = 2

# None uses the number selected by the MILP/greedy initial solution.
# Set an integer to force exactly that number of modes.
k_modes = None

n_restarts = 3
annealing_steps_per_restart = 1200

temperature_start = 0.05
temperature_end = 1e-4

max_local_swap_passes = 4
improvement_tolerance = 1e-6

rng = np.random.default_rng(random_seed)

n_modes = len(mode_numbers)
all_mode_indices = np.arange(n_modes)

if scaled_mode_series.shape[0] != n_modes:
    raise ValueError(
        "scaled_mode_series and mode_numbers contain different "
        "numbers of modes."
    )


# ---------------------------------------------------------------------
# TRUE COHERENT PSD OBJECTIVE
# ---------------------------------------------------------------------

psd_epsilon = (
    1e-15
    * np.nanmax(chaos_psd_plot[reliable_mask])
)


def True_PSD_And_Loss(gnm_total):
    """
    Compute the actual coherent PSD and RMS log10 error over reliable_mask.
    """

    actual_psd, actual_f = Lowes_Degree_PSD_All_Degrees(
        gnm_total,
        a=r_earth,
        r=r_choice,
    )

    if not np.allclose(actual_f, chaos_f):
        raise ValueError(
            "Candidate and CHAOS frequency axes do not match."
        )

    actual_psd_plot = actual_psd[g_mask][:, f_mask]

    valid = (
        reliable_mask
        & np.isfinite(actual_psd_plot)
        & np.isfinite(chaos_psd_plot)
        & (actual_psd_plot >= 0)
        & (chaos_psd_plot > 0)
    )

    log_error = np.log10(
        (actual_psd_plot[valid] + psd_epsilon)
        / (chaos_psd_plot[valid] + psd_epsilon)
    )

    loss = np.sqrt(np.mean(log_error**2))

    return loss, actual_psd_plot


# Cache losses for mode sets already evaluated
loss_cache = {}


def Evaluate_Mode_Set(mode_set, gnm_total=None):
    """
    Evaluate one fixed-size mode subset using the true coherent PSD.
    """

    key = tuple(sorted(mode_set))

    if key in loss_cache:
        return loss_cache[key]

    if gnm_total is None:
        gnm_total = np.sum(
            scaled_mode_series[list(key)],
            axis=0,
        )

    loss, _ = True_PSD_And_Loss(gnm_total)

    loss_cache[key] = loss

    return loss


# ---------------------------------------------------------------------
# INITIAL SUBSET
# ---------------------------------------------------------------------

initial_indices = None

if (
    "milp_selected_indices" in globals()
    and len(milp_selected_indices) > 0
):
    initial_indices = np.asarray(
        milp_selected_indices,
        dtype=int,
    )

elif (
    "selected_indices" in globals()
    and len(selected_indices) > 0
):
    initial_indices = np.asarray(
        selected_indices,
        dtype=int,
    )


# Fallback: construct a linear-PSD greedy initial subset
if initial_indices is None:

    if k_modes is None:
        k_modes = 12

    greedy_selected = []
    greedy_remaining = list(all_mode_indices)
    greedy_psd = np.zeros_like(chaos_psd_plot)

    for _ in range(k_modes):

        best_idx = None
        best_loss = np.inf

        for mode_idx in greedy_remaining:

            trial_psd = (
                greedy_psd
                + scaled_mode_psds[mode_idx]
            )

            trial_error = np.log10(
                (
                    trial_psd[reliable_mask]
                    + psd_epsilon
                )
                / (
                    chaos_psd_plot[reliable_mask]
                    + psd_epsilon
                )
            )

            trial_loss = np.sqrt(
                np.mean(trial_error**2)
            )

            if trial_loss < best_loss:
                best_loss = trial_loss
                best_idx = mode_idx

        greedy_selected.append(best_idx)
        greedy_remaining.remove(best_idx)

        greedy_psd += scaled_mode_psds[best_idx]

    initial_indices = np.asarray(
        greedy_selected,
        dtype=int,
    )


if k_modes is None:
    k_modes = len(initial_indices)

if not 1 <= k_modes < n_modes:
    raise ValueError(
        "k_modes must be between 1 and n_modes - 1."
    )


# Adjust initial solution if a different k_modes was requested
initial_indices = np.unique(initial_indices)

if len(initial_indices) > k_modes:
    initial_indices = rng.choice(
        initial_indices,
        size=k_modes,
        replace=False,
    )

elif len(initial_indices) < k_modes:

    available = np.setdiff1d(
        all_mode_indices,
        initial_indices,
    )

    additions = rng.choice(
        available,
        size=k_modes - len(initial_indices),
        replace=False,
    )

    initial_indices = np.concatenate([
        initial_indices,
        additions,
    ])

initial_set = set(initial_indices.tolist())

initial_gnm_total = np.sum(
    scaled_mode_series[list(initial_set)],
    axis=0,
)

initial_loss = Evaluate_Mode_Set(
    initial_set,
    initial_gnm_total,
)

print(
    f"Initial coherent RMS log10 error: "
    f"{initial_loss:.4f}"
)
print(
    f"Optimising exactly {k_modes} modes."
)


# ---------------------------------------------------------------------
# SIMULATED ANNEALING
# ---------------------------------------------------------------------

global_best_set = initial_set.copy()
global_best_gnm = initial_gnm_total.copy()
global_best_loss = initial_loss


for restart in range(n_restarts):

    current_set = initial_set.copy()

    # Perturb later restarts so that they explore different basins
    if restart > 0:

        n_perturb = max(
            1,
            int(np.ceil(0.25 * k_modes)),
        )

        for _ in range(n_perturb):

            selected = np.asarray(
                list(current_set),
                dtype=int,
            )

            unselected = np.setdiff1d(
                all_mode_indices,
                selected,
            )

            mode_out = rng.choice(selected)
            mode_in = rng.choice(unselected)

            current_set.remove(int(mode_out))
            current_set.add(int(mode_in))

    current_gnm = np.sum(
        scaled_mode_series[list(current_set)],
        axis=0,
    )

    current_loss = Evaluate_Mode_Set(
        current_set,
        current_gnm,
    )

    for iteration in range(
        annealing_steps_per_restart
    ):

        fraction = (
            iteration
            / max(annealing_steps_per_restart - 1, 1)
        )

        temperature = (
            temperature_start
            * (
                temperature_end
                / temperature_start
            ) ** fraction
        )

        selected = np.asarray(
            list(current_set),
            dtype=int,
        )

        unselected = np.setdiff1d(
            all_mode_indices,
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

        candidate_loss = Evaluate_Mode_Set(
            candidate_set,
            candidate_gnm,
        )

        loss_change = (
            candidate_loss
            - current_loss
        )

        accept = (
            loss_change < 0
            or rng.random()
            < np.exp(-loss_change / temperature)
        )

        if accept:

            current_set = candidate_set
            current_gnm = candidate_gnm
            current_loss = candidate_loss

        if current_loss < global_best_loss:

            global_best_set = current_set.copy()
            global_best_gnm = current_gnm.copy()
            global_best_loss = current_loss

    print(
        f"Annealing restart {restart + 1}/{n_restarts}: "
        f"best RMS error = {global_best_loss:.4f}"
    )


# ---------------------------------------------------------------------
# DETERMINISTIC BEST ONE-FOR-ONE SWAP REFINEMENT
# ---------------------------------------------------------------------

for local_pass in range(max_local_swap_passes):

    selected = np.asarray(
        sorted(global_best_set),
        dtype=int,
    )

    unselected = np.setdiff1d(
        all_mode_indices,
        selected,
    )

    best_swap = None
    best_swap_loss = global_best_loss
    best_swap_gnm = None

    for mode_out in selected:

        for mode_in in unselected:

            candidate_set = global_best_set.copy()
            candidate_set.remove(int(mode_out))
            candidate_set.add(int(mode_in))

            candidate_gnm = (
                global_best_gnm
                - scaled_mode_series[mode_out]
                + scaled_mode_series[mode_in]
            )

            candidate_loss = Evaluate_Mode_Set(
                candidate_set,
                candidate_gnm,
            )

            if (
                candidate_loss
                < best_swap_loss
                - improvement_tolerance
            ):

                best_swap_loss = candidate_loss
                best_swap = (
                    int(mode_out),
                    int(mode_in),
                    candidate_set,
                )
                best_swap_gnm = candidate_gnm

    if best_swap is None:

        print(
            "Swap refinement stopped: "
            "no improving one-mode replacement."
        )
        break

    mode_out, mode_in, candidate_set = best_swap

    global_best_set = candidate_set
    global_best_gnm = best_swap_gnm
    global_best_loss = best_swap_loss

    print(
        f"Swap pass {local_pass + 1}: "
        f"mode {mode_numbers[mode_out]} -> "
        f"mode {mode_numbers[mode_in]}, "
        f"RMS error = {global_best_loss:.4f}"
    )


# ---------------------------------------------------------------------
# FINAL TRUE COHERENT PSD AND METRICS
# ---------------------------------------------------------------------

optimal_indices = np.asarray(
    sorted(global_best_set),
    dtype=int,
)

optimal_mode_numbers = mode_numbers[
    optimal_indices
]

optimal_periods = mode_periods[
    optimal_indices
]

optimal_gnm_total = global_best_gnm

optimal_loss, optimal_actual_psd_plot = (
    True_PSD_And_Loss(optimal_gnm_total)
)

optimal_log_error = np.full_like(
    chaos_psd_plot,
    np.nan,
    dtype=float,
)

valid_final = (
    reliable_mask
    & np.isfinite(optimal_actual_psd_plot)
    & (optimal_actual_psd_plot >= 0)
)

optimal_log_error[valid_final] = np.log10(
    (
        optimal_actual_psd_plot[valid_final]
        + psd_epsilon
    )
    / (
        chaos_psd_plot[valid_final]
        + psd_epsilon
    )
)

error_values = optimal_log_error[reliable_mask]
absolute_error = np.abs(error_values)

print("\nFinal coherent optimisation result")
print("----------------------------------")
print("Selected modes:", optimal_mode_numbers)
print("Selected periods:", optimal_periods)
print(f"Number of modes:             {len(optimal_indices)}")
print(f"Initial RMS log10 error:     {initial_loss:.4f}")
print(f"Final RMS log10 error:       {optimal_loss:.4f}")
print(
    f"Median absolute log error:  "
    f"{np.nanmedian(absolute_error):.4f}"
)
print(
    f"Mean signed log bias:       "
    f"{np.nanmean(error_values):+.4f}"
)
print(
    f"Cells within factor 2:      "
    f"{np.nanmean(absolute_error <= np.log10(2)):.1%}"
)
print(
    f"Cells within factor 3:      "
    f"{np.nanmean(absolute_error <= np.log10(3)):.1%}"
)
print(
    f"Cells within factor 10:     "
    f"{np.nanmean(absolute_error <= 1):.1%}"
)
print(
    f"Unique coherent subsets evaluated: "
    f"{len(loss_cache)}"
)


# ---------------------------------------------------------------------
# DISCRETE LOG10 ERROR HEATMAP
# ---------------------------------------------------------------------

factor_2 = np.log10(2)
factor_3 = np.log10(3)

max_abs_error = np.nanmax(
    np.abs(optimal_log_error)
)

plot_cap = max(
    1.01,
    max_abs_error + 1e-6,
)

bounds = [
    -plot_cap,
    -1,
    -factor_3,
    -factor_2,
     factor_2,
     factor_3,
     1,
     plot_cap,
]

colors = [
    "#08306B",  # more than 10x too low
    "#3F77B5",  # 3x to 10x too low
    "#9ECAE1",  # 2x to 3x too low
    "#FFFFFF",  # within factor 2
    "#F4A582",  # 2x to 3x too high
    "#E76F51",  # 3x to 10x too high
    "#B2182B",  # more than 10x too high
]

discrete_cmap = ListedColormap(colors)
discrete_cmap.set_bad("lightgrey")

discrete_norm = BoundaryNorm(
    bounds,
    discrete_cmap.N,
    clip=True,
)

fig, ax = plt.subplots(figsize=(11, 6))

im = ax.imshow(
    np.ma.masked_invalid(optimal_log_error),
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=[
        frequencies_plot[0],
        frequencies_plot[-1],
        degrees_plot[0] - 0.5,
        degrees_plot[-1] + 0.5,
    ],
    cmap=discrete_cmap,
    norm=discrete_norm,
)

ax.set_xlabel("Frequency [yr$^{-1}$]")
ax.set_ylabel("Spherical-harmonic degree $n$")
ax.set_yticks(degrees_plot)

ax.set_title(
    "True coherent optimised-mode PSD relative to CHAOS-8.6\n"
    r"$\log_{10}(S_{\mathrm{optimised}}/S_{\mathrm{CHAOS}})$"
)

cbar = fig.colorbar(
    im,
    ax=ax,
    boundaries=bounds,
    ticks=[
        -1,
        -factor_3,
        -factor_2,
         factor_2,
         factor_3,
         1,
    ],
)

cbar.set_ticklabels([
    r"$-1$",
    r"$-\log_{10}(3)$",
    r"$-\log_{10}(2)$",
    r"$\log_{10}(2)$",
    r"$\log_{10}(3)$",
    r"$1$",
])

cbar.set_label(
    r"$\log_{10}(S_{\mathrm{optimised}}/S_{\mathrm{CHAOS}})$"
)

plt.tight_layout()
plt.show()
# %%
# %% THREE-PANEL CMB PSD COMPARISON OVER OPTIMISATION PATCH

from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

# ---------------------------------------------------------------------
# Recompute true coherent PSD explicitly at the CMB
# ---------------------------------------------------------------------

optimal_cmb_psd, optimal_cmb_f = Lowes_Degree_PSD_All_Degrees(
    optimal_gnm_total,
    a=r_earth,
    r=r_cmb,
)

if not np.allclose(optimal_cmb_f, chaos_f):
    raise ValueError(
        "Optimal-mode and CHAOS frequency axes do not match."
    )

optimal_cmb_psd_plot = optimal_cmb_psd[g_mask][:, f_mask]

# Ensure CHAOS target was also evaluated at the CMB
chaos_cmb_psd, chaos_cmb_f = Lowes_Degree_PSD_All_Degrees(
    chaos_good,
    a=r_earth,
    r=r_cmb,
)

if not np.allclose(chaos_cmb_f, chaos_f):
    raise ValueError(
        "Recomputed CHAOS and existing frequency axes do not match."
    )

chaos_cmb_psd_plot = chaos_cmb_psd[g_mask][:, f_mask]


# ---------------------------------------------------------------------
# Extract only the degree-frequency patch used in optimisation
# ---------------------------------------------------------------------

degree_patch = degrees_plot[degree_fit_mask]
frequency_patch = frequencies_plot[frequency_fit_mask]

chaos_patch = chaos_cmb_psd_plot[
    np.ix_(degree_fit_mask, frequency_fit_mask)
]

optimal_patch = optimal_cmb_psd_plot[
    np.ix_(degree_fit_mask, frequency_fit_mask)
]

reliable_patch = reliable_mask[
    np.ix_(degree_fit_mask, frequency_fit_mask)
]

if degree_patch.size == 0 or frequency_patch.size == 0:
    raise ValueError("The optimisation patch is empty.")

# Mask cells excluded from the actual optimisation
chaos_patch_masked = np.ma.masked_where(
    ~reliable_patch | ~np.isfinite(chaos_patch) | (chaos_patch <= 0),
    chaos_patch,
)

optimal_patch_masked = np.ma.masked_where(
    ~reliable_patch | ~np.isfinite(optimal_patch) | (optimal_patch <= 0),
    optimal_patch,
)


# ---------------------------------------------------------------------
# Shared logarithmic PSD colour scale
# ---------------------------------------------------------------------

joint_positive_values = np.concatenate([
    chaos_patch_masked.compressed(),
    optimal_patch_masked.compressed(),
])

if joint_positive_values.size == 0:
    raise ValueError("No positive PSD values exist in the selected patch.")

psd_vmin = np.nanmin(joint_positive_values)
psd_vmax = np.nanmax(joint_positive_values)

psd_norm = LogNorm(
    vmin=psd_vmin,
    vmax=psd_vmax,
)

psd_cmap = plt.get_cmap("viridis").copy()
psd_cmap.set_bad("lightgrey")


# ---------------------------------------------------------------------
# Signed log10 error and discrete error colour map
# ---------------------------------------------------------------------

eps = 1e-20

log_error_patch = np.full_like(
    chaos_patch,
    np.nan,
    dtype=float,
)

error_valid = (
    reliable_patch
    & np.isfinite(chaos_patch)
    & np.isfinite(optimal_patch)
    & (chaos_patch > 0)
    & (optimal_patch >= 0)
)

log_error_patch[error_valid] = np.log10(
    (optimal_patch[error_valid] + eps)
    / (chaos_patch[error_valid] + eps)
)

factor_2 = np.log10(2)
factor_3 = np.log10(3)

finite_error = np.abs(
    log_error_patch[np.isfinite(log_error_patch)]
)

plot_cap = max(
    1.01,
    finite_error.max() + 1e-6 if finite_error.size else 1.01,
)

error_bounds = [
    -plot_cap,
    -1,
    -factor_3,
    -factor_2,
     factor_2,
     factor_3,
     1,
     plot_cap,
]

error_colors = [
    "#08306B",  # >10 times too low
    "#2171B5",  # 3–10 times too low
    "#9ECAE1",  # 2–3 times too low
    "#FFFFFF",  # within factor 2
    "#FCAE91",  # 2–3 times too high
    "#FB6A4A",  # 3–10 times too high
    "#CB181D",  # >10 times too high
]

error_cmap = ListedColormap(error_colors)
error_cmap.set_bad("lightgrey")

error_norm = BoundaryNorm(
    error_bounds,
    error_cmap.N,
    clip=True,
)


# ---------------------------------------------------------------------
# Cell-centred image extent
# ---------------------------------------------------------------------

if frequency_patch.size > 1:
    df = frequency_patch[1] - frequency_patch[0]
else:
    df = 1.0

if degree_patch.size > 1:
    dn = degree_patch[1] - degree_patch[0]
else:
    dn = 1.0

extent = [
    frequency_patch[0] - df / 2,
    frequency_patch[-1] + df / 2,
    degree_patch[0] - dn / 2,
    degree_patch[-1] + dn / 2,
]


# ---------------------------------------------------------------------
# Three-panel figure
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5),
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
    cmap=psd_cmap,
    norm=psd_norm,
)

optimal_im = axes[1].imshow(
    optimal_patch_masked,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=extent,
    cmap=psd_cmap,
    norm=psd_norm,
)

error_im = axes[2].imshow(
    np.ma.masked_invalid(log_error_patch),
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=extent,
    cmap=error_cmap,
    norm=error_norm,
)

axes[0].set_title("CHAOS-8.6 SV PSD at CMB")
axes[1].set_title("Optimal coherent mode PSD at CMB")
axes[2].set_title(
    r"$\log_{10}(S_{\mathrm{optimal}}/S_{\mathrm{CHAOS}})$"
)

for ax in axes:
    ax.set_xlabel("Frequency [yr$^{-1}$]")
    ax.set_yticks(degree_patch)

axes[0].set_ylabel("Spherical-harmonic degree $n$")

# One shared PSD colourbar for panels 1 and 2
psd_cbar = fig.colorbar(
    chaos_im,
    ax=axes[:2],
    location="bottom",
    fraction=0.08,
    pad=0.12,
)

psd_cbar.set_label(
    r"Lowes-weighted SV PSD at CMB "
    r"$[(\mathrm{nT\,yr^{-1}})^2/(\mathrm{yr^{-1}})]$"
)

# Discrete error colourbar
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
    r"$\log_{10}(S_{\mathrm{optimal}}/S_{\mathrm{CHAOS}})$"
)

fig.suptitle(
    "CHAOS-8.6 and optimised coherent wave spectrum over fitted CMB domain",
    fontsize=14,
)

plt.show()
# %%
