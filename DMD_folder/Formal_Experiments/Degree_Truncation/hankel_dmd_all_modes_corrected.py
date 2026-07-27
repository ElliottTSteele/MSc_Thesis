"""
Combined-mode Hankel DMD synthetic recovery test.

Goal
----
1. Load every theoretical mode listed in ``mode_numbers``.
2. Apply the existing amplitude scaling and CHAOS resolution representation.
3. Sum all modes into ONE synthetic Gauss-coefficient time series.
4. Optionally generate covariance perturbation realisations around that combined
   signal using the existing ``Perturbation_Generate`` utility.
5. For each spherical-harmonic degree band:
      - convert the selected Gauss coefficients to the SV grid,
      - fit Hankel DMD to the complete combined signal,
      - recover oscillatory DMD modes,
      - map the delay-augmented Hankel modes back to the original physical
        SV-grid state space,
      - match each known theoretical input mode to the recovered mode with the
        highest complex-phasor spatial similarity.
6. Plot spatial-similarity and signed-period-error diagnostics.

There is deliberately NO "each wave in isolation" branch in this script.
"""

# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from pydmd import HankelDMD
from tqdm import tqdm

# Project paths / utilities
from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *


# %% ------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------

# Every mode listed here is included simultaneously in the synthetic dataset.
mode_numbers = [1, 6, 13, 21, 32, 45, 53, 60]
mode_numbers = [str(mode_number) for mode_number in mode_numbers]


# ---------------------------------------------------------
# HANKEL DMD PARAMETERS
# ---------------------------------------------------------
#
# ``d`` is the delay-embedding depth.
#
# svd_rank:
#   -1 : no SVD truncation
#    0 : automatic rank selection
#   >0 : fixed rank
# 0<x<1: retain enough singular values for that energy fraction
#
# reconstruction_method controls PyDMD's reconstruction of overlapping
# delay-embedded snapshots. It does NOT automatically map ``dmd.modes`` back
# into the original physical state space, so that mapping is handled explicitly
# below before spatial phasor comparison.
#
hankel_d = 5
hankel_svd_rank = -1
hankel_tlsq_rank = 0
hankel_exact = True
hankel_opt = False
hankel_rescale_mode = None
hankel_forward_backward = False
hankel_sorted_eigs = False
hankel_reconstruction_method = "first"
hankel_tikhonov_regularization = None


# ---------------------------------------------------------
# HOW TO MAP AUGMENTED HANKEL MODES BACK TO PHYSICAL SPACE
# ---------------------------------------------------------
#
# "first":
#     use the first n_physical rows of each delay-augmented DMD mode.
#     This corresponds to the 0-th / first delay block and is the safest,
#     least-assumptive representation for direct comparison with the original
#     SV-grid phasor.
#
# At present this script intentionally supports only "first".
#
hankel_physical_mode_method = "first"


# ---------------------------------------------------------
# PERTURBATION ENSEMBLE
# ---------------------------------------------------------

ensemble_flag = True
n_realisations = 10

noise_temporal_model = "independent"
noise_tau_years = 1.5


# ---------------------------------------------------------
# TIME / DEGREE SETTINGS
# ---------------------------------------------------------

high_q_flag = True
n_skip = 3

Nmax = 20

# Four-degree bands:
#   1-4, 5-8, 9-12, 13-16, 17-20
degree_truncations_tested = np.arange(4, 21, 4)

# Only accept finite oscillatory DMD periods inside these broad bounds.
candidate_period_min = 1e-5
candidate_period_max = 1000.0


# %% ------------------------------------------------------
# VALIDATION / HELPERS
# ------------------------------------------------------

if not mode_numbers:
    raise ValueError("mode_numbers must contain at least one mode.")

if int(hankel_d) != hankel_d or hankel_d < 1:
    raise ValueError("hankel_d must be a positive integer.")

if int(n_skip) != n_skip or n_skip < 1:
    raise ValueError("n_skip must be a positive integer.")

if ensemble_flag and n_realisations < 1:
    raise ValueError(
        "n_realisations must be >= 1 when ensemble_flag=True."
    )

if hankel_physical_mode_method != "first":
    raise ValueError(
        "This script currently supports only "
        "hankel_physical_mode_method='first'."
    )


def build_hankel_dmd():
    """Construct the configured PyDMD HankelDMD instance."""

    return HankelDMD(
        svd_rank=hankel_svd_rank,
        tlsq_rank=hankel_tlsq_rank,
        exact=hankel_exact,
        opt=hankel_opt,
        rescale_mode=hankel_rescale_mode,
        forward_backward=hankel_forward_backward,
        d=hankel_d,
        sorted_eigs=hankel_sorted_eigs,
        reconstruction_method=hankel_reconstruction_method,
        tikhonov_regularization=hankel_tikhonov_regularization,
    )


def physicalise_hankel_phasors(
    augmented_phasors,
    n_physical,
    expected_d,
):
    """
    Map delay-augmented Hankel DMD phasors to the original physical state.

    Parameters
    ----------
    augmented_phasors : ndarray, shape (n_augmented, n_modes)
        Phasors returned after ``DMD_Mode_Pair`` has operated on HankelDMD.
    n_physical : int
        Number of spatial degrees of freedom in the original SV-grid snapshots.
    expected_d : int
        Configured Hankel delay depth.

    Returns
    -------
    physical_phasors : ndarray, shape (n_physical, n_modes)
        Phasors in the same physical SV-grid space as the theoretical phasors.
    """

    augmented_phasors = np.asarray(augmented_phasors)

    if augmented_phasors.ndim != 2:
        raise ValueError(
            "Expected recovered phasors to be a 2-D array with shape "
            "(n_state, n_modes); received "
            f"{augmented_phasors.shape}."
        )

    n_augmented = augmented_phasors.shape[0]

    # d=1, or an implementation returning physical-space modes directly.
    if n_augmented == n_physical:
        return augmented_phasors

    if n_augmented % n_physical != 0:
        raise ValueError(
            "Recovered Hankel mode dimension is not an integer multiple of "
            "the original physical state dimension. "
            f"Recovered rows={n_augmented}, physical rows={n_physical}."
        )

    inferred_d = n_augmented // n_physical

    if inferred_d != expected_d:
        raise ValueError(
            "Recovered Hankel mode dimension implies a delay depth different "
            "from the configured hankel_d. "
            f"inferred_d={inferred_d}, configured hankel_d={expected_d}, "
            f"mode_shape={augmented_phasors.shape}, "
            f"n_physical={n_physical}."
        )

    # PyDMD's time-delay state stacks d copies of the physical state.
    # For direct spatial comparison with a theoretical phasor in the original
    # grid space, use the first (0-th delay) physical block.
    physical_phasors = augmented_phasors[:n_physical, :]

    return physical_phasors


def extract_hankel_candidates(
    dmd,
    dt_snapshot,
    n_physical,
):
    """
    Extract oscillatory Hankel-DMD candidates in original physical state space.

    ``DMD_Mode_Pair`` is retained from the existing thesis workflow so the
    eigenvalue convention and conjugate-pair handling remain consistent with
    the Exact-DMD experiments.
    """

    recovered_dict = DMD_Mode_Pair(
        dmd,
        dt_snapshot,
    )

    candidate_eigs = np.asarray(
        recovered_dict["continuous_eigenvalues"]
    )

    candidate_modes_augmented = np.asarray(
        recovered_dict["phasors"]
    )

    if candidate_eigs.size == 0:
        return (
            np.asarray([], dtype=complex),
            np.empty((n_physical, 0), dtype=complex),
            np.asarray([], dtype=float),
        )

    candidate_modes = physicalise_hankel_phasors(
        candidate_modes_augmented,
        n_physical=n_physical,
        expected_d=hankel_d,
    )

    if candidate_modes.shape[1] != candidate_eigs.size:
        raise ValueError(
            "Number of recovered phasor columns does not match the number of "
            "continuous eigenvalues: "
            f"{candidate_modes.shape[1]} vs {candidate_eigs.size}."
        )

    candidate_periods = np.full(
        candidate_eigs.size,
        np.nan,
        dtype=float,
    )

    imag = np.abs(np.imag(candidate_eigs))
    oscillatory = imag > 0

    candidate_periods[oscillatory] = (
        2.0 * np.pi / imag[oscillatory]
    )

    valid = (
        np.isfinite(candidate_periods)
        & (candidate_periods > candidate_period_min)
        & (candidate_periods < candidate_period_max)
    )

    return (
        candidate_eigs[valid],
        candidate_modes[:, valid],
        candidate_periods[valid],
    )


def best_spatial_match(
    candidate_modes,
    candidate_eigs,
    candidate_periods,
    target_phasor,
):
    """
    Return the candidate with maximum finite complex-phasor similarity.

    Returns None if no valid candidate produces a finite similarity.
    """

    if candidate_eigs.size == 0:
        return None

    best = None
    best_similarity = -np.inf

    for idx in range(candidate_eigs.size):

        similarity = Complex_Phasor_Compare(
            candidate_modes[:, idx],
            target_phasor,
        )

        if not np.isfinite(similarity):
            continue

        if similarity > best_similarity:
            best_similarity = float(similarity)
            best = {
                "similarity": float(similarity),
                "eigenvalue": candidate_eigs[idx],
                "recovered_period": float(
                    candidate_periods[idx]
                ),
            }

    return best


def finite_quantiles(
    values,
    quantiles=(0.05, 0.5, 0.95),
):
    """Return finite-data quantiles, or None if no finite values exist."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return np.quantile(values, quantiles)


# %% ------------------------------------------------------
# REPORT CONFIGURATION
# ------------------------------------------------------

print(
    "Running combined-mode Hankel DMD with modes:",
    mode_numbers,
)

print(
    "Hankel settings:",
    {
        "d": hankel_d,
        "svd_rank": hankel_svd_rank,
        "tlsq_rank": hankel_tlsq_rank,
        "exact": hankel_exact,
        "opt": hankel_opt,
        "forward_backward": hankel_forward_backward,
        "sorted_eigs": hankel_sorted_eigs,
        "reconstruction_method": hankel_reconstruction_method,
        "physical_mode_method": hankel_physical_mode_method,
    },
)


# %% ------------------------------------------------------
# BUILD ONE COMBINED SYNTHETIC DATASET
# ------------------------------------------------------

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

if not file_path.exists():
    raise FileNotFoundError(
        f"Could not find resolved synthetic spline file: {file_path}"
    )

A_r_Nmax = Truncate_Gauss_Coeffs(
    A_20_dict["r"],
    tmax=Nmax,
)

DMD_recovery = {}
synthetic_suite_info = {}
gnm_total_res_list = []

with h5py.File(file_path, "r") as h5_file:

    for mode_number in tqdm(
        mode_numbers,
        desc="Loading combined modes",
    ):

        mode_data = Component_Load_SV(
            mode_number
        )

        eigenvalue = mode_data["eigenvalue"]
        gnm_phasor = mode_data["gnm"]

        true_period = (
            2.0
            * np.pi
            / np.abs(eigenvalue.imag)
        )

        gnm_phasor_Nmax = Truncate_Gauss_Coeffs(
            gnm_phasor,
            tmax=Nmax,
        )

        try:
            amp_scaler = mode_amp_scalings[
                str(mode_number)
            ]
        except KeyError as exc:
            raise KeyError(
                f"No amplitude scaling found for mode {mode_number}."
            ) from exc

        gnm_phasor_scaled = (
            amp_scaler
            * gnm_phasor_Nmax
        )

        dataset_name = (
            f"mode_{mode_number}/without_decay"
        )

        if dataset_name not in h5_file:
            raise KeyError(
                f"Missing HDF5 dataset: {dataset_name}"
            )

        gnm_spline = np.asarray(
            h5_file[dataset_name][()]
        )

        gnm_spline_scaled = (
            amp_scaler
            * gnm_spline
        )

        # Existing project resolution mapping.
        gnm_mode_res = (
            H_sv
            @ gnm_spline_scaled
        )

        gnm_total_res_list.append(
            gnm_mode_res
        )

        synthetic_suite_info[
            mode_number
        ] = {
            "true_period": true_period,
            "gnm_phasor": gnm_phasor_scaled,
        }

        DMD_recovery[
            mode_number
        ] = {
            "true_period": true_period,
            "DMD": {},
            "DMD_noised": {},
        }


gnm_total_res = np.sum(
    np.asarray(gnm_total_res_list),
    axis=0,
)

del gnm_total_res_list


# %% ------------------------------------------------------
# GENERATE PERTURBED COMBINED SIGNALS
# ------------------------------------------------------

if ensemble_flag:

    noised = Perturbation_Generate(
        gnm_total_res,
        n_realisations=n_realisations,
        temporal_z=noise_temporal_model,
        tau=noise_tau_years,
        dt=dt_years,
    )

    queue = [
        ("no_noise", gnm_total_res)
    ]

    queue.extend(
        ("noised", realisation)
        for realisation in noised
    )

else:

    queue = [
        ("no_noise", gnm_total_res)
    ]


for mode_number in mode_numbers:

    for truncation_degree in degree_truncations_tested:

        DMD_recovery[
            mode_number
        ]["DMD_noised"][
            int(truncation_degree)
        ] = {
            "similarity": [],
            "eigenvalue": [],
            "recovered_period": [],
        }


# %% ------------------------------------------------------
# DEGREE-BAND HANKEL DMD TESTS
# ------------------------------------------------------

dt_snapshot = (
    n_skip
    * dt_years
)

for truncation_degree in tqdm(
    degree_truncations_tested,
    desc="Degree bands",
):

    truncation_degree = int(
        truncation_degree
    )

    tmin_current = max(
        1,
        truncation_degree - 3,
    )

    tmax_current = truncation_degree

    A_r_current = Truncate_Gauss_Coeffs(
        A_20_dict["r"],
        tmax=tmax_current,
        tmin=tmin_current,
    )

    # Precompute every target theoretical phasor once for this degree band.
    target_phasors = {}

    for mode_number in mode_numbers:

        gnm_target = (
            synthetic_suite_info[
                mode_number
            ]["gnm_phasor"]
        )

        gnm_target_band = Truncate_Gauss_Coeffs(
            gnm_target,
            tmax=tmax_current,
            tmin=tmin_current,
        )

        target_phasors[
            mode_number
        ] = (
            A_r_current
            @ gnm_target_band.T
        )

    for result_type, gnm_input in tqdm(
        queue,
        desc=f"Band {tmin_current}-{tmax_current}",
        leave=False,
    ):

        # ---------------------------------------------
        # TIME WINDOW
        # ---------------------------------------------

        if high_q_flag:

            gnm_input_windowed = (
                gnm_input[
                    good_record_slice,
                    :
                ]
            )

        else:

            gnm_input_windowed = gnm_input

        # ---------------------------------------------
        # SELECT CURRENT DEGREE BAND
        # ---------------------------------------------

        gnm_input_band = Truncate_Gauss_Coeffs(
            gnm_input_windowed,
            tmax=tmax_current,
            tmin=tmin_current,
        )

        # ---------------------------------------------
        # GAUSS -> SV GRID
        # ---------------------------------------------

        sv_input_all_steps = (
            A_r_current
            @ gnm_input_band.T
        )

        # Temporal subsampling.
        sv_input = (
            sv_input_all_steps[
                :,
                ::n_skip
            ]
        )

        n_physical = (
            sv_input.shape[0]
        )

        n_snapshots = (
            sv_input.shape[1]
        )

        if n_snapshots <= hankel_d:
            raise ValueError(
                "Not enough temporally subsampled snapshots for the selected "
                "Hankel delay depth. "
                f"n_snapshots={n_snapshots}, hankel_d={hankel_d}. "
                "Reduce hankel_d, reduce n_skip, or use a longer time window."
            )

        # ---------------------------------------------
        # FIT HANKEL DMD
        # ---------------------------------------------

        dmd = build_hankel_dmd()

        dmd.fit(
            sv_input
        )

        # Runtime guard: this should never be ordinary DMD.
        if not isinstance(
            dmd,
            HankelDMD,
        ):
            raise TypeError(
                "Internal error: fitted model is not a HankelDMD instance."
            )

        # Extract / pair modes ONCE per fitted dataset, then compare that
        # recovered candidate set to every known theoretical input mode.
        (
            candidate_eigs,
            candidate_modes,
            candidate_periods,
        ) = extract_hankel_candidates(
            dmd=dmd,
            dt_snapshot=dt_snapshot,
            n_physical=n_physical,
        )

        # Diagnostic sanity check on the first clean fit of each band.
        if result_type == "no_noise":
            print(
                f"Band {tmin_current}-{tmax_current}: "
                f"type={type(dmd).__name__}, "
                f"physical state={n_physical}, "
                f"Hankel mode rows={np.asarray(dmd.modes).shape[0]}, "
                f"valid oscillatory candidates={candidate_eigs.size}"
            )

        # ---------------------------------------------
        # MATCH TO EVERY KNOWN INPUT MODE
        # ---------------------------------------------

        for mode_number in mode_numbers:

            match = best_spatial_match(
                candidate_modes=candidate_modes,
                candidate_eigs=candidate_eigs,
                candidate_periods=candidate_periods,
                target_phasor=target_phasors[
                    mode_number
                ],
            )

            # No eligible oscillatory candidate / no finite similarity.
            if match is None:
                continue

            if result_type == "no_noise":

                DMD_recovery[
                    mode_number
                ]["DMD"][
                    truncation_degree
                ] = {
                    **match,
                    "degree_min": tmin_current,
                    "degree_max": tmax_current,
                }

            else:

                noised_store = (
                    DMD_recovery[
                        mode_number
                    ]["DMD_noised"][
                        truncation_degree
                    ]
                )

                noised_store[
                    "similarity"
                ].append(
                    match["similarity"]
                )

                noised_store[
                    "eigenvalue"
                ].append(
                    match["eigenvalue"]
                )

                noised_store[
                    "recovered_period"
                ].append(
                    match["recovered_period"]
                )


# %% ------------------------------------------------------
# PLOT DMD RECOVERY AS FUNCTION OF DEGREE BAND
# ------------------------------------------------------

degrees = np.asarray(
    sorted(
        int(degree)
        for degree in degree_truncations_tested
    ),
    dtype=int,
)

cmap = plt.colormaps[
    "viridis"
].resampled(
    len(degrees)
)

if len(degrees) > 1:
    degree_step = np.median(
        np.diff(degrees)
    )
else:
    degree_step = 1

boundaries = np.concatenate([
    [
        degrees[0]
        - degree_step / 2
    ],
    (
        degrees[:-1]
        + degrees[1:]
    ) / 2,
    [
        degrees[-1]
        + degree_step / 2
    ],
])

norm = BoundaryNorm(
    boundaries,
    cmap.N,
)


marker_handles = [
    Line2D(
        [0],
        [0],
        marker="x",
        linestyle="None",
        color="black",
        markersize=7,
        label="No perturbation",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="None",
        color="black",
        markersize=6,
        label="Perturbed median",
    ),
    Line2D(
        [0],
        [0],
        linestyle="-",
        color="black",
        linewidth=1.5,
        label="Perturbed p05-p95",
    ),
]


# =========================================================
# FIGURE 1: SPATIAL SIMILARITY VS TRUE PERIOD
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)

for degree in degrees:

    colour = cmap(
        norm(degree)
    )

    for mode_number, results in DMD_recovery.items():

        true_period = (
            results["true_period"]
        )

        # Perturbed ensemble interval + median.
        noised_result = (
            results[
                "DMD_noised"
            ][degree]
        )

        similarity_q = finite_quantiles(
            noised_result[
                "similarity"
            ]
        )

        if similarity_q is not None:

            p05, median, p95 = (
                similarity_q
            )

            ax.vlines(
                true_period,
                p05,
                p95,
                color=colour,
                linewidth=1.5,
                alpha=0.8,
                zorder=1,
            )

            ax.scatter(
                true_period,
                median,
                marker="s",
                s=35,
                color=colour,
                zorder=2,
            )

        # Clean / no-perturbation result on top.
        exact_result = (
            results[
                "DMD"
            ].get(
                degree
            )
        )

        if exact_result is not None:

            similarity = (
                exact_result[
                    "similarity"
                ]
            )

            if np.isfinite(
                similarity
            ):

                # White under-stroke helps distinguish a cross lying directly
                # over an ensemble median square.
                ax.scatter(
                    true_period,
                    similarity,
                    marker="x",
                    s=65,
                    linewidths=3.2,
                    color="white",
                    zorder=3,
                )

                ax.scatter(
                    true_period,
                    similarity,
                    marker="x",
                    s=65,
                    linewidths=1.7,
                    color=colour,
                    zorder=4,
                )


ax.set_xlabel(
    "True period (years)"
)

ax.set_ylabel(
    "Spatial similarity"
)

ax.set_title(
    f"Hankel DMD recovery "
    f"(d={hankel_d}, rank={hankel_svd_rank})"
)

ax.set_ylim(
    0,
    1.05,
)

ax.grid(
    alpha=0.25
)

ax.legend(
    handles=marker_handles,
    loc="best",
)

sm = cm.ScalarMappable(
    cmap=cmap,
    norm=norm,
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    boundaries=boundaries,
    ticks=degrees,
    spacing="uniform",
)

cbar.set_label(
    "Upper spherical harmonic degree of 4-degree band"
)

fig.tight_layout()
plt.show()


# =========================================================
# FIGURE 2: SIGNED PERIOD ERROR VS TRUE PERIOD
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)

for degree in degrees:

    colour = cmap(
        norm(degree)
    )

    for mode_number, results in DMD_recovery.items():

        true_period = (
            results["true_period"]
        )

        # Perturbed ensemble.
        recovered_periods = np.asarray(
            results[
                "DMD_noised"
            ][degree][
                "recovered_period"
            ],
            dtype=float,
        )

        period_errors = (
            100.0
            * (
                recovered_periods
                - true_period
            )
            / true_period
        )

        period_q = finite_quantiles(
            period_errors
        )

        if period_q is not None:

            p05, median, p95 = (
                period_q
            )

            ax.vlines(
                true_period,
                p05,
                p95,
                color=colour,
                linewidth=1.5,
                alpha=0.8,
                zorder=1,
            )

            ax.scatter(
                true_period,
                median,
                marker="s",
                s=35,
                color=colour,
                zorder=2,
            )

        # Clean / no-perturbation result.
        exact_result = (
            results[
                "DMD"
            ].get(
                degree
            )
        )

        if exact_result is not None:

            recovered_period = (
                exact_result[
                    "recovered_period"
                ]
            )

            if np.isfinite(
                recovered_period
            ):

                period_error = (
                    100.0
                    * (
                        recovered_period
                        - true_period
                    )
                    / true_period
                )

                ax.scatter(
                    true_period,
                    period_error,
                    marker="x",
                    s=65,
                    linewidths=3.2,
                    color="white",
                    zorder=3,
                )

                ax.scatter(
                    true_period,
                    period_error,
                    marker="x",
                    s=65,
                    linewidths=1.7,
                    color=colour,
                    zorder=4,
                )


ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
)

ax.set_xlabel(
    "True period (years)"
)

ax.set_ylabel(
    "Recovered period error (%)"
)

ax.set_title(
    f"Hankel DMD period recovery "
    f"(d={hankel_d}, rank={hankel_svd_rank})"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    handles=marker_handles,
    loc="best",
)

sm = cm.ScalarMappable(
    cmap=cmap,
    norm=norm,
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    boundaries=boundaries,
    ticks=degrees,
    spacing="uniform",
)

cbar.set_label(
    "Upper spherical harmonic degree of 4-degree band"
)

fig.tight_layout()
plt.show()


# %% ------------------------------------------------------
# NUMERICAL RECOVERY SUMMARY
# ------------------------------------------------------

print(
    "\nHankel DMD recovery summary"
)

for degree in degrees:

    tmin_current = max(
        1,
        degree - 3,
    )

    n_clean_matches = 0
    ensemble_counts = []

    for results in DMD_recovery.values():

        if degree in results["DMD"]:
            n_clean_matches += 1

        ensemble_counts.append(
            len(
                results[
                    "DMD_noised"
                ][degree][
                    "recovered_period"
                ]
            )
        )

    if ensemble_flag:
        ensemble_min = min(
            ensemble_counts
        )
        ensemble_max = max(
            ensemble_counts
        )

        print(
            f"Degree band {tmin_current}-{degree}: "
            f"{n_clean_matches}/{len(mode_numbers)} clean target matches; "
            f"successful perturbed matches per target = "
            f"{ensemble_min}-{ensemble_max}/{n_realisations}"
        )

    else:

        print(
            f"Degree band {tmin_current}-{degree}: "
            f"{n_clean_matches}/{len(mode_numbers)} clean target matches"
        )
