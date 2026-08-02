"""Helpers for true-amplitude DMD recovery videos.

The functions assume that every DMD adapter has already returned the common
candidate representation used in the thesis workflow:

    candidate_eigs    : continuous-time eigenvalues (1 / year)
    candidate_modes   : amplitude-scaled complex phasors in physical grid space
    candidate_periods : periods in years

No phase shifting, spatial rescaling, or amplitude matching is performed.
Recovered modes are evaluated directly as

    Re[phasor * exp(lambda * t)]

using the fitted amplitude/phase and continuous eigenvalue.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from tqdm import tqdm

try:
    import chaosmagpy as cp
except ImportError:  # only needed for MJD2000 labels
    cp = None


def evaluate_continuous_phasor(phasor, eigenvalue, times):
    """Evaluate Re[phasor * exp(lambda*t)] without any phase/amplitude fitting.

    Parameters
    ----------
    phasor : ndarray, shape (n_space,)
        Amplitude-scaled complex spatial phasor at t=0.
    eigenvalue : complex
        Continuous-time eigenvalue sigma + i*omega, in 1/year.
    times : ndarray, shape (n_times,)
        Times in years relative to the phasor's own t=0.

    Returns
    -------
    values : ndarray, shape (n_times, n_space)
        Real-valued field at each requested time.
    """
    phasor = np.asarray(phasor, dtype=complex).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    temporal = np.exp(complex(eigenvalue) * times)
    return np.real(temporal[:, None] * phasor[None, :])


def build_annual_frame_indices(
    model_time_years,
    dt_years,
    high_q_flag=False,
    good_record_slice=None,
    frame_spacing_years=1.0,
):
    """Choose source-grid samples separated by an exact requested year spacing.

    This deliberately samples the existing model time grid rather than
    interpolating the resolved signal, so resolved amplitudes/phases are not
    altered by an extra interpolation step.
    """
    model_time_years = np.asarray(model_time_years, dtype=float)
    all_indices = np.arange(model_time_years.size)

    if high_q_flag:
        if good_record_slice is None:
            raise ValueError(
                "good_record_slice must be supplied when high_q_flag=True."
            )
        active_indices = all_indices[good_record_slice]
    else:
        active_indices = all_indices

    if active_indices.size == 0:
        raise ValueError("Selected time window contains no samples.")

    step_float = frame_spacing_years / float(dt_years)
    step_samples = int(round(step_float))

    if step_samples < 1 or not np.isclose(
        step_samples * dt_years,
        frame_spacing_years,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError(
            "frame_spacing_years must be an integer multiple of dt_years "
            "when sampling without interpolation. "
            f"Received frame_spacing_years={frame_spacing_years}, "
            f"dt_years={dt_years}."
        )

    # Keep the first active sample as the video phase/time reference and move
    # forward by the requested number of source samples.
    frame_indices = active_indices[::step_samples]
    frame_years = model_time_years[frame_indices]

    return frame_indices, frame_years


def prepare_recovery_video_data(
    *,
    mode_numbers,
    DMD_recovery,
    synthetic_suite_info,
    target_phasors,
    clean_candidate_eigs,
    clean_candidate_modes,
    A_r_current,
    model_time_years,
    dt_years,
    Nmax,
    truncate_gauss_coeffs,
    recovered_sum_candidate_eigs=None,
    recovered_sum_candidate_modes=None,
    clean_candidate_ids=None,
    recovered_sum_candidate_ids=None,
    high_q_flag=False,
    good_record_slice=None,
    truth_t0_year=None,
    truth_include_growth=False,
    frame_spacing_years=1.0,
    compact_input_threshold=8,
):
    """Precompute exact, resolved and recovered fields for the clean DMD video.

    Required additions to the main pipeline
    ---------------------------------------
    1. ``synthetic_suite_info[mode]["gnm_resolved"]`` must contain the
       individually resolved Gauss-coefficient time series for that input mode
       (before all modes are summed).
    2. The clean/no-perturbation DMD candidate arrays must be cached as
       ``clean_candidate_eigs`` and ``clean_candidate_modes``.
    3. Each clean match in ``DMD_recovery[mode]["DMD"]`` must contain
       ``candidate_index`` identifying which column of ``clean_candidate_modes``
       was selected by spatial matching.
    4. ``recovered_sum_candidate_eigs`` and
       ``recovered_sum_candidate_modes`` may supply a separate candidate suite
       for the overall recovered total. This allows static candidates to remain
       in that total while the per-mode matching suite obeys a finite-period
       filter. If omitted, the clean matching suite is used for both purposes.
    5. ``clean_candidate_ids`` and ``recovered_sum_candidate_ids`` identify
       candidates before any period filtering. They are required to reconcile
       the two suites in compact videos when the suites differ.

    Compact layout
    --------------
    If the number of active input modes exceeds ``compact_input_threshold``,
    this function stores lightweight row recipes rather than dense
    mode-by-frame-by-space fields. Inputs selecting the same recovered
    candidate are grouped into one row, followed by retained candidates that
    no input selected. The renderer evaluates these recipes one frame at a
    time.

    Truth convention
    ----------------
    The exact input is evaluated directly from the original theoretical complex
    phasor. If the resolved HDF5 dataset was generated from a "without_decay"
    synthetic signal, leave ``truth_include_growth=False`` so the true temporal
    eigenvalue used for the exact input is i*Im(lambda). Set ``truth_t0_year``
    to the exact phase origin that was used when those synthetic splines were
    generated.
    """
    mode_numbers = [str(m) for m in mode_numbers]
    model_time_years = np.asarray(model_time_years, dtype=float)
    clean_candidate_eigs = np.asarray(clean_candidate_eigs, dtype=complex)
    clean_candidate_modes = np.asarray(clean_candidate_modes, dtype=complex)

    if clean_candidate_modes.ndim != 2:
        raise ValueError("clean_candidate_modes must be 2-D (n_space, n_modes).")
    if clean_candidate_modes.shape[1] != clean_candidate_eigs.size:
        raise ValueError(
            "clean_candidate_modes columns must equal clean_candidate_eigs size."
        )

    if clean_candidate_ids is None:
        clean_candidate_ids = np.arange(
            clean_candidate_eigs.size,
            dtype=int,
        )
    else:
        clean_candidate_ids = np.asarray(
            clean_candidate_ids,
            dtype=int,
        ).ravel()

    if clean_candidate_ids.size != clean_candidate_eigs.size:
        raise ValueError(
            "clean_candidate_ids must contain one ID per clean candidate."
        )
    if np.unique(clean_candidate_ids).size != clean_candidate_ids.size:
        raise ValueError("clean_candidate_ids must be unique.")

    if (
        recovered_sum_candidate_eigs is None
        and recovered_sum_candidate_modes is None
    ):
        recovered_sum_candidate_eigs = (
            clean_candidate_eigs
        )
        recovered_sum_candidate_modes = (
            clean_candidate_modes
        )
        recovered_sum_candidate_ids = (
            clean_candidate_ids
        )
    elif (
        recovered_sum_candidate_eigs is None
        or recovered_sum_candidate_modes is None
    ):
        raise ValueError(
            "recovered_sum_candidate_eigs and "
            "recovered_sum_candidate_modes must either both be "
            "supplied or both be omitted."
        )

    recovered_sum_candidate_eigs = np.asarray(
        recovered_sum_candidate_eigs,
        dtype=complex,
    )
    recovered_sum_candidate_modes = np.asarray(
        recovered_sum_candidate_modes,
        dtype=complex,
    )

    if recovered_sum_candidate_ids is None:
        if recovered_sum_candidate_eigs.size != clean_candidate_eigs.size:
            raise ValueError(
                "recovered_sum_candidate_ids are required when the clean "
                "matching and recovered-sum candidate suites differ."
            )
        recovered_sum_candidate_ids = clean_candidate_ids.copy()
    else:
        recovered_sum_candidate_ids = np.asarray(
            recovered_sum_candidate_ids,
            dtype=int,
        ).ravel()

    if recovered_sum_candidate_modes.ndim != 2:
        raise ValueError(
            "recovered_sum_candidate_modes must be 2-D "
            "(n_space, n_modes)."
        )
    if (
        recovered_sum_candidate_modes.shape[1]
        != recovered_sum_candidate_eigs.size
    ):
        raise ValueError(
            "recovered_sum_candidate_modes columns must equal "
            "recovered_sum_candidate_eigs size."
        )
    if (
        recovered_sum_candidate_ids.size
        != recovered_sum_candidate_eigs.size
    ):
        raise ValueError(
            "recovered_sum_candidate_ids must contain one ID per "
            "recovered-sum candidate."
        )
    if (
        np.unique(recovered_sum_candidate_ids).size
        != recovered_sum_candidate_ids.size
    ):
        raise ValueError("recovered_sum_candidate_ids must be unique.")
    if (
        recovered_sum_candidate_modes.shape[0]
        != clean_candidate_modes.shape[0]
    ):
        raise ValueError(
            "The clean matching and recovered-sum candidate suites "
            "must have the same physical-space dimension."
        )

    frame_indices, frame_years = build_annual_frame_indices(
        model_time_years=model_time_years,
        dt_years=dt_years,
        high_q_flag=high_q_flag,
        good_record_slice=good_record_slice,
        frame_spacing_years=frame_spacing_years,
    )

    if high_q_flag:
        active_indices = np.arange(model_time_years.size)[good_record_slice]
    else:
        active_indices = np.arange(model_time_years.size)

    fit_t0_year = float(model_time_years[active_indices[0]])
    dmd_times = frame_years - fit_t0_year

    if truth_t0_year is None:
        truth_t0_year = float(model_time_years[0])
    truth_times = frame_years - float(truth_t0_year)

    n_modes = len(mode_numbers)
    n_frames = frame_years.size
    n_physical = clean_candidate_modes.shape[0]

    if (
        isinstance(
            compact_input_threshold,
            (bool, np.bool_),
        )
        or not isinstance(
            compact_input_threshold,
            (int, np.integer),
        )
        or compact_input_threshold < 1
    ):
        raise ValueError(
            "compact_input_threshold must be a positive integer."
        )

    if n_modes > int(compact_input_threshold):
        return _prepare_compact_recovery_video_data(
            mode_numbers=mode_numbers,
            DMD_recovery=DMD_recovery,
            synthetic_suite_info=synthetic_suite_info,
            target_phasors=target_phasors,
            clean_candidate_eigs=clean_candidate_eigs,
            clean_candidate_ids=clean_candidate_ids,
            recovered_sum_candidate_eigs=
                recovered_sum_candidate_eigs,
            recovered_sum_candidate_modes=
                recovered_sum_candidate_modes,
            recovered_sum_candidate_ids=
                recovered_sum_candidate_ids,
            A_r_current=A_r_current,
            frame_indices=frame_indices,
            frame_years=frame_years,
            frame_mjd2000=None,
            model_time_years=model_time_years,
            fit_t0_year=fit_t0_year,
            truth_t0_year=float(truth_t0_year),
            dmd_times=dmd_times,
            truth_times=truth_times,
            Nmax=Nmax,
            truncate_gauss_coeffs=truncate_gauss_coeffs,
            truth_include_growth=truth_include_growth,
            compact_input_threshold=int(
                compact_input_threshold
            ),
        )

    exact = np.empty((n_modes, n_frames, n_physical), dtype=np.float64)
    resolved = np.empty_like(exact)
    recovered = np.zeros_like(exact)
    matched_candidate_indices = {}

    for mode_i, mode_number in enumerate(mode_numbers):
        if mode_number not in synthetic_suite_info:
            raise KeyError(f"Missing synthetic_suite_info for mode {mode_number}.")
        if mode_number not in target_phasors:
            raise KeyError(f"Missing target_phasors entry for mode {mode_number}.")

        info = synthetic_suite_info[mode_number]
        true_eig = complex(info["true_eigenvalue"])

        if truth_include_growth:
            truth_eig = true_eig
        else:
            truth_eig = 1j * true_eig.imag

        # target_phasors are already in physical grid space and preserve the
        # theoretical complex amplitude/phase. No spatial scaling is applied.
        exact[mode_i] = evaluate_continuous_phasor(
            target_phasors[mode_number],
            truth_eig,
            truth_times,
        )

        if "gnm_resolved" not in info:
            raise KeyError(
                f"Mode {mode_number} is missing synthetic_suite_info[mode]"
                "['gnm_resolved']. Store the individually resolved time series "
                "when building the synthetic suite."
            )

        gnm_resolved = np.asarray(info["gnm_resolved"])
        gnm_resolved_nmax = truncate_gauss_coeffs(
            gnm_resolved,
            tmax=Nmax,
        )

        if gnm_resolved_nmax.shape[0] != model_time_years.size:
            raise ValueError(
                f"Mode {mode_number}: resolved time-series length "
                f"{gnm_resolved_nmax.shape[0]} does not match model_time_years "
                f"length {model_time_years.size}."
            )

        resolved[mode_i] = (
            A_r_current @ gnm_resolved_nmax[frame_indices].T
        ).T

        clean_match = DMD_recovery[mode_number].get("DMD", {})
        candidate_index = clean_match.get("candidate_index", None)

        if candidate_index is None:
            matched_candidate_indices[mode_number] = None
            # Leave the recovered contribution as zero: this makes a missing
            # recovery visually explicit rather than silently matching/scaling.
            continue

        candidate_index = int(candidate_index)
        if not (0 <= candidate_index < clean_candidate_eigs.size):
            raise IndexError(
                f"Mode {mode_number}: candidate_index={candidate_index} is out "
                f"of range for {clean_candidate_eigs.size} clean candidates."
            )

        matched_candidate_indices[mode_number] = candidate_index
        recovered[mode_i] = evaluate_continuous_phasor(
            clean_candidate_modes[:, candidate_index],
            clean_candidate_eigs[candidate_index],
            dmd_times,
        )

    # Overall input sums are sums of the individually known synthetic modes.
    exact_sum = np.sum(exact, axis=0)
    resolved_sum = np.sum(resolved, axis=0)

    # Overall recovered motion is the sum of every candidate in the dedicated
    # video-total suite exactly once. This avoids double-counting when two
    # theoretical waves both select the same recovered candidate during
    # matching, while allowing intentionally retained static candidates.
    if recovered_sum_candidate_eigs.size:
        temporal = np.exp(
            np.outer(
                recovered_sum_candidate_eigs,
                dmd_times,
            )
        )
        recovered_sum = np.real(
            (
                recovered_sum_candidate_modes
                @ temporal
            ).T
        )
    else:
        recovered_sum = np.zeros_like(exact_sum)

    # CHAOS uses modified Julian date 2000 (0 = 2000-01-01). Use the official
    # ChaosMagPy converter when available.
    if cp is not None:
        frame_mjd2000 = np.asarray(
            cp.data_utils.dyear_to_mjd(frame_years),
            dtype=float,
        )
    else:
        frame_mjd2000 = np.full(frame_years.shape, np.nan)

    return {
        "layout": "full",
        "mode_numbers": mode_numbers,
        "frame_indices": frame_indices,
        "frame_years": frame_years,
        "frame_mjd2000": frame_mjd2000,
        "fit_t0_year": fit_t0_year,
        "truth_t0_year": float(truth_t0_year),
        "exact": exact,
        "resolved": resolved,
        "recovered": recovered,
        "exact_sum": exact_sum,
        "resolved_sum": resolved_sum,
        "recovered_sum": recovered_sum,
        "matched_candidate_indices": matched_candidate_indices,
    }


def _candidate_period_from_eigenvalue(eigenvalue):
    """Return an oscillatory period in years, or infinity for a static mode."""

    omega = abs(complex(eigenvalue).imag)
    if omega == 0.0:
        return np.inf
    return float(2.0 * np.pi / omega)


def _prepare_compact_recovery_video_data(
    *,
    mode_numbers,
    DMD_recovery,
    synthetic_suite_info,
    target_phasors,
    clean_candidate_eigs,
    clean_candidate_ids,
    recovered_sum_candidate_eigs,
    recovered_sum_candidate_modes,
    recovered_sum_candidate_ids,
    A_r_current,
    frame_indices,
    frame_years,
    frame_mjd2000,
    model_time_years,
    fit_t0_year,
    truth_t0_year,
    dmd_times,
    truth_times,
    Nmax,
    truncate_gauss_coeffs,
    truth_include_growth,
    compact_input_threshold,
):
    """Build compact row recipes without dense physical-space frame arrays."""

    candidate_groups = {}
    unmatched_input_modes = []
    overall_truth_components = []
    overall_resolved_coefficients = None

    for mode_number in mode_numbers:
        if mode_number not in synthetic_suite_info:
            raise KeyError(
                f"Missing synthetic_suite_info for mode {mode_number}."
            )
        if mode_number not in target_phasors:
            raise KeyError(
                f"Missing target_phasors entry for mode {mode_number}."
            )

        info = synthetic_suite_info[mode_number]
        true_eig = complex(info["true_eigenvalue"])
        truth_eig = (
            true_eig
            if truth_include_growth
            else 1j * true_eig.imag
        )
        target_phasor = np.asarray(
            target_phasors[mode_number],
            dtype=complex,
        ).reshape(-1)

        if target_phasor.size != recovered_sum_candidate_modes.shape[0]:
            raise ValueError(
                f"Mode {mode_number}: target phasor size "
                f"{target_phasor.size} does not match the recovered "
                f"physical-space size "
                f"{recovered_sum_candidate_modes.shape[0]}."
            )

        truth_component = {
            "mode_number": mode_number,
            "phasor": target_phasor,
            "eigenvalue": truth_eig,
        }
        overall_truth_components.append(
            truth_component
        )

        if "gnm_resolved" not in info:
            raise KeyError(
                f"Mode {mode_number} is missing synthetic_suite_info[mode]"
                "['gnm_resolved']. Store the individually resolved time "
                "series when building the synthetic suite."
            )

        gnm_resolved = np.asarray(
            info["gnm_resolved"]
        )
        gnm_resolved_nmax = np.asarray(
            truncate_gauss_coeffs(
                gnm_resolved,
                tmax=Nmax,
            )
        )

        if gnm_resolved_nmax.shape[0] != model_time_years.size:
            raise ValueError(
                f"Mode {mode_number}: resolved time-series length "
                f"{gnm_resolved_nmax.shape[0]} does not match "
                f"model_time_years length {model_time_years.size}."
            )

        resolved_coefficients = (
            gnm_resolved_nmax[
                frame_indices
            ].copy()
        )

        if overall_resolved_coefficients is None:
            overall_resolved_coefficients = np.zeros_like(
                resolved_coefficients
            )
        elif (
            resolved_coefficients.shape
            != overall_resolved_coefficients.shape
        ):
            raise ValueError(
                "Individually resolved input modes do not share one "
                "Gauss-coefficient shape."
            )

        overall_resolved_coefficients += (
            resolved_coefficients
        )

        clean_match = DMD_recovery[
            mode_number
        ].get("DMD", {})
        candidate_index = clean_match.get(
            "candidate_index"
        )

        if candidate_index is None:
            unmatched_input_modes.append(
                mode_number
            )
            continue

        candidate_index = int(
            candidate_index
        )
        if not (
            0
            <= candidate_index
            < clean_candidate_ids.size
        ):
            raise IndexError(
                f"Mode {mode_number}: candidate_index="
                f"{candidate_index} is outside the clean candidate "
                f"suite of size {clean_candidate_ids.size}."
            )

        candidate_id = int(
            clean_candidate_ids[
                candidate_index
            ]
        )

        if candidate_id not in candidate_groups:
            candidate_groups[
                candidate_id
            ] = {
                "candidate_id": candidate_id,
                "clean_candidate_index":
                    candidate_index,
                "input_modes": [],
                "similarities": [],
                "truth_components": [],
                "resolved_coefficients":
                    np.zeros_like(
                        resolved_coefficients
                    ),
            }

        group = candidate_groups[
            candidate_id
        ]
        group["input_modes"].append(
            mode_number
        )
        group["similarities"].append(
            float(
                clean_match.get(
                    "similarity",
                    np.nan,
                )
            )
        )
        group["truth_components"].append(
            truth_component
        )
        group["resolved_coefficients"] += (
            resolved_coefficients
        )

    recovered_sum_index_by_id = {
        int(candidate_id): candidate_index
        for candidate_index, candidate_id in enumerate(
            recovered_sum_candidate_ids
        )
    }

    compact_rows = []

    for candidate_id, group in candidate_groups.items():
        if candidate_id not in recovered_sum_index_by_id:
            raise ValueError(
                "A clean matched candidate is missing from the "
                "recovered-sum video suite: "
                f"candidate ID {candidate_id}."
            )

        recovered_index = recovered_sum_index_by_id[
            candidate_id
        ]
        eigenvalue = recovered_sum_candidate_eigs[
            recovered_index
        ]

        compact_rows.append({
            "kind": "matched",
            **group,
            "recovered_sum_candidate_index":
                int(recovered_index),
            "eigenvalue": eigenvalue,
            "period": (
                float(
                    DMD_recovery[
                        group["input_modes"][0]
                    ]["DMD"].get(
                        "recovered_period",
                        _candidate_period_from_eigenvalue(
                            eigenvalue
                        ),
                    )
                )
            ),
            "phasor": recovered_sum_candidate_modes[
                :,
                recovered_index,
            ],
        })

    matched_candidate_ids = set(
        candidate_groups
    )
    unmatched_dmd_candidate_ids = []

    for recovered_index, candidate_id in enumerate(
        recovered_sum_candidate_ids
    ):
        candidate_id = int(candidate_id)
        if candidate_id in matched_candidate_ids:
            continue

        unmatched_dmd_candidate_ids.append(
            candidate_id
        )
        eigenvalue = recovered_sum_candidate_eigs[
            recovered_index
        ]
        compact_rows.append({
            "kind": "unmatched_dmd",
            "candidate_id": candidate_id,
            "recovered_sum_candidate_index":
                int(recovered_index),
            "input_modes": [],
            "similarities": [],
            "truth_components": [],
            "resolved_coefficients": None,
            "eigenvalue": eigenvalue,
            "period":
                _candidate_period_from_eigenvalue(
                    eigenvalue
                ),
            "phasor": recovered_sum_candidate_modes[
                :,
                recovered_index,
            ],
        })

    if frame_mjd2000 is None:
        if cp is not None:
            frame_mjd2000 = np.asarray(
                cp.data_utils.dyear_to_mjd(
                    frame_years
                ),
                dtype=float,
            )
        else:
            frame_mjd2000 = np.full(
                frame_years.shape,
                np.nan,
            )

    matched_input_count = (
        len(mode_numbers)
        - len(unmatched_input_modes)
    )
    unique_matched_candidate_count = len(
        candidate_groups
    )
    shared_input_count = max(
        0,
        matched_input_count
        - unique_matched_candidate_count,
    )

    return {
        "layout": "compact",
        "compact_input_threshold":
            compact_input_threshold,
        "mode_numbers": mode_numbers,
        "frame_indices": frame_indices,
        "frame_years": frame_years,
        "frame_mjd2000": frame_mjd2000,
        "fit_t0_year": fit_t0_year,
        "truth_t0_year": truth_t0_year,
        "truth_times": truth_times,
        "dmd_times": dmd_times,
        "A_r_current": np.asarray(
            A_r_current
        ),
        "compact_rows": compact_rows,
        "overall_truth_components":
            overall_truth_components,
        "overall_resolved_coefficients":
            overall_resolved_coefficients,
        "recovered_sum_candidate_eigs":
            recovered_sum_candidate_eigs,
        "recovered_sum_candidate_modes":
            recovered_sum_candidate_modes,
        "recovered_sum_candidate_ids":
            recovered_sum_candidate_ids,
        "summary": {
            "input_count": len(mode_numbers),
            "matched_input_count":
                matched_input_count,
            "unmatched_input_count":
                len(unmatched_input_modes),
            "unmatched_input_modes":
                unmatched_input_modes,
            "unique_matched_dmd_count":
                unique_matched_candidate_count,
            "shared_input_count":
                shared_input_count,
            "unmatched_dmd_count":
                len(
                    unmatched_dmd_candidate_ids
                ),
            "unmatched_dmd_candidate_ids":
                unmatched_dmd_candidate_ids,
        },
    }


def _evaluate_truth_components(
    truth_components,
    truth_time,
    n_physical,
):
    """Evaluate and sum exact input components at one video time."""

    field = np.zeros(
        n_physical,
        dtype=float,
    )
    for component in truth_components:
        field += np.real(
            component["phasor"]
            * np.exp(
                component["eigenvalue"]
                * truth_time
            )
        )
    return field


def _compact_recovery_frame_fields(
    video_data,
    frame_i,
):
    """Evaluate all compact rows at one frame without retaining a frame cube."""

    n_physical = video_data[
        "recovered_sum_candidate_modes"
    ].shape[0]
    truth_time = video_data[
        "truth_times"
    ][frame_i]
    dmd_time = video_data[
        "dmd_times"
    ][frame_i]
    projection = video_data[
        "A_r_current"
    ]
    zero_field = np.zeros(
        n_physical,
        dtype=float,
    )
    rows = []

    for row in video_data[
        "compact_rows"
    ]:
        recovered = np.real(
            row["phasor"]
            * np.exp(
                row["eigenvalue"]
                * dmd_time
            )
        )

        if row["kind"] == "matched":
            exact = _evaluate_truth_components(
                row["truth_components"],
                truth_time,
                n_physical,
            )
            resolved = (
                projection
                @ row[
                    "resolved_coefficients"
                ][frame_i]
            )
            difference = recovered - exact
        else:
            exact = zero_field
            resolved = zero_field
            difference = zero_field

        rows.append(
            (
                exact,
                resolved,
                recovered,
                difference,
            )
        )

    exact_sum = _evaluate_truth_components(
        video_data[
            "overall_truth_components"
        ],
        truth_time,
        n_physical,
    )
    resolved_sum = (
        projection
        @ video_data[
            "overall_resolved_coefficients"
        ][frame_i]
    )
    temporal = np.exp(
        video_data[
            "recovered_sum_candidate_eigs"
        ]
        * dmd_time
    )
    recovered_sum = np.real(
        video_data[
            "recovered_sum_candidate_modes"
        ]
        @ temporal
    )
    rows.append(
        (
            exact_sum,
            resolved_sum,
            recovered_sum,
            recovered_sum - exact_sum,
        )
    )

    return rows


def compute_video_row_limits(video_data, minimum_vmax=1e-12):
    """Compute one fixed symmetric colour scale per row over ALL video frames.

    Per-mode difference is recovered - exact.
    Overall difference is recovered_sum - exact_sum.

    No percentile clipping or frame-by-frame normalization is used, preserving
    true relative amplitudes through time and across the four panels in a row.
    """
    if video_data.get("layout", "full") == "compact":
        n_rows = (
            len(
                video_data[
                    "compact_rows"
                ]
            )
            + 1
        )
        row_vmax = np.full(
            n_rows,
            float(minimum_vmax),
            dtype=float,
        )

        for frame_i in tqdm(
            range(
                len(
                    video_data[
                        "frame_years"
                    ]
                )
            ),
            desc="Scanning recovery video limits",
        ):
            rows = _compact_recovery_frame_fields(
                video_data,
                frame_i,
            )
            for row_i, fields in enumerate(
                rows
            ):
                for field in fields:
                    finite = np.isfinite(
                        field
                    )
                    if np.any(finite):
                        row_vmax[row_i] = max(
                            row_vmax[row_i],
                            float(
                                np.max(
                                    np.abs(
                                        field[
                                            finite
                                        ]
                                    )
                                )
                            ),
                        )

        return row_vmax

    exact = np.asarray(video_data["exact"])
    resolved = np.asarray(video_data["resolved"])
    recovered = np.asarray(video_data["recovered"])

    n_modes = exact.shape[0]
    row_vmax = np.empty(n_modes + 1, dtype=float)

    for i in range(n_modes):
        difference = recovered[i] - exact[i]
        row_vmax[i] = max(
            np.nanmax(np.abs(exact[i])),
            np.nanmax(np.abs(resolved[i])),
            np.nanmax(np.abs(recovered[i])),
            np.nanmax(np.abs(difference)),
            minimum_vmax,
        )

    overall_difference = (
        video_data["recovered_sum"] - video_data["exact_sum"]
    )
    row_vmax[-1] = max(
        np.nanmax(np.abs(video_data["exact_sum"])),
        np.nanmax(np.abs(video_data["resolved_sum"])),
        np.nanmax(np.abs(video_data["recovered_sum"])),
        np.nanmax(np.abs(overall_difference)),
        minimum_vmax,
    )

    return row_vmax


def _reshape_global_field(vector, nlat, nlon, roll_longitude=True):
    field = np.asarray(vector).reshape(nlat, nlon)
    if roll_longitude:
        field = np.roll(field, nlon // 2, axis=1)
    return field


def make_dmd_recovery_video(
    *,
    video_data,
    output_path,
    DMD_algorithm,
    Nmax,
    svd_rank,
    nlat,
    nlon,
    hankel_embedding_d=None,
    high_q_flag=False,
    n_skip=None,
    fps=1,
    cmap="seismic",
    roll_longitude=True,
    dpi=100,
    figsize=None,
    bitrate=5000,
):
    """Render the requested multi-row true-amplitude DMD recovery MP4.

    Layout
    ------
    Full layout (up to the configured compact-input threshold):
        one row per theoretical wave mode:
        exact input | resolved input | matched DMD output | DMD - exact

    Compact layout:
        one row per unique matched DMD candidate, using the summed exact and
        resolved fields of every input that selected that candidate, followed
        by every retained DMD candidate that no input selected.

    Final row:
        exact sum | resolved sum | all retained recovered modes |
        recovered - exact

    Each row has one fixed symmetric colour scale shared by its four panels and
    fixed for the complete animation. A separate colourbar is placed at the
    right of every row and labelled nT/yr.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compact_layout = (
        video_data.get(
            "layout",
            "full",
        )
        == "compact"
    )
    mode_numbers = video_data["mode_numbers"]
    if compact_layout:
        n_mode_rows = len(
            video_data[
                "compact_rows"
            ]
        )
    else:
        n_mode_rows = len(mode_numbers)
    n_rows = n_mode_rows + 1
    n_frames = len(video_data["frame_years"])

    if figsize is None:
        if compact_layout:
            figsize = (
                18.0,
                1.75 * n_rows + 2.0,
            )
        else:
            figsize = (
                18.0,
                2.05 * n_rows + 1.6,
            )

    row_vmax = compute_video_row_limits(video_data)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        n_rows,
        5,
        width_ratios=[1, 1, 1, 1, 0.045],
        hspace=0.34,
        wspace=0.08,
    )
    if compact_layout:
        fig.subplots_adjust(
            bottom=0.045,
            top=0.955,
        )

    axes = np.empty((n_rows, 4), dtype=object)
    images = np.empty((n_rows, 4), dtype=object)
    cbars = []

    extent = (-180.0, 180.0, -90.0, 90.0)

    def frame_fields(frame_i):
        if compact_layout:
            return _compact_recovery_frame_fields(
                video_data,
                frame_i,
            )

        rows = []
        for mode_i in range(n_mode_rows):
            exact = video_data["exact"][mode_i, frame_i]
            resolved = video_data["resolved"][mode_i, frame_i]
            recovered = video_data["recovered"][mode_i, frame_i]
            difference = recovered - exact
            rows.append((exact, resolved, recovered, difference))

        exact_sum = video_data["exact_sum"][frame_i]
        resolved_sum = video_data["resolved_sum"][frame_i]
        recovered_sum = video_data["recovered_sum"][frame_i]
        overall_difference = recovered_sum - exact_sum
        rows.append(
            (exact_sum, resolved_sum, recovered_sum, overall_difference)
        )
        return rows

    first_rows = frame_fields(0)

    def format_input_modes(input_modes):
        chunks = []
        for start in range(
            0,
            len(input_modes),
            4,
        ):
            chunks.append(
                ", ".join(
                    str(mode)
                    for mode in input_modes[
                        start:start + 4
                    ]
                )
            )
        return "\n".join(chunks)

    for row_i in range(n_rows):
        vmax = row_vmax[row_i]
        norm = Normalize(vmin=-vmax, vmax=vmax)
        unavailable_columns = ()

        if row_i < n_mode_rows:
            if compact_layout:
                compact_row = video_data[
                    "compact_rows"
                ][row_i]
                candidate_id = compact_row[
                    "candidate_id"
                ]
                period = compact_row[
                    "period"
                ]
                period_text = (
                    "static"
                    if np.isinf(period)
                    else f"{period:.2f} yr"
                )

                if (
                    compact_row["kind"]
                    == "matched"
                ):
                    input_modes = compact_row[
                        "input_modes"
                    ]
                    input_label = (
                        format_input_modes(
                            input_modes
                        )
                    )
                    plural = (
                        "s"
                        if len(input_modes) != 1
                        else ""
                    )
                    titles = (
                        f"Exact counterpart{plural}",
                        f"Resolved counterpart{plural}",
                        "Matched DMD",
                        "DMD - exact counterpart(s)",
                    )
                    row_label = (
                        f"DMD candidate {candidate_id}\n"
                        f"period={period_text}\n"
                        f"Best for input{plural}:\n"
                        f"{input_label}"
                    )
                else:
                    titles = (
                        "No matched input",
                        "No matched input",
                        "Unmatched recovered DMD",
                        "No matched-input difference",
                    )
                    row_label = (
                        f"DMD candidate {candidate_id}\n"
                        f"period={period_text}\n"
                        "No input selected this candidate"
                    )
                    unavailable_columns = (
                        0,
                        1,
                        3,
                    )
            else:
                mode_number = mode_numbers[
                    row_i
                ]
                titles = (
                    "Exact input",
                    "Resolved input",
                    "Matched DMD",
                    "DMD - exact",
                )
                row_label = (
                    f"Mode {mode_number}"
                )
        else:
            titles = (
                "Exact sum",
                "Resolved sum",
                "Recovered sum",
                "Recovered - exact",
            )
            row_label = "Overall"

        for col_i in range(4):
            ax = fig.add_subplot(gs[row_i, col_i])
            axes[row_i, col_i] = ax

            field = _reshape_global_field(
                first_rows[row_i][col_i],
                nlat=nlat,
                nlon=nlon,
                roll_longitude=roll_longitude,
            )

            image = ax.imshow(
                field,
                origin="upper",
                extent=extent,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="auto",
            )
            images[row_i, col_i] = image

            ax.set_title(titles[col_i], fontsize=8)
            if col_i in unavailable_columns:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.35",
                )
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-90, -45, 0, 45, 90])
            ax.tick_params(labelsize=6, length=2)

            if col_i == 0:
                ax.set_ylabel(
                    f"{row_label}\nLatitude (deg)",
                    fontsize=7,
                )
            else:
                ax.set_yticklabels([])

            if row_i == n_rows - 1:
                ax.set_xlabel("Longitude (deg)", fontsize=7)
            else:
                ax.set_xticklabels([])

        cax = fig.add_subplot(gs[row_i, 4])
        cbar = fig.colorbar(images[row_i, 0], cax=cax)
        cbar.set_label("nT/yr", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        cbars.append(cbar)

    if compact_layout:
        summary = video_data[
            "summary"
        ]
        unmatched_input_text = (
            str(
                summary[
                    "unmatched_input_count"
                ]
            )
        )
        if summary[
            "unmatched_input_modes"
        ]:
            unmatched_input_text += (
                " ["
                + ", ".join(
                    str(mode)
                    for mode in summary[
                        "unmatched_input_modes"
                    ]
                )
                + "]"
            )

        footer_text = (
            f"Input modes: {summary['input_count']} | "
            "inputs with no finite DMD match: "
            f"{unmatched_input_text} | "
            "unique matched DMD candidates: "
            f"{summary['unique_matched_dmd_count']} | "
            "additional inputs sharing a candidate: "
            f"{summary['shared_input_count']} | "
            "unmatched retained DMD candidates: "
            f"{summary['unmatched_dmd_count']}"
        )
        fig.text(
            0.5,
            0.012,
            footer_text,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    algorithm_label = str(DMD_algorithm).upper()
    parameter_bits = [
        f"{algorithm_label}",
        f"Nmax={Nmax}",
        f"rank={svd_rank}",
    ]
    if hankel_embedding_d is not None:
        parameter_bits.append(
            f"Hankel embedding d={hankel_embedding_d}"
        )
    else:
        parameter_bits.append(
            "no Hankel embedding"
        )
    if n_skip is not None:
        parameter_bits.append(f"n_skip={n_skip}")
    parameter_bits.append(
        "high-quality window" if high_q_flag else "full time window"
    )
    parameter_summary = " | ".join(parameter_bits)

    title_artist = fig.suptitle("", fontsize=11, y=0.995)

    def update_frame(frame_i):
        rows = frame_fields(frame_i)
        for row_i in range(n_rows):
            for col_i in range(4):
                images[row_i, col_i].set_data(
                    _reshape_global_field(
                        rows[row_i][col_i],
                        nlat=nlat,
                        nlon=nlon,
                        roll_longitude=roll_longitude,
                    )
                )

        dyear = video_data["frame_years"][frame_i]
        mjd2000 = video_data["frame_mjd2000"][frame_i]

        if np.isfinite(mjd2000):
            time_text = (
                f"decimal year={dyear:.2f} | MJD2000={mjd2000:.2f}"
            )
        else:
            time_text = f"decimal year={dyear:.2f}"

        title_artist.set_text(
            f"Synthetic wave recovery: {parameter_summary}\n{time_text}"
        )

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=bitrate,
        metadata={"title": "DMD synthetic wave recovery"},
    )

    with writer.saving(fig, str(output_path), dpi=dpi):
        for frame_i in tqdm(range(n_frames), desc="Rendering recovery video"):
            update_frame(frame_i)
            writer.grab_frame()

    plt.close(fig)
    return output_path


def prepare_chaos_reconstruction_video_data(
    *,
    baseline_sv_input,
    analysis_times_absolute,
    candidate_eigenvalues,
    candidate_modes,
    frame_spacing_years,
    reconstruction_sv=None,
):
    """Prepare processed CHAOS input and unique-candidate DMD reconstruction."""

    baseline_sv_input = np.asarray(
        baseline_sv_input,
        dtype=float,
    )
    analysis_times_absolute = np.asarray(
        analysis_times_absolute,
        dtype=float,
    ).ravel()
    candidate_eigenvalues = np.asarray(
        candidate_eigenvalues,
        dtype=complex,
    ).ravel()
    candidate_modes = np.asarray(
        candidate_modes,
        dtype=complex,
    )

    if baseline_sv_input.ndim != 2:
        raise ValueError(
            "baseline_sv_input must have shape (n_space, n_times)."
        )
    if (
        baseline_sv_input.shape[1]
        != analysis_times_absolute.size
    ):
        raise ValueError(
            "CHAOS input snapshots and analysis times must align."
        )
    if (
        candidate_modes.ndim != 2
        or candidate_modes.shape[0]
        != baseline_sv_input.shape[0]
        or candidate_modes.shape[1]
        != candidate_eigenvalues.size
    ):
        raise ValueError(
            "CHAOS candidate modes/eigenvalues must align with the "
            "physical input dimension."
        )
    if analysis_times_absolute.size < 2:
        raise ValueError(
            "At least two CHAOS snapshots are required for a video."
        )
    if not (
        np.all(
            np.isfinite(
                baseline_sv_input
            )
        )
        and np.all(
            np.isfinite(
                analysis_times_absolute
            )
        )
        and np.all(
            np.isfinite(
                candidate_eigenvalues.real
            )
            & np.isfinite(
                candidate_eigenvalues.imag
            )
        )
        and np.all(
            np.isfinite(
                candidate_modes.real
            )
            & np.isfinite(
                candidate_modes.imag
            )
        )
    ):
        raise ValueError(
            "CHAOS video inputs must contain only finite values."
        )

    time_differences = np.diff(
        analysis_times_absolute
    )
    dt_snapshot = float(
        np.median(
            time_differences
        )
    )

    if not np.allclose(
        time_differences,
        dt_snapshot,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            "CHAOS video input times must be uniformly sampled."
        )

    frame_step_float = (
        float(frame_spacing_years)
        / dt_snapshot
    )
    frame_step = int(
        round(
            frame_step_float
        )
    )

    if (
        frame_step < 1
        or not np.isclose(
            frame_step
            * dt_snapshot,
            frame_spacing_years,
            rtol=1e-10,
            atol=1e-10,
        )
    ):
        raise ValueError(
            "frame_spacing_years must be an integer multiple of the "
            "selected CHAOS snapshot spacing. "
            f"Received frame_spacing_years={frame_spacing_years}, "
            f"dt_snapshot={dt_snapshot}."
        )

    frame_indices = np.arange(
        analysis_times_absolute.size
    )[
        ::frame_step
    ]
    frame_years = (
        analysis_times_absolute[
            frame_indices
        ]
    )
    evaluation_times = (
        frame_years
        - analysis_times_absolute[0]
    )

    input_frames = (
        baseline_sv_input[
            :,
            ::frame_step,
        ].T
    )

    if reconstruction_sv is not None:
        reconstruction_sv = np.asarray(
            reconstruction_sv,
            dtype=float,
        )

        if (
            reconstruction_sv.shape
            != baseline_sv_input.shape
            or not np.all(
                np.isfinite(
                    reconstruction_sv
                )
            )
        ):
            raise ValueError(
                "reconstruction_sv must be finite and have the same "
                "shape as baseline_sv_input."
            )

        reconstruction_frames = (
            reconstruction_sv[
                :,
                ::frame_step,
            ].T
        )
    elif candidate_eigenvalues.size:
        temporal = np.exp(
            np.outer(
                candidate_eigenvalues,
                evaluation_times,
            )
        )
        reconstruction_frames = np.real(
            (
                candidate_modes
                @ temporal
            ).T
        )
    else:
        reconstruction_frames = np.zeros_like(
            input_frames
        )

    if not np.all(
        np.isfinite(
            reconstruction_frames
        )
    ):
        raise FloatingPointError(
            "The retained CHAOS candidates produced non-finite "
            "video reconstruction values."
        )

    if cp is not None:
        frame_mjd2000 = np.asarray(
            cp.data_utils.dyear_to_mjd(
                frame_years
            ),
            dtype=float,
        )
    else:
        frame_mjd2000 = np.full(
            frame_years.shape,
            np.nan,
        )

    return {
        "frame_indices": frame_indices,
        "frame_years": frame_years,
        "frame_mjd2000": frame_mjd2000,
        "input": input_frames,
        "reconstruction": (
            reconstruction_frames
        ),
        "dt_snapshot": dt_snapshot,
        "n_candidates": int(
            candidate_eigenvalues.size
        ),
    }


def prepare_chaos_all_modes_video_data(
    *,
    analysis_times_absolute,
    candidate_eigenvalues,
    candidate_modes,
    candidate_periods,
    frame_spacing_years,
    scale_time_chunk=16,
    minimum_vmax=1e-12,
):
    """Prepare unique retained CHAOS modes for a low-memory tiled video."""

    analysis_times_absolute = np.asarray(
        analysis_times_absolute,
        dtype=float,
    ).ravel()
    candidate_eigenvalues = np.asarray(
        candidate_eigenvalues,
        dtype=complex,
    ).ravel()
    candidate_modes = np.asarray(
        candidate_modes,
        dtype=complex,
    )
    candidate_periods = np.asarray(
        candidate_periods,
        dtype=float,
    ).ravel()

    if analysis_times_absolute.size < 2:
        raise ValueError(
            "At least two CHAOS snapshots are required for a video."
        )
    if (
        candidate_modes.ndim != 2
        or candidate_modes.shape[1]
        != candidate_eigenvalues.size
        or candidate_periods.size
        != candidate_eigenvalues.size
    ):
        raise ValueError(
            "CHAOS candidate modes, eigenvalues, and periods must align."
        )
    if candidate_eigenvalues.size == 0:
        raise ValueError(
            "At least one retained CHAOS DMD mode is required."
        )
    if not (
        np.all(
            np.isfinite(
                analysis_times_absolute
            )
        )
        and np.all(
            np.isfinite(
                candidate_eigenvalues.real
            )
            & np.isfinite(
                candidate_eigenvalues.imag
            )
        )
        and np.all(
            np.isfinite(
                candidate_modes.real
            )
            & np.isfinite(
                candidate_modes.imag
            )
        )
        and np.all(
            (
                np.isfinite(
                    candidate_periods
                )
                & (
                    candidate_periods
                    > 0.0
                )
            )
            | np.isposinf(
                candidate_periods
            )
        )
    ):
        raise ValueError(
            "CHAOS all-modes video inputs contain invalid values."
        )
    if (
        not isinstance(
            scale_time_chunk,
            (
                int,
                np.integer,
            ),
        )
        or scale_time_chunk < 1
    ):
        raise ValueError(
            "scale_time_chunk must be a positive integer."
        )
    if (
        not np.isfinite(
            minimum_vmax
        )
        or minimum_vmax <= 0.0
    ):
        raise ValueError(
            "minimum_vmax must be finite and positive."
        )

    try:
        frame_spacing_years = float(
            frame_spacing_years
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "frame_spacing_years must be numeric."
        ) from error

    if (
        not np.isfinite(
            frame_spacing_years
        )
        or frame_spacing_years <= 0.0
    ):
        raise ValueError(
            "frame_spacing_years must be finite and positive."
        )

    time_differences = np.diff(
        analysis_times_absolute
    )
    dt_snapshot = float(
        np.median(
            time_differences
        )
    )

    if not np.allclose(
        time_differences,
        dt_snapshot,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            "CHAOS video input times must be uniformly sampled."
        )

    frame_step_float = (
        frame_spacing_years
        / dt_snapshot
    )
    frame_step = int(
        round(
            frame_step_float
        )
    )

    if (
        frame_step < 1
        or not np.isclose(
            frame_step
            * dt_snapshot,
            frame_spacing_years,
            rtol=1e-10,
            atol=1e-10,
        )
    ):
        raise ValueError(
            "frame_spacing_years must be an integer multiple of the "
            "selected CHAOS snapshot spacing. "
            f"Received frame_spacing_years={frame_spacing_years}, "
            f"dt_snapshot={dt_snapshot}."
        )

    frame_indices = np.arange(
        analysis_times_absolute.size
    )[
        ::frame_step
    ]
    frame_years = (
        analysis_times_absolute[
            frame_indices
        ]
    )
    evaluation_times = (
        frame_years
        - analysis_times_absolute[0]
    )

    mode_vmax = np.full(
        candidate_eigenvalues.size,
        float(minimum_vmax),
        dtype=float,
    )

    for mode_index, eigenvalue in enumerate(
        candidate_eigenvalues
    ):
        for start_index in range(
            0,
            evaluation_times.size,
            scale_time_chunk,
        ):
            stop_index = min(
                start_index
                + scale_time_chunk,
                evaluation_times.size,
            )

            with np.errstate(
                over="ignore",
                invalid="ignore",
            ):
                contribution = (
                    evaluate_continuous_phasor(
                        candidate_modes[
                            :,
                            mode_index,
                        ],
                        eigenvalue,
                        evaluation_times[
                            start_index:
                            stop_index
                        ],
                    )
                )

            if not np.all(
                np.isfinite(
                    contribution
                )
            ):
                raise FloatingPointError(
                    "A retained CHAOS DMD mode produced non-finite "
                    "values while calculating its video colour scale."
                )

            mode_vmax[
                mode_index
            ] = max(
                mode_vmax[
                    mode_index
                ],
                float(
                    np.max(
                        np.abs(
                            contribution
                        )
                    )
                ),
            )

    if cp is not None:
        frame_mjd2000 = np.asarray(
            cp.data_utils.dyear_to_mjd(
                frame_years
            ),
            dtype=float,
        )
    else:
        frame_mjd2000 = np.full(
            frame_years.shape,
            np.nan,
        )

    return {
        "frame_indices": frame_indices,
        "frame_years": frame_years,
        "frame_mjd2000": frame_mjd2000,
        "evaluation_times": evaluation_times,
        "candidate_eigenvalues": (
            candidate_eigenvalues
        ),
        "candidate_modes": candidate_modes,
        "candidate_periods": (
            candidate_periods
        ),
        "mode_vmax": mode_vmax,
        "dt_snapshot": dt_snapshot,
        "n_candidates": int(
            candidate_eigenvalues.size
        ),
    }


def make_chaos_reconstruction_video(
    *,
    video_data,
    output_path,
    parameter_summary,
    nlat,
    nlon,
    fps=5,
    cmap="seismic",
    roll_longitude=True,
    dpi=100,
    figsize=(15, 5.2),
    bitrate=5000,
):
    """Render processed CHAOS input, DMD reconstruction, and their difference."""

    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_frames = np.asarray(
        video_data[
            "input"
        ],
        dtype=float,
    )
    reconstruction_frames = np.asarray(
        video_data[
            "reconstruction"
        ],
        dtype=float,
    )

    if (
        input_frames.shape
        != reconstruction_frames.shape
        or input_frames.ndim != 2
    ):
        raise ValueError(
            "CHAOS video input and reconstruction arrays must have "
            "equal shape (n_frames, n_space)."
        )
    if (
        input_frames.shape[1]
        != nlat
        * nlon
    ):
        raise ValueError(
            "CHAOS video grid dimensions do not match the physical "
            "state-vector length."
        )
    if (
        not isinstance(
            fps,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
        or not np.isfinite(
            fps
        )
        or fps <= 0
    ):
        raise ValueError(
            "fps must be a finite positive number."
        )

    difference_vmax = max(
        (
            np.max(
                np.abs(
                    reconstruction_frames[
                        frame_index
                    ]
                    - input_frames[
                        frame_index
                    ]
                )
            )
            for frame_index in range(
                input_frames.shape[0]
            )
        ),
        default=0.0,
    )
    vmax = max(
        np.max(
            np.abs(
                input_frames
            )
        ),
        np.max(
            np.abs(
                reconstruction_frames
            )
        ),
        difference_vmax,
        1e-12,
    )
    norm = Normalize(
        vmin=-vmax,
        vmax=vmax,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.88,
        bottom=0.15,
        top=0.64,
        wspace=0.18,
    )
    extent = (
        -180.0,
        180.0,
        -90.0,
        90.0,
    )
    first_fields = (
        input_frames[0],
        reconstruction_frames[0],
        (
            reconstruction_frames[0]
            - input_frames[0]
        ),
    )
    titles = (
        "Processed CHAOS input",
        "Retained DMD reconstruction",
        "Reconstruction - input",
    )
    images = []

    for axis, field, title in zip(
        axes,
        first_fields,
        titles,
    ):
        image = axis.imshow(
            _reshape_global_field(
                field,
                nlat=nlat,
                nlon=nlon,
                roll_longitude=
                    roll_longitude,
            ),
            origin="upper",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        images.append(
            image
        )
        axis.set_title(
            title
        )
        axis.set_xlim(
            -180,
            180,
        )
        axis.set_ylim(
            -90,
            90,
        )
        axis.set_xlabel(
            "Longitude [degrees]"
        )
        axis.set_xticks(
            [
                -180,
                -90,
                0,
                90,
                180,
            ]
        )
        axis.set_yticks(
            [
                -90,
                -45,
                0,
                45,
                90,
            ]
        )

    axes[0].set_ylabel(
        "Latitude [degrees]"
    )
    fig.colorbar(
        images[0],
        ax=axes,
        label="Radial SV [nT/yr]",
        shrink=0.82,
    )
    title_artist = fig.suptitle(
        "",
        fontsize=11,
    )

    def update_frame(
        frame_index,
    ):
        fields = (
            input_frames[
                frame_index
            ],
            reconstruction_frames[
                frame_index
            ],
            (
                reconstruction_frames[
                    frame_index
                ]
                - input_frames[
                    frame_index
                ]
            ),
        )

        for image, field in zip(
            images,
            fields,
        ):
            image.set_data(
                _reshape_global_field(
                    field,
                    nlat=nlat,
                    nlon=nlon,
                    roll_longitude=
                        roll_longitude,
                )
            )

        decimal_year = video_data[
            "frame_years"
        ][frame_index]
        mjd2000 = video_data[
            "frame_mjd2000"
        ][frame_index]
        time_label = (
            f"decimal year={decimal_year:.2f}"
        )

        if np.isfinite(
            mjd2000
        ):
            time_label += (
                f" | MJD2000={mjd2000:.2f}"
            )

        title_artist.set_text(
            "CHAOS retained-mode reconstruction\n"
            f"{parameter_summary}\n"
            f"{time_label}"
        )

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=bitrate,
        metadata={
            "title": (
                "CHAOS DMD reconstruction"
            )
        },
    )

    with writer.saving(
        fig,
        str(output_path),
        dpi=dpi,
    ):
        for frame_index in tqdm(
            range(
                input_frames.shape[0]
            ),
            desc=(
                "Rendering CHAOS reconstruction video"
            ),
        ):
            update_frame(
                frame_index
            )
            writer.grab_frame()

    plt.close(
        fig
    )

    return output_path


def make_chaos_all_modes_video(
    *,
    video_data,
    output_path,
    parameter_summary,
    nlat,
    nlon,
    fps=5,
    cmap="seismic",
    roll_longitude=True,
    dpi=100,
    max_columns=4,
    bitrate=5000,
):
    """Render every retained CHAOS DMD mode in one tiled MP4."""

    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_eigenvalues = np.asarray(
        video_data[
            "candidate_eigenvalues"
        ],
        dtype=complex,
    ).ravel()
    candidate_modes = np.asarray(
        video_data[
            "candidate_modes"
        ],
        dtype=complex,
    )
    candidate_periods = np.asarray(
        video_data[
            "candidate_periods"
        ],
        dtype=float,
    ).ravel()
    mode_vmax = np.asarray(
        video_data[
            "mode_vmax"
        ],
        dtype=float,
    ).ravel()
    evaluation_times = np.asarray(
        video_data[
            "evaluation_times"
        ],
        dtype=float,
    ).ravel()

    n_modes = candidate_eigenvalues.size

    if (
        candidate_modes.ndim != 2
        or candidate_modes.shape
        != (
            nlat * nlon,
            n_modes,
        )
        or candidate_periods.size
        != n_modes
        or mode_vmax.size
        != n_modes
        or evaluation_times.size
        != np.asarray(
            video_data[
                "frame_years"
            ]
        ).size
    ):
        raise ValueError(
            "CHAOS all-modes video arrays have inconsistent shapes."
        )
    if n_modes == 0:
        raise ValueError(
            "At least one retained CHAOS DMD mode is required."
        )
    if not (
        np.all(
            np.isfinite(
                mode_vmax
            )
        )
        and np.all(
            mode_vmax
            > 0.0
        )
    ):
        raise ValueError(
            "Every CHAOS video mode requires a finite positive scale."
        )
    if (
        not isinstance(
            max_columns,
            (
                int,
                np.integer,
            ),
        )
        or max_columns < 1
    ):
        raise ValueError(
            "max_columns must be a positive integer."
        )
    if (
        not isinstance(
            fps,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
        or not np.isfinite(
            fps
        )
        or fps <= 0
    ):
        raise ValueError(
            "fps must be a finite positive number."
        )

    n_columns = min(
        int(max_columns),
        n_modes,
    )
    n_rows = int(
        np.ceil(
            n_modes
            / n_columns
        )
    )
    figsize = (
        max(
            12.0,
            4.0
            * n_columns,
        ),
        2.8
        * n_rows
        + 1.6,
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.82,
        hspace=0.72,
        wspace=0.18,
    )
    flat_axes = axes.ravel()
    extent = (
        -180.0,
        180.0,
        -90.0,
        90.0,
    )
    images = []

    for mode_index in range(
        n_modes
    ):
        axis = flat_axes[
            mode_index
        ]
        period = candidate_periods[
            mode_index
        ]
        growth_rate = (
            candidate_eigenvalues[
                mode_index
            ].real
        )

        if np.isposinf(
            period
        ):
            period_label = (
                r"$T=\infty$ (static)"
            )
        else:
            period_label = (
                f"T={period:.3g} yr"
            )

        if np.isclose(
            growth_rate,
            0.0,
            rtol=0.0,
            atol=1e-12,
        ):
            growth_label = (
                r"growth/decay=0 yr$^{-1}$"
            )
        elif growth_rate > 0.0:
            growth_label = (
                f"growth={growth_rate:+.3e} "
                r"yr$^{-1}$"
            )
        else:
            growth_label = (
                f"decay={growth_rate:+.3e} "
                r"yr$^{-1}$"
            )

        image = axis.imshow(
            _reshape_global_field(
                np.real(
                    candidate_modes[
                        :,
                        mode_index,
                    ]
                ),
                nlat=nlat,
                nlon=nlon,
                roll_longitude=
                    roll_longitude,
            ),
            origin="upper",
            extent=extent,
            cmap=cmap,
            norm=Normalize(
                vmin=-mode_vmax[
                    mode_index
                ],
                vmax=mode_vmax[
                    mode_index
                ],
            ),
            interpolation="nearest",
            aspect="equal",
        )
        images.append(
            image
        )
        axis.set_title(
            (
                f"Mode {mode_index + 1} | "
                f"{period_label}\n"
                f"{growth_label}\n"
                r"fixed scale $\pm$"
                f"{mode_vmax[mode_index]:.3g} nT/yr"
            ),
            fontsize=8,
        )
        axis.set_xlim(
            -180,
            180,
        )
        axis.set_ylim(
            -90,
            90,
        )
        axis.set_xticks(
            [
                -180,
                0,
                180,
            ]
        )
        axis.set_yticks(
            [
                -90,
                0,
                90,
            ]
        )
        axis.tick_params(
            labelsize=7
        )

        row_index, column_index = divmod(
            mode_index,
            n_columns,
        )

        if (
            row_index
            == n_rows - 1
        ):
            axis.set_xlabel(
                "Longitude [degrees]",
                fontsize=8,
            )
        if column_index == 0:
            axis.set_ylabel(
                "Latitude [degrees]",
                fontsize=8,
            )

    for axis in flat_axes[
        n_modes:
    ]:
        axis.set_axis_off()

    title_artist = fig.suptitle(
        "",
        fontsize=10,
        wrap=True,
    )

    def update_frame(
        frame_index,
    ):
        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            temporal = np.exp(
                candidate_eigenvalues
                * evaluation_times[
                    frame_index
                ]
            )
            fields = np.real(
                candidate_modes
                * temporal[
                    None,
                    :
                ]
            )

        if not np.all(
            np.isfinite(
                fields
            )
        ):
            raise FloatingPointError(
                "A retained CHAOS DMD mode produced non-finite "
                "values while rendering the video."
            )

        for mode_index, image in enumerate(
            images
        ):
            image.set_data(
                _reshape_global_field(
                    fields[
                        :,
                        mode_index,
                    ],
                    nlat=nlat,
                    nlon=nlon,
                    roll_longitude=
                        roll_longitude,
                )
            )

        decimal_year = video_data[
            "frame_years"
        ][frame_index]
        mjd2000 = video_data[
            "frame_mjd2000"
        ][frame_index]
        time_label = (
            f"decimal year={decimal_year:.2f}"
        )

        if np.isfinite(
            mjd2000
        ):
            time_label += (
                f" | MJD2000={mjd2000:.2f}"
            )

        title_artist.set_text(
            "CHAOS retained DMD modes: individual contributions\n"
            f"{parameter_summary}\n"
            f"{time_label}"
        )

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=bitrate,
        metadata={
            "title": (
                "CHAOS individual DMD modes"
            )
        },
    )

    with writer.saving(
        fig,
        str(output_path),
        dpi=dpi,
    ):
        for frame_index in tqdm(
            range(
                evaluation_times.size
            ),
            desc=(
                "Rendering CHAOS all-modes video"
            ),
        ):
            update_frame(
                frame_index
            )
            writer.grab_frame()

    plt.close(
        fig
    )

    return output_path
