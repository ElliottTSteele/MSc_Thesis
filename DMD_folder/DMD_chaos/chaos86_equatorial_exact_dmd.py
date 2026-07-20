"""
Exploratory exact-DMD analysis of equatorial CHAOS-8.6 radial SV.

This is deliberately a first-look script rather than a significance test. It:

1. evaluates CHAOS-8.6 SV Gauss coefficients;
2. retains a configurable spherical-harmonic degree window;
3. synthesises radial SV only between +/- LATITUDE_LIMIT;
4. fits standard exact DMD;
5. combines conjugate DMD pairs into one physical oscillatory phasor;
6. plots the discrete eigenvalue spectrum, mode power versus period, and the
   real/imaginary parts of every retained equatorial mode phasor.
"""

# %% SETTING UP AUTOUPDATES

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chaosmagpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from pydmd import DMD

from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *


# %% ------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------

# CHAOS SV degree window, inclusive.
NMIN = 1
NMAX = 14

# Equatorial strip: -LATITUDE_LIMIT <= latitude <= LATITUDE_LIMIT.
LATITUDE_LIMIT = 25.0

# Use the higher-quality satellite record if good_record_slice is defined in
# the project setup. Both the data and time coordinates receive the same slice.
HIGH_Q_FLAG = True

# Temporal subsampling after the optional record selection.
N_SKIP = 1

# Standard exact DMD rank. Set to 0 for no truncation, or choose a modest rank
# for this exploratory analysis.
SVD_RANK = 20

# Optional removal of the temporal mean at every grid point before DMD.
# False gives exact DMD of the raw SV snapshots.
REMOVE_TEMPORAL_MEAN = False

# Period interval retained as oscillatory "wave" candidates.
MIN_PERIOD_YEARS = 1.0
MAX_PERIOD_YEARS = 100.0

# Imaginary continuous-time eigenvalues below this tolerance are treated as
# non-oscillatory and excluded from the wave-phasor compilation.
FREQUENCY_TOLERANCE = 1e-8

# Conjugate-pair matching tolerance.
PAIR_RTOL = 1e-5
PAIR_ATOL = 1e-8

# Robust colour limit for each phasor plot.
PHASOR_COLOUR_PERCENTILE = 99.0

# Optional figure saving.
SAVE_FIGURES = False
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chaos86_equatorial_exact_dmd"


# %% ------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def degree_slice(tmin, tmax):
    """Coefficient slice for inclusive degrees tmin through tmax."""

    if tmin < 1:
        raise ValueError("NMIN must be at least 1")
    if tmax < tmin:
        raise ValueError("NMAX must be greater than or equal to NMIN")

    return slice(
        tmin**2 - 1,
        (tmax + 1)**2 - 1,
    )


def select_record(data, times):
    """Apply the same record selection and subsampling to data and time."""

    if HIGH_Q_FLAG:
        data = data[good_record_slice]
        times = times[good_record_slice]

    return data[::N_SKIP], times[::N_SKIP]


def combine_exact_dmd_pairs(
    dmd,
    dt,
    data_scale,
    pair_rtol=1e-5,
    pair_atol=1e-8,
):
    """
    Combine each conjugate exact-DMD pair into one physical wave phasor.

    PyDMD returns discrete-time eigenvalues mu. Continuous eigenvalues are
    omega = log(mu) / dt. For a real dataset, oscillatory modes should occur in
    conjugate pairs. The amplitude-scaled pair is combined so that

        x_pair(t) = Re{phasor * exp(omega*t)}.
    """

    discrete_eigenvalues = np.asarray(dmd.eigs)
    continuous_eigenvalues = np.log(discrete_eigenvalues) / dt

    modes = np.asarray(dmd.modes)
    amplitudes = np.asarray(dmd.amplitudes)

    # Restore the scaling removed before fitting.
    scaled_modes = (
        modes
        * amplitudes[np.newaxis, :]
        * data_scale
    )

    used = set()
    recovered = []

    for i, omega_i in enumerate(continuous_eigenvalues):
        if i in used or omega_i.imag <= FREQUENCY_TOLERANCE:
            continue

        candidates = [
            j
            for j, omega_j in enumerate(continuous_eigenvalues)
            if j not in used
            and j != i
            and omega_j.imag < -FREQUENCY_TOLERANCE
        ]

        if not candidates:
            continue

        errors = np.asarray([
            np.abs(continuous_eigenvalues[j] - np.conj(omega_i))
            for j in candidates
        ])
        j = candidates[int(np.argmin(errors))]

        if not np.isclose(
            continuous_eigenvalues[j],
            np.conj(omega_i),
            rtol=pair_rtol,
            atol=pair_atol,
        ):
            continue

        # This equals 2*q_positive for an exact conjugate pair, where
        # q_positive = b_positive * phi_positive.
        phasor = scaled_modes[:, i] + np.conj(scaled_modes[:, j])

        recovered.append({
            "positive_index": i,
            "negative_index": j,
            "discrete_eigenvalue": discrete_eigenvalues[i],
            "continuous_eigenvalue": omega_i,
            "phasor": phasor,
            "pair_error": errors.min(),
        })

        used.add(i)
        used.add(j)

    return recovered, continuous_eigenvalues


def phase_align_phasor(phasor):
    """Choose display phase so the largest-magnitude point is real-positive."""

    phasor = np.asarray(phasor)
    reference_index = np.argmax(np.abs(phasor))
    reference_phase = np.angle(phasor[reference_index])
    return phasor * np.exp(-1j * reference_phase)


def mode_mean_square_power(phasor, omega, times, latitudes, nlon):
    """
    Area-weighted mean-square power of the real DMD-mode reconstruction.

    The finite fitted time window is used, so growth/decay in omega is included.
    """

    times_relative = times - times[0]
    temporal_component = np.exp(
        omega * times_relative
    )

    reconstruction = np.real(
        phasor[:, None]
        * temporal_component[None, :]
    )

    point_power = np.mean(reconstruction**2, axis=1)
    latitude_weights = np.repeat(
        np.cos(np.deg2rad(latitudes)),
        nlon,
    )

    return np.average(
        point_power,
        weights=latitude_weights,
    )


# %% ------------------------------------------------------
# LOAD CHAOS-8.6 SV
# ---------------------------------------------------------

chaos_path = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
chaos_model = cp.load_CHAOS_matfile(str(chaos_path))

# Evaluate all coefficients through NMAX first, then compactly select NMIN:NMAX.
chaos_sv_full = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=NMAX,
    deriv=1,
    extrapolate="off",
)

times_used = np.asarray(times_mjd2000)
chaos_sv_full, times_used = select_record(
    chaos_sv_full,
    times_used,
)

coefficient_slice = degree_slice(NMIN, NMAX)
chaos_sv = chaos_sv_full[..., coefficient_slice]

# Convert MJD2000 days into years relative to the first retained snapshot.
_, time_years = select_record(chaos_sv_full, times_dyear)

if len(time_years) < 2:
    raise ValueError("At least two CHAOS snapshots are required")

dt_dmd = float(np.median(np.diff(time_years)))

if not np.allclose(
    np.diff(time_years),
    dt_dmd,
    rtol=1e-5,
    atol=1e-8,
):
    raise ValueError(
        "Exact DMD assumes uniformly sampled snapshots, but the selected "
        "CHAOS times are not uniform."
    )


# %% ------------------------------------------------------
# SYNTHESISE THE EQUATORIAL SV STRIP
# ---------------------------------------------------------

A_r_full = A_20_dict["r"]
A_r_degree_window = A_r_full[:, coefficient_slice]

latitude = np.asarray(latitude)
longitude = np.asarray(longitude)

latitude_mask = (
    (latitude >= -LATITUDE_LIMIT)
    & (latitude <= LATITUDE_LIMIT)
)

latitude_equatorial = latitude[latitude_mask]

full_grid_indices = np.arange(
    len(latitude) * len(longitude)
).reshape(len(latitude), len(longitude))

equatorial_grid_indices = full_grid_indices[
    latitude_mask,
    :,
].ravel()

A_r_equatorial = A_r_degree_window[
    equatorial_grid_indices,
    :,
]

# DMD expects shape (space, time).
sv_equatorial = A_r_equatorial @ chaos_sv.T

if REMOVE_TEMPORAL_MEAN:
    sv_equatorial = (
        sv_equatorial
        - np.mean(sv_equatorial, axis=1, keepdims=True)
    )

print(f"CHAOS SV coefficients: {chaos_sv.shape}")
print(f"Equatorial DMD snapshots: {sv_equatorial.shape}")
print(f"DMD sampling interval: {dt_dmd:.4f} yr")
print(f"DMD record length: {time_years[-1] - time_years[0]:.2f} yr")


# %% ------------------------------------------------------
# STANDARD EXACT DMD
# ---------------------------------------------------------

data_scale = np.linalg.norm(sv_equatorial)
if data_scale == 0:
    raise ValueError("The selected CHAOS SV data have zero norm")

X_dmd = sv_equatorial / data_scale

dmd = DMD(
    svd_rank=SVD_RANK,
    tlsq_rank=0,
    exact=True,
    opt=False,
)
dmd.fit(X_dmd)

recovered_modes, all_continuous_eigenvalues = combine_exact_dmd_pairs(
    dmd,
    dt=dt_dmd,
    data_scale=data_scale,
    pair_rtol=PAIR_RTOL,
    pair_atol=PAIR_ATOL,
)

# Retain finite periods in the requested exploratory interval.
wave_modes = []

for mode in recovered_modes:
    omega = mode["continuous_eigenvalue"]
    period = 2.0 * np.pi / np.abs(omega.imag)

    if not (MIN_PERIOD_YEARS <= period <= MAX_PERIOD_YEARS):
        continue

    mode["period"] = period
    mode["growth_rate"] = omega.real
    mode["power"] = mode_mean_square_power(
        phasor=mode["phasor"],
        omega=omega,
        times=time_years,
        latitudes=latitude_equatorial,
        nlon=len(longitude),
    )
    wave_modes.append(mode)

wave_modes = sorted(
    wave_modes,
    key=lambda item: item["period"],
)

if not wave_modes:
    raise RuntimeError(
        "No conjugate oscillatory DMD modes fall inside the requested period "
        "interval. Try changing SVD_RANK or the period limits."
    )

total_wave_power = np.sum([
    mode["power"] for mode in wave_modes
])

for mode_number, mode in enumerate(wave_modes, start=1):
    mode["mode_number"] = mode_number
    mode["power_fraction"] = (
        mode["power"] / total_wave_power
        if total_wave_power > 0 else np.nan
    )

print(f"Retained oscillatory mode pairs: {len(wave_modes)}")
for mode in wave_modes:
    print(
        f"Mode {mode['mode_number']:2d}: "
        f"period={mode['period']:7.3f} yr, "
        f"growth={mode['growth_rate']:+.4f} yr^-1, "
        f"power={mode['power']:.4e}"
    )


# %% ------------------------------------------------------
# FIGURE 1: ALL DMD EIGENVALUES
# ---------------------------------------------------------

discrete_eigenvalues = np.asarray(dmd.eigs)

fig_eigs, ax = plt.subplots(figsize=(6.5, 6.0))

unit_angle = np.linspace(0, 2.0 * np.pi, 500)
ax.plot(
    np.cos(unit_angle),
    np.sin(unit_angle),
    color="grey",
    linestyle="--",
    linewidth=1,
    label="Unit circle",
)

ax.scatter(
    discrete_eigenvalues.real,
    discrete_eigenvalues.imag,
    color="0.65",
    edgecolor="black",
    linewidth=0.4,
    s=42,
    label="All exact-DMD eigenvalues",
    zorder=2,
)

period_values = np.asarray([
    mode["period"] for mode in wave_modes
])

retained_discrete = np.asarray([
    mode["discrete_eigenvalue"] for mode in wave_modes
])

scatter = ax.scatter(
    retained_discrete.real,
    retained_discrete.imag,
    c=period_values,
    cmap="viridis",
    s=75,
    edgecolor="black",
    linewidth=0.7,
    label="Retained positive-frequency waves",
    zorder=3,
)

for mode in wave_modes:
    mu = mode["discrete_eigenvalue"]
    ax.annotate(
        str(mode["mode_number"]),
        (mu.real, mu.imag),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax.axhline(0, color="black", linewidth=0.7)
ax.axvline(0, color="black", linewidth=0.7)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"Re$(\mu)$")
ax.set_ylabel(r"Im$(\mu)$")
ax.set_title("CHAOS-8.6 equatorial SV: exact-DMD eigenvalues")
ax.legend(loc="best", fontsize=8)
fig_eigs.colorbar(scatter, ax=ax, label="Period [yr]")
fig_eigs.tight_layout()


# %% ------------------------------------------------------
# FIGURE 2: MODE POWER VERSUS PERIOD
# ---------------------------------------------------------

power_values = np.asarray([
    mode["power"] for mode in wave_modes
])

growth_values = np.asarray([
    mode["growth_rate"] for mode in wave_modes
])

growth_limit = np.max(np.abs(growth_values))
growth_limit = max(growth_limit, 1e-12)

fig_power, ax = plt.subplots(figsize=(7.5, 5.0))

power_scatter = ax.scatter(
    period_values,
    power_values,
    c=growth_values,
    cmap="coolwarm",
    norm=Normalize(vmin=-growth_limit, vmax=growth_limit),
    s=70,
    edgecolor="black",
    linewidth=0.6,
)

for mode in wave_modes:
    ax.annotate(
        str(mode["mode_number"]),
        (mode["period"], mode["power"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax.axvspan(
    6.0,
    8.0,
    color="gold",
    alpha=0.18,
    label="6-8 yr reference band",
)
ax.set_yscale("log")
ax.set_xlabel("DMD period [yr]")
ax.set_ylabel(r"Area-weighted mean-square mode power [nT$^2$ yr$^{-2}$]")
ax.set_title("Oscillatory exact-DMD mode power")
ax.grid(alpha=0.25)
ax.legend()
fig_power.colorbar(
    power_scatter,
    ax=ax,
    label=r"Continuous growth rate [yr$^{-1}$]",
)
fig_power.tight_layout()


# %% ------------------------------------------------------
# FIGURE 3: REAL AND IMAGINARY EQUATORIAL MODE PHASORS
# ---------------------------------------------------------

n_wave_modes = len(wave_modes)

fig_modes, axes = plt.subplots(
    n_wave_modes,
    2,
    figsize=(13, max(2.5 * n_wave_modes, 3.5)),
    sharex=True,
    sharey=True,
    squeeze=False,
)

longitude_grid, latitude_grid = np.meshgrid(
    longitude,
    latitude_equatorial,
)

for row, mode in enumerate(wave_modes):
    phasor_aligned = phase_align_phasor(mode["phasor"])
    phasor_grid = phasor_aligned.reshape(
        len(latitude_equatorial),
        len(longitude),
    )

    colour_values = np.concatenate([
        np.abs(phasor_grid.real).ravel(),
        np.abs(phasor_grid.imag).ravel(),
    ])
    colour_limit = np.percentile(
        colour_values,
        PHASOR_COLOUR_PERCENTILE,
    )
    colour_limit = max(colour_limit, np.finfo(float).eps)

    image_real = axes[row, 0].pcolormesh(
        longitude_grid,
        latitude_grid,
        phasor_grid.real,
        cmap="RdBu_r",
        vmin=-colour_limit,
        vmax=colour_limit,
        shading="auto",
    )

    image_imag = axes[row, 1].pcolormesh(
        longitude_grid,
        latitude_grid,
        phasor_grid.imag,
        cmap="RdBu_r",
        vmin=-colour_limit,
        vmax=colour_limit,
        shading="auto",
    )

    axes[row, 0].set_ylabel("Latitude [deg]")
    axes[row, 0].set_title(
        f"Mode {mode['mode_number']}: real; "
        f"T={mode['period']:.2f} yr"
    )
    axes[row, 1].set_title(
        f"Mode {mode['mode_number']}: imaginary; "
        f"power fraction={mode['power_fraction']:.3f}"
    )

    fig_modes.colorbar(
        image_real,
        ax=axes[row, :],
        fraction=0.018,
        pad=0.015,
        label=r"Radial SV phasor [nT yr$^{-1}$]",
    )

for ax in axes[-1, :]:
    ax.set_xlabel("Longitude [degrees]")

for ax in axes.ravel():
    ax.set_xlim(longitude.min(), longitude.max())
    ax.set_ylim(-LATITUDE_LIMIT, LATITUDE_LIMIT)

fig_modes.suptitle(
    "CHAOS-8.6 equatorial radial-SV exact-DMD phasors",
    y=1.002,
)
fig_modes.subplots_adjust(
    left=0.07,
    right=0.91,
    bottom=0.05,
    top=0.96,
    hspace=0.42,
    wspace=0.15,
)


# %% ------------------------------------------------------
# OPTIONAL OUTPUT
# ---------------------------------------------------------

if SAVE_FIGURES:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_eigs.savefig(
        OUTPUT_DIR / "exact_dmd_eigenvalues.png",
        dpi=250,
        bbox_inches="tight",
    )
    fig_power.savefig(
        OUTPUT_DIR / "exact_dmd_power_vs_period.png",
        dpi=250,
        bbox_inches="tight",
    )
    fig_modes.savefig(
        OUTPUT_DIR / "exact_dmd_equatorial_phasors.png",
        dpi=250,
        bbox_inches="tight",
    )

plt.show()

# %% ------------------------------------------------------
# ANIMATE EACH RECOVERED MODE
# ---------------------------------------------------------

from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import Video, display


# Animation settings
ANIMATION_FPS = 10
ANIMATION_DPI = 150
ANIMATION_FRAME_SKIP = 1

VIDEO_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "chaos86_equatorial_exact_dmd"
    / "mode_videos"
)

VIDEO_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def reconstruct_dmd_mode(
    mode,
    times,
):
    """
    Reconstruct one real-valued oscillatory DMD mode through time.

    Returns
    -------
    reconstruction : ndarray, shape (nspace, nt)
    """

    phasor = mode["phasor"]
    omega = mode["continuous_eigenvalue"]

    times_relative = times - times[0]

    reconstruction = np.real(
        phasor[:, None]
        * np.exp(
            omega * times_relative[None, :]
        )
    )

    return reconstruction


# ---------------------------------------------------------
# Reconstruct every retained oscillatory mode
# ---------------------------------------------------------

mode_reconstructions = []

for mode in wave_modes:

    reconstruction = reconstruct_dmd_mode(
        mode=mode,
        times=time_years,
    )

    mode_reconstructions.append(
        reconstruction
    )


# Sum of all retained oscillatory modes
combined_wave_reconstruction = np.sum(
    mode_reconstructions,
    axis=0,
)

# Actual CHAOS data passed to DMD
chaos_reconstruction_target = (
    sv_equatorial.copy()
)


# ---------------------------------------------------------
# Reshape space-time arrays onto the equatorial grid
# ---------------------------------------------------------

nlat_equatorial = len(
    latitude_equatorial
)

nlon = len(longitude)

nt = len(time_years)

combined_wave_cube = (
    combined_wave_reconstruction.T.reshape(
        nt,
        nlat_equatorial,
        nlon,
    )
)

chaos_cube = (
    chaos_reconstruction_target.T.reshape(
        nt,
        nlat_equatorial,
        nlon,
    )
)

mode_reconstruction_cubes = [
    reconstruction.T.reshape(
        nt,
        nlat_equatorial,
        nlon,
    )
    for reconstruction in mode_reconstructions
]


# Approximate decimal-year labels from MJD2000
decimal_years = (
    2000.0 + times_used / 365.25
)

frame_indices = np.arange(
    0,
    nt,
    ANIMATION_FRAME_SKIP,
)


# ---------------------------------------------------------
# Shared colour scale for combined DMD and CHAOS panels
# ---------------------------------------------------------

comparison_values = np.concatenate([
    np.abs(combined_wave_cube).ravel(),
    np.abs(chaos_cube).ravel(),
])

comparison_limit = np.nanpercentile(
    comparison_values,
    99,
)

comparison_limit = max(
    comparison_limit,
    np.finfo(float).eps,
)


def animate_one_mode(
    mode,
    mode_cube,
    output_path,
):
    """
    Create a video showing one mode, the combined wave reconstruction,
    and the corresponding CHAOS SV.
    """

    mode_number = mode["mode_number"]
    period = mode["period"]
    growth_rate = mode["growth_rate"]
    power_fraction = mode["power_fraction"]

    # Give the individual mode its own colour scale so that weak modes
    # remain visible.
    mode_limit = np.nanpercentile(
        np.abs(mode_cube),
        99,
    )

    mode_limit = max(
        mode_limit,
        np.finfo(float).eps,
    )

    fig = plt.figure(
        figsize=(13, 8),
    )

    grid_spec = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 1.0],
        hspace=0.32,
        wspace=0.18,
    )

    # Individual mode spans the upper row
    ax_mode = fig.add_subplot(
        grid_spec[0, :]
    )

    ax_combined = fig.add_subplot(
        grid_spec[1, 0]
    )

    ax_chaos = fig.add_subplot(
        grid_spec[1, 1]
    )

    image_extent = [
        longitude.min(),
        longitude.max(),
        latitude_equatorial.min(),
        latitude_equatorial.max(),
    ]

    first_frame = frame_indices[0]

    mode_image = ax_mode.imshow(
        mode_cube[first_frame],
        origin="lower",
        extent=image_extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-mode_limit,
        vmax=mode_limit,
        interpolation="nearest",
    )

    combined_image = ax_combined.imshow(
        combined_wave_cube[first_frame],
        origin="lower",
        extent=image_extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-comparison_limit,
        vmax=comparison_limit,
        interpolation="nearest",
    )

    chaos_image = ax_chaos.imshow(
        chaos_cube[first_frame],
        origin="lower",
        extent=image_extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-comparison_limit,
        vmax=comparison_limit,
        interpolation="nearest",
    )

    ax_mode.set_title(
        (
            f"DMD mode {mode_number}: "
            f"T = {period:.2f} yr, "
            f"growth = {growth_rate:+.3f} yr$^{{-1}}$, "
            f"power fraction = {power_fraction:.3f}"
        )
    )

    ax_combined.set_title(
        "Combined retained oscillatory DMD modes"
    )

    if REMOVE_TEMPORAL_MEAN:
        chaos_title = "Temporally centred CHAOS-8.6 SV"
    else:
        chaos_title = "CHAOS-8.6 SV"

    ax_chaos.set_title(
        chaos_title
    )

    for ax in [
        ax_mode,
        ax_combined,
        ax_chaos,
    ]:
        ax.set_xlabel(
            "Longitude [degrees]"
        )

        ax.set_ylabel(
            "Latitude [degrees]"
        )

        ax.set_xlim(
            longitude.min(),
            longitude.max(),
        )

        ax.set_ylim(
            -LATITUDE_LIMIT,
            LATITUDE_LIMIT,
        )

    mode_colourbar = fig.colorbar(
        mode_image,
        ax=ax_mode,
        orientation="vertical",
        pad=0.015,
        fraction=0.025,
    )

    mode_colourbar.set_label(
        r"Mode radial SV [nT yr$^{-1}$]"
    )

    comparison_colourbar = fig.colorbar(
        chaos_image,
        ax=[
            ax_combined,
            ax_chaos,
        ],
        orientation="horizontal",
        pad=0.14,
        fraction=0.07,
    )

    comparison_colourbar.set_label(
        r"Radial SV [nT yr$^{-1}$]"
    )

    time_text = fig.text(
        0.5,
        0.965,
        "",
        ha="center",
        va="top",
        fontsize=12,
    )

    def update(frame_position):

        frame = frame_indices[
            frame_position
        ]

        mode_image.set_data(
            mode_cube[frame]
        )

        combined_image.set_data(
            combined_wave_cube[frame]
        )

        chaos_image.set_data(
            chaos_cube[frame]
        )

        time_text.set_text(
            (
                f"Time = {decimal_years[frame]:.2f} "
                f"(t = {time_years[frame]:.2f} yr)"
            )
        )

        return (
            mode_image,
            combined_image,
            chaos_image,
            time_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )

    writer = FFMpegWriter(
        fps=ANIMATION_FPS,
        bitrate=3000,
    )

    animation.save(
        output_path,
        writer=writer,
        dpi=ANIMATION_DPI,
    )

    plt.close(fig)

    return output_path


# ---------------------------------------------------------
# Generate one video for every retained wave mode
# ---------------------------------------------------------

video_paths = []

for mode, mode_cube in zip(
    wave_modes,
    mode_reconstruction_cubes,
):

    video_path = (
        VIDEO_OUTPUT_DIR
        / (
            f"mode_{mode['mode_number']:02d}"
            f"_period_{mode['period']:.2f}yr.mp4"
        )
    )

    print(
        f"Rendering mode {mode['mode_number']} "
        f"with period {mode['period']:.2f} yr"
    )

    animate_one_mode(
        mode=mode,
        mode_cube=mode_cube,
        output_path=video_path,
    )

    video_paths.append(
        video_path
    )


# ---------------------------------------------------------
# Display videos in a notebook or VS Code interactive window
# ---------------------------------------------------------

for mode, video_path in zip(
    wave_modes,
    video_paths,
):

    print(
        f"Mode {mode['mode_number']}: "
        f"period = {mode['period']:.2f} yr"
    )

    display(
        Video(
            str(video_path),
            embed=True,
        )
    )
# %%
