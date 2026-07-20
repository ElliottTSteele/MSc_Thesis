# %% IMPORTS

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm

from pydmd import DMD

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *


# %% ------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

mode_number = "43"

Nmax = 20
n_skip = 3
high_q_flag = True

degree_blocks = [4, 8, 12, 16, 20]   # 1-4, 5-8, ..., 17-20

n_frames = 10
fps = 1
output_filename = f"mode_{mode_number}_degree_block_DMD.mp4"

# Grid assumptions used elsewhere in the project
nlon = 360
nlat = A_20_dict["r"].shape[0] // nlon
latitudes = np.linspace(90, -90, nlat)
longitudes = np.linspace(0, 360, nlon, endpoint=False)

def DMD_Mode_Pair_With_Amplitude(dmd, dt):

    discrete_eigs = np.asarray(dmd.eigs)
    continuous_eigs = np.log(discrete_eigs) / dt

    modes = np.asarray(dmd.modes)
    amplitudes = np.asarray(dmd.amplitudes)

    phasors = []
    eigs_out = []

    # Keep one member of each oscillatory conjugate pair
    positive_indices = np.where(
        continuous_eigs.imag > 1e-10
    )[0]

    for idx in positive_indices:

        eig = continuous_eigs[idx]

        # DMD contribution of conjugate pair:
        #
        # phi*b*exp(lambda*t) + conjugate
        # = 2 Re[phi*b*exp(lambda*t)]
        #
        # Therefore the complex phasor used with
        # Re[phasor * exp(lambda*t)] is 2*phi*b.

        phasor = (
            2.0
            * modes[:, idx]
            * amplitudes[idx]
        )

        phasors.append(phasor)
        eigs_out.append(eig)

    return {
        "continuous_eigenvalues": np.asarray(eigs_out),
        "phasors": np.column_stack(phasors),
    }
# %% ------------------------------------------------------
# LOAD ONE MODE AND RUN DMD DEGREE-BLOCK-WISE
# ---------------------------------------------------------

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

mode_data = Component_Load_SV(mode_number)
eigenvalue_true = mode_data["eigenvalue"]
true_period = 2 * np.pi / np.abs(eigenvalue_true.imag)
true_omega = 2 * np.pi / true_period

amp_scaler = mode_amp_scalings[mode_number]

gnm_phasor = Truncate_Gauss_Coeffs(
    mode_data["gnm"],
    tmax=Nmax,
    tmin=1,
) * amp_scaler

with h5py.File(file_path, "r") as h5_file:
    gnm_spl = np.asarray(
        h5_file[f"mode_{mode_number}/without_decay"][()]
    )

gnm_mode_res = H_sv @ (amp_scaler * gnm_spl)

if high_q_flag:
    gnm_mode_res = gnm_mode_res[good_record_slice, :]

results = []

for tmax_current in tqdm(degree_blocks, desc="Degree blocks"):

    tmin_current = max(1, tmax_current - 3)

    A_r_current = Truncate_Gauss_Coeffs(
        A_20_dict["r"],
        tmax=tmax_current,
        tmin=tmin_current,
    )

    gnm_input_current = Truncate_Gauss_Coeffs(
        gnm_mode_res,
        tmax=tmax_current,
        tmin=tmin_current,
    )

    gnm_phasor_current = Truncate_Gauss_Coeffs(
        gnm_phasor,
        tmax=tmax_current,
        tmin=tmin_current,
    )

    sv_input_all_steps = A_r_current @ gnm_input_current.T
    sv_input = sv_input_all_steps[:, ::n_skip]

    sv_phasor_true = A_r_current @ gnm_phasor_current.T

    dmd = DMD()
    dmd.fit(sv_input)

    recovered_dict = DMD_Mode_Pair_With_Amplitude(
        dmd,
        n_skip * dt_years,
    )

    candidate_eigs = np.asarray(
        recovered_dict["continuous_eigenvalues"]
    )
    candidate_modes = recovered_dict["phasors"]

    candidate_periods = np.asarray([
        2 * np.pi / np.abs(eig.imag)
        if np.abs(eig.imag) > 0 else np.inf
        for eig in candidate_eigs
    ])

    valid = np.where(
        np.isfinite(candidate_periods)
        & (candidate_periods > 1e-5)
        & (candidate_periods < 1000)
    )[0]

    similarities = np.asarray([
        Complex_Phasor_Compare(
            candidate_modes[:, idx],
            sv_phasor_true,
        )
        for idx in valid
    ])

    match_idx = valid[np.argmax(similarities)]

    recovered_phasor = candidate_modes[:, match_idx].copy()
    recovered_eig = candidate_eigs[match_idx]
    recovered_period = candidate_periods[match_idx]
    similarity = similarities[np.argmax(similarities)]

    # Global complex phase alignment only.
    # This preserves the recovered spatial pattern/amplitude while choosing
    # the phase origin that best matches the theoretical phasor.
    phase_factor = np.vdot(recovered_phasor, sv_phasor_true)
    if np.abs(phase_factor) > 0:
        recovered_phasor *= np.exp(1j * np.angle(phase_factor))

    results.append({
        "tmin": tmin_current,
        "tmax": tmax_current,
        "true_phasor": sv_phasor_true,
        "recovered_phasor": recovered_phasor,
        "recovered_eig": recovered_eig,
        "recovered_period": recovered_period,
        "similarity": similarity,
    })


# %% ------------------------------------------------------
# PRECOMPUTE TWO-TRUE-PERIOD ANIMATION
# ---------------------------------------------------------

animation_times = np.linspace(
    0,
    2 * true_period,
    n_frames,
    endpoint=False,
)

for result in results:

    true_phasor = result["true_phasor"]
    recovered_phasor = result["recovered_phasor"]
    recovered_eig = result["recovered_eig"]

    true_frames = np.real(
        true_phasor[:, None]
        * np.exp(1j * true_omega * animation_times)[None, :]
    )

    recovered_frames = np.real(
        recovered_phasor[:, None]
        * np.exp(recovered_eig * animation_times)[None, :]
    )

    result["true_frames"] = true_frames
    result["recovered_frames"] = recovered_frames

    vmax = max(
        np.max(np.abs(true_frames)),
        np.max(np.abs(recovered_frames)),
    )
    result["vmax"] = vmax


# %% ------------------------------------------------------
# BUILD VIDEO FIGURE
# ---------------------------------------------------------

nrows = len(results)

fig, axes = plt.subplots(
    nrows,
    2,
    figsize=(12, 2.8 * nrows),
    constrained_layout=True,
    squeeze=False,
)

fig.suptitle(
    f"Mode {mode_number}: degree-block DMD recovery",
    fontsize=14,
)

images = []

for row, result in enumerate(results):

    vmax = result["vmax"]
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    true_grid = result["true_frames"][:, 0].reshape(
        nlat,
        nlon,
    )

    recovered_grid = result["recovered_frames"][:, 0].reshape(
        nlat,
        nlon,
    )

    im_true = axes[row, 0].imshow(
        true_grid,
        extent=[0, 360, -90, 90],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
    )

    im_recovered = axes[row, 1].imshow(
        recovered_grid,
        extent=[0, 360, -90, 90],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
    )

    axes[row, 0].set_title(
        f"Input, n={result['tmin']}-{result['tmax']}"
    )

    axes[row, 1].set_title(
        f"Recovered DMD | "
        f"Ttrue={true_period:.2f} yr, "
        f"TDMD={result['recovered_period']:.2f} yr, "
        f"sim={result['similarity']:.3f}"
    )

    axes[row, 0].set_ylabel("Latitude (°)")

    for ax in axes[row]:
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_yticks([-60, -30, 0, 30, 60])

    if row == nrows - 1:
        axes[row, 0].set_xlabel("Longitude (°)")
        axes[row, 1].set_xlabel("Longitude (°)")

    cbar = fig.colorbar(
        im_true,
        ax=axes[row, :],
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("SV")

    images.append((im_true, im_recovered))


time_text = fig.text(
    0.5,
    0.995,
    "",
    ha="center",
    va="top",
)


def update(frame):

    t = animation_times[frame]

    for row, result in enumerate(results):

        true_grid = result["true_frames"][:, frame].reshape(
            nlat,
            nlon,
        )

        recovered_grid = result["recovered_frames"][:, frame].reshape(
            nlat,
            nlon,
        )

        images[row][0].set_data(true_grid)
        images[row][1].set_data(recovered_grid)

    time_text.set_text(
        f"Animation time = {t:.2f} yr "
        f"({t / true_period:.2f} true periods)"
    )

    return [
        artist
        for pair in images
        for artist in pair
    ] + [time_text]


animation = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=1000 / fps,
    blit=False,
)

output_path = Path(FIG_DIR) / output_filename

animation.save(
    output_path,
    writer="ffmpeg",
    fps=fps,
    dpi=150,
)

plt.close(fig)

print(f"Saved video to: {output_path}")
