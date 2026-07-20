'''
Plan:

this presents the base case with exact dmd + uncertainty ensemble
'''

# PREAMBLE
# %% SETTING UP AUTOUPDATES

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from pathlib import Path
print(sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
from matplotlib import cm


import numpy as np
import matplotlib.pyplot as plt
import chaosmagpy as cp

import os
import sys
import pydmd
import cmath
import copy
import h5py
import math
import pickle
import gc
import pydmd

from pathlib import Path
from scipy.signal import periodogram
from matplotlib.animation import FuncAnimation
from IPython.display import Video
from tqdm import tqdm
from scipy.interpolate import make_interp_spline
from chaosmagpy.chaos import BaseModel

# importing from pyDMD
from pydmd import DMD

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# import simulation setup
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *

import numpy as np


# %% ------------------------------------------------------
# SINGLE-MODE DEGREE-BLOCK DMD + NOISE ENSEMBLE
# ------------------------------------------------------

mode_number = "43"

Nmax = 20
n_skip = 3
high_q_flag = True

n_realisations = 10
noise_tau = 1.5

degree_truncations_tested = np.arange(4, 21, 4)

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

DMD_recovery = {}


with h5py.File(file_path, "r") as h5_file:

    # ------------------------------------------------------
    # LOAD SINGLE MODE
    # ------------------------------------------------------

    mode_data = Component_Load_SV(mode_number)

    eigenvalue = mode_data["eigenvalue"]
    true_period = 2 * np.pi / np.abs(eigenvalue.imag)

    gnm_phasor = Truncate_Gauss_Coeffs(
        mode_data["gnm"],
        tmax=Nmax,
    )

    amp_scaler = mode_amp_scalings[mode_number]

    gnm_phasor *= amp_scaler

    gnm_spl = np.asarray(
        h5_file[
            f"mode_{mode_number}/without_decay"
        ][()]
    )

    gnm_mode_res = H_sv @ (
        amp_scaler * gnm_spl
    )

    # ------------------------------------------------------
    # GENERATE NOISE ENSEMBLE ON THIS MODE ONLY
    # ------------------------------------------------------

    noised = Perturbation_Generate(
        gnm_mode_res,
        n_realisations=n_realisations,
        temporal_z="independent",
        tau=noise_tau,
        dt=dt_years,
    )

    queue = [
        ("no_noise", gnm_mode_res)
    ]

    queue += [
        ("noised", realisation)
        for realisation in noised
    ]

    # ------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------

    DMD_recovery = {
        "true_period": true_period,
        "degree_blocks": {},
    }

    for degree in degree_truncations_tested:

        DMD_recovery[
            "degree_blocks"
        ][degree] = {

            "no_noise": None,

            "noised_periods": [],
            "noised_similarities": [],
        }

    # ======================================================
    # DEGREE-BLOCK LOOP
    # ======================================================

    for truncation_degree in tqdm(
        degree_truncations_tested,
        desc="Degree blocks",
    ):

        tmax_current = truncation_degree
        tmin_current = max(
            1,
            truncation_degree - 3,
        )

        A_r_current = Truncate_Gauss_Coeffs(
            A_20_dict["r"],
            tmax=tmax_current,
            tmin=tmin_current,
        )

        # Theoretical phasor in this degree block
        gnm_phasor_current = Truncate_Gauss_Coeffs(
            gnm_phasor,
            tmax=tmax_current,
            tmin=tmin_current,
        )

        sv_phasor_current = (
            A_r_current
            @ gnm_phasor_current.T
        )

        # --------------------------------------------------
        # NO-NOISE + EACH NOISE REALISATION
        # --------------------------------------------------

        for result_type, gnm_input in queue:

            if high_q_flag:

                gnm_input_windowed = (
                    gnm_input[
                        good_record_slice,
                        :
                    ]
                )

            else:

                gnm_input_windowed = (
                    gnm_input
                )

            gnm_input_current = Truncate_Gauss_Coeffs(
                gnm_input_windowed,
                tmax=tmax_current,
                tmin=tmin_current,
            )

            sv_input = (
                A_r_current
                @ gnm_input_current.T
            )

            sv_input = (
                sv_input[:, ::n_skip]
            )

            # ----------------------------------------------
            # DMD
            # ----------------------------------------------

            dmd = DMD(forward_backward=True)
            dmd.fit(sv_input)

            recovered_dict = DMD_Mode_Pair(
                dmd,
                n_skip * dt_years,
            )

            candidate_eigs = np.asarray(
                recovered_dict[
                    "continuous_eigenvalues"
                ]
            )

            candidate_modes = (
                recovered_dict["phasors"]
            )

            candidate_periods = np.asarray([

                2 * np.pi
                / np.abs(eig.imag)

                if np.abs(eig.imag) > 0

                else np.inf

                for eig in candidate_eigs
            ])

            valid = np.where(
                np.isfinite(candidate_periods)
                & (candidate_periods > 1e-5)
                & (candidate_periods < 1000)
            )[0]

            # ----------------------------------------------
            # BEST PHASOR MATCH
            # ----------------------------------------------

            similarities = np.asarray([

                Complex_Phasor_Compare(
                    candidate_modes[:, idx],
                    sv_phasor_current,
                )

                for idx in valid
            ])

            if len(similarities) == 0:
                continue

            best_local = np.argmax(
                similarities
            )

            match_idx = valid[
                best_local
            ]

            recovered_period = (
                candidate_periods[
                    match_idx
                ]
            )

            similarity = (
                similarities[
                    best_local
                ]
            )

            # ----------------------------------------------
            # STORE
            # ----------------------------------------------

            results = DMD_recovery[
                "degree_blocks"
            ][truncation_degree]

            if result_type == "no_noise":

                results["no_noise"] = {
                    "period":
                        recovered_period,

                    "similarity":
                        similarity,

                    "eigenvalue":
                        candidate_eigs[
                            match_idx
                        ],
                }

            else:

                results[
                    "noised_periods"
                ].append(
                    recovered_period
                )

                results[
                    "noised_similarities"
                ].append(
                    similarity
                )

# %% ------------------------------------------------------
# PLOT PERIOD VS PHASOR SIMILARITY BY DEGREE BLOCK
# ------------------------------------------------------

from matplotlib import cm
from matplotlib.colors import BoundaryNorm

degrees = np.asarray(
    sorted(
        degree_truncations_tested
    ),
    dtype=int,
)

cmap = plt.colormaps[
    "viridis"
].resampled(
    len(degrees)
)

degree_step = np.median(
    np.diff(degrees)
)

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


fig, ax = plt.subplots(
    figsize=(8, 5.5)
)


# ---------------------------------------------------------
# TRUE / EXACT REFERENCE
# ---------------------------------------------------------

ax.scatter(
    true_period,
    1.0,
    marker="o",
    s=65,
    facecolor="none",
    edgecolor="black",
    linewidth=1.5,
    label="Exact",
    zorder=10,
)


# ---------------------------------------------------------
# DEGREE BLOCK RESULTS
# ---------------------------------------------------------

for degree in degrees:

    results = DMD_recovery[
        "degree_blocks"
    ][degree]

    colour = cmap(
        norm(degree)
    )

    # -----------------------------------------
    # NO-NOISE DMD
    # -----------------------------------------

    if results[
        "no_noise"
    ] is not None:

        ax.scatter(
            results[
                "no_noise"
            ]["period"],

            results[
                "no_noise"
            ]["similarity"],

            marker="s",
            s=60,
            color=colour,
            edgecolor="black",
            linewidth=0.5,
            zorder=8,
        )

    # -----------------------------------------
    # NOISE ENSEMBLE
    # -----------------------------------------

    periods = np.asarray(
        results[
            "noised_periods"
        ]
    )

    similarities = np.asarray(
        results[
            "noised_similarities"
        ]
    )

    ax.scatter(
        periods,
        similarities,
        marker="x",
        s=35,
        color=colour,
        alpha=0.5,
    )

    # -----------------------------------------
    # JOINT MEDIAN POINT
    #
    # Median period and median similarity
    # of ensemble for this degree block.
    # -----------------------------------------

    if len(periods) > 0:

        ax.scatter(
            np.median(periods),
            np.median(similarities),

            marker="^",
            s=85,
            color=colour,
            edgecolor="black",
            linewidth=0.8,
            zorder=9,
        )


# ---------------------------------------------------------
# MARKER LEGEND
# ---------------------------------------------------------

from matplotlib.lines import Line2D

result_handles = [

    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=7,
        label="Exact",
    ),

    Line2D(
        [0], [0],
        marker="s",
        linestyle="none",
        color="grey",
        markersize=7,
        label="DMD, no noise",
    ),

    Line2D(
        [0], [0],
        marker="x",
        linestyle="none",
        color="grey",
        markersize=7,
        label="DMD, noised",
    ),

    Line2D(
        [0], [0],
        marker="^",
        linestyle="none",
        markerfacecolor="grey",
        markeredgecolor="black",
        markersize=8,
        label="Noised median",
    ),
]

legend = ax.legend(
    handles=result_handles,
    title="Result",
    loc="lower right",
)

ax.add_artist(
    legend
)


# ---------------------------------------------------------
# DISCRETE DEGREE-BLOCK COLOURBAR
# ---------------------------------------------------------

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
    pad=0.02,
)

cbar.ax.set_yticklabels([
    f"{max(1, d - 3)}-{d}"
    for d in degrees
])

cbar.set_label(
    "Spherical harmonic degree block"
)


ax.set_xlabel(
    "Recovered period [yr]"
)

ax.set_ylabel(
    "Phasor similarity"
)

ax.set_ylim(
    0,
    1.03,
)

ax.grid(
    alpha=0.25
)

ax.set_title(
    f"Mode {mode_number} — true period = "
    f"{true_period:.2f} yr"
)

fig.tight_layout()

plt.show()