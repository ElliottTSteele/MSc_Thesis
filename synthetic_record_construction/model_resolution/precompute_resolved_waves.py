'''
g_res_syn_make.py

purpose:
- R is expensive to compute each time
- ideally want to precompute as much as possible to circumvent this
- by examining B_res = R P (g_input), can see it is a linear operation
- therefore can precompmute B_res for all input waves.
- following this the evaluation of g_res(c), c=[mf, sv, sa] is cheap 
- NOTE: all of the values are arbitrary, as scaling is applied post resolution

# scaling can happen in here, using the g_res(sv) for each mode

- then Cr = Ar @ g_res(c) can be computed for each component (slightly expensive)
- this is relevant for testing DMD in physical gridded space

considerations:
- this just stores one realisation of the wave
- i.e. period, decay, phase, and 'start time' are all set

decisions:
- period = default of wave ('true' period)
- decay = default of wave
- phase = 0

- start time = 0 decimal years - why? start of record actually quite poor quality
therefore unlikely to use for DMD, however it is used in the resolution matrix
(even if contribution to other times is negligible), so keep there for now.

FOR NOW:
compute and store:
- B_res: default eigenvalue, 0 phase, 0 dyear start time
- B_res_no_decay: default period + no decay, 0 phase, 0 dyear start time

both are unscaled/ in arbitrary physical units.
'''

# PREAMBLE
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

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *
# import the shared synthetic definitions used by the testing programs
from src.msc_thesis.synUtils import (
    G_Time_Series_Eval,
    H_mf,
    H_sa,
    H_sv,
    P,
    R_Apply_Two_Halves,
    Truncate_Gauss_Coeffs,
)
# retain the original raw-mode loading convention used to build these records
from src.msc_thesis.synConstruct import Component_Load

# %%
# TEST THE PROCEDURE

# LOAD IN MODE DATA

# get gauss phasor, eigenvalue
i = str("50")
mode_i_data = Component_Load(i)
g_phasor_full = mode_i_data["gnm"]
eigenvalue = mode_i_data["eigenvalue"]

# truncate to N=20
g_phasor = Truncate_Gauss_Coeffs(g_phasor_full, tmax=20)

# evaluate gauss coefficient time series (decay)
gnm_decay = G_Time_Series_Eval(g_phasor, eigenvalue)
# evaluate gauss coefficient time series (no decay)
eigenvalue_no_decay = 0 + 1j*eigenvalue.copy().imag

gnm_no_decay = G_Time_Series_Eval(g_phasor, eigenvalue_no_decay)

# for both time series, compute resolved b spline
gnm_ts_dict = {
    "With decay":gnm_decay, 
    "Without decay":gnm_no_decay
}

gnm_spl_res_dict = {}

for key in gnm_ts_dict:
    gnm = gnm_ts_dict[key]

    # project to b spline
    gnm_spl = P @ gnm

    # apply resolution (ram safe)
    gnm_spl_res = R_Apply_Two_Halves(gnm_spl)

    gnm_spl_res_dict[key] = gnm_spl_res

derivative_keys = ["MF", "SV", "SA"]
H_dict = {
    "MF":H_mf,
    "SV":H_sv,
    "SA":H_sa
}

def Gnm_BA_Res_Show(gnm_spl_res_dict):
    fig, axes = plt.subplots(3, 2)

    for i, k in enumerate(derivative_keys):

        gnm_no_decay = H_dict[k] @ gnm_spl_res_dict["Without decay"]
        gnm_decay = H_dict[k] @ gnm_spl_res_dict["With decay"]

        axes[i, 0].imshow(gnm_no_decay)
        axes[i, 1].imshow(gnm_decay)

    plt.show()

Gnm_BA_Res_Show(gnm_spl_res_dict)

# writing to HDF5

file_path_test = Path(f"{FELIX_DIR}/R_splines_arbitrary_TEST.h5")

with h5py.File(file_path_test, "a") as f:

    for i, mode in enumerate([i]):

        group = f.require_group(f"mode_{i}")

        group.require_dataset(
            "without_decay",
            data=gnm_spl_res_dict["Without decay"],
            shape=gnm_spl_res_dict["Without decay"].shape,
            dtype=gnm_spl_res_dict["Without decay"].dtype
        )

        group.require_dataset(
            "with_decay",
            data=gnm_spl_res_dict["With decay"],
            shape=gnm_spl_res_dict["With decay"].shape,
            dtype=gnm_spl_res_dict["With decay"].dtype
        )

# reading test
with h5py.File(file_path_test, "r") as f:
    gnm_spl = f[f"mode_{i}/with_decay"][:]
    plt.imshow(gnm_spl)
# %% APPLYING TO FULL DATA SET
# Set up output file
file_path = Path(f"{FELIX_DIR}/R_splines_arbitrary.h5")

mode_numbers = np.arange(1, 63).astype(str)

with h5py.File(file_path, "a") as f:

    for mode_key in mode_numbers:

        print(f"Processing mode {mode_key}")

        # Load Gauss phasor and eigenvalue
        mode_data = Component_Load(mode_key)

        g_phasor_full = mode_data["gnm"]
        eigenvalue = mode_data["eigenvalue"]

        # Truncate to degree 20
        g_phasor = Truncate_Gauss_Coeffs(
            g_phasor_full,
            tmax=20
        )

        # Time series with decay
        gnm_decay = G_Time_Series_Eval(
            g_phasor,
            eigenvalue
        )

        # Time series without decay
        eigenvalue_no_decay = 1j * eigenvalue.imag

        gnm_no_decay = G_Time_Series_Eval(
            g_phasor,
            eigenvalue_no_decay
        )

        gnm_ts_dict = {
            "with_decay": gnm_decay,
            "without_decay": gnm_no_decay,
        }

        # Create/retrieve mode group
        group = f.require_group(f"mode_{mode_key}")

        for dataset_name, gnm in gnm_ts_dict.items():

            # Project Gauss coefficients to spline coefficients
            gnm_spl = P @ gnm

            # Apply resolution matrix
            gnm_spl_res = R_Apply_Two_Halves(gnm_spl)

            # Create or retrieve compatible dataset
            dset = group.require_dataset(
                dataset_name,
                shape=gnm_spl_res.shape,
                dtype=gnm_spl_res.dtype,
            )

            # Explicitly write/overwrite the values
            dset[...] = gnm_spl_res

# %%
# %% APPLYING TO FULL DATA SET
# Set up output file
file_path = Path(f"{FELIX_DIR}/R_splines_arbitrary.h5")

mode_numbers = np.arange(1, 63).astype(str)

with h5py.File(file_path, "a") as f:

    for mode_key in mode_numbers:

        print(f"Processing mode {mode_key}")

        # Load Gauss phasor and eigenvalue
        mode_data = Component_Load(mode_key)

        g_phasor_full = mode_data["gnm"]
        eigenvalue = mode_data["eigenvalue"]

        # Truncate to degree 20
        g_phasor = Truncate_Gauss_Coeffs(
            g_phasor_full,
            tmax=20
        )

        # Time series with decay
        gnm_decay = G_Time_Series_Eval(
            g_phasor,
            eigenvalue
        )

        # Time series without decay
        eigenvalue_no_decay = 1j * eigenvalue.imag

        gnm_no_decay = G_Time_Series_Eval(
            g_phasor,
            eigenvalue_no_decay
        )

        gnm_ts_dict = {
            "with_decay": gnm_decay,
            "without_decay": gnm_no_decay,
        }

        # Create/retrieve mode group
        group = f.require_group(f"mode_{mode_key}")

        for dataset_name, gnm in gnm_ts_dict.items():

            # Project Gauss coefficients to spline coefficients
            gnm_spl = P @ gnm

            # Apply resolution matrix
            gnm_spl_res = R_Apply_Two_Halves(gnm_spl)

            # Create or retrieve compatible dataset
            dset = group.require_dataset(
                dataset_name,
                shape=gnm_spl_res.shape,
                dtype=gnm_spl_res.dtype,
            )

            # Explicitly write/overwrite the values
            dset[...] = gnm_spl_res


# code to generate the quadrature imaginary part
# %% STORE IMAGINARY, NO-DECAY SPLINE COUNTERPARTS

file_path = Path(f"{FELIX_DIR}/R_splines_arbitrary.h5")

mode_numbers = np.arange(1, 63).astype(str)

new_dataset_name = "without_decay_imaginary"

with h5py.File(file_path, "a") as f:

    for mode_key in mode_numbers:

        group_name = f"mode_{mode_key}"
        print(f"Processing {group_name}")

        # Retrieve the existing mode group.
        # Using f[group_name], rather than require_group(), ensures that an
        # accidentally missing group is reported instead of silently created.
        if group_name not in f:
            raise KeyError(
                f"Expected group '{group_name}' was not found in {file_path}"
            )

        group = f[group_name]

        # Never overwrite an existing imaginary dataset.
        if new_dataset_name in group:
            print(
                f"  '{new_dataset_name}' already exists — leaving unchanged."
            )
            continue

        # Load the original complex Gauss-coefficient phasor.
        mode_data = Component_Load(mode_key)
        g_phasor_full = mode_data["gnm"]
        eigenvalue = mode_data["eigenvalue"]

        # Truncate consistently with the existing real-component calculation.
        g_phasor = Truncate_Gauss_Coeffs(
            g_phasor_full,
            tmax=20,
        )

        # Remove modal growth/decay while retaining the angular frequency.
        eigenvalue_no_decay = 1j * eigenvalue.imag

        # If G_Time_Series_Eval(phi, lambda) evaluates
        #
        #     Re{phi exp(lambda t)},
        #
        # then replacing phi by -i*phi gives
        #
        #     Re{-i phi exp(lambda t)}
        #       = Im{phi exp(lambda t)}.
        gnm_no_decay_imaginary = G_Time_Series_Eval(
            -1j * g_phasor,
            eigenvalue_no_decay,
        )

        # Project into exactly the same B-spline representation.
        gnm_spl_imaginary = P @ gnm_no_decay_imaginary

        # Apply the same resolution matrix used for the real counterpart.
        gnm_spl_res_imaginary = R_Apply_Two_Halves(
            gnm_spl_imaginary
        )

        # create_dataset() raises an error if this name already exists.
        # The explicit check above makes this append-only and restartable.
        dset = group.create_dataset(
            new_dataset_name,
            data=gnm_spl_res_imaginary,
        )

        # Optional metadata documenting exactly what this dataset contains.
        dset.attrs["quadrature"] = "imaginary"
        dset.attrs["decay_applied"] = False
        dset.attrs["maximum_spherical_harmonic_degree"] = 20
        dset.attrs["construction"] = (
            "G_Time_Series_Eval(-1j * g_phasor, "
            "1j * imag(eigenvalue)), followed by P and R"
        )

        print(
            f"  Added '{new_dataset_name}' "
            f"with shape {dset.shape} and dtype {dset.dtype}"
        )

print("Finished without overwriting any existing datasets.")
# %%
