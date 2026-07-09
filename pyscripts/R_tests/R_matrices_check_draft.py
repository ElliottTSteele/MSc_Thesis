'''
R_matrices_check.py

Purpose:
To understand which P, R and H are the same/ different across the files provided at each component
'''

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import chaosmagpy as cp

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# %% Code to read in the resolution matrix data

def R_Read_In(R_part, which):
    f = h5py.File((f'{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5'),"r")

    R_sub_matrix = np.asarray(f[R_part])

    f.close()
    return R_sub_matrix

# %% reading in and comparing

derivatives = ["MF", "SV", "SA"]
R_parts = ["P", "H"]

for R_part in R_parts:
    part_list = []
    for derivative in derivatives:
        part_deriv = R_Read_In(R_part, derivative)
        part_list.append(part_deriv)
    if np.allclose(part_list[0], part_list[1]):
        if np.allclose(part_list[0], part_list[2]):
            print(f"{R_part} matrices are all identical")
        else:
            print(f"{R_part} matrices are all different")
    else:
        print(f"{R_part} matrices are all different")




# %% comparing the R matrices memory efficiently


def Compare_R_Matrices_Blockwise(
    R_part="R",
    which_list=("MF", "SV", "SA"),
    block_size=500,
    rtol=0.0,
    atol=0.0,
    verbose=True,
):
    """
    Compare CHAOS resolution R matrices block-by-block without loading
    the full matrices into RAM.

    Parameters
    ----------
    R_part : str
        Dataset name inside the HDF5 file, probably "R".
    which_list : tuple
        Which resolution files to compare, e.g. ("MF", "SV", "SA").
    block_size : int
        Number of rows loaded at once.
    rtol, atol : float
        Tolerances passed to np.allclose.
        rtol=0, atol=0 gives exact equality for float values.
    verbose : bool
        Print progress and diagnostics.

    Returns
    -------
    bool
        True if all matrices match within tolerance, False otherwise.
    """

    file_paths = {
        which: Path(f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5")
        for which in which_list
    }

    files = {}
    datasets = {}

    try:
        # Open all files and get dataset handles
        for which, path in file_paths.items():
            files[which] = h5py.File(path, "r")
            datasets[which] = files[which][R_part]

        # Check shapes
        shapes = {which: datasets[which].shape for which in which_list}
        dtypes = {which: datasets[which].dtype for which in which_list}

        if verbose:
            print("Shapes:")
            for which in which_list:
                print(f"  {which}: shape={shapes[which]}, dtype={dtypes[which]}")

        first_shape = shapes[which_list[0]]
        if any(shape != first_shape for shape in shapes.values()):
            print("Shape mismatch:")
            for which, shape in shapes.items():
                print(f"  {which}: {shape}")
            return False

        n_rows, n_cols = first_shape

        # Compare every matrix to the first one
        reference_name = which_list[0]
        reference = datasets[reference_name]

        global_max_abs_diff = 0.0
        global_max_info = None

        for i0 in range(0, n_rows, block_size):
            i1 = min(i0 + block_size, n_rows)

            if verbose:
                print(f"Checking rows {i0}:{i1} / {n_rows}")

            ref_block = reference[i0:i1, :]

            for which in which_list[1:]:
                test_block = datasets[which][i0:i1, :]

                same = np.allclose(
                    ref_block,
                    test_block,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )

                if not same:
                    diff = np.abs(ref_block - test_block)
                    local_max = np.max(diff)
                    local_idx = np.unravel_index(np.argmax(diff), diff.shape)
                    global_idx = (i0 + local_idx[0], local_idx[1])

                    print()
                    print("Difference found.")
                    print(f"Comparison: {reference_name} vs {which}")
                    print(f"Row block: {i0}:{i1}")
                    print(f"Max abs diff in block: {local_max}")
                    print(f"Location: {global_idx}")
                    print(f"{reference_name} value: {reference[global_idx]}")
                    print(f"{which} value: {datasets[which][global_idx]}")

                    return False

                # Optional: track maximum absolute difference even if within tolerance
                if rtol != 0.0 or atol != 0.0:
                    diff = np.abs(ref_block - test_block)
                    local_max = np.max(diff)

                    if local_max > global_max_abs_diff:
                        local_idx = np.unravel_index(np.argmax(diff), diff.shape)
                        global_max_abs_diff = local_max
                        global_max_info = (
                            reference_name,
                            which,
                            i0 + local_idx[0],
                            local_idx[1],
                        )

        print()
        print(f"All {R_part} matrices match for {which_list}.")

        if global_max_info is not None:
            print(f"Global max abs diff within tolerance: {global_max_abs_diff}")
            print(f"Location/info: {global_max_info}")

        return True

    finally:
        for f in files.values():
            f.close()

Compare_R_Matrices_Blockwise()
# %% want to examine exactly the temporal structure of R, how different coefficients bleed into others
import h5py
import numpy as np
from tqdm import tqdm


def make_Tkl_abs_from_chaos_R(
    n_splines=64,
    chunk_rows=256,
    dtype=np.float64,
):
    """
    Forms:

        T[k, l] = sum_{i,j} |(R_ij)[k,l]|

    without loading the full R matrix into RAM.

    Assumes ravelled beta ordering is coefficient-major, i.e.

        beta = [coeff 0 splines, coeff 1 splines, ...]

    so spline index is global_index % n_splines.
    """

    filepath = f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_MF.h5"

    with h5py.File(filepath, "r") as f:
        R = f["R"]

        n_rows, n_cols = R.shape

        if n_rows != n_cols:
            raise ValueError(f"R must be square, got {R.shape}")

        if n_rows % n_splines != 0:
            raise ValueError(
                f"R size {n_rows} is not divisible by n_splines={n_splines}"
            )

        n_gauss = n_rows // n_splines
        print(f"R shape: {R.shape}")
        print(f"n_gauss: {n_gauss}")
        print(f"n_splines: {n_splines}")

        Tkl = np.zeros((n_splines, n_splines), dtype=dtype)

        col_spline_idx = np.arange(n_cols) % n_splines

        for row_start in tqdm(range(0, n_rows, chunk_rows), desc=f"Building Tkl MF"):
            row_end = min(row_start + chunk_rows, n_rows)

            R_chunk = R[row_start:row_end, :]
            abs_chunk = np.abs(R_chunk)

            # Sum over all input Gauss coefficients j,
            # while preserving input spline index l.
            chunk_by_l = np.zeros((row_end - row_start, n_splines), dtype=dtype)
            np.add.at(chunk_by_l, (slice(None), col_spline_idx), abs_chunk)

            # Sum over all output Gauss coefficients i,
            # while preserving output spline index k.
            row_spline_idx = np.arange(row_start, row_end) % n_splines
            np.add.at(Tkl, row_spline_idx, chunk_by_l)

    return Tkl

# %%
T_kl = make_Tkl_abs_from_chaos_R()
# %%
fig, ax = plt.subplots(figsize=(6, 6))

pcm = ax.pcolormesh(T_kl, shading="auto")

ax.set_box_aspect(1)
ax.set_xlabel("input spline index $\\ell$")
ax.set_ylabel("output spline index $k$")
ax.set_title(r"$T_{k\ell} = \sum_{i,j} |(R_{ij})_{k\ell}|$")

fig.colorbar(pcm, ax=ax, label="absolute summed resolution")
# %%
print(np.shape(T_kl))
# %%
P = R_Read_In("P", "MF")
H_MF = R_Read_In("H", "MF")
H_SV = R_Read_In("H", "SV")
H_SA = R_Read_In("H", "SA")
# %%
plt.imshow(P)
# %%
plt.imshow(H_MF @ T_kl @ P)
# %%
plt.imshow(H_SV @ T_kl @ P)

# %%

plt.imshow(H_SA @ T_kl @ P)

# %% now looking at how only a certain range of degree inputs effects total outputs:

import h5py
import numpy as np
from tqdm import tqdm


def gauss_degrees_from_nmax(nmax):
    """
    Standard Gauss coefficient ordering:
        n=1: g10, g11, h11
        n=2: g20, g21, h21, g22, h22
        ...
    Returns degree for each Gauss coefficient index.
    """
    degrees = []

    for n in range(1, nmax + 1):
        degrees.extend([n] * (2*n + 1))

    return np.array(degrees)


def infer_nmax_from_n_gauss(n_gauss):
    """
    For internal field coefficients:
        n_gauss = nmax * (nmax + 2)
    """
    nmax = int(np.sqrt(n_gauss + 1) - 1)

    if nmax * (nmax + 2) != n_gauss:
        raise ValueError(
            f"Could not infer nmax from n_gauss={n_gauss}. "
            "Check coefficient ordering/size."
        )

    return nmax


def make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=1,
    n_max=None,
    chunk_rows=256,
    dtype=np.float64,
):
    """
    Forms:

        T[k, l] = sum_i sum_{j in selected degrees} |(R_ij)[k,l]|

    without loading the full R matrix into RAM.

    This loops over all output coefficients i, but only includes input
    Gauss coefficients j with spherical harmonic degree:

        n_min <= n <= n_max

    Assumes beta ordering:

        [coeff 0 splines, coeff 1 splines, coeff 2 splines, ...]

    so spline index is:

        global_index % n_splines
    """

    filepath = f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_MF.h5"

    with h5py.File(filepath, "r") as f:
        R = f["R"]

        n_rows, n_cols = R.shape

        if n_rows != n_cols:
            raise ValueError(f"R must be square, got {R.shape}")

        if n_rows % n_splines != 0:
            raise ValueError(
                f"R size {n_rows} is not divisible by n_splines={n_splines}"
            )

        n_gauss = n_rows // n_splines
        inferred_nmax = infer_nmax_from_n_gauss(n_gauss)

        if n_max is None:
            n_max = inferred_nmax

        degrees = gauss_degrees_from_nmax(inferred_nmax)

        input_coeff_mask = (degrees >= n_min) & (degrees <= n_max)
        input_coeff_indices = np.where(input_coeff_mask)[0]

        # Convert selected Gauss coefficient indices into selected beta-vector columns
        selected_cols = []
        for j in input_coeff_indices:
            start = j * n_splines
            end = (j + 1) * n_splines
            selected_cols.extend(range(start, end))

        selected_cols = np.array(selected_cols, dtype=int)

        print(f"R shape: {R.shape}")
        print(f"n_gauss: {n_gauss}")
        print(f"n_splines: {n_splines}")
        print(f"inferred nmax: {inferred_nmax}")
        print(f"input degree range: n={n_min} to n={n_max}")
        print(f"number of selected input Gauss coefficients: {len(input_coeff_indices)}")
        print(f"number of selected input spline columns: {len(selected_cols)}")

        Tkl = np.zeros((n_splines, n_splines), dtype=dtype)

        selected_col_spline_idx = selected_cols % n_splines

        for row_start in tqdm(range(0, n_rows, chunk_rows), desc=f"Building Tkl n={n_min}-{n_max}"):
            row_end = min(row_start + chunk_rows, n_rows)

            # Read all output rows, but only selected input-degree columns
            R_chunk = R[row_start:row_end, selected_cols]
            abs_chunk = np.abs(R_chunk)

            # Sum over selected input Gauss coefficients j,
            # preserving input spline index l.
            chunk_by_l = np.zeros((row_end - row_start, n_splines), dtype=dtype)
            np.add.at(chunk_by_l, (slice(None), selected_col_spline_idx), abs_chunk)

            # Sum over all output Gauss coefficients i,
            # preserving output spline index k.
            row_spline_idx = np.arange(row_start, row_end) % n_splines
            np.add.at(Tkl, row_spline_idx, chunk_by_l)

    return Tkl

def make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=1,
    n_max=None,
    chunk_rows=256,
    dtype=np.float64,
):
    """
    Forms:

        T[k, l] = sum_i sum_{j in selected degrees} |(R_ij)[k,l]|

    without loading the full R matrix into RAM.

    This loops over all output coefficients i, but only includes input
    Gauss coefficients j with spherical harmonic degree:

        n_min <= n <= n_max

    Assumes beta ordering:

        [coeff 0 splines, coeff 1 splines, coeff 2 splines, ...]

    so spline index is:

        global_index % n_splines
    """

    filepath = f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_MF.h5"

    with h5py.File(filepath, "r") as f:
        R = f["R"]

        n_rows, n_cols = R.shape

        if n_rows != n_cols:
            raise ValueError(f"R must be square, got {R.shape}")

        if n_rows % n_splines != 0:
            raise ValueError(
                f"R size {n_rows} is not divisible by n_splines={n_splines}"
            )

        n_gauss = n_rows // n_splines
        inferred_nmax = n_gauss

        if n_max is None:
            n_max = inferred_nmax

        degrees = gauss_degrees_from_nmax(inferred_nmax)

        input_coeff_mask = (degrees >= n_min) & (degrees <= n_max)
        input_coeff_indices = np.where(input_coeff_mask)[0]

        # Convert selected Gauss coefficient indices into selected beta-vector columns
        selected_cols = []
        for j in input_coeff_indices:
            start = j * n_splines
            end = (j + 1) * n_splines
            selected_cols.extend(range(start, end))

        selected_cols = np.array(selected_cols, dtype=int)

        print(f"R shape: {R.shape}")
        print(f"n_gauss: {n_gauss}")
        print(f"n_splines: {n_splines}")
        print(f"inferred nmax: {inferred_nmax}")
        print(f"input degree range: n={n_min} to n={n_max}")
        print(f"number of selected input Gauss coefficients: {len(input_coeff_indices)}")
        print(f"number of selected input spline columns: {len(selected_cols)}")

        Tkl = np.zeros((n_splines, n_splines), dtype=dtype)

        selected_col_spline_idx = selected_cols % n_splines

        for row_start in tqdm(range(0, n_rows, chunk_rows), desc=f"Building Tkl n={n_min}-{n_max}"):
            row_end = min(row_start + chunk_rows, n_rows)

            # Read all output rows, but only selected input-degree columns
            R_chunk = R[row_start:row_end, selected_cols]
            abs_chunk = np.abs(R_chunk)

            # Sum over selected input Gauss coefficients j,
            # preserving input spline index l.
            chunk_by_l = np.zeros((row_end - row_start, n_splines), dtype=dtype)
            np.add.at(chunk_by_l, (slice(None), selected_col_spline_idx), abs_chunk)

            # Sum over all output Gauss coefficients i,
            # preserving output spline index k.
            row_spline_idx = np.arange(row_start, row_end) % n_splines
            np.add.at(Tkl, row_spline_idx, chunk_by_l)

    return Tkl
# %% running this for 5 degree intervals

Tkl_n1_5 = make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=1,
    n_max=5,
    chunk_rows=256,
)
Tkl_n6_10 = make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=6,
    n_max=10,
    chunk_rows=256,
)
Tkl_n11_15 = make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=11,
    n_max=15,
    chunk_rows=256,
)
Tkl_n16_20 = make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=16,
    n_max=20,
    chunk_rows=256,
)

# %%
import matplotlib.pyplot as plt
import numpy as np

Tkl_list = [Tkl_n1_5, Tkl_n6_10, Tkl_n11_15, Tkl_n16_20]
Tkl_list_t = [(H_MF @ Tkl_i @ P)[20:130, 20:130] for Tkl_i in Tkl_list.copy()]
titles = ["n = 1–5", "n = 6–10", "n = 11–15", "n = 16–20"]

# shared colour scale across all four panels
vmin = min(np.min(T) for T in Tkl_list)
vmax = max(np.max(T) for T in Tkl_list)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.ravel()

for ax, Tkl, title in zip(axes, Tkl_list_t, titles):
    pcm = ax.pcolormesh(Tkl, shading="auto")
    ax.set_title(title)
    ax.set_xlabel(r"input spline index $\ell$")
    ax.set_ylabel(r"output spline index $k$")
    ax.set_box_aspect(1)

# one shared colourbar
fig.colorbar(
    pcm,
    ax=axes,
    label=r"$T_{k\ell}=\sum_i \sum_{j\in \mathrm{range}} |(R_{ij})_{k\ell}|$",
    shrink=0.9
)

plt.show()
# %% examining how one of these matrices works:

def Simple_G_Generate_One(s, f, times):
    gnm = np.zeros((148, 440)) # N = 20 gauss time series template

    g_i = np.cos(2 * np.pi * f * times)

    gnm[:, s] = g_i

    return(gnm)

# %%
def make_Tkl_abs_from_single_input_gauss_R(
    input_gauss_idx,
    n_splines=64,
    chunk_rows=256,
    dtype=np.float64,
):
    """
    Forms:

        T[k, l] = sum_i |(R_ij)[k,l]|

    for one selected input Gauss coefficient j = input_gauss_idx.

    Loops over all output coefficients i, but only includes the selected
    input Gauss coefficient.

    input_gauss_idx is assumed to be zero-based.
    """

    filepath = f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_MF.h5"

    with h5py.File(filepath, "r") as f:
        R = f["R"]

        n_rows, n_cols = R.shape

        if n_rows != n_cols:
            raise ValueError(f"R must be square, got {R.shape}")

        if n_rows % n_splines != 0:
            raise ValueError(
                f"R size {n_rows} is not divisible by n_splines={n_splines}"
            )

        n_gauss = n_rows // n_splines

        if input_gauss_idx < 0 or input_gauss_idx >= n_gauss:
            raise ValueError(
                f"input_gauss_idx={input_gauss_idx} is outside valid range "
                f"0 to {n_gauss - 1}"
            )

        selected_cols = np.arange(
            input_gauss_idx * n_splines,
            (input_gauss_idx + 1) * n_splines,
            dtype=int,
        )

        selected_col_spline_idx = selected_cols % n_splines

        print(f"R shape: {R.shape}")
        print(f"n_gauss: {n_gauss}")
        print(f"n_splines: {n_splines}")
        print(f"selected input Gauss index: {input_gauss_idx}")
        print(f"selected input spline columns: {selected_cols[0]} to {selected_cols[-1]}")

        Tkl = np.zeros((n_splines, n_splines), dtype=dtype)

        for row_start in tqdm(
            range(0, n_rows, chunk_rows),
            desc=f"Building Tkl input idx={input_gauss_idx}",
        ):
            row_end = min(row_start + chunk_rows, n_rows)

            # all output rows, only the selected input Gauss coefficient columns
            R_chunk = R[row_start:row_end, selected_cols]
            abs_chunk = (R_chunk)

            # preserve input spline index l
            chunk_by_l = np.zeros((row_end - row_start, n_splines), dtype=dtype)
            np.add.at(chunk_by_l, (slice(None), selected_col_spline_idx), abs_chunk)

            # sum over all output Gauss coefficients i, preserving output spline index k
            row_spline_idx = np.arange(row_start, row_end) % n_splines
            np.add.at(Tkl, row_spline_idx, chunk_by_l)

    return Tkl

Tkl_idx420 = make_Tkl_abs_from_single_input_gauss_R(
    input_gauss_idx=420,
    n_splines=64,
    chunk_rows=256,
)

fig, ax = plt.subplots(figsize=(6, 6))

pcm = ax.pcolormesh(Tkl_idx420, shading="auto")

ax.set_box_aspect(1)
ax.set_xlabel(r"input spline index $\ell$")
ax.set_ylabel(r"output spline index $k$")
ax.set_title(r"$T_{k\ell}$ for input Gauss index 420")

fig.colorbar(pcm, ax=ax, label=r"$\sum_i |(R_{ij})_{k\ell}|$")
plt.show()
# %%
gnm_test = Simple_G_Generate_One(420, 0.1, np.arange(0,148,1)*0.2)

# seeing what happens
T_test = Tkl_idx420

plt.imshow((Tkl_idx420 @ P @ gnm_test))
# %%
plt.imshow((gnm_test))
# %%

plt.imshow(H_MF @ P)

# %%
