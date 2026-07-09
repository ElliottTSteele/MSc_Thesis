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

# code to add degree markers on plot
degrees = np.arange(1, 21, 1)

def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)

degrees_idx = n_Gauss_Coeffs(degrees)-1

# Code to read in the resolution matrix data

def R_Read_In(R_part, which):
    f = h5py.File((f'{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5'),"r")

    R_sub_matrix = np.asarray(f[R_part])

    f.close()
    return R_sub_matrix

# reading in and comparing

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

# %% outputting the overal T_kl (i.e. examining leakage with white gauss input)
T_kl = make_Tkl_abs_from_chaos_R()
# %% reading in B-spline projection matrices
P = R_Read_In("P", "MF")
H_MF = R_Read_In("H", "MF")
H_SV = R_Read_In("H", "SV")
H_SA = R_Read_In("H", "SA")
# %% RECORDING IMPORTANT TIME POINTS HERE FOR FUTURE USE:
t_r_start = 1997.1
dt_years = 0.2

notable_times_raw = {
    "CHAMP Start (2000/08)": 2000 + 8/12,
    "CHAMP Start (2010/09)": 2010 + 9/12,
    "SWARM End (2013/11)": 2013 + 11/12,
    "(2026/01)": 2025
}

notable_times_relative = {}
notable_times_aspline = {}

for event in notable_times_raw:
    notable_times_relative[event] = \
        (notable_times_raw[event] - t_r_start) / dt_years
    
# %%

print(notable_times_aspline)
# %% getting function to say given time idx t, which spline
# should in theory be contributing most (i.e. spline index k)
plt.imshow(H_MF)


def max_abs_row_index_per_column(A):
    A = np.asarray(A)
    return np.argmax(np.abs(A), axis=0)

# function to convert time to approximate spline index
time_to_spline = max_abs_row_index_per_column(P)
print(time_to_spline)
notable_times_aspline = {}
for event in notable_times_relative:
    time_dyear_idx = notable_times_relative[event]
    
    t_idx = int(np.ceil(time_dyear_idx))
    notable_times_aspline[event] = time_to_spline[t_idx]

# %% from this can record rough spline times (be careful - 
# this is based on MF!) that correspond to bad data periods
s_list = []
full_spline_idx = np.arange(1, 64, 1)
for event in notable_times_aspline:
    s_idx = notable_times_aspline[event]
    s_list.append(s_idx)

def split_spline_quality_indices(s_list, n_splines=64, include_end=False):

    s1, s2, s3, s4 = s_list

    if include_end:
        high_quality_idx = np.r_[s1:s2 + 1, s3:s4 + 1]
    else:
        high_quality_idx = np.r_[s1:s2, s3:s4]

    all_idx = np.arange(n_splines)

    poor_quality_idx = np.setdiff1d(
        all_idx,
        high_quality_idx
    )

    return high_quality_idx, poor_quality_idx

high_quality_idx, poor_quality_idx = split_spline_quality_indices(
    s_list=s_list,
    n_splines=64,
    include_end=False
)
# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# -------------------------
# 1) T_kl in spline space
# -------------------------
pcm0 = ax[0].pcolormesh(
    T_kl/440,
    shading="auto",
    vmin=0,
    vmax=1
)

for event, t_rel in notable_times_aspline.items():

    ax[0].axhline(t_rel, linewidth=2, color='white', linestyle='--')

    ax[0].text(
        x=0.02,              # near left of plot, in axes coordinates
        y=t_rel + 0.5,        # just above the line, in data coordinates
        s=event,
        transform=ax[0].get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=12,
        color='white',
        bbox=dict(
            facecolor="black",
            alpha=0.65,
            edgecolor="none",
            boxstyle="round,pad=0.2"
        )
    )

ax[0].set_box_aspect(1)
ax[0].set_xlabel(r"input spline index $\ell$")
ax[0].set_ylabel(r"output spline index $k$")
ax[0].set_title(r"Spline-space $T_{k\ell}$")

fig.colorbar(
    pcm0,
    ax=ax[0],
    label=r"$T_{k\ell}$"
)


# -----------------------------------
# 2) H_mf @ T_kl / 440 @ P in gnm space
# -----------------------------------
T_kl_gnm = H_MF @ (T_kl / 440) @ P

pcm1 = ax[1].pcolormesh(
    np.abs(T_kl_gnm),
    shading="auto",
    vmin=0,
    vmax=1
)

for event, t_rel in notable_times_relative.items():

    ax[1].axhline(t_rel, linewidth=2, color='white', linestyle='--')

    ax[1].text(
        x=0.02,              # near left of plot, in axes coordinates
        y=t_rel + 0.5,        # just above the line, in data coordinates
        s=event,
        transform=ax[1].get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=12,
        color='white',
        bbox=dict(
            facecolor="black",
            alpha=0.65,
            edgecolor="none",
            boxstyle="round,pad=0.2"
        )
    )

ax[1].set_box_aspect(1)
ax[1].set_xlabel(r"input time index $\ell$")
ax[1].set_ylabel(r"output time index $k$")
ax[1].set_title(r"Time-domain $H_{\mathrm{mf}} (T_{k\ell}) P$")

ax[1].legend(loc="upper left")

fig.colorbar(
    pcm1,
    ax=ax[1],
    label="$H_{\mathrm{mf}} (T_{k\ell}) P$"
)

plt.show()


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
    
    return Tkl / len(input_coeff_indices)
# %% running this for 5 degree intervals

Tkl_n1 = make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=1,
    n_max=1,
    chunk_rows=256,
)

Tkl_n20 = make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=20,
    n_max=20,
    chunk_rows=256,
)
Tkl_n10 = make_Tkl_abs_from_chaos_R(
    n_splines=64,
    n_min=10,
    n_max=10,
    chunk_rows=256,
)

# %%

Tkl_list = [Tkl_n1, Tkl_n10, Tkl_n20]
Tkl_list_t = [(Tkl_i) for Tkl_i in Tkl_list.copy()]
titles = ["n = 1", "n = 10", "n = 20"]

# shared colour scale across all four panels
vmin = 0
vmax = 1

fig, axes = plt.subplots(1, len(Tkl_list), figsize=(14, 4), constrained_layout=True)
axes = axes.ravel()

for ax, Tkl, title in zip(axes, Tkl_list_t, titles):
    pcm = ax.pcolormesh(Tkl, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(r"input spline index $\ell$")
    ax.set_ylabel(r"output spline index $k$")
    ax.set_box_aspect(1)
    for event, t_rel in notable_times_aspline.items():

        ax.axhline(t_rel, linewidth=2, color='white', linestyle='--')

        

# one shared colourbar
fig.colorbar(
    pcm,
    ax=axes,
    label=r"$T_{k\ell}^{j=n}$",
)

plt.show()

# %%
# but what about P @ H_MF? i.e. what do splines look like
centre_idx = int(70)

MF_line = np.abs((H_MF @ P)[centre_idx, :])
SV_line = np.abs((H_SV @ P)[centre_idx, :])
SA_line = np.abs((H_SA @ P)[centre_idx, :])

n_input = MF_line.size

# Relative input time-step axis:
# centre_idx is labelled 0,
# indices after it are +1, +2, ...
# indices before it are -1, -2, ...
x_rel = np.arange(n_input) - centre_idx

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x_rel, MF_line/np.max(MF_line), label="MF")
ax.plot(x_rel, SV_line/np.max(SV_line), label="SV")
ax.plot(x_rel, SA_line/np.max(SA_line), label="SA")

ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
    label=r"input time step $k_j$"
)

ax.set_xlabel(r"input time step relative to $k_j$")
ax.set_ylabel(r"relative contribution to output time step $k_j^r$")
ax.set_title(r"Time-domain contribution kernels centred on output time step $k_j^r$")
ax.set_xlim((-35, 35))
ax.grid(True, alpha=0.3)
ax.legend()

plt.show()
# %% visualising R from the perspective of R_ij block norms
import h5py
import numpy as np
from tqdm import tqdm


def make_Rij_block_norm_matrix(
    n_splines=64,
    chunk_rows_blocks=8,
    norm="fro",
    dtype=np.float64,
):
    """
    Builds a (n_gauss, n_gauss) matrix M where:

        M[i, j] = ||R_ij||

    where R_ij is the n_splines x n_splines block mapping
    input Gauss coefficient j to output Gauss coefficient i.

    Does not load the full R matrix into RAM.

    Parameters
    ----------
    n_splines : int
        Number of spline coefficients per Gauss coefficient.

    chunk_rows_blocks : int
        Number of output Gauss-coefficient blocks to read at once.

    norm : str
        "fro"  -> Frobenius norm of each block.
        "l1"   -> sum(abs(block)).
        "max"  -> max(abs(block)).

    dtype : type
        Output matrix dtype.
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
        print(f"block norm: {norm}")

        M = np.zeros((n_gauss, n_gauss), dtype=dtype)

        rows_per_chunk = chunk_rows_blocks * n_splines

        for row_start in tqdm(
            range(0, n_rows, rows_per_chunk),
            desc="Building Rij block norm matrix",
        ):
            row_end = min(row_start + rows_per_chunk, n_rows)

            R_chunk = R[row_start:row_end, :]  # shape: (chunk_rows, n_cols)

            n_i_chunk = (row_end - row_start) // n_splines
            i_start = row_start // n_splines
            i_end = i_start + n_i_chunk

            # reshape into:
            # (output_gauss_i, output_spline_k, input_gauss_j, input_spline_l)
            R_blocks = R_chunk.reshape(
                n_i_chunk,
                n_splines,
                n_gauss,
                n_splines,
            )

            if norm == "fro":
                M[i_start:i_end, :] = np.sqrt(
                    np.sum(R_blocks**2, axis=(1, 3))
                )

            elif norm == "l1":
                M[i_start:i_end, :] = np.sum(
                    np.abs(R_blocks), axis=(1, 3)
                )

            elif norm == "max":
                M[i_start:i_end, :] = np.max(
                    np.abs(R_blocks), axis=(1, 3)
                )

            else:
                raise ValueError("norm must be one of: 'fro', 'l1', 'max'")

    return M

# %% function to normalise matrix elements by column sum
import numpy as np

def normalise_columns(A, kind):
    A = np.asarray(A, dtype=float)

    if kind=="sum":
        col_scale = A.sum(axis=0, keepdims=True)
    elif kind=="max":
        col_scale = A.max(axis=0, keepdims=True)

    A_norm = np.divide(
        A,
        col_scale,
        out=np.zeros_like(A),
        where=col_scale != 0
    )

    return A_norm

# %%
Rij_norm = make_Rij_block_norm_matrix(
    n_splines=64,
    chunk_rows_blocks=8,
    norm="fro",
)

# %%
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

pcm = ax.imshow(
    Rij_norm/np.max(Rij_norm),
    origin="lower",
    aspect="equal"
)

for idx in degrees_idx:
    if idx < Rij_norm.shape[1]:
        ax.axvline(idx - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.9)
    if idx < Rij_norm.shape[0]:
        ax.axhline(idx - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.9)

ax.set_xlabel(r"input Gauss coefficient index $j$")
ax.set_ylabel(r"output Gauss coefficient index $i$")
ax.set_title(r"Full $R_{ij}$ block norm matrix")

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r"block norm $\|R_{ij}\|$")

plt.show()

# %%

col_sums = Rij_norm.sum(axis=0, keepdims=True)
col_maxes = Rij_norm.max(axis=0, keepdims=True)

Rij_norm_col = np.divide(
    Rij_norm,
    col_maxes,
    out=np.zeros_like(Rij_norm),
    where=col_sums != 0,
)
fig, ax = plt.subplots(figsize=(7, 7))

pcm = ax.pcolormesh((Rij_norm), shading="auto", vmin=0, vmax=1, cmap='binary')

ax.set_aspect("equal", adjustable='box')
ax.invert_yaxis()

ax.set_xlabel("input Gauss coefficient index $j$")
ax.set_ylabel("output Gauss coefficient index $i$")
ax.set_title("Column-normalised block norm matrix")

fig.colorbar(pcm, ax=ax, label="fraction of input leakage")
plt.show()

# %%
def make_Rij_block_norm_matrix_spline_windowed(
    k_indices,
    n_splines=64,
    chunk_rows_blocks=8,
    norm="fro",
    normalise=True,
    dtype=np.float64,
):
    """
    Builds a (n_gauss, n_gauss) matrix M where:

        M[i, j] = || R_ij[k_indices, :] ||

    where R_ij is the n_splines x n_splines block mapping
    input Gauss coefficient j to output Gauss coefficient i.

    This examines how input Gauss coefficient j influences output Gauss
    coefficient i, but only over a specified window of output spline
    coefficient indices k.

    Parameters
    ----------
    k_indices : array-like
        Output spline coefficient indices to include in the block norm.

        Example:
            k_indices = np.arange(20, 31)

        This keeps rows k=20,...,30 of each R_ij block.

    n_splines : int
        Number of spline coefficients per Gauss coefficient.

    chunk_rows_blocks : int
        Number of output Gauss-coefficient blocks to read at once.

    norm : str
        "fro" -> Frobenius norm over R_ij[k_indices, :]
        "l1"  -> sum(abs(...))
        "max" -> max(abs(...))

    normalise : bool
        If True, normalises by the number of included elements so that
        different window sizes are comparable.

        For "fro":
            sqrt(sum(x^2) / N)

        For "l1":
            sum(abs(x)) / N

        For "max":
            unchanged

    dtype : type
        Output matrix dtype.
    """

    filepath = f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_MF.h5"

    k_indices = np.asarray(k_indices, dtype=int)

    if np.any(k_indices < 0) or np.any(k_indices >= n_splines):
        raise ValueError(
            f"k_indices must be between 0 and {n_splines - 1}"
        )

    n_k = len(k_indices)
    n_norm_elements = n_k * n_splines

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
        print(f"output spline k window: {k_indices[0]} to {k_indices[-1]}")
        print(f"number of included k values: {n_k}")
        print(f"block norm: {norm}")
        print(f"normalise: {normalise}")

        M = np.zeros((n_gauss, n_gauss), dtype=dtype)

        rows_per_chunk = chunk_rows_blocks * n_splines

        for row_start in tqdm(
            range(0, n_rows, rows_per_chunk),
            desc="Building spline-windowed Rij block norm matrix",
        ):
            row_end = min(row_start + rows_per_chunk, n_rows)

            R_chunk = R[row_start:row_end, :]

            n_i_chunk = (row_end - row_start) // n_splines
            i_start = row_start // n_splines
            i_end = i_start + n_i_chunk

            # Shape:
            # (output_gauss_i, output_spline_k, input_gauss_j, input_spline_l)
            R_blocks = R_chunk.reshape(
                n_i_chunk,
                n_splines,
                n_gauss,
                n_splines,
            )

            # Keep only selected output spline rows k
            R_blocks_windowed = R_blocks[:, k_indices, :, :]

            if norm == "fro":
                vals = np.sqrt(
                    np.sum(R_blocks_windowed**2, axis=(1, 3))
                )

                if normalise:
                    vals = vals / np.sqrt(n_norm_elements)

                M[i_start:i_end, :] = vals

            elif norm == "l1":
                vals = np.sum(
                    np.abs(R_blocks_windowed), axis=(1, 3)
                )

                if normalise:
                    vals = vals / n_norm_elements

                M[i_start:i_end, :] = vals

            elif norm == "max":
                M[i_start:i_end, :] = np.max(
                    np.abs(R_blocks_windowed), axis=(1, 3)
                )

            else:
                raise ValueError("norm must be one of: 'fro', 'l1', 'max'")

    return M

# %% running windowed R block matrix

spline_windowed_good = make_Rij_block_norm_matrix_spline_windowed(
    k_indices=high_quality_idx,
    n_splines=64,
    chunk_rows_blocks=8,
    norm="fro",
    normalise=True,
)

# %%

spline_windowed_poor = make_Rij_block_norm_matrix_spline_windowed(
    k_indices=poor_quality_idx,
    n_splines=64,
    chunk_rows_blocks=8,
    norm="fro",
    normalise=True,
)
# %% normalise relative to R00 best recovery norm
spline_windowed_good_0max =\
      spline_windowed_good / np.max(spline_windowed_good)
spline_windowed_poor_0max =\
      spline_windowed_poor / np.max(spline_windowed_good)



# %%

# If your normalise_columns function does NOT take a second argument,
# just remove ,"max" from the two lines below.
spline_windowed_diff =\
      spline_windowed_poor_0max - spline_windowed_good_0max

# Symmetric colour scale for the difference plot
diff_abs_max = np.max(np.abs(spline_windowed_diff))

fig, ax = plt.subplots(
    1, 3,
    figsize=(14, 4),
    constrained_layout=True,
    sharex=True,
    sharey=True
)

# -------------------
# Panel 1: good
# -------------------
pcm0 = ax[0].pcolormesh(
    spline_windowed_good_0max,
    shading="auto",
    vmin=0,
    vmax=1
)

ax[0].set_title("High-quality spline window")
ax[0].set_xlabel("input Gauss coefficient index $j$")
ax[0].set_ylabel("output Gauss coefficient index $i$")
ax[0].set_box_aspect(1)

# -------------------
# Panel 2: poor
# -------------------
pcm1 = ax[1].pcolormesh(
    spline_windowed_poor_0max,
    shading="auto",
    vmin=0,
    vmax=1
)

ax[1].set_title("Poor-quality spline window")
ax[1].set_xlabel("input Gauss coefficient index $j$")
ax[1].set_ylabel("output Gauss coefficient index $i$")
ax[1].set_box_aspect(1)

# -------------------
# Panel 3: difference
# -------------------
pcm2 = ax[2].pcolormesh(
    spline_windowed_diff,
    shading="auto",
    vmin=-diff_abs_max,
    vmax=diff_abs_max,
    cmap="RdBu_r"
)

ax[2].set_title("Poor - high quality")
ax[2].set_xlabel("input Gauss coefficient index $j$")
ax[2].set_ylabel("output Gauss coefficient index $i$")
ax[2].set_box_aspect(1)

# -------------------
# Degree boundary lines
# -------------------
for a in ax:
    for idx in degrees_idx:
        # only draw boundaries that lie inside the plotted matrix extent
        if idx < spline_windowed_norm_g.shape[1]:
            a.axvline(idx, color="white", linestyle="--", linewidth=0.8, alpha=0.9)
        if idx < spline_windowed_norm_g.shape[0]:
            a.axhline(idx, color="white", linestyle="--", linewidth=0.8, alpha=0.9)

# -------------------
# Colorbars
# -------------------
fig.colorbar(pcm0, ax=ax[0], label="normalised windowed block norm")
fig.colorbar(pcm1, ax=ax[1], label="normalised windowed block norm")
fig.colorbar(pcm2, ax=ax[2], label="difference in normalised block norm")

plt.show()

# %% now get columns for plotting


# -------------------------------------------------
# Choose the two injected/input Gauss coefficient indices
# (edit these to whichever two you want to compare)
# -------------------------------------------------
coeff_idx_1 = 0
coeff_idx_2 = 220
coeff_idx_3 = 439

# If you want the second one to be 200 instead, use:
# coeff_idx_2 = 200

# -------------------------------------------------
# Degree boundary indices
# -------------------------------------------------
degrees = np.arange(1, 21)

def n_Gauss_Coeffs(lmax):
    return lmax * (lmax + 2)

degrees_idx = n_Gauss_Coeffs(degrees)

# -------------------------------------------------
# Helper plotting function
# -------------------------------------------------
def plot_leakage_panel(ax, M, col_idx, title, degrees_idx):
    """
    Plots the leakage of injected input coefficient j=col_idx
    into all output coefficients i, i.e. M[:, col_idx].
    """
    y = M[:, col_idx]
    x = np.arange(len(y))

    ax.plot(x, y, linewidth=1.8, marker='x',\
             label=fr"input coeff $j={col_idx}$")

    # black dashed line at injected coefficient index
    ax.axvline(
        col_idx,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="injected coefficient"
    )

    # red dashed lines at spherical harmonic degree boundaries
    first_deg_line = True
    for idx in degrees_idx:
        if idx < len(y):
            ax.axvline(
                idx,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="degree boundary" if first_deg_line else None
            )
            first_deg_line = False

    ax.set_title(title)
    ax.set_xlabel("output Gauss coefficient index $i$")
    ax.set_ylabel(r"leakage amplitude / normalised block norm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

cols_to_plot = [coeff_idx_1, coeff_idx_2, coeff_idx_3]

y_global_max = max(
    np.max(spline_windowed_poor_0max[:, cols_to_plot]),
    np.max(spline_windowed_good_0max[:, cols_to_plot])
)

y_global_min = min(
    np.min(spline_windowed_poor_0max[:, cols_to_plot]),
    np.min(spline_windowed_good_0max[:, cols_to_plot])
)
# -------------------------------------------------
# 2x2 figure
# top row    = poor
# bottom row = good
# col 1      = first chosen coefficient
# col 2      = second chosen coefficient
# -------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True, sharex=True)

# Top row: poor
plot_leakage_panel(
    ax[0, 0],
    spline_windowed_poor_0max,
    coeff_idx_1,
    title=fr"Poor record: injected coefficient $j={coeff_idx_1}$",
    degrees_idx=degrees_idx
)

plot_leakage_panel(
    ax[0, 1],
    spline_windowed_poor_0max,
    coeff_idx_2,
    title=fr"Poor record: injected coefficient $j={coeff_idx_2}$",
    degrees_idx=degrees_idx
)

plot_leakage_panel(
    ax[0, 2],
    spline_windowed_poor_0max,
    coeff_idx_3,
    title=fr"Poor record: injected coefficient $j={coeff_idx_2}$",
    degrees_idx=degrees_idx
)

# Bottom row: good
plot_leakage_panel(
    ax[1, 0],
    spline_windowed_good_0max,
    coeff_idx_1,
    title=fr"Good record: injected coefficient $j={coeff_idx_1}$",
    degrees_idx=degrees_idx
)

plot_leakage_panel(
    ax[1, 1],
    spline_windowed_good_0max,
    coeff_idx_2,
    title=fr"Good record: injected coefficient $j={coeff_idx_2}$",
    degrees_idx=degrees_idx
)

plot_leakage_panel(
    ax[1, 2],
    spline_windowed_good_0max,
    coeff_idx_3,
    title=fr"Good record: injected coefficient $j={coeff_idx_2}$",
    degrees_idx=degrees_idx
)
for a in ax.ravel():
    a.set_ylim(y_global_min, y_global_max)
fig.suptitle("Leakage of injected Gauss coefficients into output coefficients", fontsize=14)

plt.show()
# %% need code that can retrieve the real order of a gauss_i
# then find the multiples of that
