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
from matplotlib import cm
import os
import warnings

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# import simulation setup
from src.msc_thesis.synSetup import *


# ---------------------------------------------------------------
# Decomposition related
# ---------------------------------------------------------------
# function to compute economical SVD and return its component matrices
def Econ_SVD(data):

    # compute economical SVD of data matrix F
    U_econ, s_econ, Vh_econ = np.linalg.svd(data, full_matrices=False)

    econ_svd = [U_econ, s_econ, Vh_econ]

    return(econ_svd) # return final truncated SVD


# function to truncate (k/ variance) SVD and return output
def Truncate_SVD(svd, 
                 truncate_by = "variance", # truncate by variance by default
                 var_target = 0.99, # want 99% of variance by default
                 k=0 # use all non zero singular values by default
                    ):

    U, s, Vh = svd

    # code that truncates based on energy, defined k etc.

    # directly define number of singular values to truncate to
    if truncate_by == 'k':
        U_truncated = U[:, :k]
        s_truncated = s[:k]
        Vh_truncated = Vh[:k, :]

    # define by variance, using variance threshhold
    elif truncate_by == 'variance':

        # getting variance (squared singular values)
        energy = s**2
        # getting the fraction relative to total for each
        variance_fraction = energy / energy.sum()
        # get cumulative sum array to identify where
        # target variance is met
        cumulative_variance = np.cumsum(variance_fraction)
        
        # r-1 = lowest rank sum that meets variance target
        r = np.searchsorted(cumulative_variance, 0.99) + 1

        # truncate to r
        U_truncated = U[:, :r]
        s_truncated = s[:r]
        Vh_truncated = Vh[:r, :]

    truncated_svd = [U_truncated, s_truncated, Vh_truncated]
    
    return(truncated_svd) # return final truncated SVD

def Exact_DMD(data,
              # below are for internal SVD
              truncate_by = None, # None, variance, k
              var_target = 0.99, # want 99% of variance by default
              k=0): # use all non zero singular values by default):

    # establishing number of time steps from data
    n_times = np.shape(data)[1]

    # first split data into X and X'
    X = data[:, :-1] # exclude final time step
    X_prime = data[:, 1:] # exclude first time step

    # from this have U = modes, s = coefficients, Vt = dynamics
    if truncate_by == None:
        data_svd = Econ_SVD(X)
    else:
        data_svd = Truncate_SVD(Econ_SVD(X), 
                                truncate_by=truncate_by, 
                                var_target=var_target, 
                                k=k)
    
    U, s, Vh = data_svd
    
    # constructing A_tilde
    Uh = U.conj().T
    V = Vh.conj().T
    s_inv = 1/s
    A_tilde = Uh @ X_prime @ V * s_inv[None, :]
    
    # getting eigen solutions (L = values, W = vectors) of
    # reduced order system (k-truncated earlier) 
    # L = full order coefficients
    L, W = np.linalg.eig(A_tilde)
    
    # getting PHI = high dimensional modes
    PHI = X_prime @ V * s_inv[None, :] @ W
    
    # dynamics and coefficients
    
    # obtaining dynamics:
    '''
    dynamics:
    - each time step is advanced by multiplying by lambda
    - initial time state = initial amplitude (b_i)
    - each additional time state = previous * lambda_i
    Therefore can can make a vector v_i for each mode, where 
    each element is the next time step
    '''
    
    # getting initial amplitudes
    # X = PHI @ b
    # therefore b = np.inv(PHI, X)
    x0 = X[:, 0]
    b0, *_ = np.linalg.lstsq(PHI, x0, rcond=None)
    PHI = b0 * PHI # all modes are scaled to initial amplitudes

    # storing DMD time evolution coefficients for each mode
    # DONT NEED TO STORE, EASY TO RECONSTRUCT IN LOOP WITH IDX I AND L
    #n_times = new_snap_shape[1] # how many time steps
    #time_indices = np.arange(n_times+1) # array listing each time step
    #dynamics = b[:, None] * L[:, None]**time_indices[None, :]

    exact_DMD = [PHI,  L]

    return(exact_DMD)


def DMD_Mode_Pair(
    dmd,
    dt,
    atol=1e-8,
    rtol=1e-6,
    include_unpaired=False, # dont include unpaired modes
):

    eigenvalues = np.asarray(dmd.eigs, dtype=complex)
    modes = np.asarray(dmd.modes, dtype=complex)
    amplitudes = np.asarray(dmd.amplitudes, dtype=complex)

    if modes.shape[1] != eigenvalues.size:
        raise ValueError(
            "The number of columns in dmd.modes must equal the number "
            "of eigenvalues."
        )

    if amplitudes.size != eigenvalues.size:
        raise ValueError(
            "dmd.amplitudes must contain one amplitude per DMD mode."
        )

    # Complete dynamically scaled modal coefficients b_j phi_j.
    scaled_modes = modes * amplitudes[np.newaxis, :]

    used = set()
    output_phasors = []
    output_eigenvalues = []
    mode_information = []

    for i, eig_i in enumerate(eigenvalues):

        if i in used:
            continue

        # ---------------------------------------------------------
        # Real/non-oscillatory eigenvalue
        # ---------------------------------------------------------
        if np.isclose(eig_i.imag, 0.0, atol=atol, rtol=rtol):

            output_phasors.append(scaled_modes[:, i])
            output_eigenvalues.append(eig_i)

            mode_information.append({
                "type": "real",
                "indices": (i,),
                "pair_error": np.nan,
            })

            used.add(i)
            continue

        # ---------------------------------------------------------
        # Find the closest unused conjugate eigenvalue
        # ---------------------------------------------------------
        candidate_indices = [
            j
            for j in range(eigenvalues.size)
            if j != i and j not in used
        ]

        if candidate_indices:

            conjugate_target = np.conj(eig_i)

            errors = np.array([
                np.abs(eigenvalues[j] - conjugate_target)
                for j in candidate_indices
            ])

            best_position = np.argmin(errors)
            j = candidate_indices[best_position]
            pair_error = errors[best_position]

            is_pair = np.isclose(
                eigenvalues[j],
                conjugate_target,
                atol=atol,
                rtol=rtol,
            )

        else:
            j = None
            pair_error = np.inf
            is_pair = False

        # ---------------------------------------------------------
        # Combine a conjugate pair
        # ---------------------------------------------------------
        if is_pair:

            # Positive frequency corresponds to positive discrete angle.
            if np.angle(eig_i) > 0:
                positive_idx = i
                negative_idx = j
            else:
                positive_idx = j
                negative_idx = i

            eig_positive = eigenvalues[positive_idx]

            # If the pair were exact:
            #
            # scaled_negative = conj(scaled_positive)
            #
            # and therefore this equals 2 * scaled_positive.
            phasor = (
                scaled_modes[:, positive_idx]
                + np.conj(scaled_modes[:, negative_idx])
            )

            output_phasors.append(phasor)
            output_eigenvalues.append(eig_positive)

            mode_information.append({
                "type": "conjugate_pair",
                "indices": (positive_idx, negative_idx),
                "pair_error": pair_error,
            })

            used.add(i)
            used.add(j)

        # ---------------------------------------------------------
        # Complex mode without a conjugate partner
        # ---------------------------------------------------------
        else:

            used.add(i)

            if include_unpaired:

                output_phasors.append(scaled_modes[:, i])
                output_eigenvalues.append(eig_i)

                mode_information.append({
                    "type": "unpaired_complex",
                    "indices": (i,),
                    "pair_error": pair_error,
                })

                warnings.warn(
                    f"DMD mode {i} with eigenvalue {eig_i} has no "
                    "conjugate partner.",
                    RuntimeWarning,
                )

    output_phasors = np.column_stack(output_phasors)
    output_eigenvalues = np.asarray(output_eigenvalues)

    continuous_eigenvalues = np.asarray(
        D_To_C_Eigenvalue_Converter(
            output_eigenvalues,
            dt=dt,
        )
    )

    return {
        "phasors": output_phasors,
        "discrete_eigenvalues": output_eigenvalues,
        "continuous_eigenvalues": continuous_eigenvalues,
        "mode_information": mode_information,
    }

def DMD_Mode_Pair_Old(dmd_out, tol=1e-8):

    modes, eigenvalues = dmd_out
    eigenvalues = np.asarray(eigenvalues)

    used = set()
    mode_indices = []

    # for each eigenvalue returned in decomposition
    for idx, eig in enumerate(eigenvalues):

        # if index already examined, skip
        if idx in used:
            continue

        # real/static/non-oscillatory mode
        # i.e. if no imaginary part, skip
        if abs(eig.imag) < tol:
            mode_indices.append(idx)
            used.add(idx)
            continue

        # look for unused conjugate partner
        conj_eig = np.conjugate(eig)

        candidates = []
        for j, eig_j in enumerate(eigenvalues):
            if j == idx or j in used:
                continue

            if np.isclose(eig_j, conj_eig, atol=tol, rtol=tol):
                candidates.append(j)

        if len(candidates) > 0:
            idx_pair = candidates[0]
            mode_indices.append((idx, idx_pair))
            used.add(idx)
            used.add(idx_pair)
        else:
            # no pair found
            mode_indices.append(idx)
            used.add(idx)

    return mode_indices

# evaluates total recomposition from DMD
def Total_Output_Evaluate(decomposition, 
                          times
                                  ):

    modes, eigenvalues = decomposition

    n_times = len(times) # how many time steps
    powers = np.arange(n_times) # array of time step number

    # essentially computing how each mode amplitude evolves through time
    dynamics = eigenvalues[:, None]**powers[None, :]

    # vectorised construction of modes through time with dynamics
    total_reconstructed_output = modes @ dynamics

    # make real as numerical reality likely leaves imaginary residuals
    return np.real(total_reconstructed_output)

# converts a discrete (i.e. DMD output eigenvalue) into a continuous type form
# see notes for explanation
def D_To_C_Eigenvalue_Converter(eigenvalue_d, dt, tol=1e-10):
    eigenvalue_d = np.asarray(eigenvalue_d, dtype=complex)

    sigma_c = np.log(np.abs(eigenvalue_d)) / dt
    omega_c = np.angle(eigenvalue_d) / dt

    # getting rid of floating point error:
    omega_c = np.where(np.abs(omega_c) < tol, 0.0, omega_c)
    sigma_c = np.where(np.abs(sigma_c) < tol, 0.0, sigma_c)

    return sigma_c + 1j * omega_c

# converts a continuous eigenvalue into a discrete type form
def C_To_D_Eigenvalue_Converter(eigenvalue_c, dt, tol=1e-10):
    eigenvalue_c = np.asarray(eigenvalue_c, dtype=complex)

    # first getting r and theta
    r = np.exp(eigenvalue_c.real * dt)
    theta = eigenvalue_c.imag * dt
    # then converting to x + iy
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # getting rid of floating point error:
    x = np.where(np.abs(x) < tol, 0.0, x)
    y = np.where(np.abs(y) < tol, 0.0, y)

    return x + 1j * y

# ---------------------------------------------------------------
# success metric related
# ---------------------------------------------------------------


# function for comparing 2 complex phasors of same shape
def Complex_Phasor_Compare(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return np.nan

    return np.abs(np.vdot(x, y)) / denominator

def Spatial_Relative_Error(input_phasor, output_phasor):
    return np.linalg.norm(output_phasor - input_phasor) / np.linalg.norm(input_phasor)

'''def period_relative_error(input_eigenvalue, output_eigenvalue):
    T_input = 2 * np.pi / np.abs(input_eigenvalue.imag)
    T_output = 2 * np.pi / np.abs(output_eigenvalue.imag)

    return abs(T_output - T_input) / T_input'''


def period_relative_error(input_eigenvalue, output_eigenvalues, tol=1e-12):
    omega_input = np.abs(input_eigenvalue.imag)
    omega_output = np.abs(np.asarray(output_eigenvalues).imag)

    # initialise all outputs as invalid
    errors = np.full_like(omega_output, np.inf, dtype=float)

    # if input frequency is zero, all period errors are undefined
    if np.isclose(omega_input, 0.0, atol=tol):
        return errors

    # valid output modes are non-static
    valid = ~np.isclose(omega_output, 0.0, atol=tol)

    errors[valid] = np.abs((omega_input / omega_output[valid]) - 1)

    return errors

# ---------------------------------------------------------------
# summary related
# ---------------------------------------------------------------


def plot_performance_summary(performance_summary, mode_labels=None, save_dir=FIG_DIR, show=True):
    """
    For each input mode in performance_summary, create one row figure with:
      1) eigenvalue plot in the complex plane
      2) spatial correlation vs normalised spatial error scatter plot

    Individual noise realisations are plotted in component colours.
    Component-wise averages across noise realisations are shown as black versions
    of the same component markers.
    """

    component_styles = {
        "sv": {
            "color": "blue",
            "marker": "s",
            "label": "SV"
        },
        "gnm_sv": {
            "color": "brown",
            "marker": "s",
            "label": "SV (gnm)"
        },
        "utheta": {
            "color": "green",
            "marker": "^",
            "label": r"$u_\theta$"
        },
        "uphi": {
            "color": "red",
            "marker": "o",
            "label": r"$u_\phi$"
        },
    }

    components = ["sv", "gnm_sv", "utheta", "uphi"]

    if mode_labels is None:
        mode_labels = [f"{i+1}" for i in range(len(performance_summary))]

    for i, wave_summary in enumerate(performance_summary):

        mode_label = mode_labels[i]
        true_eig = wave_summary["true_eigenvalue"]

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 6),
            dpi=200
        )

        ax_eig, ax_perf = axes

        # ==========================================================
        # LEFT PLOT: EIGENVALUES
        # ==========================================================

        ax_eig.scatter(
            np.real(true_eig),
            np.imag(true_eig),
            color="black",
            marker="x",
            s=140,
            linewidths=2.5,
            label="Truth"
        )

        for comp in components:

            style = component_styles[comp]
            eig_list = wave_summary[comp]["recovered_eigenvalues"]

            eig_real = []
            eig_imag = []

            for eig in eig_list:

                if eig is None:
                    continue

                r = np.real(eig)
                im = np.imag(eig)

                if np.isfinite(r) and np.isfinite(im):
                    eig_real.append(r)
                    eig_imag.append(im)

            eig_real = np.asarray(eig_real)
            eig_imag = np.asarray(eig_imag)

            if eig_real.size > 0:

                # individual realisations
                ax_eig.scatter(
                    eig_real,
                    eig_imag,
                    color=style["color"],
                    marker=style["marker"],
                    s=65,
                    alpha=0.65,
                    label=style["label"]
                )

                # average recovery
                ax_eig.scatter(
                    np.nanmean(eig_real),
                    np.nanmean(eig_imag),
                    color="black",
                    marker=style["marker"],
                    s=130,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"{style['label']} mean"
                )

        ax_eig.set_xlabel(r"Re($\omega$) [yr$^{-1}$]")
        ax_eig.set_ylabel(r"Im($\omega$) [rad yr$^{-1}$]")
        ax_eig.set_title("Recovered eigenvalues")
        ax_eig.grid(True)
        ax_eig.legend(fontsize=8)

        # ==========================================================
        # RIGHT PLOT: CORRELATION VS NORMALISED SPATIAL ERROR
        # ==========================================================

        # perfect spatial recovery
        ax_perf.scatter(
            1.0,
            0.0,
            color="black",
            marker="x",
            s=140,
            linewidths=2.5,
            label="Perfect match"
        )

        for comp in components:

            style = component_styles[comp]

            corr = np.asarray(
                wave_summary[comp]["spatial_correlations"],
                dtype=float
            )

            err = np.asarray(
                wave_summary[comp]["norm_spatial_errors"],
                dtype=float
            )

            mask = np.isfinite(corr) & np.isfinite(err)

            if np.any(mask):

                # individual realisations
                ax_perf.scatter(
                    corr[mask],
                    err[mask],
                    color=style["color"],
                    marker=style["marker"],
                    s=65,
                    alpha=0.65,
                    label=style["label"]
                )

                # average recovery
                ax_perf.scatter(
                    np.nanmean(corr[mask]),
                    np.nanmean(err[mask]),
                    color="black",
                    marker=style["marker"],
                    s=130,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"{style['label']} mean"
                )

        ax_perf.set_xlabel("Spatial correlation")
        ax_perf.set_ylabel("Normalised spatial error")
        ax_perf.set_title("Spatial recovery")
        ax_perf.grid(True)
        ax_perf.legend(fontsize=8)

        # sensible default limits
        ax_perf.set_xlim(0, 1.05)

        y_max = 1.0
        for comp in components:
            err = np.asarray(
                wave_summary[comp]["norm_spatial_errors"],
                dtype=float
            )
            if np.any(np.isfinite(err)):
                y_max = max(y_max, np.nanmax(err))

        ax_perf.set_ylim(0, 1.1 * y_max)

        fig.suptitle(f"Input mode {mode_label}", fontsize=14)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(
                os.path.join(save_dir, f"mode_{mode_label}_performance_summary.png"),
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close(fig)