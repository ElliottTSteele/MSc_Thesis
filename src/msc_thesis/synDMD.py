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
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.signal import (
    butter,
    buttord,
    filtfilt,
    freqz,
)

import warnings

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# import simulation setup
from src.msc_thesis.synSetup import *

# importing from pyDMD
from pydmd import HankelDMD, DMD, FbDMD, BOPDMD
from pydmd.preprocessing import hankel_preprocessing


# ---------------------------------------------------------------
# Universal limit choices applied across all
# ---------------------------------------------------------------

# By default, retain static modes and oscillatory modes of every period.
# Optional finite bounds remain supported by finalise_candidate_suite for
# experiments that explicitly request them.
candidate_period_min = None
candidate_period_max = None

# ---------------------------------------------------------
# covariance derived perturbation generation
# ---------------------------------------------------------

file_path = Path(f"{CHAOS_COV_DIR}/CHAOS_Cov_1997_2026_0806_SV.h5")

with h5py.File(file_path, "r") as cov_file:

    Cov_full = np.asarray(cov_file["Cnm"])
    print(f"C has shape dimensions: {np.shape(Cov_full)}")

def factor_time_covariance(covariance):

    covariance = np.asarray(covariance, dtype=np.float64)

    # NumPy performs this over all time steps.
    L = np.linalg.cholesky(covariance)

    # checks if needed:
    '''print(np.shape(L))
    print(np.allclose(Cov_full, L @ L.swapaxes(-1, -2)))'''

    return L

def Perturbation_Generate(gnm_input,
                          dt=dt_sample,
                          n_realisations=100,
                          seed=42, 
                          covariance_used=Cov_full, 
                          temporal_z = "independent", 
                          tau=5,
                          just_noise=False):
    
    Nt, Ng = np.shape(gnm_input)
    
    L = factor_time_covariance(covariance_used)

    rng = np.random.default_rng(seed)

    # Store the covariance-correlated perturbation realisations.
    perturbed_gauss = np.empty(
        (n_realisations, Nt, Ng),
        dtype=np.float64,
    )

    if temporal_z == "independent":
        # New independent z for every realisation and every time step.
        z = rng.standard_normal(
            (n_realisations, Nt, Ng)
        )

        for t in range(Nt):
            perturbed_gauss[:, t, :] = z[:, t, :] @ L[t].T

    elif temporal_z == "constant":
        # One latent z per realisation, reused at every time step.
        z_constant = rng.standard_normal(
            (n_realisations, Ng)
        )

        for t in range(Nt):
            perturbed_gauss[:, t, :] = z_constant @ L[t].T

    elif temporal_z == "ar1":
        if tau is None:
            raise ValueError(
                "tau must be provided when temporal_z='ar1'"
            )

        if tau <= 0:
            raise ValueError("tau must be positive")

        # AR(1) coefficient corresponding to the specified
        # e-folding correlation time tau.
        rho = np.exp(-dt / tau)

        # Innovation scaling ensures that every z[t] has stationary
        # marginal variance equal to one.
        innovation_scale = np.sqrt(1.0 - rho**2)

        z_ar1 = np.empty(
            (n_realisations, Nt, Ng),
            dtype=np.float64,
        )

        # Draw the initial state from the stationary N(0, I)
        # distribution.
        z_ar1[:, 0, :] = rng.standard_normal(
            (n_realisations, Ng)
        )

        # Generate temporally correlated latent vectors.
        for t in range(1, Nt):
            innovation = rng.standard_normal(
                (n_realisations, Ng)
            )

            z_ar1[:, t, :] = (
                rho * z_ar1[:, t - 1, :]
                + innovation_scale * innovation
            )

        # Apply the time-dependent Gauss covariance factors.
        for t in range(Nt):
            perturbed_gauss[:, t, :] = (
                z_ar1[:, t, :] @ L[t].T
            )

    else:
        raise ValueError(
            "temporal_z must be one of "
            "'independent', 'constant', or 'ar1'"
        )

    noise_gauss = perturbed_gauss.copy()

    if just_noise:
        return(perturbed_gauss)

    # Add the mean CHAOS model.
    perturbed_gauss += gnm_input[None, :, :]

    # return ensemble of original models + noise realisations
    return(perturbed_gauss)



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

# ---------------------------------------------------------
# BUIDLING DMD CLASS OBJECT 
# ---------------------------------------------------------

def Build_Exact_DMD(svd_rank=-1, fbdmd=False):

    return DMD(
        svd_rank=svd_rank,
        exact=True,
        forward_backward=fbdmd,
    )

def Build_BOPDMD(svd_rank=-1, bopdmd=False, bopdmd_params=None):

    if bopdmd:
        return BOPDMD(
                svd_rank=svd_rank,
                num_trials=bopdmd_params["num_trials"],
                trial_size=bopdmd_params["trial_size"],
                remove_bad_bags=bopdmd_params["remove_bad_bags"],
                eig_constraints={"conjugate_pairs", "imag"},
                varpro_opts_dict={
                    "maxiter": 100,
                    "tol": 1e-1,
                    "verbose": False,
                },
            )
    else:
        return BOPDMD(
                svd_rank=svd_rank,
                eig_constraints={"conjugate_pairs", "imag"},
            )


# ---------------------------------------------------------
# CONVERT DMD OUTPUT -> PHYSICAL MODES
# ---------------------------------------------------------

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

# physicalise phasor by taking first embedded state vector
def Physicalise_Phasor(embedded_phasor, embedding_d):

    embedded_length = len(embedded_phasor)
    physical_length = int(embedded_length / embedding_d)
    physical_phasor = embedded_phasor[:physical_length]

    return(physical_phasor)

# takes in dmd object, and time spacing
def DMD_Mode_Process(
    dmd,
    dmd_type_used,
    dt,
    hankel,
    embedding_d,
    atol=1e-4,
    rtol=1e-4
    ):

    # gets eigenvalues, modes and amplitues (raw output)
    eigenvalues = np.asarray(dmd.eigs, dtype=complex)
    modes = np.asarray(dmd.modes, dtype=complex)
    amplitudes = np.asarray(dmd.amplitudes, dtype=complex)

    # Complete dynamically scaled modal coefficients b_j phi_j.
    scaled_modes = modes * amplitudes[np.newaxis, :]

    # initialise objects to handle pairing algorithm
    used = set()
    output_phasors = []
    output_eigenvalues = []
    mode_information = []

    # for each eigenvalue in modes
    for i, eig_i in enumerate(eigenvalues):

        # if eigenvalue already 'ticked off' from examination, skip
        if i in used:
            continue

        # ---------------------------------------------------------
        # Real/non-oscillatory eigenvalue - i.e. if eigenvalue has no frequency
        # ---------------------------------------------------------
        if np.isclose(eig_i.imag, 0.0, atol=atol, rtol=rtol):

            output_phasors.append(scaled_modes[:, i])
            output_eigenvalues.append(
                complex(eig_i.real, 0.0)
            )

            mode_information.append({
                "type": "real",
                "indices": (i,),
                "pair_error": np.nan,
            })

            used.add(i)
            continue

        # ---------------------------------------------------------
        # Find the closest unused conjugate eigenvalue - if mode not static
        # ---------------------------------------------------------

        # conjugate candidates = not the current mode and not those already examined
        candidate_indices = [
            j
            for j in range(eigenvalues.size)
            if j != i and j not in used
        ]

        # if viable candidates exist
        if candidate_indices:

            # the desired conjugate value
            conjugate_target = np.conj(eig_i)

            errors = np.array([
                np.abs(eigenvalues[j] - conjugate_target)
                for j in candidate_indices
            ])

            best_position = np.argmin(errors)
            j = candidate_indices[best_position]
            pair_error = errors[best_position]

            # if eigenvalue conjugate found, then there is a match
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

        # if conjugate match found
        if is_pair:

            # Positive frequency corresponds to positive discrete angle.
            # i.e. which one of pair has positive frequency
            if np.angle(eig_i) > 0:
                positive_idx = i
                negative_idx = j
            else:
                positive_idx = j
                negative_idx = i

            # taking positive frequency eigenvalue
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

    # physicalise hankel embedded state vectors
    if hankel:
        output_phasors = [Physicalise_Phasor(phasor, embedding_d) for phasor in output_phasors]

    # storing continuous vs discrete eigenvalues appropriately
    # (DMD output type varies with method)

    if dmd_type_used == 'exact':

        discrete_eigenvalues = output_eigenvalues

        continuous_eigenvalues = np.asarray(
            D_To_C_Eigenvalue_Converter(
                output_eigenvalues,
                dt=dt,
            )
        )

    elif dmd_type_used == 'opdmd':

        continuous_eigenvalues = output_eigenvalues

        discrete_eigenvalues = np.asarray(
            C_To_D_Eigenvalue_Converter(
                output_eigenvalues,
                dt=dt,
            )
        )

    # getting periods from continuous eigenvalues
    periods = [2.0 * np.pi / np.abs(eig.imag) for eig in continuous_eigenvalues]

    return {
        "phasors": output_phasors,
        "discrete_eigenvalues": discrete_eigenvalues,
        "continuous_eigenvalues": continuous_eigenvalues,
        "periods": periods,
        "mode_information": mode_information
    }


# ---------------------------------------------------------------
# success metric related
# ---------------------------------------------------------------


# returns quantiles of data set
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

def SV_Grid_Phasor_Record_Power(
    phasor,
    eigenvalue,
    evaluation_times,
    spatial_weights=None,
):
    """
    Return mean-square SV power over a specified finite record.

    For

        x(t) = Re[phasor * exp(lambda*t)],

    the metric is

        mean_t[sum(w * x(t)^2) / sum(w)].

    ``evaluation_times`` must be expressed relative to the reference time of
    the supplied phasor. The calculation is exact at those discrete times and
    includes initial amplitude, phase, oscillation, and exponential
    growth/decay. Static modes are handled by the same expression.

    The implementation evaluates the weighted spatial quadratic form
    analytically, avoiding construction of an ``(n_time, n_space)`` array.
    Supplying spherical cell-area weights such as sin(colatitude) gives an
    area-weighted global mean square. Equal weights are used if omitted.
    """

    phasor = np.asarray(
        phasor,
        dtype=complex,
    ).ravel()

    if phasor.size == 0:
        raise ValueError(
            "phasor must contain at least one spatial value."
        )

    if not np.all(
        np.isfinite(phasor.real)
        & np.isfinite(phasor.imag)
    ):
        return np.nan

    evaluation_times = np.asarray(
        evaluation_times,
        dtype=float,
    ).ravel()

    if evaluation_times.size == 0:
        raise ValueError(
            "evaluation_times must contain at least one time."
        )

    if not np.all(
        np.isfinite(evaluation_times)
    ):
        raise ValueError(
            "evaluation_times must be finite."
        )

    if spatial_weights is None:
        weights = np.ones(
            phasor.size,
            dtype=float,
        )
    else:
        weights = np.asarray(
            spatial_weights,
            dtype=float,
        ).ravel()

        if weights.size != phasor.size:
            raise ValueError(
                "spatial_weights must contain one weight per phasor "
                f"value: {weights.size} vs {phasor.size}."
            )

    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError(
            "spatial_weights must be finite and non-negative."
        )

    weight_sum = np.sum(weights)

    if weight_sum <= 0.0:
        raise ValueError(
            "spatial_weights must have a positive sum."
        )

    eigenvalue = complex(
        eigenvalue
    )

    if not (
        np.isfinite(eigenvalue.real)
        and np.isfinite(eigenvalue.imag)
    ):
        return np.nan

    weighted_abs_square = (
        np.sum(
            weights
            * np.abs(phasor)**2
        )
        / weight_sum
    )

    weighted_complex_square = (
        np.sum(
            weights
            * phasor**2
        )
        / weight_sum
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        temporal_factor = np.exp(
            2.0
            * eigenvalue
            * evaluation_times
        )

        record_power = (
            0.5
            * np.exp(
                2.0
                * eigenvalue.real
                * evaluation_times
            )
            * weighted_abs_square
            + 0.5
            * np.real(
                weighted_complex_square
                * temporal_factor
            )
        )

    if np.any(
        np.isposinf(record_power)
    ):
        return np.inf

    if not np.all(
        np.isfinite(record_power)
    ):
        return np.nan

    mean_power = float(
        np.mean(record_power)
    )

    # Roundoff can produce an extremely small negative value when the true
    # mean square is zero.
    if (
        mean_power < 0.0
        and np.isclose(
            mean_power,
            0.0,
            atol=1e-14,
            rtol=0.0,
        )
    ):
        mean_power = 0.0

    return mean_power

def Mode_Quality_Factor(
    eigenvalues,
    static_tol=1e-12,
):
    """
    Return Q = |omega| / (2|sigma|) for continuous eigenvalues.

    Static modes return NaN. Undamped oscillatory modes return infinity.
    """

    eigenvalues = np.asarray(
        eigenvalues,
        dtype=complex,
    )

    omega = np.abs(eigenvalues.imag)
    sigma = np.abs(eigenvalues.real)

    quality = np.full(
        eigenvalues.shape,
        np.nan,
        dtype=float,
    )

    oscillatory = omega > static_tol
    damped = oscillatory & (sigma > 0.0)
    undamped = oscillatory & (sigma == 0.0)

    quality[damped] = (
        omega[damped]
        / (2.0 * sigma[damped])
    )
    quality[undamped] = np.inf

    return quality

# comparing the complex phasor similarity of candidates vs true phasor
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
                "candidate_index": int(idx),
                "similarity": float(similarity),
                "eigenvalue": candidate_eigs[idx],
                "recovered_period": float(
                    candidate_periods[idx]
                ),
            }

    return best

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

# period error
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
