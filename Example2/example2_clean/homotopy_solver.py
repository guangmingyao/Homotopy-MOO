"""
Homotopy continuation code for the degenerate, disconnected, nonconvex
many-objective test problem with equality constraints.

Problem:
    minimize F(x) = (f1, f2, f3, f4)
    subject to h1(x)=x4=0,
               h2(x)=x3^2 - x2^2=0,
               g0(x)=(x1^2-1)^2+x2^2-0.08 <= 0,
               -2 <= x1 <= 2,  -1 <= x2,x3,x4 <= 1.

The code follows the same structure as the user's two-objective script, but it is
written in vector/matrix form to handle 4 objectives, 2 equalities, and multiple
inequality constraints.
"""

import itertools
import os
from pathlib import Path
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

# -----------------------------------------------------------------------------
# Problem dimensions
# -----------------------------------------------------------------------------
N = 4   # variables: x1,x2,x3,x4
M = 4   # objectives
Q = 2   # equalities: h1,h2
P = 9   # inequalities: nonlinear g0 plus 8 bound constraints

# -----------------------------------------------------------------------------
# Output and run controls
# -----------------------------------------------------------------------------
RESULTS_DIR = Path("homotopy_manyobj_results")

FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"




# Use FAST_MODE=True while debugging. For final paper-quality runs, set False
# or call solve_homotopy_manyobj(fast_mode=False).
# FAST_MODE = True
# SAVE_PDF = False  # PNG files are faster and sufficient for quick debugging.

FAST_MODE = False  
SAVE_PDF = True  # PNG files are faster and sufficient for quick debugging.
# -----------------------------------------------------------------------------
# Objective, equality, and inequality functions
# -----------------------------------------------------------------------------
def objectives(x):
    x1, x2, x3, x4 = x
    f1 = (x1 + 1.0) ** 2 + x2 ** 2 + 0.05 * np.sin(8.0 * x1)
    f2 = (x1 - 1.0) ** 2 + x2 ** 2 + 0.05 * np.sin(8.0 * x1)
    f3 = x2 ** 2 + 0.1 * (1.0 - np.cos(6.0 * x1))
    f4 = (x1 ** 2 - 1.0) ** 2 + x2 ** 2
    return np.array([f1, f2, f3, f4], dtype=float)


def equalities(x):
    x1, x2, x3, x4 = x
    return np.array([
        x4,                 # h1(x)=0
        x3 ** 2 - x2 ** 2   # h2(x)=0
    ], dtype=float)


def inequalities(x):
    """All inequalities are written as g_i(x) <= 0."""
    x1, x2, x3, x4 = x
    g0 = (x1 ** 2 - 1.0) ** 2 + x2 ** 2 - 0.08

    # box bounds in the form lower - x <= 0 and x - upper <= 0
    return np.array([
        g0,
        -2.0 - x1, x1 - 2.0,
        -1.0 - x2, x2 - 1.0,
        -1.0 - x3, x3 - 1.0,
        -1.0 - x4, x4 - 1.0,
    ], dtype=float)


# -----------------------------------------------------------------------------
# Gradients and Hessians
# -----------------------------------------------------------------------------
def grad_objectives(x):
    """Return Df as an M x N matrix; row i is grad f_i."""
    x1, x2, x3, x4 = x
    Df = np.zeros((M, N), dtype=float)

    Df[0, :] = [2.0 * (x1 + 1.0) + 0.4 * np.cos(8.0 * x1), 2.0 * x2, 0.0, 0.0]
    Df[1, :] = [2.0 * (x1 - 1.0) + 0.4 * np.cos(8.0 * x1), 2.0 * x2, 0.0, 0.0]
    Df[2, :] = [0.6 * np.sin(6.0 * x1), 2.0 * x2, 0.0, 0.0]
    Df[3, :] = [4.0 * x1 * (x1 ** 2 - 1.0), 2.0 * x2, 0.0, 0.0]
    return Df


def hess_objectives(x):
    """Return a list of M Hessian matrices."""
    x1, x2, x3, x4 = x
    H = []

    H1 = np.zeros((N, N)); H1[0, 0] = 2.0 - 3.2 * np.sin(8.0 * x1); H1[1, 1] = 2.0
    H2 = np.zeros((N, N)); H2[0, 0] = 2.0 - 3.2 * np.sin(8.0 * x1); H2[1, 1] = 2.0
    H3 = np.zeros((N, N)); H3[0, 0] = 3.6 * np.cos(6.0 * x1); H3[1, 1] = 2.0
    H4 = np.zeros((N, N)); H4[0, 0] = 12.0 * x1 ** 2 - 4.0; H4[1, 1] = 2.0

    H.extend([H1, H2, H3, H4])
    return H


def grad_equalities(x):
    """Return Dh as a Q x N matrix; row j is grad h_j."""
    x1, x2, x3, x4 = x
    Dh = np.zeros((Q, N), dtype=float)
    Dh[0, :] = [0.0, 0.0, 0.0, 1.0]
    Dh[1, :] = [0.0, -2.0 * x2, 2.0 * x3, 0.0]
    return Dh


def hess_equalities(x):
    Hh1 = np.zeros((N, N))
    Hh2 = np.zeros((N, N)); Hh2[1, 1] = -2.0; Hh2[2, 2] = 2.0
    return [Hh1, Hh2]


def grad_inequalities(x):
    """Return Dg as a P x N matrix; row i is grad g_i."""
    x1, x2, x3, x4 = x
    Dg = np.zeros((P, N), dtype=float)

    Dg[0, :] = [4.0 * x1 * (x1 ** 2 - 1.0), 2.0 * x2, 0.0, 0.0]
    Dg[1, :] = [-1.0, 0.0, 0.0, 0.0]  # -2 - x1 <= 0
    Dg[2, :] = [ 1.0, 0.0, 0.0, 0.0]  # x1 - 2 <= 0
    Dg[3, :] = [0.0, -1.0, 0.0, 0.0]
    Dg[4, :] = [0.0,  1.0, 0.0, 0.0]
    Dg[5, :] = [0.0, 0.0, -1.0, 0.0]
    Dg[6, :] = [0.0, 0.0,  1.0, 0.0]
    Dg[7, :] = [0.0, 0.0, 0.0, -1.0]
    Dg[8, :] = [0.0, 0.0, 0.0,  1.0]
    return Dg


def hess_inequalities(x):
    x1, x2, x3, x4 = x
    Hg = []
    H0 = np.zeros((N, N)); H0[0, 0] = 12.0 * x1 ** 2 - 4.0; H0[1, 1] = 2.0
    Hg.append(H0)
    for _ in range(P - 1):
        Hg.append(np.zeros((N, N)))
    return Hg


# -----------------------------------------------------------------------------
# Homotopy H and Jacobian DH
# -----------------------------------------------------------------------------
def H_map(x, x0, lam, lam0, u, u0, v, mu):
    x = np.asarray(x, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    lam = np.asarray(lam, dtype=float)
    lam0 = np.asarray(lam0, dtype=float)
    u = np.asarray(u, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    v = np.asarray(v, dtype=float)

    Df = grad_objectives(x)
    Dg = grad_inequalities(x)
    Dh = grad_equalities(x)
    h = equalities(x)
    g = inequalities(x)
    g0 = inequalities(x0)

    # Stationarity homotopy
    H1 = (1.0 - mu) * (Df.T @ lam + Dg.T @ u) + Dh.T @ v + mu * (x - x0)

    # Equality constraints
    H2 = h

    # Complementarity homotopy for all inequalities
    H3 = u * g - mu * u0 * g0

    # Objective weight normalization homotopy
    lpow = np.power(np.clip(lam, 0.0, None), 1.5)
    l0pow = np.power(np.clip(lam0, 0.0, None), 1.5)
    H4_common = (1.0 - mu) * (1.0 - np.sum(lam))
    H4 = H4_common * np.ones(M) - mu * (lpow - l0pow)

    return np.concatenate([H1, H2, H3, H4])


def DH_map(x, x0, lam, lam0, u, u0, v, mu):
    x = np.asarray(x, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    lam = np.asarray(lam, dtype=float)
    lam0 = np.asarray(lam0, dtype=float)
    u = np.asarray(u, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    v = np.asarray(v, dtype=float)

    Df = grad_objectives(x)
    Dg = grad_inequalities(x)
    Dh = grad_equalities(x)
    g = inequalities(x)
    g0 = inequalities(x0)

    Hf = hess_objectives(x)
    Hg = hess_inequalities(x)
    Hh = hess_equalities(x)

    # Hessian contribution in stationarity row
    D2L = np.zeros((N, N), dtype=float)
    for i in range(M):
        D2L += lam[i] * Hf[i]
    for j in range(P):
        D2L += u[j] * Hg[j]
    for k in range(Q):
        D2L += v[k] * Hh[k]

    dH1_dx = (1.0 - mu) * (sum(lam[i] * Hf[i] for i in range(M)) +
                            sum(u[j] * Hg[j] for j in range(P))) + \
              sum(v[k] * Hh[k] for k in range(Q)) + mu * np.eye(N)
    dH1_dlam = (1.0 - mu) * Df.T                    # N x M
    dH1_du = (1.0 - mu) * Dg.T                      # N x P
    dH1_dv = Dh.T                                   # N x Q
    dH1_dmu = (-(Df.T @ lam + Dg.T @ u) + (x - x0)).reshape(N, 1)
    R1 = np.hstack([dH1_dx, dH1_dlam, dH1_du, dH1_dv, dH1_dmu])

    dH2_dx = Dh
    R2 = np.hstack([dH2_dx, np.zeros((Q, M)), np.zeros((Q, P)), np.zeros((Q, Q)), np.zeros((Q, 1))])

    dH3_dx = u[:, None] * Dg
    dH3_dlam = np.zeros((P, M))
    dH3_du = np.diag(g)
    dH3_dv = np.zeros((P, Q))
    dH3_dmu = (-u0 * g0).reshape(P, 1)
    R3 = np.hstack([dH3_dx, dH3_dlam, dH3_du, dH3_dv, dH3_dmu])

    sqrt_l = np.sqrt(np.clip(lam, 0.0, None))
    dH4_dlam = -(1.0 - mu) * np.ones((M, M)) - mu * np.diag(1.5 * sqrt_l)
    lpow = np.power(np.clip(lam, 0.0, None), 1.5)
    l0pow = np.power(np.clip(lam0, 0.0, None), 1.5)
    dH4_dmu = (-(1.0 - np.sum(lam)) * np.ones(M) - (lpow - l0pow)).reshape(M, 1)
    R4 = np.hstack([np.zeros((M, N)), dH4_dlam, np.zeros((M, P)), np.zeros((M, Q)), dH4_dmu])

    return np.vstack([R1, R2, R3, R4])


# -----------------------------------------------------------------------------
# Linear algebra and continuation helpers
# -----------------------------------------------------------------------------
def unpack_omega(omega):
    x = omega[:N]
    lam = omega[N:N+M]
    u = omega[N+M:N+M+P]
    v = omega[N+M+P:N+M+P+Q]
    mu = omega[-1]
    return x, lam, u, v, mu


def make_omega(x, lam, u, v, mu):
    return np.concatenate([x, lam, u, v, [mu]])


def nullspace_right_vector(A):
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    xi = Vt[-1, :]
    nrm = np.linalg.norm(xi)
    if nrm == 0 or not np.isfinite(nrm):
        raise np.linalg.LinAlgError("Zero or nonfinite null vector returned by SVD.")
    return xi / nrm


def safe_solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def omega_residual(omega, x0, lam0, u0):
    x, lam, u, v, mu = unpack_omega(omega)
    return H_map(x, x0, lam, lam0, u, u0, v, mu)


def omega_jacobian(omega, x0, lam0, u0):
    x, lam, u, v, mu = unpack_omega(omega)
    return DH_map(x, x0, lam, lam0, u, u0, v, mu)


def stabilize_omega(omega):
    """Soft numerical safeguards. Do not project x; only keep dual variables finite."""
    omega = np.asarray(omega, dtype=float).copy()
    omega[N:N+M] = np.maximum(omega[N:N+M], 1e-12)
    omega[N+M:N+M+P] = np.maximum(omega[N+M:N+M+P], 1e-12)
    omega = np.nan_to_num(omega, nan=0.0, posinf=1e6, neginf=-1e6)
    return omega


def pseudo_arclength_corrector(omega_pred, omega_old, tangent, x0, lam0, u0,
                               max_newton=12, tol=1e-9, line_search=True):
    """
    Newton corrector for the pseudo-arclength system

        H(omega)=0,
        tangent^T(omega - omega_pred)=0.

    This replaces the determinant-orientation correction and is much more stable
    near vertical folds in mu or weakly singular path regions.
    """
    omega = stabilize_omega(omega_pred)
    tangent = tangent / max(np.linalg.norm(tangent), 1e-15)

    for it in range(max_newton):
        Hval = omega_residual(omega, x0, lam0, u0)
        arc = np.array([np.dot(tangent, omega - omega_pred)])
        R = np.concatenate([Hval, arc])
        Rnorm = np.linalg.norm(R)
        if Rnorm < tol:
            return omega, True, it + 1, Rnorm

        J = omega_jacobian(omega, x0, lam0, u0)
        J_aug = np.vstack([J, tangent.reshape(1, -1)])
        delta = safe_solve(J_aug, R)

        # Damped Newton helps prevent the corrector from jumping branches.
        alpha = 1.0
        if line_search:
            best = omega - delta
            best_norm = np.inf
            for _ in range(8):
                trial = stabilize_omega(omega - alpha * delta)
                H_trial = omega_residual(trial, x0, lam0, u0)
                arc_trial = np.array([np.dot(tangent, trial - omega_pred)])
                trial_norm = np.linalg.norm(np.concatenate([H_trial, arc_trial]))
                if trial_norm < Rnorm:
                    omega = trial
                    break
                if trial_norm < best_norm:
                    best = trial
                    best_norm = trial_norm
                alpha *= 0.5
            else:
                omega = best
        else:
            omega = stabilize_omega(omega - delta)

    Hval = omega_residual(omega, x0, lam0, u0)
    arc = np.array([np.dot(tangent, omega - omega_pred)])
    Rnorm = np.linalg.norm(np.concatenate([Hval, arc]))
    return omega, Rnorm < 10 * tol, max_newton, Rnorm


def initial_tangent(omega, x0, lam0, u0, previous_tangent=None):
    J = omega_jacobian(omega, x0, lam0, u0)
    tangent = nullspace_right_vector(J)

    # Prefer motion toward smaller mu at the beginning.
    if previous_tangent is None:
        if tangent[-1] > 0:
            tangent = -tangent
    else:
        # Keep a continuous orientation instead of determinant orientation.
        if np.dot(tangent, previous_tangent) < 0:
            tangent = -tangent
    return tangent / max(np.linalg.norm(tangent), 1e-15)



# -----------------------------------------------------------------------------
# Output, reference set, and quality indicator helpers
# -----------------------------------------------------------------------------
def ensure_output_dirs(results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    data_dir = results_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, fig_dir, data_dir


def save_current_figure(fig_dir, basename, dpi=220, show=False, save_pdf=SAVE_PDF):
    """Save the active matplotlib figure, then close it to avoid memory buildup."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{basename}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        plt.savefig(fig_dir / f"{basename}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def component_side(component):
    """Collapse left_plus/left_minus and right_plus/right_minus into left/right."""
    component = str(component)
    if component.startswith("left"):
        return "left"
    if component.startswith("right"):
        return "right"
    return component


def nondominated_mask(F, eps=1e-12):
    """Return mask of nondominated points for a minimization problem."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by_i = np.all(F[i] <= F + eps, axis=1) & np.any(F[i] < F - eps, axis=1)
        dominated_by_i[i] = False
        keep[dominated_by_i] = False
        if np.any(np.all(F[keep] <= F[i] + eps, axis=1) & np.any(F[keep] < F[i] - eps, axis=1)):
            keep[i] = False
    return keep


def filter_nondominated_df(df, obj_cols=("f1", "f2", "f3", "f4")):
    if len(df) == 0:
        return df.copy()
    mask = nondominated_mask(df[list(obj_cols)].to_numpy(float))
    return df.loc[mask].copy()


def normalize_points(F, ideal=None, nadir=None):
    F = np.asarray(F, dtype=float)
    if ideal is None:
        ideal = np.min(F, axis=0)
    if nadir is None:
        nadir = np.max(F, axis=0)
    denom = np.maximum(nadir - ideal, 1e-12)
    return (F - ideal) / denom, ideal, nadir


def nearest_distances(A, B, chunk=2000):
    """Euclidean nearest-neighbor distances from each point in A to set B."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if len(A) == 0 or len(B) == 0:
        return np.array([], dtype=float)
    out = np.empty(len(A), dtype=float)
    for i in range(0, len(A), chunk):
        D2 = np.sum((A[i:i+chunk, None, :] - B[None, :, :])**2, axis=2)
        out[i:i+chunk] = np.sqrt(np.min(D2, axis=1))
    return out


def averaged_hausdorff_distance(A, B):
    """
    Averaged Hausdorff distance between two point sets:
        max(mean_{a in A} d(a,B), mean_{b in B} d(b,A)).
    Lower is better. Here it is computed in normalized objective space.
    """
    if len(A) == 0 or len(B) == 0:
        return np.inf, np.inf, np.inf
    d_ab = nearest_distances(A, B)
    d_ba = nearest_distances(B, A)
    gd = float(np.mean(d_ab))
    igd = float(np.mean(d_ba))
    ahd = float(max(gd, igd))
    return ahd, gd, igd


def hypervolume_recursive_2d(P2, ref2):
    """Exact 2D hypervolume for nondominated minimization points."""
    if len(P2) == 0:
        return 0.0
    P2 = P2[nondominated_mask(P2)]
    P2 = P2[np.argsort(P2[:, 0])]
    hv = 0.0
    prev_y = ref2[1]
    for x, y in P2:
        width = max(ref2[0] - x, 0.0)
        height = max(prev_y - y, 0.0)
        hv += width * height
        prev_y = min(prev_y, y)
    return float(max(hv, 0.0))


def hypervolume_recursive(P, ref):
    """Exact hypervolume for small/medium nondominated sets in minimization form."""
    P = np.asarray(P, dtype=float)
    ref = np.asarray(ref, dtype=float)
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[nondominated_mask(P)]
    d = P.shape[1]
    if d == 1:
        return float(max(ref[0] - np.min(P[:, 0]), 0.0))
    if d == 2:
        return hypervolume_recursive_2d(P, ref)
    order = np.argsort(P[:, 0])
    P = P[order]
    hv = 0.0
    prev = ref[0]
    # Sweep from largest to smallest first coordinate.
    for i in range(len(P) - 1, -1, -1):
        x = P[i, 0]
        width = max(prev - x, 0.0)
        if width > 0:
            slice_points = P[:i+1, 1:]
            hv += width * hypervolume_recursive(slice_points, ref[1:])
            prev = x
    return float(max(hv, 0.0))

def hypervolume_monte_carlo(P, ref, lower=None, n_samples=200000, seed=12345, chunk=50000):
    """
    Deterministic Monte Carlo hypervolume estimate for minimization.
    This is much faster than exact recursive HV for large 4-objective reference sets.
    """
    P = np.asarray(P, dtype=float)
    ref = np.asarray(ref, dtype=float)
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[nondominated_mask(P)]
    if lower is None:
        lower = np.minimum(np.min(P, axis=0), np.zeros(P.shape[1]))
    lower = np.asarray(lower, dtype=float)
    box = ref - lower
    volume = float(np.prod(box))
    rng = np.random.default_rng(seed)
    dominated_count = 0
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        S = lower + rng.random((m, P.shape[1])) * box
        dom = np.zeros(m, dtype=bool)
        # A sample is dominated if some computed point is <= it in all objectives.
        for i in range(0, len(P), 1000):
            dom |= np.any(np.all(P[i:i+1000, None, :] <= S[None, :, :], axis=2), axis=0)
            if np.all(dom):
                break
        dominated_count += int(np.sum(dom))
        done += m
    return volume * dominated_count / n_samples


def generate_reference_front(n_x1=500, n_x2=120):
    """
    Build a dense feasible reference approximation by sampling the exact feasible
    structure: x4=0, x3=+/-x2, and g0<=0. This is used only for quality metrics.
    """
    points = []
    for center in (-1.0, 1.0):
        x1_grid = np.linspace(center - 0.35, center + 0.35, n_x1)
        for x1 in x1_grid:
            y2 = 0.08 - (x1**2 - 1.0)**2
            if y2 < 0:
                continue
            xmax = np.sqrt(y2)
            for x2 in np.linspace(-xmax, xmax, n_x2):
                for sgn in (-1.0, 1.0):
                    x = np.array([x1, x2, sgn*x2, 0.0])
                    points.append(objectives(x))
    F = np.asarray(points, dtype=float)
    F = F[nondominated_mask(F)]
    return F


def compute_quality_indicators(df, data_dir=DATA_DIR, reference_F=None):
    """
    Compute quality indicators for the obtained solution set.
    Metrics are computed in normalized objective space to avoid scale bias.
    """
    obj_cols = ["f1", "f2", "f3", "f4"]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    feasible = df[(df["success"] == True) & (df["max_g"] <= 1e-7) &
                  (df["h1"].abs() <= 1e-7) & (df["h2"].abs() <= 1e-7)].copy()
    approx = filter_nondominated_df(feasible, obj_cols)

    if reference_F is None:
        reference_F = generate_reference_front(n_x1=100, n_x2=25)
    reference_F = reference_F[nondominated_mask(reference_F)]

    if len(approx) == 0:
        metrics = {
            "n_total_rows": len(df), "n_success": int(df["success"].sum()),
            "n_feasible_success": len(feasible), "n_approx_nondominated": 0,
            "n_reference_nondominated": len(reference_F),
            "hypervolume_normalized": np.nan,
            "reference_hypervolume_normalized": np.nan,
            "hypervolume_ratio": np.nan,
            "averaged_hausdorff_distance": np.inf,
            "generational_distance": np.inf,
            "inverted_generational_distance": np.inf,
            "max_equality_residual": np.nan,
            "max_inequality_violation": np.nan,
            "median_Hnorm": np.nan,
        }
        pd.DataFrame([metrics]).to_csv(data_dir / "quality_indicators.csv", index=False)
        return metrics, approx, reference_F

    F_approx = approx[obj_cols].to_numpy(float)
    F_all = np.vstack([F_approx, reference_F])
    _, ideal, nadir = normalize_points(F_all)
    A, _, _ = normalize_points(F_approx, ideal, nadir)
    R, _, _ = normalize_points(reference_F, ideal, nadir)

    # Reference point slightly worse than the normalized nadir.
    ref = np.ones(M) * 1.10
    A_nd = A[nondominated_mask(A)]
    R_nd = R[nondominated_mask(R)]
    hv = hypervolume_monte_carlo(A_nd, ref, n_samples=300000, seed=12345)
    hv_ref = hypervolume_monte_carlo(R_nd, ref, n_samples=300000, seed=12345)
    ahd, gd, igd = averaged_hausdorff_distance(A_nd, R_nd)

    max_eq = float(max(approx["h1"].abs().max(), approx["h2"].abs().max()))
    max_ineq = float(np.maximum(approx["max_g"].max(), 0.0))
    metrics = {
        "n_total_rows": int(len(df)),
        "n_success": int(df["success"].sum()),
        "n_feasible_success": int(len(feasible)),
        "n_approx_nondominated": int(len(approx)),
        "n_reference_nondominated": int(len(reference_F)),
        "hypervolume_normalized": float(hv),
        "reference_hypervolume_normalized": float(hv_ref),
        "hypervolume_ratio": float(hv / hv_ref) if hv_ref > 0 else np.nan,
        "averaged_hausdorff_distance": float(ahd),
        "generational_distance": float(gd),
        "inverted_generational_distance": float(igd),
        "max_equality_residual": max_eq,
        "max_inequality_violation": max_ineq,
        "median_Hnorm": float(approx["Hnorm"].median()),
    }

    pd.DataFrame([metrics]).to_csv(data_dir / "quality_indicators.csv", index=False)
    approx.to_csv(data_dir / "nondominated_successful_solutions.csv", index=False)
    pd.DataFrame(reference_F, columns=obj_cols).to_csv(data_dir / "reference_front_sample.csv", index=False)
    return metrics, approx, reference_F

# -----------------------------------------------------------------------------
# Weight generation and robust solver
# -----------------------------------------------------------------------------
def simplex_weights_4obj(level=20, lower=0.02):
    """Generate positive weights on the 4-objective simplex."""
    raw = []
    for a in range(1, level):
        for b in range(1, level):
            for c in range(1, level):
                d = level - a - b - c
                if d >= 1:
                    w = np.array([a, b, c, d], dtype=float) / level
                    if np.all(w >= lower):
                        raw.append(w)
    if not raw:
        raw = [np.array([0.25, 0.25, 0.25, 0.25])]
    return raw


def branch_restarts(compact=False):
    """
    Multiple feasible initial points on both disconnected components and on both
    equality branches x3=+x2 and x3=-x2. This is essential because local
    continuation cannot jump between disconnected components.
    """
    starts = []
    x2_values = [0.08] if compact else [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, -0.04, -0.08, -0.12, -0.15]
    for name, center in [("left", -1.0), ("right", 1.0)]:
        for x2 in x2_values:
            # Keep points strictly inside g0 <= 0.
            if (center**2 - 1.0)**2 + x2**2 < 0.08:
                starts.append((f"{name}_plus",  np.array([center, x2,  x2, 0.0])))
                starts.append((f"{name}_minus", np.array([center, x2, -x2, 0.0])))
    return starts


def run_single_path(component, x0, lam0, u0_scale=0.05,
                    step0=0.08, step_min=1e-4, step_max=0.12,
                    eps_mu=1e-4, max_steps=600):
    """
    Robust path tracker with:
      1. adaptive step size,
      2. multiple damped Newton corrector iterations,
      3. pseudo-arclength continuation,
      4. branch restarting from externally supplied x0.
    """
    u0 = u0_scale * np.ones(P)
    v0 = np.zeros(Q)                  # makes H(omega0)=0 at mu=1
    omega = make_omega(x0.copy(), lam0.copy(), u0.copy(), v0.copy(), 1.0)
    omega = stabilize_omega(omega)

    step = step0
    tangent = initial_tangent(omega, x0, lam0, u0)
    accepted = 0
    rejected = 0
    min_mu_seen = omega[-1]
    best_omega = omega.copy()
    best_mu_abs = abs(omega[-1])
    last_residual = np.linalg.norm(omega_residual(omega, x0, lam0, u0))

    for k in range(1, max_steps + 1):
        if abs(omega[-1]) <= eps_mu:
            return omega, True, k, accepted, rejected, step, last_residual

        # Recompute tangent and orient continuously.
        try:
            tangent_new = initial_tangent(omega, x0, lam0, u0, previous_tangent=tangent)
        except np.linalg.LinAlgError:
            step *= 0.5
            rejected += 1
            if step < step_min:
                break
            continue

        # Prefer decreasing mu, but use dot-product continuity first.
        if omega[-1] > eps_mu and tangent_new[-1] > 0:
            tangent_new = -tangent_new
        tangent = tangent_new

        trial_step = step
        accepted_this_iteration = False

        for attempt in range(10):
            omega_pred = omega + trial_step * tangent
            omega_corr, ok, newton_its, Rnorm = pseudo_arclength_corrector(
                omega_pred, omega, tangent, x0, lam0, u0,
                max_newton=15, tol=1e-9, line_search=True
            )
            x_c, lam_c, u_c, v_c, mu_c = unpack_omega(omega_corr)

            finite = np.all(np.isfinite(omega_corr))
            reasonable_mu = (-0.25 <= mu_c <= 1.25) or abs(mu_c) < abs(omega[-1])
            residual_ok = Rnorm < max(1e-6, 20.0 * max(last_residual, 1e-12))
            progress_ok = abs(mu_c) <= abs(omega[-1]) + 5e-3

            if ok and finite and reasonable_mu and residual_ok and progress_ok:
                omega = stabilize_omega(omega_corr)
                last_residual = np.linalg.norm(omega_residual(omega, x0, lam0, u0))
                accepted += 1
                accepted_this_iteration = True

                if abs(omega[-1]) < best_mu_abs:
                    best_mu_abs = abs(omega[-1])
                    best_omega = omega.copy()
                min_mu_seen = min(min_mu_seen, omega[-1])

                # Adaptive step-size control.
                if newton_its <= 4 and Rnorm < 1e-10:
                    step = min(step * 1.25, step_max)
                elif newton_its <= 8:
                    step = min(step * 1.05, step_max)
                else:
                    step = max(step * 0.75, step_min)

                # Smaller step near the target endpoint mu=0.
                if abs(omega[-1]) < 0.05:
                    step = min(step, 0.02)
                if abs(omega[-1]) < 0.01:
                    step = min(step, 0.006)
                break

            trial_step *= 0.5
            rejected += 1
            if trial_step < step_min:
                break

        if not accepted_this_iteration:
            step *= 0.5
            if step < step_min:
                break

    # Return the best near-mu=0 point even if the path did not fully succeed.
    omega = best_omega
    return omega, abs(omega[-1]) <= 10 * eps_mu, k, accepted, rejected, step, last_residual


def solve_homotopy_manyobj(fast_mode=FAST_MODE, results_dir=RESULTS_DIR, show_plots=False, native_analysis=True):
    start_time = time.time()
    rows = []

    results_dir, fig_dir, data_dir = ensure_output_dirs(results_dir)

    # Faster default settings for debugging and repeated experiments.
    # For a denser final run, call solve_homotopy_manyobj(fast_mode=False).
    if fast_mode:
        starts = branch_restarts(compact=True)
        weights = simplex_weights_4obj(level=5, lower=0.05)
        max_steps = 350
        step0, step_max = 0.08, 0.13
    else:
        starts = branch_restarts(compact=False)
        weights = simplex_weights_4obj(level=10, lower=0.03)
        max_steps = 900
        step0, step_max = 0.06, 0.10

    print(f"Using {len(starts)} branch restarts and {len(weights)} weights per restart.")
    print(f"Saving outputs to: {results_dir.resolve()}")

    for component, x0 in starts:
        for idx, lam0 in enumerate(weights):
            omega, success, k, accepted, rejected, step_final, res_final = run_single_path(
                component, x0, lam0,
                u0_scale=0.05,
                step0=step0,
                step_min=1e-5,
                step_max=step_max,
                eps_mu=1e-4,
                max_steps=max_steps,
            )

            x, lam, u, v, mu = unpack_omega(omega)
            f = objectives(x)
            h = equalities(x)
            g = inequalities(x)
            Hnorm = np.linalg.norm(H_map(x, x0, lam, lam0, u, 0.05*np.ones(P), v, mu))

            rows.append({
                "component": component,
                "success": success,
                "k": k,
                "accepted_steps": accepted,
                "rejected_steps": rejected,
                "step_final": step_final,
                "mu": mu,
                "Hnorm": Hnorm,
                "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3],
                "L1": lam[0], "L2": lam[1], "L3": lam[2], "L4": lam[3],
                "f1": f[0], "f2": f[1], "f3": f[2], "f4": f[3],
                "h1": h[0], "h2": h[1],
                "max_g": np.max(g),
                "g0_disconnected": g[0],
                "u_g0": u[0],
                "w1": lam0[0], "w2": lam0[1], "w3": lam0[2], "w4": lam0[3],
            })
            print(f"{component:11s} weight {idx+1:03d}/{len(weights):03d}: "
                  f"success={success} mu={mu:.2e} |H|={Hnorm:.1e} "
                  f"acc/rej={accepted}/{rejected} "
                  f"f=({f[0]:.3f},{f[1]:.3f},{f[2]:.3f},{f[3]:.3f})")

    df = pd.DataFrame(rows)

    # Remove near-duplicates caused by multiple branch restarts converging to the same point.
    obj_cols = ["f1", "f2", "f3", "f4"]
    df_success = df[df["success"] == True].copy()
    if len(df_success) > 0:
        rounded = df_success[obj_cols].round(5).astype(str).agg("_".join, axis=1)
        df_success = df_success.loc[~rounded.duplicated()].copy()
        df_failed = df[df["success"] == False].copy()
        df = pd.concat([df_success, df_failed], ignore_index=True)

    df.to_csv(data_dir / "solution_homotopy_manyobj_degenerate_disconnected_robust.csv", index=False)

    if native_analysis:
        metrics, df_nd, reference_F = compute_quality_indicators(df, data_dir=data_dir)
        pd.DataFrame([metrics]).to_csv(data_dir / "quality_indicators.csv", index=False)
    else:
        metrics = {"n_approx_nondominated": float("nan"), "hypervolume_ratio": float("nan"), "averaged_hausdorff_distance": float("nan")}

    print(f"\nSuccessful paths: {int(df['success'].sum())} / {len(df)}")
    print(f"Nondominated feasible solutions: {metrics['n_approx_nondominated']}")
    print(f"Hypervolume ratio: {metrics['hypervolume_ratio']:.6f}")
    print(f"Averaged Hausdorff distance: {metrics['averaged_hausdorff_distance']:.6e}")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    if native_analysis:
        display(df)
        plot_results(df, fig_dir=fig_dir, show=show_plots)
    return df, metrics


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(df, fig_dir=FIGURES_DIR, show=False):
    """
    Publication-oriented figures for the 4-objective test problem.

    The problem has four objectives, so no single 3D figure can show the full
    objective space. We therefore use:
      (1) a 3D projection (f1,f2,f3) with f4 encoded by color,
      (2) all six pairwise objective projections colored by disconnected component,
      (3) decision-space views showing the disconnected feasible set and equality
          constraint structure,
      (4) cross-section plots showing nonconvexity of the objective functions.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["component_label"] = df["component"].astype(str)
    df["component_side"] = df["component"].map(component_side)

    # Use successful rows for the main scientific figures. If there are too few,
    # include near-successful rows with small |mu| and small constraint violation.
    df_plot = df[df["success"] == True].copy()
    if len(df_plot) < 10:
        near = df[(df["mu"].abs() < 5e-2) & (df["Hnorm"] < 1e-4) & (df["max_g"] <= 1e-6)].copy()
        df_plot = pd.concat([df_plot, near], ignore_index=True).drop_duplicates()
    if len(df_plot) == 0:
        print("Warning: no successful or near-successful paths found; plotting all computed rows.")
        df_plot = df.copy()

    # -------------------------------------------------------------------------
    # Figure 1: 3D objective projection with the fourth objective encoded by color
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df_plot["f1"], df_plot["f2"], df_plot["f3"],
        c=df_plot["f4"], cmap="viridis", s=55, edgecolor="k", linewidth=0.25
    )
    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")
    ax.set_zlabel(r"$f_3$")
    ax.set_title(r"Four-objective visualization: $(f_1,f_2,f_3)$ projection colored by $f_4$")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label(r"$f_4$")
    plt.tight_layout()
    save_current_figure(
        fig_dir,
        "manyobj_3d_projection_colored_by_f4",
        show=show
    )
    
    # -------------------------------------------------------------------------
    
    # Figure 3: pairwise objective projections, colored by omitted objective
    # -------------------------------------------------------------------------
    pairs = [("f1","f2","f4"), ("f1","f3","f4"), ("f1","f4","f3"),
            ("f2","f3","f4"), ("f2","f4","f3"), ("f3","f4","f2")]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.08, top=0.90,
                       wspace=0.42, hspace=0.38)

    for ax, (a, b, cobj) in zip(axes.ravel(), pairs):
        sc = ax.scatter(df_plot[a], df_plot[b], c=df_plot[cobj],
                        cmap="viridis", s=40, edgecolor="k", linewidth=0.2)

        ax.set_xlabel(rf"${a}$")
        ax.set_ylabel(rf"${b}$")
        ax.set_title(rf"${a}\ \mathrm{{vs.}}\ {b}$"+ "\n" + rf"$\mathrm{{colored\ by}}\ {cobj}$",fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.ticklabel_format(axis="both", style="plain", useOffset=False)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.02)
        cbar.set_label(rf"${cobj}$")

    fig.suptitle("Pairwise objective-space projections colored by an omitted objective",
                 fontsize=13)

    save_current_figure(
        fig_dir,
        "manyobj_pairwise_objective_projections_colored_by_omitted_objective",
        show=show
    )
    
    # -------------------------------------------------------------------------
    # Figure 5: disconnected feasible set in the (x1,x2)-plane
    # The disconnected structure is created by g0=(x1^2-1)^2+x2^2-0.08 <= 0.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    x1_grid = np.linspace(-1.35, 1.35, 1500)
    y2 = 0.08 - (x1_grid ** 2 - 1.0) ** 2
    mask = y2 >= 0
    plt.fill_between(
        x1_grid[mask], -np.sqrt(y2[mask]), np.sqrt(y2[mask]),
        alpha=0.25, label=r"$g_0(x)=(x_1^2-1)^2+x_2^2-0.08\leq0$"
    )
    for comp, part in df_plot.groupby("component_side"):
        plt.scatter(part["x1"],part["x2"],s=48,
                    label=(
                        r"branch near $x_1 \approx -1$"
                        if "left" in comp
                        else r"branch near $x_1 \approx 1$"
                        )
                    )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Disconnected feasible components in the $(x_1,x_2)$ subspace")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.savefig(
        fig_dir / "manyobj_3d_projection_components.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.3  
    )

    # -------------------------------------------------------------------------
    # Figure 6: equality-constraint degeneracy in the (x2,x3)-plane
    # h2=x3^2-x2^2=0 means x3=+x2 or x3=-x2.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    z = np.linspace(-1, 1, 400)
    plt.plot(z, z, linestyle="--", linewidth=1.5, label=r"$x_3=x_2$")
    plt.plot(z, -z, linestyle=":", linewidth=1.8, label=r"$x_3=-x_2$")
    for comp, part in df_plot.groupby("component_side"):
        plt.scatter(part["x2"],part["x3"],s=55,alpha=0.85,edgecolor="k",linewidth=0.3,
                    label=(r"branch near $x_1 \approx -1$"
                        if "left" in comp
                        else r"branch near $x_1 \approx 1$"
                    )
                )
    plt.xlabel(r"$x_2$")
    plt.ylabel(r"$x_3$")
    plt.title(r"Equality-constraint structure: $h_2=x_3^2-x_2^2=0$")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    save_current_figure(fig_dir, "manyobj_decision_space_x2_x3_equality_degeneracy", show=show)

    # -------------------------------------------------------------------------
    # Figure 7: x4 equality constraint residual, showing h1=x4=0
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for comp, part in df_plot.groupby("component_side"):
        plt.scatter(part["x1"],part["x4"],s=48,
                    label=(
                        r"branch near $x_1 \approx -1$"
                        if "left" in comp
                        else r"branch near $x_1 \approx 1$"
                        )
                    )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_4$")
    plt.title(r"Equality constraint $h_1=x_4=0$")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    save_current_figure(fig_dir, "manyobj_decision_space_x1_x4_equality", show=show)

   

    # -------------------------------------------------------------------------
    # Figure 8: 3D decision-space projection (x1,x2,x3), with x4 omitted
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for comp, part in df_plot.groupby("component_side"):
        ax.scatter(
            part["x1"],
            part["x2"],
            part["x3"],
            s=55,
            alpha=0.90,
            label=(
                r"branch near $x_1 \approx -1$"
                if "left" in comp
                else r"branch near $x_1 \approx 1$"
            )
        )

    ax.set_xlabel(r"$x_1$", fontsize=13, labelpad=12)
    ax.set_ylabel(r"$x_2$", fontsize=13, labelpad=12)
    ax.set_zlabel(r"$x_3$", fontsize=13, labelpad=20)

    # important for visibility of x3 axis/label
    ax.view_init(elev=28, azim=-60)

    ax.set_title(
        r"Decision-space projection $(x_1,x_2,x_3)$",
        fontsize=15,
        pad=18
    )

    ax.legend(loc="upper right")

    plt.tight_layout()

    save_current_figure(
        fig_dir,
        "manyobj_decision_space_x1_x2_x3",
        show=show
    )
    # -------------------------------------------------------------------------
    # Figure 9: nonconvex objective cross-sections.
    # This is the slice x2=x3=x4=0, with x1 varied.
    # -------------------------------------------------------------------------
    x1_grid = np.linspace(-1.35, 1.35, 1200)
    vals = np.array([objectives(np.array([z, 0.0, 0.0, 0.0])) for z in x1_grid])
    plt.figure(figsize=(9, 6))
    plt.plot(x1_grid, vals[:, 0], label=r"$f_1(x_1,0,0,0)$")
    plt.plot(x1_grid, vals[:, 1], label=r"$f_2(x_1,0,0,0)$")
    plt.plot(x1_grid, vals[:, 2], label=r"$f_3(x_1,0,0,0)$")
    plt.plot(x1_grid, vals[:, 3], label=r"$f_4(x_1,0,0,0)$")
    plt.xlabel(r"$x_1$")
    plt.ylabel("Objective value")
    plt.title(r"Nonconvex objective cross-sections on $x_2=x_3=x_4=0$")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    save_current_figure(fig_dir, "manyobj_nonconvex_cross_sections_x1", show=show)

    # -------------------------------------------------------------------------
    # Figure 10: feasibility diagnostics
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    idx = np.arange(len(df_plot))
    plt.semilogy(idx, np.maximum(np.abs(df_plot["h1"].values), 1e-16), marker="o", linestyle="", label=r"$|h_1|=|x_4|$")
    plt.semilogy(idx, np.maximum(np.abs(df_plot["h2"].values), 1e-16), marker="s", linestyle="", label=r"$|h_2|=|x_3^2-x_2^2|$")
    plt.semilogy(idx, np.maximum(np.maximum(df_plot["max_g"].values, 0.0), 1e-16), marker="^", linestyle="", label=r"$\max(0,\max_i g_i)$")
    plt.xlabel("computed solution index")
    plt.ylabel("residual / violation")
    plt.title("Constraint feasibility diagnostics")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    save_current_figure(fig_dir, "manyobj_constraint_feasibility_diagnostics", show=show)


if __name__ == "__main__":
    df, metrics = solve_homotopy_manyobj(fast_mode=FAST_MODE, results_dir=RESULTS_DIR, show_plots=False)
