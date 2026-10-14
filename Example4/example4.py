"""
Example 4: Three-objective problems with analytically known Pareto fronts

(a) Convex-front quadratic simplex problem
(b) Concave-front quadratic simplex problem

Decision variables:
    x = (x1,x2)

Feasible triangle:
    x1 >= 0, x2 >= 0, x1+x2 <= 1

Let z = (x1, x2, 1-x1-x2).  Then
    convex:  F+(x) = ( z1^2,  z2^2,  z3^2)
    concave: F-(x) = (-z1^2, -z2^2, -z3^2)

Every feasible point is Pareto optimal in both cases.  The exact fronts are
    sqrt(f1)+sqrt(f2)+sqrt(f3)=1                 [convex case]
    sqrt(-f1)+sqrt(-f2)+sqrt(-f3)=1             [concave case]

For lambda_i > 0, sum lambda_i = 1, the interior Pareto-critical point is
    z_i = (1/lambda_i) / sum_j(1/lambda_j).

This file is completely self-contained.  It does NOT import any DTLZ code.

The continuation implementation follows the same Algorithm-2 structure used in
the previous production code:
  * pseudo-arclength predictor/corrector,
  * SVD null-space tangent,
  * adaptive step size,
  * endpoint rule |mu| <= eps_mu,
  * no separate exact-mu=0 endpoint correction.

All objective gradients and Hessians are analytic.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Global dimensions and numerical controls
# ============================================================================

N = 2                  # decision variables
M = 3                  # objectives
P = 3                  # inequalities
Q = 0                  # equalities

RESULTS_DIR = Path("example4_known_pf_results")
SAVE_PDF = True

EPS_MU = 1.0e-4
U0_SCALE = 0.05

STEP0 = 0.06
STEP_MIN = 1.0e-5
STEP_MAX = 0.10
MAX_STEPS = 900

NEWTON_TOL = 1.0e-8
MAX_NEWTON = 15

FEAS_TOL = 1.0e-7
BOUNDARY_TOL = 1.0e-4
NEAR_EXACT_TOL = 1.0e-3


# ============================================================================
# Problem definitions
# ============================================================================

def z_from_x(x):
    """Barycentric/simplex coordinates z=(x1,x2,1-x1-x2)."""
    x = np.asarray(x, dtype=float)
    return np.array([x[0], x[1], 1.0-x[0]-x[1]], dtype=float)


def objectives(x, problem):
    """Three objectives."""
    z = z_from_x(x)

    if problem == "convex":
        return z**2

    if problem == "concave":
        return -(z**2)

    raise ValueError(f"Unknown problem: {problem}")


def grad_objectives(x, problem):
    """
    Objective Jacobian Df.
    Rows are gradients of f1,f2,f3 with respect to (x1,x2).
    """
    x1, x2 = np.asarray(x, dtype=float)
    z3 = 1.0-x1-x2

    G = np.array([
        [ 2.0*x1, 0.0      ],
        [ 0.0,      2.0*x2 ],
        [-2.0*z3,  -2.0*z3 ]
    ], dtype=float)

    if problem == "convex":
        return G

    if problem == "concave":
        return -G

    raise ValueError(f"Unknown problem: {problem}")


def hess_objectives(x, problem):
    """Exact Hessian of each objective."""
    H = [
        np.array([[2.0, 0.0],
                  [0.0, 0.0]], dtype=float),

        np.array([[0.0, 0.0],
                  [0.0, 2.0]], dtype=float),

        np.array([[2.0, 2.0],
                  [2.0, 2.0]], dtype=float),
    ]

    if problem == "convex":
        return H

    if problem == "concave":
        return [-A for A in H]

    raise ValueError(f"Unknown problem: {problem}")


def inequalities(x):
    """
    g(x) <= 0:
        g1 = -x1
        g2 = -x2
        g3 = x1+x2-1
    """
    x1, x2 = np.asarray(x, dtype=float)
    return np.array([
        -x1,
        -x2,
        x1+x2-1.0
    ], dtype=float)


def grad_inequalities(x):
    """Rows are gradients of g1,g2,g3."""
    return np.array([
        [-1.0,  0.0],
        [ 0.0, -1.0],
        [ 1.0,  1.0]
    ], dtype=float)


def hess_inequalities(x):
    """All three inequalities are linear."""
    return [np.zeros((N, N), dtype=float) for _ in range(P)]


# ============================================================================
# Exact Pareto-critical solution associated with a positive weight
# ============================================================================

def exact_z_from_lambda(lam):
    lam = np.asarray(lam, dtype=float)

    if np.any(lam <= 0.0):
        raise ValueError("Exact interior formula requires strictly positive weights.")

    inv = 1.0/lam
    return inv/np.sum(inv)


def exact_x_from_lambda(lam):
    z = exact_z_from_lambda(lam)
    return np.array([z[0], z[1]], dtype=float)


def exact_F_from_lambda(lam, problem):
    return objectives(exact_x_from_lambda(lam), problem)


# ============================================================================
# Homotopy
#
# omega = [x, lambda, u, mu]
#
# Unknowns:
#   N + M + P + 1 = 2+3+3+1 = 9
#
# Equations:
#   H1: stationarity homotopy       N = 2
#   H2: complementarity homotopy    P = 3
#   H3: weight homotopy             M = 3
#   total                               8
#
# Hence the regular zero set is one-dimensional.
# ============================================================================

def make_omega(x, lam, u, mu):
    return np.concatenate([
        np.asarray(x, dtype=float),
        np.asarray(lam, dtype=float),
        np.asarray(u, dtype=float),
        np.array([mu], dtype=float)
    ])


def unpack_omega(omega):
    omega = np.asarray(omega, dtype=float)

    x = omega[:N]
    lam = omega[N:N+M]
    u = omega[N+M:N+M+P]
    mu = omega[-1]

    return x, lam, u, mu


def H_map(x, x0, lam, lam0, u, u0, mu, problem):
    """
    H1 = (1-mu)(Df^T lambda + Dg^T u) + mu(x-x0)
    H2 = u .* g(x) - mu u0 .* g(x0)
    H3_i = (1-mu)(1-sum lambda)
           - mu(lambda_i^(3/2)-lambda0_i^(3/2))
    """
    Df = grad_objectives(x, problem)
    Dg = grad_inequalities(x)

    g = inequalities(x)
    g0 = inequalities(x0)

    H1 = (
        (1.0-mu)*(Df.T @ lam + Dg.T @ u)
        + mu*(x-x0)
    )

    H2 = u*g - mu*u0*g0

    lpow = np.power(np.clip(lam, 0.0, None), 1.5)
    l0pow = np.power(np.clip(lam0, 0.0, None), 1.5)

    H3_common = (1.0-mu)*(1.0-np.sum(lam))
    H3 = H3_common*np.ones(M) - mu*(lpow-l0pow)

    return np.concatenate([H1, H2, H3])


def weighted_objective_hessian(x, lam, problem):
    Hf = hess_objectives(x, problem)

    H = np.zeros((N, N), dtype=float)
    for i in range(M):
        H += lam[i]*Hf[i]

    return H


def DH_map(x, x0, lam, lam0, u, u0, mu, problem):
    """Exact Jacobian of the homotopy with respect to omega."""

    Df = grad_objectives(x, problem)
    Dg = grad_inequalities(x)

    g = inequalities(x)
    g0 = inequalities(x0)

    # ------------------------------------------------------------------
    # H1 derivatives
    # ------------------------------------------------------------------
    Hobj = weighted_objective_hessian(x, lam, problem)

    # Inequality Hessians are zero here.
    dH1_dx = (1.0-mu)*Hobj + mu*np.eye(N)
    dH1_dlam = (1.0-mu)*Df.T
    dH1_du = (1.0-mu)*Dg.T

    stationarity_term = Df.T @ lam + Dg.T @ u
    dH1_dmu = (-stationarity_term + (x-x0)).reshape(N, 1)

    R1 = np.hstack([
        dH1_dx,
        dH1_dlam,
        dH1_du,
        dH1_dmu
    ])

    # ------------------------------------------------------------------
    # H2 derivatives
    # ------------------------------------------------------------------
    dH2_dx = u[:, None]*Dg
    dH2_dlam = np.zeros((P, M))
    dH2_du = np.diag(g)
    dH2_dmu = (-u0*g0).reshape(P, 1)

    R2 = np.hstack([
        dH2_dx,
        dH2_dlam,
        dH2_du,
        dH2_dmu
    ])

    # ------------------------------------------------------------------
    # H3 derivatives
    # ------------------------------------------------------------------
    sqrt_l = np.sqrt(np.clip(lam, 0.0, None))

    dH3_dlam = (
        -(1.0-mu)*np.ones((M, M))
        - mu*np.diag(1.5*sqrt_l)
    )

    lpow = np.power(np.clip(lam, 0.0, None), 1.5)
    l0pow = np.power(np.clip(lam0, 0.0, None), 1.5)

    dH3_dmu = (
        -(1.0-np.sum(lam))*np.ones(M)
        - (lpow-l0pow)
    ).reshape(M, 1)

    R3 = np.hstack([
        np.zeros((M, N)),
        dH3_dlam,
        np.zeros((M, P)),
        dH3_dmu
    ])

    return np.vstack([R1, R2, R3])


def omega_residual(omega, x0, lam0, u0, problem):
    x, lam, u, mu = unpack_omega(omega)
    return H_map(x, x0, lam, lam0, u, u0, mu, problem)


def omega_jacobian(omega, x0, lam0, u0, problem):
    x, lam, u, mu = unpack_omega(omega)
    return DH_map(x, x0, lam, lam0, u, u0, mu, problem)


# ============================================================================
# Algorithm 2: pseudo-arclength continuation
# ============================================================================

def nullspace_right_vector(A):
    _, _, Vt = np.linalg.svd(A, full_matrices=True)

    xi = Vt[-1, :]
    nrm = np.linalg.norm(xi)

    if nrm == 0.0 or not np.isfinite(nrm):
        raise np.linalg.LinAlgError("Invalid null vector.")

    return xi/nrm


def safe_solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def stabilize_omega(omega):
    """
    Preserve positivity of lambda and u during numerical continuation.
    This is the same numerical safeguard used in the production tracker.
    """
    omega = np.asarray(omega, dtype=float).copy()

    omega[N:N+M] = np.maximum(
        omega[N:N+M], 1.0e-12
    )

    omega[N+M:N+M+P] = np.maximum(
        omega[N+M:N+M+P], 1.0e-12
    )

    return np.nan_to_num(
        omega,
        nan=0.0,
        posinf=1.0e6,
        neginf=-1.0e6
    )


def tangent_vector(
    omega,
    x0,
    lam0,
    u0,
    problem,
    previous_tangent=None
):
    J = omega_jacobian(
        omega, x0, lam0, u0, problem
    )

    tangent = nullspace_right_vector(J)

    if previous_tangent is None:
        # Initial orientation: move toward decreasing mu.
        if tangent[-1] > 0.0:
            tangent = -tangent
    else:
        # Preserve orientation from the previous accepted tangent.
        if np.dot(tangent, previous_tangent) < 0.0:
            tangent = -tangent

    return tangent/max(
        np.linalg.norm(tangent), 1.0e-15
    )


def pseudo_arclength_corrector(
    omega_pred,
    tangent,
    x0,
    lam0,
    u0,
    problem,
    max_newton=MAX_NEWTON,
    tol=NEWTON_TOL,
    line_search=True
):
    omega = stabilize_omega(omega_pred)

    tangent = tangent/max(
        np.linalg.norm(tangent), 1.0e-15
    )

    for it in range(max_newton):

        Hval = omega_residual(
            omega, x0, lam0, u0, problem
        )

        arc = np.array([
            np.dot(tangent, omega-omega_pred)
        ])

        R = np.concatenate([Hval, arc])
        Rnorm = np.linalg.norm(R)

        if Rnorm < tol:
            return omega, True, it+1, Rnorm

        J = omega_jacobian(
            omega, x0, lam0, u0, problem
        )

        Jaug = np.vstack([
            J,
            tangent.reshape(1, -1)
        ])

        delta = safe_solve(Jaug, R)

        if not line_search:
            omega = stabilize_omega(omega-delta)
            continue

        alpha = 1.0
        best = omega.copy()
        best_norm = np.inf
        accepted = False

        for _ in range(8):

            trial = stabilize_omega(
                omega-alpha*delta
            )

            Ht = omega_residual(
                trial, x0, lam0, u0, problem
            )

            at = np.array([
                np.dot(tangent, trial-omega_pred)
            ])

            trial_norm = np.linalg.norm(
                np.concatenate([Ht, at])
            )

            if trial_norm < Rnorm:
                omega = trial
                accepted = True
                break

            if trial_norm < best_norm:
                best = trial
                best_norm = trial_norm

            alpha *= 0.5

        if not accepted:
            omega = best

    Hval = omega_residual(
        omega, x0, lam0, u0, problem
    )

    arc = np.array([
        np.dot(tangent, omega-omega_pred)
    ])

    Rnorm = np.linalg.norm(
        np.concatenate([Hval, arc])
    )

    return (
        omega,
        Rnorm < 10.0*tol,
        max_newton,
        Rnorm
    )


def run_single_path(
    problem,
    x0,
    lam0,
    u0_scale=U0_SCALE,
    step0=STEP0,
    step_min=STEP_MIN,
    step_max=STEP_MAX,
    eps_mu=EPS_MU,
    max_steps=MAX_STEPS
):
    """
    Follow one homotopy path.

    Endpoint rule:
        stop when |mu| <= eps_mu.

    No separate exact-mu=0 endpoint correction is applied.
    """

    x0 = np.asarray(x0, dtype=float)
    lam0 = np.asarray(lam0, dtype=float)

    u0 = u0_scale*np.ones(P)

    omega = make_omega(
        x0,
        lam0,
        u0,
        1.0
    )

    omega = stabilize_omega(omega)

    step = step0

    tangent = tangent_vector(
        omega,
        x0,
        lam0,
        u0,
        problem
    )

    accepted = 0
    rejected = 0

    best_omega = omega.copy()
    best_mu_abs = abs(omega[-1])

    last_residual = np.linalg.norm(
        omega_residual(
            omega,
            x0,
            lam0,
            u0,
            problem
        )
    )

    k = 0

    for k in range(1, max_steps+1):

        if abs(omega[-1]) <= eps_mu:
            return (
                omega,
                True,
                k,
                accepted,
                rejected,
                step,
                last_residual
            )

        try:
            tangent_new = tangent_vector(
                omega,
                x0,
                lam0,
                u0,
                problem,
                previous_tangent=tangent
            )
        except np.linalg.LinAlgError:

            step *= 0.5
            rejected += 1

            if step < step_min:
                break

            continue

        # Keep orientation toward decreasing mu while mu is positive.
        if (
            omega[-1] > eps_mu
            and tangent_new[-1] > 0.0
        ):
            tangent_new = -tangent_new

        tangent = tangent_new

        trial_step = step
        accepted_here = False

        for _ in range(10):

            omega_pred = omega + trial_step*tangent

            (
                omega_corr,
                ok,
                nits,
                Rnorm
            ) = pseudo_arclength_corrector(
                omega_pred,
                tangent,
                x0,
                lam0,
                u0,
                problem,
                max_newton=MAX_NEWTON,
                tol=NEWTON_TOL,
                line_search=True
            )

            _, _, _, mu_c = unpack_omega(
                omega_corr
            )

            finite = np.all(
                np.isfinite(omega_corr)
            )

            reasonable_mu = (
                (-0.25 <= mu_c <= 1.25)
                or abs(mu_c) < abs(omega[-1])
            )

            residual_ok = (
                Rnorm
                < max(
                    1.0e-6,
                    20.0*max(
                        last_residual,
                        1.0e-12
                    )
                )
            )

            progress_ok = (
                abs(mu_c)
                <= abs(omega[-1]) + 5.0e-3
            )

            if (
                ok
                and finite
                and reasonable_mu
                and residual_ok
                and progress_ok
            ):

                omega = stabilize_omega(
                    omega_corr
                )

                last_residual = np.linalg.norm(
                    omega_residual(
                        omega,
                        x0,
                        lam0,
                        u0,
                        problem
                    )
                )

                accepted += 1
                accepted_here = True

                if abs(omega[-1]) < best_mu_abs:
                    best_mu_abs = abs(omega[-1])
                    best_omega = omega.copy()

                # Adaptive step size.
                if (
                    nits <= 4
                    and Rnorm < 1.0e-9
                ):
                    step = min(
                        step*1.25,
                        step_max
                    )

                elif nits <= 8:
                    step = min(
                        step*1.05,
                        step_max
                    )

                else:
                    step = max(
                        step*0.75,
                        step_min
                    )

                # Refine near the endpoint.
                if abs(omega[-1]) < 0.05:
                    step = min(step, 0.02)

                if abs(omega[-1]) < 0.01:
                    step = min(step, 0.006)

                break

            trial_step *= 0.5
            rejected += 1

            if trial_step < step_min:
                break

        if not accepted_here:

            step *= 0.5

            if step < step_min:
                break

    # If the loop fails, return the point with the smallest |mu|.
    omega = best_omega

    success = (
        abs(omega[-1])
        <= 10.0*eps_mu
    )

    return (
        omega,
        success,
        k,
        accepted,
        rejected,
        step,
        last_residual
    )


# ============================================================================
# Weight lattice and generic initial points
# ============================================================================

def simplex_weights_3obj(level=8):
    """
    Strictly positive simplex lattice:
        lambda_i = integer_i / level,
        integer_i >= 1,
        sum integer_i = level.

    level=8 gives 21 weights.
    """
    W = []

    for i in range(1, level):
        for j in range(1, level-i):

            k = level-i-j

            if k >= 1:
                W.append(
                    np.array(
                        [i, j, k],
                        dtype=float
                    )/level
                )

    return W


def initial_points():
    """
    Four generic feasible interior starting points.

    They do NOT use the exact Pareto-critical solution associated
    with the selected weight.
    """
    return [
        ("center", np.array([1.0/3.0, 1.0/3.0])),
        ("left",   np.array([0.15, 0.425])),
        ("right",  np.array([0.425, 0.15])),
        ("upper",  np.array([0.425, 0.425])),
    ]


# ============================================================================
# Endpoint diagnostics
# ============================================================================

def kkt_diagnostics(x, lam, u, problem):

    Df = grad_objectives(x, problem)
    Dg = grad_inequalities(x)

    g = inequalities(x)

    stationarity = np.linalg.norm(
        Df.T @ lam + Dg.T @ u
    )

    complementarity = np.max(
        np.abs(u*g)
    )

    max_g = max(
        0.0,
        float(np.max(g))
    )

    weight_sum_residual = abs(
        float(np.sum(lam)-1.0)
    )

    return (
        stationarity,
        complementarity,
        max_g,
        weight_sum_residual,
        float(np.min(lam)),
        float(np.min(u))
    )


def nondominated_mask(F, eps=1.0e-12):
    """Simple O(n^2) nondominance filter for minimization."""
    F = np.asarray(F, dtype=float)
    n = len(F)

    keep = np.ones(n, dtype=bool)

    for i in range(n):

        if not keep[i]:
            continue

        for j in range(n):

            if i == j:
                continue

            if (
                np.all(F[j] <= F[i] + eps)
                and np.any(F[j] < F[i] - eps)
            ):
                keep[i] = False
                break

    return keep


# ============================================================================
# Exact Pareto-front sampling
# ============================================================================

def exact_reference_front(problem, resolution=100):
    """
    Uniform triangular grid in z coordinates.
    Since the entire feasible triangle is Pareto optimal, this gives
    an exact reference-front discretization.
    """
    F = []
    X = []

    for i in range(resolution+1):

        z1 = i/resolution

        for j in range(resolution+1-i):

            z2 = j/resolution
            z3 = 1.0-z1-z2

            x = np.array(
                [z1, z2],
                dtype=float
            )

            X.append(x)
            F.append(
                objectives(x, problem)
            )

    return (
        np.asarray(X),
        np.asarray(F)
    )


def nearest_distances(A, B, chunk=1000):

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    out = np.empty(len(A))

    for i in range(0, len(A), chunk):

        D2 = np.sum(
            (
                A[i:i+chunk, None, :]
                - B[None, :, :]
            )**2,
            axis=2
        )

        out[i:i+chunk] = np.sqrt(
            np.min(D2, axis=1)
        )

    return out


def quality_metrics(F, Fref):

    if len(F) == 0:
        return {
            "GD": np.nan,
            "IGD": np.nan,
            "AHD": np.nan
        }

    ideal = np.min(Fref, axis=0)
    nadir = np.max(Fref, axis=0)

    scale = np.maximum(
        nadir-ideal,
        1.0e-12
    )

    Fn = (F-ideal)/scale
    Rn = (Fref-ideal)/scale

    gd = float(
        np.mean(
            nearest_distances(
                Fn,
                Rn
            )
        )
    )

    igd = float(
        np.mean(
            nearest_distances(
                Rn,
                Fn
            )
        )
    )

    return {
        "GD": gd,
        "IGD": igd,
        "AHD": max(gd, igd)
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_problem(problem, df_good, Fref, fig_dir):

    fig = plt.figure(
        figsize=(7.2, 5.8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Thin reference front for display only.
    stride = max(
        1,
        len(Fref)//2500
    )

    ax.scatter(
        Fref[::stride, 0],
        Fref[::stride, 1],
        Fref[::stride, 2],
        s=3,
        alpha=0.15,
        label="Exact PF"
    )

    if len(df_good):

        F = df_good[
            ["f1", "f2", "f3"]
        ].to_numpy(float)

        ax.scatter(
            F[:, 0],
            F[:, 1],
            F[:, 2],
            s=24,
            label="Homotopy endpoints"
        )

    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")
    ax.set_zlabel(r"$f_3$")

    title = (
        "Example 4(a): Convex known PF"
        if problem == "convex"
        else
        "Example 4(b): Concave known PF"
    )

    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        fig_dir/f"{problem}_known_pf.png",
        dpi=250,
        bbox_inches="tight"
    )

    if SAVE_PDF:
        fig.savefig(
            fig_dir/f"{problem}_known_pf.pdf",
            bbox_inches="tight"
        )

    plt.close(fig)


# ============================================================================
# Solve one subproblem
# ============================================================================

def solve_problem(
    problem,
    weight_level=8,
    eps_mu=EPS_MU
):

    problem_dir = RESULTS_DIR/problem
    data_dir = problem_dir/"data"
    fig_dir = problem_dir/"figures"

    data_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    W = simplex_weights_3obj(
        level=weight_level
    )

    starts = initial_points()

    print()
    print("="*72)
    print(problem.upper())
    print("="*72)

    print(
        f"{len(W)} weights x "
        f"{len(starts)} starts = "
        f"{len(W)*len(starts)} paths"
    )

    rows = []

    t0 = time.time()

    for wi, lam0 in enumerate(W, start=1):

        x_exact = exact_x_from_lambda(
            lam0
        )

        F_exact = exact_F_from_lambda(
            lam0,
            problem
        )

        for start_name, x0 in starts:

            (
                omega,
                success,
                iterations,
                accepted,
                rejected,
                step_final,
                last_residual
            ) = run_single_path(
                problem,
                x0,
                lam0,
                u0_scale=U0_SCALE,
                step0=STEP0,
                step_min=STEP_MIN,
                step_max=STEP_MAX,
                eps_mu=eps_mu,
                max_steps=MAX_STEPS
            )

            x, lam, u, mu = unpack_omega(
                omega
            )

            F = objectives(
                x,
                problem
            )

            u0 = U0_SCALE*np.ones(P)

            Hnorm = float(
                np.linalg.norm(
                    omega_residual(
                        omega,
                        x0,
                        lam0,
                        u0,
                        problem
                    )
                )
            )

            (
                stationarity,
                complementarity,
                max_g,
                weight_sum_residual,
                min_lambda,
                min_u
            ) = kkt_diagnostics(
                x,
                lam,
                u,
                problem
            )

            z = z_from_x(x)

            feasible = (
                max_g <= FEAS_TOL
            )

            boundary = (
                np.min(z)
                <= BOUNDARY_TOL
            )

            row = {
                "problem": problem,
                "weight_id": wi,

                "w0_1": lam0[0],
                "w0_2": lam0[1],
                "w0_3": lam0[2],

                "start": start_name,

                "success": bool(success),
                "feasible": bool(feasible),

                "iterations": iterations,
                "accepted_steps": accepted,
                "rejected_steps": rejected,
                "step_final": step_final,

                "mu": mu,
                "Hnorm": Hnorm,

                "x1": x[0],
                "x2": x[1],

                "z1": z[0],
                "z2": z[1],
                "z3": z[2],

                "f1": F[0],
                "f2": F[1],
                "f3": F[2],

                "lambda1": lam[0],
                "lambda2": lam[1],
                "lambda3": lam[2],

                "u1": u[0],
                "u2": u[1],
                "u3": u[2],

                "stationarity": stationarity,
                "complementarity": complementarity,
                "max_g": max_g,
                "weight_sum_residual": weight_sum_residual,
                "min_lambda": min_lambda,
                "min_u": min_u,

                "exact_x1": x_exact[0],
                "exact_x2": x_exact[1],

                "exact_f1": F_exact[0],
                "exact_f2": F_exact[1],
                "exact_f3": F_exact[2],

                "x_error_to_exact_kkt":
                    np.linalg.norm(
                        x-x_exact
                    ),

                "F_error_to_exact_kkt":
                    np.linalg.norm(
                        F-F_exact
                    ),

                "lambda_movement":
                    np.linalg.norm(
                        lam-lam0
                    ),

                "boundary_endpoint":
                    bool(boundary),
            }

            rows.append(row)

    runtime = time.time()-t0

    df = pd.DataFrame(rows)

    df.to_csv(
        data_dir/"all_paths.csv",
        index=False
    )

    good = df[
        (df["success"] == True)
        & (df["feasible"] == True)
    ].copy()

    if len(good):

        Fgood = good[
            ["f1", "f2", "f3"]
        ].to_numpy(float)

        nd = nondominated_mask(Fgood)

        good_nd = good.loc[
            good.index[nd]
        ].copy()

    else:
        good_nd = good.copy()

    good_nd.to_csv(
        data_dir/
        "nondominated_successful_endpoints.csv",
        index=False
    )

    Xref, Fref = exact_reference_front(
        problem,
        resolution=120
    )

    pd.DataFrame(
        Fref,
        columns=["f1", "f2", "f3"]
    ).to_csv(
        data_dir/"exact_reference_front.csv",
        index=False
    )

    metrics = quality_metrics(
        good_nd[
            ["f1", "f2", "f3"]
        ].to_numpy(float)
        if len(good_nd)
        else np.empty((0, 3)),
        Fref
    )

    successful_feasible = len(good)

    near_exact = int(
        (
            good[
                "F_error_to_exact_kkt"
            ]
            <= NEAR_EXACT_TOL
        ).sum()
    ) if len(good) else 0

    boundary_count = int(
        good[
            "boundary_endpoint"
        ].sum()
    ) if len(good) else 0

    summary = {
        "problem": problem,
        "n_weights": len(W),
        "n_starts": len(starts),
        "n_paths": len(df),

        "n_success": int(
            df["success"].sum()
        ),

        "n_successful_feasible":
            successful_feasible,

        "n_nondominated":
            len(good_nd),

        "n_near_exact_interior_kkt":
            near_exact,

        "n_boundary_endpoints":
            boundary_count,

        "median_F_error_to_exact_kkt":
            float(
                good[
                    "F_error_to_exact_kkt"
                ].median()
            )
            if len(good)
            else np.nan,

        "max_F_error_to_exact_kkt":
            float(
                good[
                    "F_error_to_exact_kkt"
                ].max()
            )
            if len(good)
            else np.nan,

        "median_lambda_movement":
            float(
                good[
                    "lambda_movement"
                ].median()
            )
            if len(good)
            else np.nan,

        "max_lambda_movement":
            float(
                good[
                    "lambda_movement"
                ].max()
            )
            if len(good)
            else np.nan,

        "max_stationarity":
            float(
                good[
                    "stationarity"
                ].max()
            )
            if len(good)
            else np.nan,

        "max_complementarity":
            float(
                good[
                    "complementarity"
                ].max()
            )
            if len(good)
            else np.nan,

        "max_Hnorm":
            float(
                good[
                    "Hnorm"
                ].max()
            )
            if len(good)
            else np.nan,

        "runtime_seconds":
            runtime,

        **metrics
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        data_dir/"summary.csv",
        index=False
    )

    # Summary by starting point.
    if len(good):

        start_summary = (
            good.assign(
                near_exact=
                    good[
                        "F_error_to_exact_kkt"
                    ]
                    <= NEAR_EXACT_TOL
            )
            .groupby("start")
            .agg(
                successful_feasible=(
                    "weight_id",
                    "count"
                ),
                near_exact=(
                    "near_exact",
                    "sum"
                ),
                boundary_endpoints=(
                    "boundary_endpoint",
                    "sum"
                ),
                median_F_error=(
                    "F_error_to_exact_kkt",
                    "median"
                ),
                max_F_error=(
                    "F_error_to_exact_kkt",
                    "max"
                )
            )
        )

        start_summary.to_csv(
            data_dir/
            "summary_by_start.csv"
        )

    plot_problem(
        problem,
        good_nd,
        Fref,
        fig_dir
    )

    print()
    print(pd.DataFrame(
        [summary]
    ).T)

    return (
        df,
        good_nd,
        Fref,
        summary
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries = []

    for problem_name in (
        "convex",
        "concave"
    ):

        (
            df,
            good,
            ref,
            summary
        ) = solve_problem(
            problem_name,
            weight_level=8,
            eps_mu=EPS_MU
        )

        summaries.append(summary)

    pd.DataFrame(
        summaries
    ).to_csv(
        RESULTS_DIR/
        "example4_summary.csv",
        index=False
    )

    print()
    print("="*72)
    print("EXAMPLE 4 COMPLETE")
    print("="*72)
    print(
        pd.DataFrame(
            summaries
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "Results written to:",
        RESULTS_DIR.resolve()
    )
