"""
Concave branch diagnostic for Example 4.

Requires:
    example4.py

This script DOES NOT modify Algorithm 2.  It imports the baseline
implementation and changes only the initial point x0 supplied to
run_single_path() for the concave problem.

Purpose
-------
For selected positive weights lambda, the exact interior Pareto-critical
point is

    z_i* = (1/lambda_i) / sum_j(1/lambda_j),
    x*   = (z_1*, z_2*).

We test:
  1. x0 = x* exactly;
  2. coordinate perturbations x* +/- eps e1, x* +/- eps e2;
  3. diagonal perturbations x* +/- eps (1,1)/sqrt(2);
  4. anti-diagonal perturbations x* +/- eps (1,-1)/sqrt(2).

Several perturbation magnitudes are used.  Only perturbations that remain
strictly inside the feasible triangle are run.

Outputs
-------
example4_concave_branch_diagnostic_results/
    all_paths.csv
    summary_by_weight.csv
    summary_by_epsilon.csv
    figures/
        ... diagnostic plots ...
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import example4 as E4


# ============================================================================
# Controls
# ============================================================================

RESULTS_DIR = Path("example4_concave_branch_diagnostic_results")
FIG_DIR = RESULTS_DIR / "figures"

PROBLEM = "concave"

# Representative interior weights, including the symmetric case.
WEIGHTS = {
    "equal": np.array([1/3, 1/3, 1/3], dtype=float),
    "w_2_3_3": np.array([0.25, 0.375, 0.375], dtype=float),
    "w_3_2_3": np.array([0.375, 0.25, 0.375], dtype=float),
    "w_3_3_2": np.array([0.375, 0.375, 0.25], dtype=float),
    "w_2_2_4": np.array([0.25, 0.25, 0.50], dtype=float),
}

# Small-to-moderate perturbations.
EPSILONS = [
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    5.0e-2,
]

DIRECTIONS = {
    "+e1": np.array([1.0, 0.0]),
    "-e1": np.array([-1.0, 0.0]),
    "+e2": np.array([0.0, 1.0]),
    "-e2": np.array([0.0, -1.0]),
    "+diag": np.array([1.0, 1.0]) / np.sqrt(2.0),
    "-diag": -np.array([1.0, 1.0]) / np.sqrt(2.0),
    "+anti": np.array([1.0, -1.0]) / np.sqrt(2.0),
    "-anti": -np.array([1.0, -1.0]) / np.sqrt(2.0),
}

INTERIOR_MARGIN = 1.0e-10
NEAR_EXACT_TOL = 1.0e-3
BOUNDARY_TOL = E4.BOUNDARY_TOL


# ============================================================================
# Helpers
# ============================================================================

def strictly_interior(x, margin=INTERIOR_MARGIN):
    z = E4.z_from_x(x)
    return bool(np.min(z) > margin)


def classify_endpoint(x, xstar):
    """
    Classify endpoint as:
      exact/interior  : near analytical interior KKT point
      edge-z1         : z1 ~ 0
      edge-z2         : z2 ~ 0
      edge-z3         : z3 ~ 0
      vertex-z1       : near (z1,z2,z3)=(1,0,0)
      vertex-z2       : near (0,1,0)
      vertex-z3       : near (0,0,1)
      interior-other  : interior but not near x*
    """
    z = E4.z_from_x(x)

    if np.linalg.norm(x - xstar) <= NEAR_EXACT_TOL:
        return "exact/interior"

    vertex_tol = 5.0e-4
    vertices = np.eye(3)

    for i in range(3):
        if np.linalg.norm(z - vertices[i]) <= vertex_tol:
            return f"vertex-z{i+1}"

    if z[0] <= BOUNDARY_TOL:
        return "edge-z1"
    if z[1] <= BOUNDARY_TOL:
        return "edge-z2"
    if z[2] <= BOUNDARY_TOL:
        return "edge-z3"

    return "interior-other"


def run_case(weight_name, lam0, start_label, x0, epsilon, direction):
    xstar = E4.exact_x_from_lambda(lam0)
    Fstar = E4.exact_F_from_lambda(lam0, PROBLEM)

    t0 = time.time()

    (
        omega,
        success,
        iterations,
        accepted,
        rejected,
        step_final,
        last_residual
    ) = E4.run_single_path(
        PROBLEM,
        x0,
        lam0,
        u0_scale=E4.U0_SCALE,
        step0=E4.STEP0,
        step_min=E4.STEP_MIN,
        step_max=E4.STEP_MAX,
        eps_mu=E4.EPS_MU,
        max_steps=E4.MAX_STEPS
    )

    runtime = time.time() - t0

    x, lam, u, mu = E4.unpack_omega(omega)
    F = E4.objectives(x, PROBLEM)
    z = E4.z_from_x(x)

    u0 = E4.U0_SCALE * np.ones(E4.P)

    Hnorm = float(np.linalg.norm(
        E4.omega_residual(
            omega, x0, lam0, u0, PROBLEM
        )
    ))

    (
        stationarity,
        complementarity,
        max_g,
        weight_sum_residual,
        min_lambda,
        min_u
    ) = E4.kkt_diagnostics(
        x, lam, u, PROBLEM
    )

    feasible = max_g <= E4.FEAS_TOL
    boundary = np.min(z) <= BOUNDARY_TOL

    xerr = float(np.linalg.norm(x - xstar))
    Ferr = float(np.linalg.norm(F - Fstar))

    return {
        "weight_name": weight_name,

        "w1": lam0[0],
        "w2": lam0[1],
        "w3": lam0[2],

        "start_label": start_label,
        "epsilon": epsilon,
        "direction": direction,

        "x0_1": x0[0],
        "x0_2": x0[1],

        "exact_x1": xstar[0],
        "exact_x2": xstar[1],

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

        "x_error_to_exact_kkt": xerr,
        "F_error_to_exact_kkt": Ferr,

        "lambda_movement":
            float(np.linalg.norm(lam - lam0)),

        "stationarity": stationarity,
        "complementarity": complementarity,
        "max_g": max_g,
        "weight_sum_residual": weight_sum_residual,
        "min_lambda": min_lambda,
        "min_u": min_u,

        "boundary_endpoint": bool(boundary),

        "near_exact_interior":
            bool(Ferr <= NEAR_EXACT_TOL),

        "endpoint_class":
            classify_endpoint(x, xstar),

        "runtime_seconds": runtime,
    }


# ============================================================================
# Plot one weight in decision/simplex coordinates
# ============================================================================

def plot_weight_diagnostic(weight_name, lam0, sub):
    xstar = E4.exact_x_from_lambda(lam0)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    # Feasible triangle.
    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])

    ax.plot(
        triangle[:, 0],
        triangle[:, 1],
        linewidth=1.2
    )

    ax.scatter(
        [xstar[0]],
        [xstar[1]],
        marker="*",
        s=160,
        label="Exact interior KKT"
    )

    # Starting points.
    ax.scatter(
        sub["x0_1"],
        sub["x0_2"],
        marker="o",
        s=22,
        alpha=0.55,
        label="Starts"
    )

    # Endpoints.
    ax.scatter(
        sub["x1"],
        sub["x2"],
        marker="x",
        s=42,
        label="Endpoints"
    )

    # Light start-to-end segments.
    for _, r in sub.iterrows():
        ax.plot(
            [r["x0_1"], r["x1"]],
            [r["x0_2"], r["x2"]],
            linewidth=0.5,
            alpha=0.20
        )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.set_title(
        "Concave branch diagnostic\n"
        + rf"$\lambda=({lam0[0]:.3f},{lam0[1]:.3f},{lam0[2]:.3f})$"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / f"{weight_name}_decision_space.png",
        dpi=250,
        bbox_inches="tight"
    )

    if E4.SAVE_PDF:
        fig.savefig(
            FIG_DIR / f"{weight_name}_decision_space.pdf",
            bbox_inches="tight"
        )

    plt.close(fig)


# ============================================================================
# Main experiment
# ============================================================================

def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    print("=" * 78)
    print("EXAMPLE 4(b): CONCAVE INTERIOR-BRANCH DIAGNOSTIC")
    print("=" * 78)
    print("Algorithm 2 is unchanged.")
    print()

    for weight_name, lam0 in WEIGHTS.items():

        xstar = E4.exact_x_from_lambda(lam0)
        zstar = E4.exact_z_from_lambda(lam0)

        print("-" * 78)
        print(
            f"{weight_name}: "
            f"lambda = {lam0}, "
            f"x* = {xstar}, "
            f"z* = {zstar}"
        )

        # ------------------------------------------------------------
        # Exact analytical interior start
        # ------------------------------------------------------------
        row = run_case(
            weight_name=weight_name,
            lam0=lam0,
            start_label="exact",
            x0=xstar.copy(),
            epsilon=0.0,
            direction="exact"
        )

        rows.append(row)

        print(
            "  exact start:"
            f" success={row['success']},"
            f" class={row['endpoint_class']},"
            f" F-error={row['F_error_to_exact_kkt']:.3e},"
            f" mu={row['mu']:.3e}"
        )

        # ------------------------------------------------------------
        # Perturbed starts
        # ------------------------------------------------------------
        for eps in EPSILONS:

            for dname, d in DIRECTIONS.items():

                x0 = xstar + eps*d

                if not strictly_interior(x0):
                    continue

                row = run_case(
                    weight_name=weight_name,
                    lam0=lam0,
                    start_label=f"{dname}_eps_{eps:.0e}",
                    x0=x0,
                    epsilon=eps,
                    direction=dname
                )

                rows.append(row)

        print()

    df = pd.DataFrame(rows)

    df.to_csv(
        RESULTS_DIR / "all_paths.csv",
        index=False
    )

    # ----------------------------------------------------------------
    # Summary by weight
    # ----------------------------------------------------------------
    good = (
        (df["success"] == True)
        & (df["feasible"] == True)
    )

    summary_weight = (
        df.assign(good=good)
        .groupby("weight_name")
        .agg(
            paths=("start_label", "count"),
            successful_feasible=("good", "sum"),
            near_exact_interior=("near_exact_interior", "sum"),
            boundary_endpoints=("boundary_endpoint", "sum"),
            median_F_error=("F_error_to_exact_kkt", "median"),
            min_F_error=("F_error_to_exact_kkt", "min"),
            max_F_error=("F_error_to_exact_kkt", "max"),
            max_Hnorm=("Hnorm", "max"),
            max_stationarity=("stationarity", "max"),
            max_complementarity=("complementarity", "max"),
        )
    )

    summary_weight.to_csv(
        RESULTS_DIR / "summary_by_weight.csv"
    )

    # ----------------------------------------------------------------
    # Summary by epsilon
    # ----------------------------------------------------------------
    perturbed = df[df["epsilon"] > 0.0].copy()

    if len(perturbed):
        summary_eps = (
            perturbed
            .groupby("epsilon")
            .agg(
                paths=("start_label", "count"),
                successful=("success", "sum"),
                near_exact_interior=("near_exact_interior", "sum"),
                boundary_endpoints=("boundary_endpoint", "sum"),
                median_F_error=("F_error_to_exact_kkt", "median"),
                min_F_error=("F_error_to_exact_kkt", "min"),
                max_F_error=("F_error_to_exact_kkt", "max"),
            )
        )

        summary_eps.to_csv(
            RESULTS_DIR / "summary_by_epsilon.csv"
        )

    # ----------------------------------------------------------------
    # Endpoint-class table
    # ----------------------------------------------------------------
    class_table = pd.crosstab(
        df["weight_name"],
        df["endpoint_class"]
    )

    class_table.to_csv(
        RESULTS_DIR / "endpoint_classes.csv"
    )

    # ----------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------
    for weight_name, lam0 in WEIGHTS.items():
        sub = df[
            df["weight_name"] == weight_name
        ].copy()

        plot_weight_diagnostic(
            weight_name,
            lam0,
            sub
        )

    # ----------------------------------------------------------------
    # Terminal report
    # ----------------------------------------------------------------
    print()
    print("=" * 78)
    print("SUMMARY BY WEIGHT")
    print("=" * 78)
    print(summary_weight.to_string())

    print()
    print("=" * 78)
    print("ENDPOINT CLASSES")
    print("=" * 78)
    print(class_table.to_string())

    if len(perturbed):
        print()
        print("=" * 78)
        print("SUMMARY BY PERTURBATION SIZE")
        print("=" * 78)
        print(summary_eps.to_string())

    print()
    print("Results written to:")
    print(RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
