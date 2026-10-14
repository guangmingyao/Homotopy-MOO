"""
Example 1: final post-processing for the six-method comparison.

This script DOES NOT rerun any optimization.  It reads raw CSV files from
``example1_results/`` (created by ``example1_run_all_methods.py``) and writes
all final metric tables and figures to ``example1_figures/``.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist


# ============================================================
# Paths and controls
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "example1_results"
OUTPUT_DIR = SCRIPT_DIR / "example1_figures"

NSGA_EQ_TOLS = [1e-1, 1e-2, 1e-3, 1e-4]
NSGA_MAIN_TOL = 1e-2
FEAS_TOL = 1.0e-8

FIG2_BASE = "figure2_six_methods_final"
NSGA_SENS_BASE = "nsga2_equality_tolerance_sensitivity_final"
TIMING_FILE = "method_timing_summary.csv"

CSV_FILES = {
    "Homotopy": "solution_homotopy_ex2_python_endpoint_corrected.csv",
    "Weighted Sum": "solution_weighted_sum.csv",
    "Global Criterion": "solution_global_criterion.csv",
    "Lexicographical": "solution_lexicographical.csv",
    "Epsilon-Constraint": "solution_epsilon_constraint.csv",
}

NSGA_CSV_FILES = {
    1e-1: "solution_nsga2_eqtol_1em01.csv",
    1e-2: "solution_nsga2_eqtol_1em02.csv",
    1e-3: "solution_nsga2_eqtol_1em03.csv",
    1e-4: "solution_nsga2_eqtol_1em04.csv",
}

REFERENCE_METHODS = [
    "Homotopy",
    "Weighted Sum",
    "Global Criterion",
    "Epsilon-Constraint",
]


# ============================================================
# Front utilities
# ============================================================
def objective_array(df):
    if df is None or len(df) == 0:
        return np.empty((0, 2))
    return df[["f1", "f2"]].to_numpy(dtype=float)


def nondominated_mask(F, atol=1e-12):
    """Minimization nondominance mask for a 2-objective array."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i] + atol, axis=1) & np.any(
            F < F[i] - atol, axis=1
        )
        if np.any(dominated):
            keep[i] = False
    return keep


def clean_front(F, decimals=12):
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return F.reshape(0, 2)
    F = F[np.all(np.isfinite(F), axis=1)]
    F = np.unique(np.round(F, decimals=decimals), axis=0)
    F = F[nondominated_mask(F)]
    return F[np.argsort(F[:, 0])]


def normalization_from_reference(R):
    lo = np.min(R, axis=0)
    hi = np.max(R, axis=0)
    scale = hi - lo
    scale[scale <= 1e-14] = 1.0
    return lo, scale


def normalize(F, lo, scale):
    return (np.asarray(F, dtype=float) - lo) / scale


def nearest_distances(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.array([])
    return np.min(cdist(A, B), axis=1)


def spacing_statistics(Fn):
    """Adjacent-point spacing after sorting by normalized f1."""
    if len(Fn) < 2:
        return np.nan, np.nan, np.nan, np.nan
    S = Fn[np.argsort(Fn[:, 0])]
    gaps = np.linalg.norm(np.diff(S, axis=0), axis=1)
    mean_gap = float(np.mean(gaps))
    std_gap = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
    cv = std_gap / mean_gap if mean_gap > 0 else np.nan
    return mean_gap, std_gap, cv, float(np.max(gaps))


def front_metrics(F, R, lo, scale):
    """Metrics in normalized objective space."""
    F = clean_front(F)
    if len(F) == 0:
        return {k: np.nan for k in [
            "n_nondominated", "GD_RMS", "IGD_mean", "Hausdorff",
            "spacing_mean", "spacing_std", "spacing_CV", "max_gap",
            "f1_coverage", "f2_coverage",
        ]}

    Fn = normalize(F, lo, scale)
    Rn = normalize(R, lo, scale)
    d_F_R = nearest_distances(Fn, Rn)
    d_R_F = nearest_distances(Rn, Fn)
    mean_gap, std_gap, cv, max_gap = spacing_statistics(Fn)

    ref_span = np.ptp(R, axis=0)
    span = np.ptp(F, axis=0)
    coverage = np.divide(
        span, ref_span,
        out=np.full(2, np.nan),
        where=ref_span > 1e-14,
    )

    return {
        "n_nondominated": len(F),
        "GD_RMS": float(np.sqrt(np.mean(d_F_R**2))),
        "IGD_mean": float(np.mean(d_R_F)),
        "Hausdorff": float(max(np.max(d_F_R), np.max(d_R_F))),
        "spacing_mean": mean_gap,
        "spacing_std": std_gap,
        "spacing_CV": cv,
        "max_gap": max_gap,
        "f1_coverage": float(coverage[0]),
        "f2_coverage": float(coverage[1]),
    }


def pairwise_hausdorff(fronts, lo, scale):
    names = list(fronts.keys())
    M = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        A = clean_front(objective_array(fronts[a]))
        if len(A) == 0:
            continue
        An = normalize(A, lo, scale)
        for j, b in enumerate(names):
            B = clean_front(objective_array(fronts[b]))
            if len(B) == 0:
                continue
            Bn = normalize(B, lo, scale)
            ab = nearest_distances(An, Bn)
            ba = nearest_distances(Bn, An)
            M[i, j] = max(np.max(ab), np.max(ba))
    return pd.DataFrame(M, index=names, columns=names)


# ============================================================
# Data loading and reference construction
# ============================================================
def load_required_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}\n"
            "Run example1_run_all_methods.py first."
        )
    return pd.read_csv(path)


def common_feasibility_mask(df, tol=FEAS_TOL):
    """Strict common feasibility: |h1|<=tol, |h2|<=tol, g<=tol."""
    if df is None or len(df) == 0:
        return np.zeros(0, dtype=bool)

    needed = {"h1", "h2", "g"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(
            "Cannot apply common feasibility filter; missing columns: "
            f"{sorted(missing)}"
        )

    return (
        (np.abs(df["h1"].to_numpy(dtype=float)) <= tol)
        & (np.abs(df["h2"].to_numpy(dtype=float)) <= tol)
        & (df["g"].to_numpy(dtype=float) <= tol)
    )


def build_highly_feasible_deterministic_reference(fronts, tol=FEAS_TOL):
    """
    Reference = nondominated union of highly feasible deterministic
    front-generating methods only.
    """
    arrays = []
    summary = []

    for name in REFERENCE_METHODS:
        df = fronts[name]
        mask = common_feasibility_mask(df, tol)
        feasible_df = df.loc[mask].copy()
        summary.append({
            "Method": name,
            "n_raw": len(df),
            "n_highly_feasible": len(feasible_df),
        })
        if len(feasible_df):
            arrays.append(objective_array(feasible_df))

    if not arrays:
        raise RuntimeError(
            "No highly feasible deterministic solutions were available "
            "to construct the empirical reference front."
        )

    return clean_front(np.vstack(arrays)), pd.DataFrame(summary)


# ============================================================
# Plotting
# ============================================================
def save_figure(fig, base):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "jpg"]:
        path = OUTPUT_DIR / f"{base}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext != "pdf":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        print(f"Saved: {path}")


def load_feasible_background():
    """
    Load the projected feasible-point background.

    Preferred location:
        example1_results/projected_feasible_solution.csv

    For compatibility, also check the Example1 root directory.
    """
    candidates = [
        DATA_DIR / "projected_feasible_solution.csv",
        SCRIPT_DIR / "projected_feasible_solution.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"Loading projected feasible background: {path}")
            return pd.read_csv(path)

    searched = "\n  ".join(str(p) for p in candidates)
    print(
        "NOTE: projected_feasible_solution.csv was not found. "
        "Searched:\n  " + searched +
        "\nFigures will be drawn without the sampled-feasible background."
    )
    return None


def figure2(fronts, feasible=None):
    titles = [
        "Homotopy", "Weighted Sum", "Global Criterion",
        rf"Genetic Algorithm NSGA-II ($\epsilon_h={NSGA_MAIN_TOL:g}$)",
        "Lexicographical", r"$\epsilon$-Constraint",
    ]
    keys = [
        "Homotopy", "Weighted Sum", "Global Criterion",
        "NSGA-II", "Lexicographical", "Epsilon-Constraint",
    ]

    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 11), sharex=True, sharey=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(wspace=0.06, hspace=0.06)

    for ax, key, title in zip(axes.flat, keys, titles):
        if feasible is not None:
            ax.scatter(
                feasible.f1, feasible.f2,
                c="goldenrod", s=2, alpha=0.25,
                edgecolors="none", rasterized=True,
            )
        df = fronts[key]
        if df is not None and len(df):
            ax.scatter(
                df.f1, df.f2,
                marker="o", facecolors="white", edgecolors="red",
                linewidths=1.2, s=38, zorder=3,
            )
        ax.set_title(title)
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$f_2$")

    handles = []
    if feasible is not None:
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor="goldenrod", markeredgecolor="goldenrod",
            markersize=7, alpha=0.6, label="Sampled feasible points",
        ))
    handles.append(Line2D(
        [0], [0], marker="o", linestyle="None",
        markerfacecolor="white", markeredgecolor="red",
        markeredgewidth=1.2, markersize=7, label="Method solution",
    ))
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        frameon=True, bbox_to_anchor=(0.5, 1.01), fontsize=18,
    )
    save_figure(fig, FIG2_BASE)
    return fig


def nsga_sensitivity_figure(nsga_runs, feasible=None):
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, tol in zip(axes.flat, NSGA_EQ_TOLS):
        if feasible is not None:
            ax.scatter(
                feasible.f1, feasible.f2,
                c="goldenrod", s=2, alpha=0.25,
                edgecolors="none", rasterized=True,
            )
        df = nsga_runs[tol]
        if len(df):
            ax.scatter(
                df.f1, df.f2,
                marker="o", facecolors="white", edgecolors="red",
                linewidths=1.0, s=32,
            )
        ax.set_title(rf"NSGA-II: $\epsilon_h={tol:g}$, $N={len(df)}$")
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$f_2$")
    save_figure(fig, NSGA_SENS_BASE)
    return fig


# ============================================================
# Final post-processing
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("FINAL POST-PROCESSING -- NO OPTIMIZATION WILL BE RERUN")
    print("=" * 72)
    print(f"Reading raw data from:\n  {DATA_DIR}")
    print(f"Writing final outputs to:\n  {OUTPUT_DIR}")

    loaded = {
        name: load_required_csv(filename)
        for name, filename in CSV_FILES.items()
    }
    nsga_runs = {
        tol: load_required_csv(filename)
        for tol, filename in NSGA_CSV_FILES.items()
    }

    main_fronts = {
        "Homotopy": loaded["Homotopy"],
        "Weighted Sum": loaded["Weighted Sum"],
        "Global Criterion": loaded["Global Criterion"],
        "NSGA-II": nsga_runs[NSGA_MAIN_TOL],
        "Lexicographical": loaded["Lexicographical"],
        "Epsilon-Constraint": loaded["Epsilon-Constraint"],
    }
    all_fronts = dict(main_fronts)
    for tol, df in nsga_runs.items():
        all_fronts[f"NSGA-II tol={tol:g}"] = df

    R, ref_summary = build_highly_feasible_deterministic_reference(
        loaded, FEAS_TOL
    )
    pd.DataFrame(R, columns=["f1", "f2"]).to_csv(
        OUTPUT_DIR / "empirical_reference_front_corrected.csv", index=False
    )
    ref_summary.to_csv(
        OUTPUT_DIR / "reference_feasibility_summary.csv", index=False
    )

    print(f"\nReference feasibility tolerance = {FEAS_TOL:.1e}")
    print("\nSolutions eligible to define the deterministic reference:")
    print(ref_summary.to_string(index=False))
    print(f"\nNondominated points in corrected reference = {len(R)}")

    lo, scale = normalization_from_reference(R)

    rows = []
    for name, df in all_fronts.items():
        Fraw = objective_array(df)
        m = front_metrics(Fraw, R, lo, scale)
        m["Method"] = name
        m["n_raw"] = len(Fraw)

        if len(df) and "max_abs_h" in df.columns:
            m["max_abs_h"] = float(df["max_abs_h"].max())
        elif len(df) and {"h1", "h2"}.issubset(df.columns):
            m["max_abs_h"] = float(np.maximum(
                np.abs(df["h1"].to_numpy(dtype=float)),
                np.abs(df["h2"].to_numpy(dtype=float)),
            ).max())
        else:
            m["max_abs_h"] = np.nan

        m["max_g"] = float(df["g"].max()) if len(df) and "g" in df.columns else np.nan

        if len(df) and {"h1", "h2", "g"}.issubset(df.columns):
            strict_mask = common_feasibility_mask(df, FEAS_TOL)
            m["n_strict_feasible"] = int(strict_mask.sum())
            m["strict_feasible_fraction"] = float(strict_mask.mean())
        else:
            m["n_strict_feasible"] = np.nan
            m["strict_feasible_fraction"] = np.nan

        rows.append(m)

    metrics = pd.DataFrame(rows).set_index("Method")
    metrics.to_csv(OUTPUT_DIR / "method_comparison_metrics_corrected.csv")

    print("\nCORRECTED NORMALIZED FRONT METRICS")
    print("(Reference = highly feasible deterministic nondominated union)")
    print(metrics.to_string(float_format=lambda x: f"{x:.6e}"))

    Hmat = pairwise_hausdorff(all_fronts, lo, scale)
    Hmat.to_csv(OUTPUT_DIR / "pairwise_hausdorff_matrix_corrected.csv")

    timing_path = DATA_DIR / TIMING_FILE
    if timing_path.exists():
        timing_df = pd.read_csv(timing_path)
        timing_df.to_csv(
            OUTPUT_DIR / "method_timing_summary_corrected.csv", index=False
        )
    else:
        timing_df = None
        print(f"\nNOTE: {timing_path} was not found; timing summary was not copied.")

    feasible = load_feasible_background()
    figure2(main_fronts, feasible)
    nsga_sensitivity_figure(nsga_runs, feasible)

    # Homotopy vs. Weighted Sum overlay
    H = objective_array(loaded["Homotopy"])
    W = objective_array(loaded["Weighted Sum"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        H[:, 0], H[:, 1], marker="o", facecolors="none",
        edgecolors="red", s=55, linewidths=1.4, label="Homotopy",
    )
    ax.scatter(
        W[:, 0], W[:, 1], marker="x", c="black",
        s=35, linewidths=1.2, label="Weighted Sum",
    )
    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")
    ax.set_title("Homotopy vs. Weighted Sum")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "homotopy_vs_weighted_sum_overlay_corrected_metrics")

    # Correctly normalized Homotopy -> Weighted Sum discrepancy diagnostic
    Hn = normalize(H, lo, scale)
    Wn = normalize(W, lo, scale)
    d = nearest_distances(Hn, Wn)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        H[:, 0], np.maximum(d, 1e-16),
        facecolors="none", edgecolors="black", s=35,
    )
    ax.set_xlabel(r"$f_1$ (Homotopy)")
    ax.set_ylabel("Normalized distance to nearest Weighted Sum point")
    ax.set_title("Homotopy--Weighted Sum Discrepancy")
    ax.set_yscale("log")
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "homotopy_weighted_sum_normalized_distance_corrected")

    print("\nFinal tables and figures written to:")
    print(f"  {OUTPUT_DIR}")
    print("\nInterpretation:")
    print(
        "  * The empirical reference is defined ONLY by Homotopy, Weighted "
        "Sum, Global Criterion, and epsilon-Constraint solutions satisfying "
        f"the common {FEAS_TOL:.0e} feasibility threshold."
    )
    print("  * NSGA-II does NOT contribute points to the reference front.")
    print(
        "  * Compare NSGA-II front metrics together with max_abs_h and "
        "strict_feasible_fraction."
    )
    print(
        "  * Lexicographical is evaluated against the reference but is not "
        "used to construct it because it naturally provides only endpoints."
    )

    plt.show()


if __name__ == "__main__":
    main()
