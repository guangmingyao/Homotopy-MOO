from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ============================================================
# Experiment A1: Retrospective feasibility reassessment
# Equality tolerance = 1e-4
# Inequality tolerance = 1e-7
# No optimization reruns
# All inputs and outputs are local to A1_reassessment
# ============================================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
EQ_TOL = 1e-4
INEQ_TOL = 1e-7

OBJ = ["f1", "f2", "f3", "f4"]
XCOL = ["x1", "x2", "x3", "x4"]

def main():
    paths = {
        "populations": ROOT / "all_final_populations.csv",
        "reference": ROOT / "reference_front.csv",
        "settings": ROOT / "settings.json"
    }

    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name}: {path}")

    data = pd.read_csv(paths["populations"])
    ref = pd.read_csv(paths["reference"])
    settings = json.loads(paths["settings"].read_text(encoding="utf-8"))

    required = set(XCOL + OBJ + ["method", "run"])
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"Missing population columns: {sorted(missing)}")

    if set(OBJ) - set(ref.columns):
        raise SystemExit("Reference front must contain f1, f2, f3, f4.")

    if data.empty or ref.empty:
        raise SystemExit("Population and reference data must not be empty.")

    # Original equality constraints
    x1, x2, x3, x4 = data[XCOL].to_numpy(float).T

    eq = np.maximum(
        np.abs(x4),
        np.abs(x3**2 - x2**2)
    )

    # Original inequality and variable bounds
    g = (x1**2 - 1)**2 + x2**2 - 0.08

    ineq = np.maximum.reduce([
        np.zeros(len(data)), g,
        -2-x1, x1-2,
        -1-x2, x2-1,
        -1-x3, x3-1,
        -1-x4, x4-1
    ])

    if "eligible" in data.columns:
        eligible = (
            data["eligible"].astype(str).str.lower()
            .isin(["true", "1", "yes"]).to_numpy()
        )
    else:
        eligible = np.ones(len(data), dtype=bool)

    feasible = eligible & (eq <= EQ_TOL) & (ineq <= INEQ_TOL)

    data["A1_eq_residual"] = eq
    data["A1_ineq_violation"] = ineq
    data["A1_feasible"] = feasible

    # Reuse Experiment A normalization exactly
    ref_f = ref[OBJ].to_numpy(float)

    for key in [
        "normalization_ideal",
        "normalization_scale",
        "normalized_hv_reference_point"
    ]:
        if key not in settings:
            raise SystemExit(f"Missing normalization setting: {key}")

    ideal = np.asarray(settings["normalization_ideal"], float)
    scale = np.asarray(settings["normalization_scale"], float)
    hv_ref_point = np.asarray(
        settings["normalized_hv_reference_point"], float
    )

    if (ideal.shape != (4,) or scale.shape != (4,)
            or hv_ref_point.shape != (4,)
            or np.any(scale <= 0)):
        raise SystemExit("Invalid normalization settings.")

    R = (ref_f - ideal) / scale
    hv_indicator = HV(ref_point=hv_ref_point)
    reference_hv = float(settings.get("reference_hv", hv_indicator(R)))
    ref_tree = cKDTree(R)

    records = []
    nd_sets = []

    for (method, run), group in data.groupby(
        ["method", "run"], sort=True
    ):
        accepted = group.loc[group["A1_feasible"]].copy()

        if len(accepted):
            F = accepted[OBJ].to_numpy(float)

            _, ids = np.unique(
                np.round(F, 12), axis=0, return_index=True
            )
            accepted = accepted.iloc[np.sort(ids)].copy()

            nd = NonDominatedSorting().do(
                accepted[OBJ].to_numpy(float),
                only_non_dominated_front=True
            )
            accepted = accepted.iloc[nd].copy()

            A = (accepted[OBJ].to_numpy(float) - ideal) / scale

            hv_value = float(hv_indicator(A))
            gd = float(ref_tree.query(A)[0].mean())
            igd = float(cKDTree(A).query(R)[0].mean())

            nd_sets.append(accepted)

        else:
            hv_value = np.nan
            gd = np.nan
            igd = np.nan

        left = bool((accepted["x1"] < 0).any())
        right = bool((accepted["x1"] > 0).any())

        records.append({
            "method": method,
            "run": run,
            "n_candidates": len(group),
            "n_feasible": int(group["A1_feasible"].sum()),
            "feasible_run": int(group["A1_feasible"].any()),
            "n_nd": len(accepted),
            "left_covered": int(left),
            "right_covered": int(right),
            "both_components": int(left and right),
            "best_eq_residual": float(
                group["A1_eq_residual"].min()
            ),
            "best_combined_violation": float(
                np.maximum(
                    group["A1_eq_residual"],
                    group["A1_ineq_violation"]
                ).min()
            ),
            "HV": hv_value,
            "HV_ratio": (
                hv_value / reference_hv
                if np.isfinite(hv_value) and reference_hv > 0
                else np.nan
            ),
            "GD": gd,
            "IGD": igd,
            "AHD": max(gd, igd) if len(accepted) else np.nan
        })

    per_run = pd.DataFrame(records)

    summary = per_run.groupby("method", sort=False).agg(
        runs=("run", "size"),
        feasible_runs=("feasible_run", "sum"),
        feasible_points_mean=("n_feasible", "mean"),
        nondominated_mean=("n_nd", "mean"),
        both_components_runs=("both_components", "sum"),
        HV_mean=("HV", "mean"),
        HV_sd=("HV", "std"),
        IGD_mean=("IGD", "mean"),
        IGD_sd=("IGD", "std"),
        GD_mean=("GD", "mean"),
        AHD_mean=("AHD", "mean"),
        best_combined_violation=("best_combined_violation", "min")
    ).reset_index()

    summary["feasible_run_rate"] = (
        summary["feasible_runs"] / summary["runs"]
    )

    OUT.mkdir(parents=True, exist_ok=True)

    per_run.to_csv(OUT / "A1_per_run.csv", index=False)
    summary.to_csv(OUT / "A1_summary.csv", index=False)

    if nd_sets:
        pd.concat(nd_sets, ignore_index=True).to_csv(
            OUT / "A1_nondominated.csv", index=False
        )

    (OUT / "A1_settings.json").write_text(
        json.dumps({
            "experiment": "A1",
            "type": "Retrospective reassessment",
            "equality_tolerance": EQ_TOL,
            "inequality_tolerance": INEQ_TOL,
            "optimization_rerun": False,
            "reference": "Copied original Experiment A reference front",
            "normalization": "Copied original Experiment A settings",
            "note": "Relaxed equality feasibility does not imply exact feasibility."
        }, indent=2),
        encoding="utf-8"
    )

    print("\nEXPERIMENT A1 COMPLETED")
    print("Equality tolerance:", EQ_TOL)
    print("Inequality tolerance:", INEQ_TOL)
    print("\n", summary.to_string(index=False))
    print("\nResults saved to:", OUT)


if __name__ == "__main__":
    main()
