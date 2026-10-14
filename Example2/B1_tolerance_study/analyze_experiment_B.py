from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = Path("original_populations.csv")
OUTPUT = Path("results/tolerance_study")
EQ_TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
INEQ_TOL = 1e-7

if not INPUT.exists():
    raise SystemExit(f"Missing input file: {INPUT.resolve()}")

df = pd.read_csv(INPUT)

required = {"method", "run", "x1", "x2", "x3", "x4"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

x1 = df["x1"].to_numpy(float)
x2 = df["x2"].to_numpy(float)
x3 = df["x3"].to_numpy(float)
x4 = df["x4"].to_numpy(float)

# Original equality constraints
eq = np.maximum(np.abs(x4), np.abs(x3**2 - x2**2))

# Original inequality and variable bounds
g = (x1**2 - 1)**2 + x2**2 - 0.08
ineq = np.maximum.reduce([
    np.zeros(len(df)), g,
    -2-x1, x1-2,
    -1-x2, x2-1,
    -1-x3, x3-1,
    -1-x4, x4-1
])

if "eligible" in df.columns:
    eligible = df["eligible"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    ).to_numpy()
else:
    eligible = np.ones(len(df), dtype=bool)

df["eq_residual_B1"] = eq
df["ineq_violation_B1"] = ineq

records = []

for tol in EQ_TOLS:
    feasible = eligible & (eq <= tol) & (ineq <= INEQ_TOL)

    for (method, run), group in df.groupby(["method", "run"]):
        ix = group.index.to_numpy()
        good = feasible[ix]

        records.append({
            "method": method,
            "run": run,
            "equality_tolerance": tol,
            "inequality_tolerance": INEQ_TOL,
            "n_candidates": len(ix),
            "n_feasible": int(good.sum()),
            "feasible_run": int(good.any()),
            "best_eq_residual": float(eq[ix].min()),
            "best_combined_violation": float(
                np.maximum(eq[ix], ineq[ix]).min()
            ),
            "left_covered": int(np.any(good & (x1[ix] < 0))),
            "right_covered": int(np.any(good & (x1[ix] > 0)))
        })

per_run = pd.DataFrame(records)

summary = per_run.groupby(
    ["method", "equality_tolerance"], as_index=False
).agg(
    runs=("run", "size"),
    feasible_runs=("feasible_run", "sum"),
    mean_feasible_points=("n_feasible", "mean"),
    best_eq_residual=("best_eq_residual", "min"),
    best_combined_violation=("best_combined_violation", "min"),
    left_coverage_runs=("left_covered", "sum"),
    right_coverage_runs=("right_covered", "sum")
)

summary["feasible_run_rate"] = (
    summary["feasible_runs"] / summary["runs"]
)

OUTPUT.mkdir(parents=True, exist_ok=True)

per_run.to_csv(OUTPUT / "tolerance_per_run.csv", index=False)
summary.to_csv(OUTPUT / "tolerance_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))

for method, group in summary.groupby("method"):
    group = group.sort_values("equality_tolerance")
    ax.semilogx(
        group["equality_tolerance"],
        group["feasible_run_rate"],
        marker="o",
        label=method
    )

ax.set_xlabel("Equality constraint tolerance")
ax.set_ylabel("Fraction of feasible runs")
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(EQ_TOLS)
ax.set_xticklabels(["1e-3", "1e-4", "1e-5", "1e-6", "1e-7"])
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

fig.savefig(OUTPUT / "feasible_run_rate.pdf")
fig.savefig(OUTPUT / "feasible_run_rate.png", dpi=200)
plt.close(fig)

print("\nExperiment B1: Equality-tolerance analysis")
print(summary.to_string(index=False))
print("\nResults saved to:", OUTPUT.resolve())
