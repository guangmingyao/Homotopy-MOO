from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

reference = pd.read_csv(
    BASE / "example2_clean" / "results" / "reference_front.csv"
)
original = pd.read_csv(
    BASE / "example2_clean" / "results" / "all_final_populations.csv"
)
b2 = pd.read_csv(
    BASE / "B2_equality_elimination" / "results" / "all_final_populations.csv"
)

homotopy = original[
    original["method"].astype(str).str.lower() == "homotopy"
].copy()

if homotopy.empty:
    raise ValueError("No homotopy solutions found in original populations.")

methods = [
    ("nsga2", "NSGA-II", "o"),
    ("nsga3", "NSGA-III", "^"),
    ("ctaea", "C-TAEA", "s"),
]

objectives = ["f1", "f2", "f3", "f4"]

fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
axes = axes.ravel()

for ax, (fx, fy) in zip(axes, combinations(objectives, 2)):

    ax.scatter(
        reference[fx], reference[fy],
        s=8, marker=".", color="0.75",
        alpha=0.65,
        label="Reference front", zorder=1
    )   

    for method, label, color in methods:
        q = b2[
            (b2["method"].astype(str).str.lower() == method)
            & (b2["run"] == 1)
        ].copy()

        if "feasible" in q.columns:
            q = q[
                q["feasible"].astype(str).str.lower().isin(["true", "1"])
            ]

        ax.scatter(
            q[fx], q[fy],
            s=24,
            marker=color,
            facecolors="none",
            edgecolors="black",
            linewidths=0.65,
            alpha=0.8,
            label=f"{label} (B2, run 1)",
            zorder=2
        )

    ax.scatter(
        homotopy[fx], homotopy[fy],
        s=25, marker="x",
        linewidths=1.0,
        color="black",
        label="Original homotopy", zorder=3
    )

    ax.set_xlabel(r"$f_{" + fx[1:] + "}$")
    ax.set_ylabel(r"$f_{" + fy[1:] + "}$")
    ax.grid(alpha=0.18)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.015)
)

fig.suptitle(
    "Example 2: Pairwise objective-space projections",
    fontsize=14
)

fig.tight_layout(rect=[0, 0.09, 1, 0.95])

for extension in ("pdf", "png"):
    path = OUT / f"objective_space_comparison.{extension}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)

plt.close(fig)

print("Reference points:", len(reference))
print("Original homotopy points:", len(homotopy))
