from pathlib import Path
from itertools import combinations

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
    raise ValueError("Original homotopy solutions not found.")

METHODS = [
    ("nsga2", "NSGA-II", "#2166ac", "o"),
    ("nsga3", "NSGA-III", "#d18b16", "^"),
    ("ctaea", "C-TAEA", "#238b45", "s"),
]

objectives = ["f1", "f2", "f3", "f4"]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
axes = axes.ravel()

for ax, (fx, fy) in zip(axes, combinations(objectives, 2)):

    # Reference front: light-gray background points
    ax.scatter(
        reference[fx], reference[fy],
        s=9, marker=".", color="0.72",
        alpha=0.65, label="Reference front",
        zorder=1, rasterized=True
    )

    # B2 evolutionary solutions: distinct colors and open markers
    for method, label, color, marker in METHODS:
        q = b2[
            (b2["method"].astype(str).str.lower() == method)
            & (b2["run"] == 1)
        ].copy()

        if "feasible" in q.columns:
            q = q[
                q["feasible"].astype(str).str.lower().isin(
                    ["true", "1"]
                )
            ]

        ax.scatter(
            q[fx], q[fy],
            s=17,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.65,
            alpha=0.65,
            label=f"{label} (B2, run 1)",
            zorder=2
        )

    # Original homotopy: red crosses
    ax.scatter(
        homotopy[fx], homotopy[fy],
        s=22,
        marker="x",
        linewidths=0.95,
        color="#a50f15",
        alpha=0.90,
        label="Original homotopy",
        zorder=3
    )

    ax.set_xlabel(r"$f_{" + fx[1:] + "}$")
    ax.set_ylabel(r"$f_{" + fy[1:] + "}$")
    ax.grid(color="0.85", linewidth=0.5, alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.01)
)

fig.suptitle(
    "Example 2: Pairwise objective-space projections",
    fontsize=13
)

fig.tight_layout(rect=[0, 0.09, 1, 0.96])

for extension in ("pdf", "png"):
    path = OUT / f"objective_space_comparison.{extension}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)

plt.close(fig)
