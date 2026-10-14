import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------
# Load CSV data
# -------------------------------------------------
df = pd.read_csv("solution_homotopy_ieee5.csv")
# -------------------------------------------------
# Create folder for figures
# -------------------------------------------------
output_dir = "pareto_figures"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------
# Objective columns
# -------------------------------------------------
f1 = "f1"   # cost
f2 = "f2"   # tracking
f3 = "f3"   # stress

# -------------------------------------------------
# Identify special solutions
# -------------------------------------------------
idx_min_cost = df[f1].idxmin()
idx_best_tracking = df[f2].idxmin()
idx_min_stress = df[f3].idxmin()

# -------------------------------------------------
# Balanced compromise solution
# Normalize objectives and choose point closest
# to the ideal point (0,0,0)
# -------------------------------------------------
obj = df[[f1, f2, f3]]

obj_norm = (obj - obj.min()) / (obj.max() - obj.min())

df["compromise_score"] = np.sqrt(
    obj_norm[f1]**2
    + obj_norm[f2]**2
    + obj_norm[f3]**2
)

idx_compromise = df["compromise_score"].idxmin()

# -------------------------------------------------
# Dictionary of highlighted solutions
# -------------------------------------------------
selected = {
    "Minimum-cost": idx_min_cost,
    "Best-tracking": idx_best_tracking,
    "Minimum-stress": idx_min_stress,
    "Balanced compromise": idx_compromise,
}

# -------------------------------------------------
# Print selected solutions
# -------------------------------------------------
print("\nSelected Solutions:\n")
print(df.loc[
    list(selected.values()),
    ["L1", "L2", "L3", "mu", "f1", "f2", "f3"]
])

# -------------------------------------------------
# 3D Pareto Plot
# -------------------------------------------------
# -------------------------------------------------
# Normalize objectives
# -------------------------------------------------
F = df[[f1, f2, f3]]

Fnorm = (F - F.min()) / (F.max() - F.min())

# -------------------------------------------------
# 3D Pareto Plot
# -------------------------------------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Plot all normalized Pareto points
ax.scatter(
    Fnorm[f1],
    Fnorm[f2],
    Fnorm[f3],
    s=22,
    alpha=0.7,
    label="Homotopy solutions"
)

# Marker styles
markers = {
    "Minimum-cost": "*",
    "Best-tracking": "^",
    "Minimum-stress": "s",
    "Balanced compromise": "D",
}

# Plot highlighted solutions USING NORMALIZED VALUES

# -------------------------------------------------
# Manual text offsets to avoid overlap
# (dx, dy, dz)
# -------------------------------------------------
offsets = {
    "Minimum-cost": (0.02, 0.02, 0.02),
    "Best-tracking": (-0.08, 0.03, 0.03),
    "Minimum-stress": (0.03, -0.06, 0.02),
    "Balanced compromise": (0.03, 0.03, 0.03),
}

# -------------------------------------------------
# Plot highlighted solutions
# -------------------------------------------------
for label, idx in selected.items():

    x = Fnorm.loc[idx, f1]
    y = Fnorm.loc[idx, f2]
    z = Fnorm.loc[idx, f3]

    # plot marker
    ax.scatter(
        x, y, z,
        s=70,
        marker=markers[label],
        label=label
    )

    # shifted label location
    dx, dy, dz = offsets[label]

    # Labels are identified in the legend.


# Labels
ax.set_xlabel("Normalized Cost Objective", labelpad=12)
ax.set_ylabel("Normalized Tracking Objective", labelpad=12)
ax.set_zlabel("Normalized Stress Objective", labelpad=18)

ax.set_title("Normalized 3D Pareto Front", pad=20)

ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))

# Add extra space on the right side
fig.subplots_adjust(left=0.02, right=0.82, bottom=0.05, top=0.92)

# Save figure
plt.savefig(
    os.path.join(output_dir, "3D_Pareto_Front.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()
# -------------------------------------------------
# 2D Pairwise Pareto Projections
# using the same four selected 3D solutions
# -------------------------------------------------

pairs = [
    (f1, f2, "Normalized Cost Objective", "Normalized Tracking Objective"),
    (f1, f3, "Normalized Cost Objective", "Normalized Stress Objective"),
    (f2, f3, "Normalized Tracking Objective", "Normalized Stress Objective"),
]

# Small label offsets for 2D plots
offsets_2d = {
    "Minimum-cost": (0.02, 0.02),
    "Best-tracking": (-0.12, 0.03),
    "Minimum-stress": (0.03, -0.06),
    "Balanced compromise": (0.03, 0.03),
}

for xcol, ycol, xlabel, ylabel in pairs:

    plt.figure(figsize=(7, 5))

    # Plot all homotopy solutions
    plt.scatter(
        Fnorm[xcol],
        Fnorm[ycol],
        s=22,
        alpha=0.7,
        label="Homotopy solutions"
    )

    # Highlight the same four selected solutions
    for label, idx in selected.items():

        x = Fnorm.loc[idx, xcol]
        y = Fnorm.loc[idx, ycol]

        plt.scatter(
            x,
            y,
            s=75,
            marker=markers[label],
            label=label
        )

        dx, dy = offsets_2d[label]

        # Use the legend instead of annotations to prevent overlap.

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs. {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Create filename automatically
    filename = f"{xcol}_vs_{ycol}.png"

    # Save figure
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


# -------------------------------------------------
# Create representative solution table
# -------------------------------------------------

table_cols = [
    "L1",
    "L2",
    "L3",
    "mu",
    "f1",
    "f2",
    "f3"
]

# Build table
rep_table = pd.DataFrame()

for label, idx in selected.items():

    row = df.loc[idx, table_cols].copy()
    row["Solution"] = label

    rep_table = pd.concat(
        [rep_table, pd.DataFrame([row])],
        ignore_index=True
    )

# Reorder columns
rep_table = rep_table[
    ["Solution", "L1", "L2", "L3", "mu", "f1", "f2", "f3"]
]

# -------------------------------------------------
# Print table
# -------------------------------------------------
print("\nRepresentative Pareto Solutions:\n")
print(rep_table)

# -------------------------------------------------
# Save CSV table
# -------------------------------------------------
rep_table.to_csv(
    os.path.join(output_dir, "representative_solutions.csv"),
    index=False
)

# -------------------------------------------------
# Export LaTeX table
# -------------------------------------------------
latex_table = rep_table.to_latex(
    index=False,
    float_format="%.6f",
    caption="Representative Pareto solutions with original objective values.",
    label="tab:representative-solutions"
)

# Save LaTeX code
with open(
    os.path.join(output_dir, "representative_solutions.tex"),
    "w"
) as f:
    f.write(latex_table)

print("\nLaTeX table saved.")

# --- Generator dispatch table ---

dispatch_table=pd.DataFrame()

for label,idx in selected.items():
    row=df.loc[idx,["P1","P2","P3","P4","P5"]].copy(); row["Solution"]=label
    dispatch_table=pd.concat([dispatch_table,pd.DataFrame([row])],ignore_index=True)

dispatch_table=dispatch_table[["Solution","P1","P2","P3","P4","P5"]]

print("\nRepresentative Generator Dispatches:\n")
print(dispatch_table)

dispatch_table.to_csv(os.path.join(output_dir,"generator_dispatches.csv"),index=False)

latex_dispatch=dispatch_table.to_latex(index=False,float_format="%.4f",caption="Generator dispatches for representative Pareto solutions of the IEEE 5-bus system.",label="tab:dispatches")

with open(os.path.join(output_dir,"generator_dispatches.tex"),"w") as f: f.write(latex_dispatch)

print("\nGenerator dispatch LaTeX table saved.")

# --- Dispatch comparison figure ---
plt.figure(figsize=(8,5))
x=np.arange(5); width=0.18
for j,(_,row) in enumerate(dispatch_table.iterrows()):
    plt.bar(x+(j-1.5)*width,[row["P1"],row["P2"],row["P3"],row["P4"],row["P5"]],width,label=row["Solution"])
plt.xticks(x,[r"$P_1$",r"$P_2$",r"$P_3$",r"$P_4$",r"$P_5$"])
plt.ylabel("Generator Output (MW)")
plt.xlabel("Generator")
plt.title("Generator Dispatch Comparison")
plt.grid(axis="y",alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"generator_dispatch_comparison.png"),dpi=300,bbox_inches="tight")
plt.show()
