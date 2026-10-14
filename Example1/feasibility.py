import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
from diagnosticSampling import diagnostic_sampling

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "example1_results"
FIGURE_DIR = SCRIPT_DIR / "example1_figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})


# Your eps values
eps_values = [1e-2, 1e-1, 1.0]
results = {}

# Create subplots (3 per row)
ncols = 3
nrows = (len(eps_values) + ncols - 1) // ncols

# fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), sharex=True, sharey=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), sharex=True, sharey=True,
                         constrained_layout=True)
fig.set_constrained_layout_pads(wspace=0.06)

axes = axes.flatten()

for i, eps in enumerate(eps_values):
    curr_df, curr_df_feas = diagnostic_sampling(eps, n_samples=100000)
    curr_df.to_csv(DATA_DIR / f'diagnostic_sampling_eps_{eps}.csv', index=False)
    curr_df_feas.to_csv(DATA_DIR / f'feasible_diagnostic_sampling_eps_{eps}.csv', index=False)

    results[eps] = len(curr_df_feas)

    ax = axes[i]
    # scatter(df_projected_feasible["f1"], df_projected_feasible["f2"], s=1, alpha=0.3)

    ax.scatter(curr_df["f1"], curr_df["f2"], s=1, alpha=0.3)   # no label
    if not curr_df_feas.empty:
        ax.scatter(curr_df_feas["f1"], curr_df_feas["f2"], s=8, c="red")  # no label

    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")
    ax.set_title(f"eps={eps}, Feasible={len(curr_df_feas)}/$10^5$")
    # ax.grid(True, linestyle="--", alpha=0.5)

# Remove unused axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# --- Manual shared legend ---
all_patch = mpatches.Patch(color="blue", alpha=0.6, label="All samples")   # solid blue
feas_patch = mpatches.Patch(color="red", label="Feasible")

# fig.legend(handles=[all_patch, feas_patch],
        #    loc="upper center", ncol=2, frameon=False)

fig.legend(handles=[all_patch, feas_patch],
           loc="lower center", ncol=2, frameon=True,
           bbox_to_anchor=(0.5, 1.02), fontsize=18)

# plt.tight_layout(rect=[0, 0, 1, 0.95])

# figure_name = "diagnostic_sampling_feasibility_100000.pdf"
# plt.savefig(figure_name, bbox_inches="tight")
for ext in ["pdf", "png", "jpg"]:
    plt.savefig(FIGURE_DIR / f"diagnostic_sampling_feasibility_100000.{ext}", bbox_inches="tight", dpi=300)

plt.show()

print(results)