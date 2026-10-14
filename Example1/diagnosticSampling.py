import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "example1_results"
FIGURE_DIR = SCRIPT_DIR / "example1_figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Diagnostic with looser equality tolerance
# -----------------------------

def diagnostic_sampling(eps, n_samples=200000):

    # -----------------------------
    # Define problem evaluation function
    # -----------------------------

    def evaluate(X):
        x1, x2, x3, x4, x5 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]

        # Objectives
        f1 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2
        f2 = 3*x1 + 2*x2 - (1/3)*x3 + 0.01*(x4 - x5)**3

        # Equality constraints
        h1 = 4*x1 - 2*x2 + 0.8*x3 + 0.6*x4 + 0.5*x5**2
        h2 = x1 + 2*x2 - x3 - 0.5*x4 + x5 - 2

        # Inequality constraint
        g1 = x1**2 + x2**2 + x3**2 + x4**2 - 10

        return np.column_stack([f1, f2]), h1, h2, g1


    X = np.random.uniform(-5, 5, size=(n_samples, 5))
    F, h1, h2, g1 = evaluate(X)
    feasible_mask = (np.abs(h1) <= eps) & (np.abs(h2) <= eps) & (g1 <= 0)
    F_feasible = F[feasible_mask]

        # Build dataframe
    df_all = pd.DataFrame({
        "f1": F[:, 0],
        "f2": F[:, 1],
        "h1": h1,
        "h2": h2,
        "g": g1,
        "x1": X[:, 0],
        "x2": X[:, 1],
        "x3": X[:, 2],
        "x4": X[:, 3],
        "x5": X[:, 4],
        "feasible": feasible_mask.astype(int)  # optional flag
    })
    # display("df_all: ", df_all)

    h1_feasible = h1[feasible_mask]
    h2_feasible = h2[feasible_mask]
    g1_feasible = g1[feasible_mask]
    X_feasible = X[feasible_mask]

    # Check if any feasible solutions are found before creating the DataFrame
    if len(F_feasible) > 0:
        df_feasible = pd.DataFrame({
            "f1": F_feasible[:, 0],
            "f2": F_feasible[:, 1],
            "h1": h1_feasible,
            "h2": h2_feasible,
            "g": g1_feasible,
            "x1": X_feasible[:, 0],
            "x2": X_feasible[:, 1],
            "x3": X_feasible[:, 2],
            "x4": X_feasible[:, 3],
            "x5": X_feasible[:, 4],
            "feasible": feasible_mask[feasible_mask].astype(int)  # optional flag
        })
    else:
        df_feasible = pd.DataFrame() # Return an empty DataFrame if no feasible solutions

    # display("df_feasible: ", df_feasible)
    return df_all, df_feasible


