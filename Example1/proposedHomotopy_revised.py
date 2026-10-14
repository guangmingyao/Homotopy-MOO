import numpy as np
import sympy as sp
import pandas as pd
import time
import matplotlib.pyplot as plt

from scipy.linalg import null_space


# ============================================================
# COMMON NUMERICAL PARAMETERS
# ============================================================

# Use the same weight grid later in Weighted Sum / Global Criterion
N_WEIGHTS = 199
WEIGHTS = np.linspace(0.0, 1.0, N_WEIGHTS + 2)[1:-1]

# Homotopy parameters
H_INITIAL = 0.10
H_NEAR_ZERO = 0.01
MU_SWITCH = 0.01
MU_TOL = 1.0e-3
MAX_STEPS = 2000

# Initial/reference quantities
X0 = np.array([1.0, 2.0, 0.0, 1.0, 1.0])

U0 = 1.0

# Revised formulation
V0 = np.array([0.0, 0.0])

MU0 = 1.0


# ============================================================
# SYMBOLIC PROBLEM AND HOMOTOPY
# ============================================================

def get_symbolic_functions():

    x1, x2, x3, x4, x5 = sp.symbols(
        'x1 x2 x3 x4 x5',
        real=True
    )

    lambda1, lambda2 = sp.symbols(
        'lambda1 lambda2',
        real=True
    )

    u_sym, u0_sym = sp.symbols(
        'u_sym u0_sym',
        real=True
    )

    v1, v2 = sp.symbols(
        'v1 v2',
        real=True
    )

    mu_sym = sp.symbols(
        'mu_sym',
        real=True
    )

    x10, x20, x30, x40, x50 = sp.symbols(
        'x10 x20 x30 x40 x50',
        real=True
    )

    lambda10, lambda20 = sp.symbols(
        'lambda10 lambda20',
        real=True
    )

    # --------------------------------------------------------
    # Vectors
    # --------------------------------------------------------

    x = sp.Matrix([
        x1, x2, x3, x4, x5
    ])

    x0 = sp.Matrix([
        x10, x20, x30, x40, x50
    ])

    lam = sp.Matrix([
        lambda1, lambda2
    ])

    lam0 = sp.Matrix([
        lambda10, lambda20
    ])

    v = sp.Matrix([
        v1, v2
    ])


    # ========================================================
    # ORIGINAL MULTIOBJECTIVE PROBLEM
    # ========================================================

    f1 = (
        x1**2
        + x2**2
        + x3**2
        + x4**2
        + x5**2
    )

    f2 = (
        3*x1
        + 2*x2
        - sp.Rational(1, 3)*x3
        + sp.Rational(1, 100)*(x4 - x5)**3
    )

    f = sp.Matrix([
        f1,
        f2
    ])


    # Mathematical inequality:
    #
    #       g(x) <= 0
    #
    g = (
        x1**2
        + x2**2
        + x3**2
        + x4**2
        - 10
    )


    # Equality constraints
    h1 = (
        4*x1
        - 2*x2
        + sp.Rational(4, 5)*x3
        + sp.Rational(3, 5)*x4
        + sp.Rational(1, 2)*x5**2
    )

    h2 = (
        x1
        + 2*x2
        - x3
        - sp.Rational(1, 2)*x4
        + x5
        - 2
    )

    h_eq = sp.Matrix([
        h1,
        h2
    ])


    # ========================================================
    # DERIVATIVES
    # ========================================================

    grad_f = f.jacobian(x)
    grad_g = sp.Matrix([g]).jacobian(x)
    grad_h = h_eq.jacobian(x)


    # ========================================================
    # HOMOTOPY SYSTEM
    # ========================================================

    # Stationarity block
    H1 = (
        (1 - mu_sym)
        * (
            grad_f.T * lam
            + grad_g.T * u_sym
        )
        + grad_h.T * v
        + mu_sym * (x - x0)
    )


    # Equality constraints
    H2 = h_eq


    # Complementarity deformation
    g0 = g.subs({
        x1: x10,
        x2: x20,
        x3: x30,
        x4: x40,
        x5: x50
    })

    H3 = sp.Matrix([
        u_sym * g
        - mu_sym * u0_sym * g0
    ])


    # Weight-normalization homotopy
    H4_base = (
        (1 - mu_sym)
        * (1 - lambda1 - lambda2)
    )

    H4_1 = (
        H4_base
        - mu_sym
        * (
            lambda1**sp.Rational(3, 2)
            - lambda10**sp.Rational(3, 2)
        )
    )

    H4_2 = (
        H4_base
        - mu_sym
        * (
            lambda2**sp.Rational(3, 2)
            - lambda20**sp.Rational(3, 2)
        )
    )

    H4 = sp.Matrix([
        H4_1,
        H4_2
    ])


    # Complete homotopy
    H = sp.Matrix([
        H1,
        H2,
        H3,
        H4
    ])


    # Unknown continuation vector:
    #
    # omega =
    # (x1,...,x5,lambda1,lambda2,u,v1,v2,mu)
    #
    omega = sp.Matrix([
        x1, x2, x3, x4, x5,
        lambda1, lambda2,
        u_sym,
        v1, v2,
        mu_sym
    ])

    DH = H.jacobian(omega)


    # ========================================================
    # NUMERICAL FUNCTIONS
    # ========================================================

    args = [
        x,
        x0,
        lam,
        lam0,
        u_sym,
        u0_sym,
        v,
        mu_sym
    ]

    H_func = sp.lambdify(
        args,
        H,
        'numpy'
    )

    DH_func = sp.lambdify(
        args,
        DH,
        'numpy'
    )

    f1_func = sp.lambdify(
        x,
        f1,
        'numpy'
    )

    f2_func = sp.lambdify(
        x,
        f2,
        'numpy'
    )

    g_func = sp.lambdify(
        x,
        g,
        'numpy'
    )

    h1_func = sp.lambdify(
        x,
        h1,
        'numpy'
    )

    h2_func = sp.lambdify(
        x,
        h2,
        'numpy'
    )

    return (
        H_func,
        DH_func,
        f1_func,
        f2_func,
        g_func,
        h1_func,
        h2_func
    )


# ============================================================
# UNPACK CONTINUATION VECTOR
# ============================================================

def unpack_omega(omega):

    x = omega[0:5]

    lam = omega[5:7]

    u = omega[7]

    v = omega[8:10]

    mu = omega[10]

    return x, lam, u, v, mu


# ============================================================
# HOMOTOPY FUNCTION CALL
# ============================================================

def evaluate_H(
    H_func,
    omega,
    x0,
    lambda0,
    u0
):

    x, lam, u, v, mu = unpack_omega(omega)

    return np.asarray(
        H_func(
            x,
            x0,
            lam,
            lambda0,
            u,
            u0,
            v,
            mu
        ),
        dtype=float
    ).flatten()


def evaluate_DH(
    DH_func,
    omega,
    x0,
    lambda0,
    u0
):

    x, lam, u, v, mu = unpack_omega(omega)

    return np.asarray(
        DH_func(
            x,
            x0,
            lam,
            lambda0,
            u,
            u0,
            v,
            mu
        ),
        dtype=float
    )


# ============================================================
# ONE-STEP NEWTON CORRECTOR
# ============================================================

def one_step_newton(
    omega_pred,
    H_func,
    DH_func,
    x0,
    lambda0,
    u0
):

    """
    Perform exactly ONE minimum-norm Newton correction
    at the PREDICTED point.

        omega_new
          =
        omega_pred
          -
        DH^T (DH DH^T)^(-1) H

    Important:
    H and DH are evaluated at omega_pred.
    """

    H_pred = evaluate_H(
        H_func,
        omega_pred,
        x0,
        lambda0,
        u0
    )

    DH_pred = evaluate_DH(
        DH_func,
        omega_pred,
        x0,
        lambda0,
        u0
    )

    A = DH_pred @ DH_pred.T

    try:
        y = np.linalg.solve(
            A,
            H_pred
        )

    except np.linalg.LinAlgError:

        # Robust fallback
        y = np.linalg.lstsq(
            A,
            H_pred,
            rcond=None
        )[0]

    correction = DH_pred.T @ y

    omega_new = (
        omega_pred
        - correction
    )

    return omega_new


# ============================================================
# COMPUTE TANGENT
# ============================================================

def compute_tangent(
    DH,
    previous_tangent=None
):

    Z = null_space(DH)

    if Z.shape[1] == 0:
        raise RuntimeError(
            "No null-space direction found."
        )

    xi = Z[:, 0]

    xi = xi / np.linalg.norm(xi)

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------
    #
    # First step:
    # choose direction toward decreasing mu.
    #
    # Subsequent steps:
    # preserve tangent orientation.
    #

    if previous_tangent is None:

        if xi[-1] > 0:
            xi = -xi

    else:

        if np.dot(
            xi,
            previous_tangent
        ) < 0:

            xi = -xi

    return xi


# ============================================================
# TRACE ONE HOMOTOPY PATH
# ============================================================

def trace_one_path(
    w,
    H_func,
    DH_func
):

    x0 = X0.copy()

    lambda0 = np.array([
        w,
        1.0 - w
    ])

    u0 = U0

    v0 = V0.copy()

    # --------------------------------------------------------
    # Initial continuation vector
    # --------------------------------------------------------

    omega = np.concatenate([
        x0,
        lambda0,
        [u0],
        v0,
        [MU0]
    ])

    h_step = H_INITIAL

    previous_tangent = None

    k = 0


    # ========================================================
    # CONTINUATION LOOP
    # ========================================================

    while abs(omega[-1]) > MU_TOL:

        if k >= MAX_STEPS:

            print(
                f"Warning: maximum number of steps "
                f"reached for w={w:.6f}"
            )

            break


        # ----------------------------------------------------
        # 1. Evaluate Jacobian at current corrected point
        # ----------------------------------------------------

        DH_curr = evaluate_DH(
            DH_func,
            omega,
            x0,
            lambda0,
            u0
        )


        # ----------------------------------------------------
        # 2. Tangent predictor direction
        # ----------------------------------------------------

        xi = compute_tangent(
            DH_curr,
            previous_tangent
        )


        # ----------------------------------------------------
        # 3. Predictor
        # ----------------------------------------------------

        omega_pred = (
            omega
            + h_step * xi
        )


        # ----------------------------------------------------
        # 4. ONE-STEP NEWTON CORRECTOR
        #
        # IMPORTANT:
        # H and DH are recomputed at omega_pred.
        # ----------------------------------------------------

        omega_new = one_step_newton(
            omega_pred,
            H_func,
            DH_func,
            x0,
            lambda0,
            u0
        )


        # ----------------------------------------------------
        # 5. Update
        # ----------------------------------------------------

        omega = omega_new

        previous_tangent = xi

        k += 1


        # ----------------------------------------------------
        # 6. Smaller step near mu = 0
        # ----------------------------------------------------

        if abs(omega[-1]) < MU_SWITCH:

            h_step = H_NEAR_ZERO


        # ----------------------------------------------------
        # Basic diagnostic
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(omega)
        ):

            print(
                f"Warning: nonfinite value "
                f"for w={w:.6f}"
            )

            break


    return omega, k, h_step


# ============================================================
# MAIN HOMOTOPY SWEEP
# ============================================================

def run_homotopy_analysis():

    start_time = time.time()

    (
        H_func,
        DH_func,
        f1_func,
        f2_func,
        g_func,
        h1_func,
        h2_func
    ) = get_symbolic_functions()


    all_results = []


    print(
        f"Starting revised homotopy analysis "
        f"for {len(WEIGHTS)} weights..."
    )


    for j, w in enumerate(WEIGHTS, start=1):

        omega, k, h_step = trace_one_path(
            w,
            H_func,
            DH_func
        )

        x, lam, u, v, mu = unpack_omega(
            omega
        )


        # ----------------------------------------------------
        # Final diagnostics
        # ----------------------------------------------------

        H_final = evaluate_H(
            H_func,
            omega,
            X0,
            np.array([w, 1.0-w]),
            U0
        )

        H_norm = np.linalg.norm(
            H_final
        )


        f1_val = float(
            f1_func(x)
        )

        f2_val = float(
            f2_func(x)
        )

        g_val = float(
            g_func(x)
        )

        h1_val = float(
            h1_func(x)
        )

        h2_val = float(
            h2_func(x)
        )


        all_results.append({
            'k': k,

            'x1': x[0],
            'x2': x[1],
            'x3': x[2],
            'x4': x[3],
            'x5': x[4],

            'w1_initial': w,
            'w2_initial': 1.0-w,

            'lambda1_final': lam[0],
            'lambda2_final': lam[1],

            'u': u,

            'v1': v[0],
            'v2': v[1],

            'mu': mu,

            'f1': f1_val,
            'f2': f2_val,

            'g': g_val,

            'h1': h1_val,
            'h2': h2_val,

            '|H|': H_norm,

            'h_step': h_step
        })


        if (
            j == 1
            or j % 20 == 0
            or j == len(WEIGHTS)
        ):

            print(
                f"{j:3d}/{len(WEIGHTS)}: "
                f"w1={w:.4f}, "
                f"mu={mu:.3e}, "
                f"|H|={H_norm:.3e}, "
                f"|h1|={abs(h1_val):.3e}, "
                f"|h2|={abs(h2_val):.3e}"
            )


    # ========================================================
    # DATA FRAME
    # ========================================================

    df = pd.DataFrame(
        all_results
    )


    elapsed = (
        time.time()
        - start_time
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "REVISED HOMOTOPY SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        f"Number of computed paths: "
        f"{len(df)}"
    )

    print(
        f"Execution time: "
        f"{elapsed:.3f} seconds"
    )

    print(
        f"Maximum |h1|: "
        f"{np.max(np.abs(df['h1'])):.3e}"
    )

    print(
        f"Maximum |h2|: "
        f"{np.max(np.abs(df['h2'])):.3e}"
    )

    print(
        f"Maximum g: "
        f"{np.max(df['g']):.3e}"
    )

    print(
        f"Maximum |H|: "
        f"{np.max(df['|H|']):.3e}"
    )

    print(
        f"mu range: "
        f"[{df['mu'].min():.3e}, "
        f"{df['mu'].max():.3e}]"
    )

    print(
        f"f1 range: "
        f"[{df['f1'].min():.8f}, "
        f"{df['f1'].max():.8f}]"
    )

    print(
        f"f2 range: "
        f"[{df['f2'].min():.8f}, "
        f"{df['f2'].max():.8f}]"
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_filename = (
        'solution_homotopy_revised.csv'
    )

    df.to_csv(
        output_filename,
        index=False
    )

    print(
        f"\nResults saved to "
        f"'{output_filename}'"
    )


    return df


# ============================================================
# PLOT
# ============================================================

def plot_pareto_front(df):

    plt.figure(
        figsize=(8, 5)
    )

    plt.scatter(
        df['f1'],
        df['f2'],
        s=30,
        facecolors='none',
        edgecolors='red'
    )

    plt.xlabel(
        r'$f_1$'
    )

    plt.ylabel(
        r'$f_2$'
    )

    plt.title(
        'Pareto Front - Revised Homotopy Method'
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    result = run_homotopy_analysis()

    plot_pareto_front(
        result
    )