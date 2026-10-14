import numpy as np
import sympy as sp
import pandas as pd
import time
from scipy.linalg import null_space
from scipy.optimize import root
import matplotlib.pyplot as plt


def get_symbolic_functions():
    """
    Original homotopy system for Example 1.
    The continuation variables are
        z = (x1,...,x5, lambda1,lambda2, u, v1,v2, mu).
    """
    x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5', real=True)
    lambda1, lambda2 = sp.symbols('lambda1 lambda2', real=True)
    u_sym, u0_sym, v1, v2, mu_sym = sp.symbols(
        'u_sym u0_sym v1 v2 mu_sym', real=True
    )
    x10, x20, x30, x40, x50 = sp.symbols(
        'x10 x20 x30 x40 x50', real=True
    )
    lambda10, lambda20 = sp.symbols(
        'lambda10 lambda20', real=True
    )

    x = sp.Matrix([x1, x2, x3, x4, x5])
    x0 = sp.Matrix([x10, x20, x30, x40, x50])
    lam = sp.Matrix([lambda1, lambda2])
    lam0 = sp.Matrix([lambda10, lambda20])
    v = sp.Matrix([v1, v2])

    # Objectives
    f1 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2
    f2 = 3*x1 + 2*x2 - sp.Rational(1, 3)*x3 + sp.Rational(1, 100)*(x4-x5)**3
    f = sp.Matrix([f1, f2])

    # Constraint convention used by the homotopy:
    # g(x) <= 0, h1(x)=h2(x)=0
    g = x1**2 + x2**2 + x3**2 + x4**2 - 10
    h1 = 4*x1 - 2*x2 + sp.Rational(4, 5)*x3 + sp.Rational(3, 5)*x4 + sp.Rational(1, 2)*x5**2
    h2 = x1 + 2*x2 - x3 - sp.Rational(1, 2)*x4 + x5 - 2
    h = sp.Matrix([h1, h2])

    grad_f = f.jacobian(x)
    grad_g = sp.Matrix([g]).jacobian(x)
    grad_h = h.jacobian(x)

    # Homotopy blocks -- kept consistent with the original implementation
    H1 = (
        (1-mu_sym)*(grad_f.T*lam + grad_g.T*u_sym)
        + grad_h.T*v
        + mu_sym*(x-x0)
    )

    H2 = h

    g0 = g.subs(list(zip(x, x0)))
    H3 = sp.Matrix([
        u_sym*g - mu_sym*u0_sym*g0
    ])

    H4_base = (1-mu_sym)*(1-(lambda1+lambda2))
    H4 = sp.Matrix([
        H4_base - mu_sym*(lambda1**(sp.Rational(3, 2))
                          - lambda10**(sp.Rational(3, 2))),
        H4_base - mu_sym*(lambda2**(sp.Rational(3, 2))
                          - lambda20**(sp.Rational(3, 2)))
    ])

    H = sp.Matrix([H1, H2, H3, H4])

    z = sp.Matrix([
        x1, x2, x3, x4, x5,
        lambda1, lambda2,
        u_sym, v1, v2, mu_sym
    ])
    DH = H.jacobian(z)

    args = [x, x0, lam, lam0, u_sym, u0_sym, v, mu_sym]

    return (
        sp.lambdify(args, H, 'numpy'),
        sp.lambdify(args, DH, 'numpy'),
        sp.lambdify(args, f1, 'numpy'),
        sp.lambdify(args, f2, 'numpy'),
        sp.lambdify(args, g, 'numpy'),
        sp.lambdify(args, h1, 'numpy'),
        sp.lambdify(args, h2, 'numpy'),
    )


def eval_func(func, x, x0, lam, lam0, u, u0, v, mu):
    return np.asarray(
        func(x, x0, lam, lam0, u, u0, v, mu),
        dtype=float
    )


def unpack_z(z):
    x = z[0:5].copy()
    lam = z[5:7].copy()
    u = float(z[7])
    v = z[8:10].copy()
    mu = float(z[10])
    return x, lam, u, v, mu


def pack_z(x, lam, u, v, mu):
    return np.concatenate([x, lam, [u], v, [mu]]).astype(float)



def correct_endpoint_mu_zero(H_func, x0, lam0, u0, z_near,
                             tol=1.0e-11, maxfev=3000):
    """
    Refine a continuation point near mu=0 by fixing mu=0 and solving
        H(x, lambda, u, v, 0) = 0
    for the 10 unknowns
        (x1,...,x5, lambda1,lambda2,u,v1,v2).

    This does not alter the continuation path. It only refines the
    endpoint of the same homotopy system.
    """
    x, lam, u, v, _ = unpack_z(z_near)
    y0 = np.concatenate([x, lam, [u], v])

    def residual(y):
        xx = y[0:5]
        ll = y[5:7]
        uu = float(y[7])
        vv = y[8:10]
        # Guard against invalid fractional powers during trial steps.
        if np.any(ll < 0.0):
            return np.full(10, 1.0e6)
        return eval_func(
            H_func, xx, x0, ll, lam0, uu, u0, vv, 0.0
        ).reshape(-1)

    sol = root(
        residual,
        y0,
        method="lm",
        options={
            "ftol": tol,
            "xtol": tol,
            "gtol": tol,
            "maxiter": maxfev
        }
    )

    y = np.asarray(sol.x, dtype=float)
    r = residual(y)

    if (not np.all(np.isfinite(y))) or (not np.all(np.isfinite(r))):
        raise RuntimeError("nonfinite endpoint correction")

    if np.any(y[5:7] < -1.0e-10):
        raise RuntimeError(
            f"endpoint correction produced negative lambda: {y[5:7]}"
        )

    return y, r, sol



def run_homotopy_analysis():
    start_time = time.time()

    # Match the weighted-sum sweep: 199 values
    weights = np.linspace(0.0, 1.0, 201)[1:-1]

    (
        H_func, DH_func,
        f1_func, f2_func,
        g_func, h1_func, h2_func
    ) = get_symbolic_functions()

    # Restore the original homotopy initialization.
    # Do NOT impose [-5,5] box constraints on the continuation variables.
    x0 = np.array([1.0, 2.0, 0.0, 1.0, 1.0])
    u0 = 1.0
    v0 = np.array([0.5, 0.2])
    mu0 = 1.0

    # Original continuation parameters
    h_initial = 0.1
    mu_tol = 1.0e-3
    max_steps = 2000

    results = []
    failures = []

    print(f"Starting homotopy analysis for {len(weights)} initial weights...")

    for idx, w0 in enumerate(weights, start=1):
        lam0 = np.array([w0, 1.0-w0])

        x = x0.copy()
        lam = lam0.copy()
        u = float(u0)
        v = v0.copy()
        mu = float(mu0)
        h_step = float(h_initial)

        status = "success"
        message = ""
        k = 1

        try:
            while abs(mu) > mu_tol:
                if k > max_steps:
                    status = "failed"
                    message = "maximum continuation steps reached"
                    break

                # Keep fractional powers real.
                if np.any(lam <= 0.0):
                    status = "failed"
                    message = f"nonpositive lambda encountered: {lam}"
                    break

                H = eval_func(
                    H_func, x, x0, lam, lam0, u, u0, v, mu
                ).reshape(-1)

                DH = eval_func(
                    DH_func, x, x0, lam, lam0, u, u0, v, mu
                )

                if not np.all(np.isfinite(H)) or not np.all(np.isfinite(DH)):
                    status = "failed"
                    message = "nonfinite H or DH"
                    break

                # The 10 x 11 Jacobian should generically have a
                # one-dimensional null space.  Use the right singular
                # vector associated with the smallest singular value.
                #
                # This is more numerically robust than rejecting the run
                # solely because scipy.linalg.null_space reports extra
                # near-null directions at a rank-sensitive point.
                U, svals, Vh = np.linalg.svd(DH, full_matrices=True)
                xi = Vh[-1, :].copy()
                xi_norm = np.linalg.norm(xi)
                if xi_norm == 0.0 or not np.isfinite(xi_norm):
                    status = "failed"
                    message = "could not compute a finite tangent"
                    break
                xi /= xi_norm

                # Orient tangent exactly as in the original determinant
                # convention when the augmented matrix is nonsingular.
                Aaug = np.vstack([DH, xi])
                sign, logabsdet = np.linalg.slogdet(Aaug)
                if sign < 0:
                    xi = -xi

                # Predictor
                z_old = pack_z(x, lam, u, v, mu)
                z_pred = z_old + h_step*xi

                # ORIGINAL correction structure retained intentionally:
                # one minimum-norm correction based on the current H,DH.
                #
                # We keep this here so the revised reviewer experiment
                # remains the same algorithm as the manuscript code.
                M = DH @ DH.T
                try:
                    y = np.linalg.solve(M, H)
                except np.linalg.LinAlgError:
                    y = np.linalg.lstsq(M, H, rcond=None)[0]

                correction = DH.T @ y
                z_new = z_pred - correction

                x, lam, u, v, mu = unpack_z(z_new)

                if mu < 0.01:
                    h_step = 0.01

                k += 1

            # Only retain successful endpoints
            if status != "success":
                raise RuntimeError(message)

            # -------------------------------------------------------
            # Endpoint refinement at the actual limiting system mu=0.
            # The continuation above is unchanged.  We now use its
            # near-endpoint only as the initial guess for H(.,0)=0.
            # -------------------------------------------------------
            z_near = pack_z(x, lam, u, v, mu)

            # Store how far the correction moves the endpoint.
            x_near = x.copy()
            lam_near = lam.copy()
            u_near = float(u)
            v_near = v.copy()
            mu_near = float(mu)

            y_corr, H_corr, endpoint_sol = correct_endpoint_mu_zero(
                H_func, x0, lam0, u0, z_near,
                tol=1.0e-11,
                maxfev=3000
            )

            x = y_corr[0:5].copy()
            lam = y_corr[5:7].copy()
            u = float(y_corr[7])
            v = y_corr[8:10].copy()
            mu = 0.0

            args = (x, x0, lam, lam0, u, u0, v, mu)

            f1_val = float(eval_func(f1_func, *args).squeeze())
            f2_val = float(eval_func(f2_func, *args).squeeze())
            g_val = float(eval_func(g_func, *args).squeeze())
            h1_val = float(eval_func(h1_func, *args).squeeze())
            h2_val = float(eval_func(h2_func, *args).squeeze())

            H_final = eval_func(H_func, *args).reshape(-1)
            H_norm = float(np.linalg.norm(H_final))
            H_inf = float(np.linalg.norm(H_final, ord=np.inf))

            endpoint_shift_x = float(np.linalg.norm(x - x_near))
            endpoint_shift_all = float(np.linalg.norm(
                y_corr - np.concatenate([
                    x_near, lam_near, [u_near], v_near
                ])
            ))

            inside_box = bool(
                np.all(x >= -5.0) and np.all(x <= 5.0)
            )

            results.append({
                "w1_initial": w0,
                "w2_initial": 1.0-w0,
                "k": k,

                "x1": x[0], "x2": x[1], "x3": x[2],
                "x4": x[3], "x5": x[4],

                "lambda1_final": lam[0],
                "lambda2_final": lam[1],
                "lambda_sum_final": lam.sum(),

                "u": u,
                "v1": v[0],
                "v2": v[1],

                "mu": mu,
                "mu_before_endpoint_correction": mu_near,

                "f1": f1_val,
                "f2": f2_val,

                "g": g_val,
                "h1": h1_val,
                "h2": h2_val,

                "H_norm": H_norm,
                "H_inf": H_inf,

                "endpoint_shift_x": endpoint_shift_x,
                "endpoint_shift_all": endpoint_shift_all,
                "endpoint_solver_success": bool(endpoint_sol.success),
                "endpoint_solver_message": str(endpoint_sol.message),

                "h_step": h_step,
                "inside_plot_box": inside_box
            })

        except Exception as exc:
            failures.append({
                "w1_initial": w0,
                "message": str(exc)
            })
            print(f"Homotopy failed for w0={w0:.6f}: {exc}")

        if idx % 20 == 0 or idx == len(weights):
            print(
                f"Completed {idx:3d}/{len(weights)}; "
                f"successful endpoints = {len(results)}"
            )

    df = pd.DataFrame(results)
    fail_df = pd.DataFrame(failures)

    df.to_csv("solution_homotopy_ex2_python_endpoint_corrected.csv", index=False)
    fail_df.to_csv("solution_homotopy_failures.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\nHomotopy: {len(df)} endpoints, {elapsed:.2f} s")
    print(f"Failures: {len(fail_df)}")

    if len(df) == 0:
        print(
            "\nNo successful homotopy endpoints were obtained. "
            "The plotting step is skipped."
        )
        return df

    print("\nEndpoint diagnostics:")
    print(f"max |h1|       = {df['h1'].abs().max():.3e}")
    print(f"max |h2|       = {df['h2'].abs().max():.3e}")
    print(f"max g           = {df['g'].max():.3e}  (feasible requires g <= 0)")
    print(f"max ||H||_2     = {df['H_norm'].max():.3e}")
    print(f"max ||H||_inf   = {df['H_inf'].max():.3e}")
    print(f"max |mu|        = {df['mu'].abs().max():.3e}")
    print(
        "max |lambda sum - 1| = "
        f"{(df['lambda_sum_final']-1.0).abs().max():.3e}"
    )
    print(
        "max endpoint shift in x = "
        f"{df['endpoint_shift_x'].max():.3e}"
    )
    print(
        "max endpoint shift (all variables) = "
        f"{df['endpoint_shift_all'].max():.3e}"
    )
    print(
        "endpoints inside [-5,5]^5 = "
        f"{int(df['inside_plot_box'].sum())}/{len(df)}"
    )

    # Clear discrete markers; no connecting line.
    plt.figure(figsize=(6, 5))
    plt.scatter(
        df["f1"], df["f2"],
        s=28,
        facecolors="none",
        edgecolors="red",
        linewidths=1.0
    )
    plt.xlabel(r"$f_1$")
    plt.ylabel(r"$f_2$")
    plt.title("Homotopy")
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    result_file = run_homotopy_analysis()
