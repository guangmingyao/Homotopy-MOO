import numpy as np
import pandas as pd
from scipy.linalg import null_space, det
import time
from IPython.display import display
import os
import matplotlib.pyplot as plt 


def nullspace_right_vector(A, rtol=1e-12):
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    xi = Vt[-1, :]
    xi /= np.linalg.norm(xi)
    return xi


def get_system_matrices():
    # Figure 13: buses 1--5, six lines, one generator per bus.
    # Full-precision DC susceptance matrix from line reactances (p.u.).
    branch_data = [
        (1, 2, 0.0281), (1, 4, 0.0304), (1, 5, 0.0064),
        (2, 3, 0.0108), (3, 4, 0.0297), (4, 5, 0.0297),
    ]
    Bbus = np.zeros((5, 5), dtype=float)
    for i_bus, j_bus, xij in branch_data:
        i, j = i_bus - 1, j_bus - 1
        bij = 1.0 / xij
        Bbus[i, i] += bij
        Bbus[j, j] += bij
        Bbus[i, j] -= bij
        Bbus[j, i] -= bij

    Ag = np.eye(5)
    Pd = np.array([0., 0., 300., 300., 400.])
    b = np.array([14., 15., 30., 40., 10.])
    Pg_ref = np.array([40., 170., 323.4948, 0., 466.5052])

    return Bbus, Ag, Pd, b, Pg_ref

def eval_physics(vars_in, u0_val, vars0_val):
    Pg = vars_in[0:5]
    th = vars_in[5:10]
    Bbus, Ag, Pd, b, Pg_ref = get_system_matrices()
    base = 100
    b12, b45 = 1.0/0.0281, 1.0/0.0297

    # Objectives
    f1 = np.dot(b, Pg)
    f2 = np.sum((Pg - Pg_ref)**2)
    f3 = (base * b12 * (th[0] - th[1]) / 400)**2 + (base * b45 * (th[3] - th[4]) / 240)**2

    # Equalities (h)
    h = Ag @ Pg - Pd - 100 * (Bbus @ th)
    h = np.append(h, th[3]) # Adding slack-like constraint th(4)=0

    # Inequalities (g)
    g = -np.array([
        400 - base*b12*(th[0]-th[1]), 400 + base*b12*(th[0]-th[1]),
        240 - base*b45*(th[3]-th[4]), 240 + base*b45*(th[3]-th[4]),
        40-Pg[0], 170-Pg[1], 520-Pg[2], 200-Pg[3], 600-Pg[4],
        Pg[0], Pg[1], Pg[2], Pg[3], Pg[4],
        np.pi-th[0], np.pi+th[0], np.pi-th[1], np.pi+th[1],
        np.pi-th[2], np.pi+th[2], np.pi-th[3], np.pi+th[3],
        np.pi-th[4], np.pi+th[4]
    ])

    return f1, f2, f3, h, g

def get_H_and_DH(vars_in, lam, u, v, mu, u0_val, vars0_val,lam0):
    Pg = vars_in[0:5]
    th = vars_in[5:10]
    Bbus, Ag, Pd, b, Pg_ref = get_system_matrices()
    f1, f2, f3, h, g = eval_physics(vars_in, u0_val, vars0_val)
    base, b12, b45 = 100, 1.0/0.0281, 1.0/0.0297

    # --- Gradients ---
    # Df (3 x 10)
    Df1 = np.concatenate([b, np.zeros(5)])
    Df2 = np.concatenate([2*(Pg - Pg_ref), np.zeros(5)])
    Df3_th0 = 2 * (base*b12/400)**2 * (th[0]-th[1])
    Df3_th3 = 2 * (base*b45/240)**2 * (th[3]-th[4])
    Df3 = np.array([0,0,0,0,0, Df3_th0, -Df3_th0, 0, Df3_th3, -Df3_th3])
    Df = np.vstack([Df1, Df2, Df3])

    # Dh (6 x 10)
    Dh_top = np.hstack([Ag, -100*Bbus])
    Dh_bot = np.zeros((1, 10))
    Dh_bot[0, 8] = 1 # th(4) index is 8
    Dh = np.vstack([Dh_top, Dh_bot])

    # Dg (24 x 10)
    Dg = np.zeros((24, 10))
    Dg[0, 5], Dg[0, 6] = base*b12, -base*b12
    Dg[1, 5], Dg[1, 6] = -base*b12, base*b12
    Dg[2, 8], Dg[2, 9] = base*b45, -base*b45
    Dg[3, 8], Dg[3, 9] = -base*b45, base*b45
    for i in range(5): Dg[4+i, i] = 1; Dg[9+i, i] = -1
    for i in range(5): Dg[14+2*i, 5+i] = 1; Dg[15+2*i, 5+i] = -1

    # --- Homotopy H ---
    H1 = (1-mu)*(Df.T @ lam + Dg.T @ u) + Dh.T @ v + mu*(vars_in - vars0_val)
    H2 = h
    _, _, _, _, g0 = eval_physics(vars0_val, u0_val, vars0_val)
    H3 = u * g - mu * u0_val * g0
    # H4: homotopy for objective weights
    H4 = ((1 - mu) * (1 - np.sum(lam)) * np.ones(3)- mu * (lam**1.5 - lam0**1.5))

    H = np.concatenate([H1, H2, H3, H4])

    # --- Jacobian DH (Manual Assembly) ---
    # DH is (43 x 44) -> [dH/dx, dH/dlam, dH/du, dH/dv, dH/dmu]
    # This part involves second derivatives for H1 w.r.t x
    D2f2 = np.zeros((10,10)); D2f2[:5,:5] = 2*np.eye(5)
    D2f3 = np.zeros((10,10))
    val3_0 = 2*(base*b12/400)**2; val3_3 = 2*(base*b45/240)**2
    D2f3[5,5]=val3_0; D2f3[5,6]=-val3_0; D2f3[6,5]=-val3_0; D2f3[6,6]=val3_0
    D2f3[8,8]=val3_3; D2f3[8,9]=-val3_3; D2f3[9,8]=-val3_3; D2f3[9,9]=val3_3

    D2L = lam[1]*D2f2 + lam[2]*D2f3

    dH1_dx = (1-mu)*D2L + mu*np.eye(10)
    dH1_dlam = (1-mu)*Df.T
    dH1_du = (1-mu)*Dg.T
    dH1_dv = Dh.T
    dH1_dmu = -(Df.T @ lam + Dg.T @ u) + (vars_in - vars0_val)

    dH2_dx = Dh

    dH3_dx = (u[:, None] * Dg)
    dH3_du = np.diag(g)
    dH3_dmu = -u0_val * g0

    # Jacobian with respect to lambda
    dH4_dlam = (-(1 - mu) * np.ones((3, 3))- mu * np.diag(1.5 * np.sqrt(lam)))
    # Jacobian with respect to mu
    dH4_dmu = ((np.sum(lam) - 1) * np.ones(3)- (lam**1.5 - lam0**1.5))

    # Assemble rows
    R1 = np.hstack([dH1_dx, dH1_dlam, dH1_du, dH1_dv, dH1_dmu.reshape(-1,1)])
    R2 = np.hstack([dH2_dx, np.zeros((6,3)), np.zeros((6,24)), np.zeros((6,6)), np.zeros((6,1))])
    R3 = np.hstack([dH3_dx, np.zeros((24,3)), dH3_du, np.zeros((24,6)), dH3_dmu.reshape(-1,1)])
    R4 = np.hstack([np.zeros((3,10)), dH4_dlam, np.zeros((3,24)), np.zeros((3,6)), dH4_dmu.reshape(-1,1)])

    DH = np.vstack([R1, R2, R3, R4])
    return H, DH



def check_full_jacobian(z, mu, u0, vars0, lam0, eps=1e-6):
    """
    Compare the full analytical Jacobian with
    central finite differences.

    z = [vars_in (10), lambda (3), u (24), v (6)]
    """

    def evaluate(zvec, muval):
        return get_H_and_DH(
            zvec[:10],
            zvec[10:13],
            zvec[13:37],
            zvec[37:43],
            muval,
            u0, vars0, lam0
        )

    H, J = evaluate(z, mu)
    J_numeric = np.zeros_like(J)

    # Derivatives with respect to the 43 unknowns
    for j in range(43):
        step = eps * max(1.0, abs(z[j]))

        zp = z.copy()
        zm = z.copy()
        zp[j] += step
        zm[j] -= step

        Hp, _ = evaluate(zp, mu)
        Hm, _ = evaluate(zm, mu)

        J_numeric[:, j] = (Hp - Hm) / (2 * step)

    # Derivative with respect to mu
    Hp, _ = evaluate(z, mu + eps)
    Hm, _ = evaluate(z, mu - eps)

    J_numeric[:, 43] = (Hp - Hm) / (2 * eps)

    error = np.abs(J - J_numeric)

    row, col = np.unravel_index(
        np.argmax(error), error.shape
    )

    print("Full Jacobian maximum error:",
          np.max(error))
    print("Worst entry: row =", row,
          "column =", col)
    print("Analytical value:", J[row, col])
    print("Numerical value:", J_numeric[row, col])


def correct_endpoint(z, u0, vars0, lam0, tol=1e-9, maxiter=30):
    """
    Endpoint Newton correction at mu = 0 with fixed
    prescribed objective weights.

    z = [x(10), lambda(3), u(24), v(6)]
    """

    z = np.asarray(z, dtype=float).copy()
    lam0 = np.asarray(lam0, dtype=float)

    def evaluate(zvec):
        x = zvec[:10]
        lam = zvec[10:13]
        u = zvec[13:37]
        v = zvec[37:43]

        H, DH = get_H_and_DH(
            x, lam, u, v, 0.0,
            u0, vars0, lam0
        )

        H = H.copy()
        J = DH[:, :43].copy()

        # Replace the final three H4 equations
        # with independent weight constraints.
        H[-3] = lam[0] - lam0[0]
        H[-2] = lam[1] - lam0[1]
        H[-1] = np.sum(lam) - 1.0

        J[-3:, :] = 0.0
        J[-3, 10] = 1.0
        J[-2, 11] = 1.0
        J[-1, 10:13] = 1.0

        return H, J

    for iteration in range(maxiter):

        H, J = evaluate(z)
        residual = np.linalg.norm(H, ord=np.inf)

        if not np.isfinite(residual):
            return z, False, residual

        if residual < tol:
            return z, True, residual

        try:
            delta = np.linalg.solve(J, -H)
        except np.linalg.LinAlgError:
            return z, False, residual

        alpha = 1.0
        accepted = False

        while alpha >= 1e-10:
            z_trial = z + alpha * delta

            # Require nonnegative objective weights
            # and inequality multipliers.
            if (np.min(z_trial[10:13]) < -1e-12 or
                np.min(z_trial[13:37]) < -1e-12):
                alpha *= 0.5
                continue

            H_trial, _ = evaluate(z_trial)
            residual_trial = np.linalg.norm(
                H_trial, ord=np.inf
            )

            if (np.isfinite(residual_trial) and
                residual_trial < residual):
                z = z_trial
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            return z, False, residual

    H, _ = evaluate(z)
    residual = np.linalg.norm(H, ord=np.inf)

    return z, residual < tol, residual


def solve_homotopy():
    # CAUTION: The original H4 homotopy equation/Jacobian require
    # separate verification. Correcting network data alone does not
    # establish convergence or mathematical validity of this solver.
    # Wset = np.array([
    #     [0.0754, 0.1508, 0.7738], [0.0754, 0.2262, 0.6984], [0.0754, 0.3016, 0.6250],
    #     [0.0754, 0.3770, 0.5476], [0.0754, 0.4524, 0.4722], [0.1508, 0.0754, 0.7738]
    #     # ... truncated for brevity, add full Wset here
    # ])

    Wset = np.array([
        # [0.075, 0.075, 0.850],
        [0.075, 0.150, 0.775], [0.075, 0.225, 0.700],
        [0.075, 0.300, 0.625], [0.075, 0.375, 0.550], [0.075, 0.450, 0.475],
        [0.075, 0.525, 0.400], [0.075, 0.600, 0.325], [0.075, 0.675, 0.250],
        [0.075, 0.750, 0.175], [0.075, 0.825, 0.100], [0.150, 0.075, 0.775],
        [0.150, 0.150, 0.700], [0.150, 0.225, 0.625], [0.150, 0.300, 0.550],
        [0.150, 0.375, 0.475], [0.150, 0.450, 0.400], [0.150, 0.525, 0.325],
        [0.150, 0.600, 0.250], [0.150, 0.675, 0.175], [0.150, 0.750, 0.100],
        [0.225, 0.075, 0.700], [0.225, 0.150, 0.625], [0.225, 0.225, 0.550],
        [0.225, 0.300, 0.475], [0.225, 0.375, 0.400], [0.225, 0.450, 0.325],
        [0.225, 0.525, 0.250], [0.225, 0.600, 0.175], [0.225, 0.675, 0.100],
        [0.300, 0.075, 0.625], [0.300, 0.150, 0.550], [0.300, 0.225, 0.475],
        [0.300, 0.300, 0.400], [0.300, 0.375, 0.325], [0.300, 0.450, 0.250],
        [0.300, 0.525, 0.175], [0.300, 0.600, 0.100], [0.375, 0.075, 0.550],
        [0.375, 0.150, 0.475], [0.375, 0.225, 0.400], [0.375, 0.300, 0.325],
        [0.375, 0.375, 0.250], [0.375, 0.450, 0.175], [0.375, 0.525, 0.100],
        [0.450, 0.075, 0.475], [0.450, 0.150, 0.400], [0.450, 0.226, 0.324],
        [0.450, 0.301, 0.249], [0.450, 0.377, 0.173]
        ])

    results = []
    start_time = time.time()

    for i, w in enumerate(Wset):
        # if i == 0:
        #     vars_curr = np.array([10, 140, 320, 7, 400, 0, -0.05, -0.09, -0.04, 0.03])
        # else:
        #     vars_curr = vars_curr.copy()
            # u_curr = u_curr.copy()
            # v_curr = v_curr.copy()



        vars_curr = np.array([10, 140, 320, 7, 400, 0, -0.05, -0.09, -0.04, 0.03])

        # print(f"vars_curr = {vars_curr}")
        lambda_curr = w.copy()
        lambda0 = w.copy()
        u_curr = 0.5 * np.ones(24)
        v_curr = np.array([0.5, 0.2, 0.5, 0.2, 0.5, 0.2])
        mu_curr = 1.0

        vars0, u0 = vars_curr.copy(), u_curr.copy()
        h_step, k = 1, 0

        H, DH = get_H_and_DH(vars_curr, lambda_curr, u_curr, v_curr, mu_curr, u0, vars0,lambda0)
        
        z_last_positive = np.concatenate([vars_curr, lambda_curr, u_curr, v_curr])
        mu_last_positive = mu_curr

        while abs(mu_curr) > 1e-3 and k < 1000:
            if mu_curr > 0:
                z_last_positive = np.concatenate([vars_curr, lambda_curr, u_curr, v_curr])
                mu_last_positive = mu_curr
            
            # Predictor
            # xi = null_space(DH)
            # xi = xi[:, 0]
            
            xi = nullspace_right_vector(DH)
            # _, _, Vh = np.linalg.svd(DH, full_matrices=False)
            # xi = Vh[-1]

            # Orientation check
            aug = np.vstack([DH, xi])
            if det(aug) < 0: xi = -xi

            omega = np.concatenate([vars_curr, lambda_curr, u_curr, v_curr, [mu_curr]]) + h_step * xi

            # Corrector (Newton)
            vars_curr, lambda_curr = omega[0:10], omega[10:13]
            u_curr, v_curr, mu_curr = omega[13:37], omega[37:43], omega[43]
            H, DH = get_H_and_DH(vars_curr, lambda_curr, u_curr, v_curr, mu_curr, u0, vars0,lambda0)
            delta = DH.T @ np.linalg.solve(DH @ DH.T, H)
            omega -= delta

            # Corrector (Newton)
            # for _ in range(3): # Small fixed number of iterations for correction
            #     v_x = omega[:10]; v_l = omega[10:13]; v_u = omega[13:37]; v_v = omega[37:43]; v_m = omega[43]
            #     Hc, DHc = get_H_and_DH(v_x, v_l, v_u, v_v, v_m, u0, vars0)
            #     # om = om - DH' * ((DH*DH')\H)
            #     delta = DHc.T @ np.linalg.solve(DHc @ DHc.T, Hc)
            #     omega -= delta

            vars_curr, lambda_curr = omega[0:10], omega[10:13]
            u_curr, v_curr, mu_curr = omega[13:37], omega[37:43], omega[43]
            k += 1

            # make h_step adaptive
            if mu_curr < 0.01:
                h_step = 0.2

        if i == 0:
            z_test = np.concatenate([vars_curr, lambda_curr, u_curr, v_curr])
            check_full_jacobian(z_test, mu_curr, u0, vars0, lambda0)

        # Save original continuation endpoint   
        z_original = np.concatenate([vars_curr, lambda_curr, u_curr, v_curr])
        mu_original = mu_curr

        # Attempt endpoint correction at mu = 0
        z_corrected, converged, endpoint_residual = correct_endpoint(z_original, u0, vars0, lambda0)
        # Retry from the last positive-mu point only if needed
        if not converged and mu_original < 0:
            z_retry, retry_converged, retry_residual = correct_endpoint(z_last_positive, u0, vars0, lambda0)

            if retry_converged:
                z_corrected = z_retry
                converged = True
                endpoint_residual = retry_residual

        if converged:
            vars_curr = z_corrected[:10].copy()
            lambda_curr = z_corrected[10:13].copy()
            u_curr = z_corrected[13:37].copy()
            v_curr = z_corrected[37:43].copy()
            mu_curr = 0.0
        else:
            # Keep original continuation result if correction fails
            vars_curr = z_original[:10].copy()
            lambda_curr = z_original[10:13].copy()
            u_curr = z_original[13:37].copy()
            v_curr = z_original[37:43].copy()
            mu_curr = mu_original

        print(f"Endpoint: converged={converged}, "f"residual={endpoint_residual:.3e}, "f"mu={mu_curr:.6e}")

        f1, f2, f3, hVal, gVal = eval_physics(vars_curr, u0, vars0)
        
        Pg = vars_curr[0:5]
        eq_max = float(np.max(np.abs(hVal)))
        ineq_max = float(max(0.0, np.max(gVal)))
        H_final, _ = get_H_and_DH(vars_curr, lambda_curr, u_curr, v_curr,mu_curr, u0, vars0, lambda0)

        res = [k, *lambda_curr, mu_curr, f1, f2, f3,*Pg, *vars_curr[5:10], eq_max, ineq_max, np.linalg.norm(H_final, ord=np.inf)]
        
        results.append(res)
        print(f"w1={w[0]:.4f} mu={mu_curr:.4e} k={k} f1={f1:.2f}")

    print(f"Elapsed: {time.time() - start_time:.2f}s")
    
    df = pd.DataFrame(results, columns=['k', 'L1', 'L2', 'L3', 'mu', 'f1', 'f2', 'f3', 'P1', 'P2', 'P3', 'P4', 'P5', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'eq_max_MW', 'ineq_max', 'Hnorm'])
    display("Homotopy Solution\n", df)
    df.to_csv('solution_homotopy_ieee5_figure13_corrected.csv', index=False)
    
def plot_pareto_3d():
    # Load the Homotopy results
    homotopy_file = 'solution_homotopy_ieee5_figure13_corrected.csv'
    if not os.path.exists(homotopy_file):
        print(f"Error: {homotopy_file} not found. Run the solver first.")
        return

    df_h = pd.read_csv(homotopy_file)

    # Setup the figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for Homotopy solutions
    sc = ax.scatter(df_h['f1'], df_h['f2'], df_h['f3'],
                    c=df_h['f1'], cmap='viridis', s=50, label='Homotopy Method', alpha=0.8)

    # Labels and Title
    ax.set_xlabel('Total Cost ($f_1$)', fontsize=10)
    ax.set_ylabel('Generation Deviation ($f_2$)', fontsize=10)
    ax.set_zlabel('Line Loading ($f_3$)', fontsize=10)
    ax.set_title('3D Pareto Front (IEEE 5-Bus System)', fontsize=14)

    # Matching MATLAB's view(45, 25)
    ax.view_init(elev=25, azim=45)

    plt.colorbar(sc, label='Cost Gradient', shrink=0.5)
    ax.legend()
    plt.grid(True)
    plt.show()

def check_H4_jacobian():
    lam0 = np.array([0.075, 0.150, 0.775])
    lam = np.array([0.20, 0.30, 0.50])
    mu = 0.4
    eps = 1e-6

    def H4(lam, mu):
        return (
            (1-mu)*(1-np.sum(lam))*np.ones(3)
            - mu*(lam**1.5 - lam0**1.5)
        )

    J_analytic = (
        -(1-mu)*np.ones((3,3))
        - mu*np.diag(1.5*np.sqrt(lam))
    )

    J_numeric = np.zeros((3,3))

    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        J_numeric[:,j] = (
            H4(lam+e, mu) - H4(lam-e, mu)
        ) / (2*eps)

    dmu_analytic = (
        (np.sum(lam)-1)*np.ones(3)
        - (lam**1.5-lam0**1.5)
    )

    dmu_numeric = (
        H4(lam, mu+eps) - H4(lam, mu-eps)
    ) / (2*eps)

    print("Lambda Jacobian error:",
          np.max(np.abs(J_analytic-J_numeric)))
    print("Mu derivative error:",
          np.max(np.abs(dmu_analytic-dmu_numeric)))

check_H4_jacobian()


# if __name__ == "__main__":
solve_homotopy()
plot_pareto_3d()