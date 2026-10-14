# Important Feasible


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

def projectedFeasibleRunner():
    # --- Problem definition ---
    def f1(x):
        return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2

    def f2(x):
        return 3*x[0] + 2*x[1] - (1/3)*x[2] + 0.01*(x[3] - x[4])**3

    def g_ineq(x):
        # constraint g(x) = x1^2 + x2^2 + x3^2 + x4^2 - 10 <= 0
        # scipy SLSQP uses 'ineq' meaning fun(x) >= 0, so return 10 - sumsq >= 0
        return 10.0 - (x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2)

    def h1(x):
        return 4*x[0] - 2*x[1] + 0.8*x[2] + 0.6*x[3] + 0.5*x[4]**2

    def h2(x):
        return x[0] + 2*x[1] - x[2] - 0.5*x[3] + x[4] - 2

    constraints = (
        {'type':'ineq', 'fun': g_ineq},
        {'type':'eq', 'fun': h1},
        {'type':'eq', 'fun': h2},
    )

    bounds = [(-10,10)]*5

    # --- optimization helpers ---
    def multi_restart_minimize(obj_fn, nstarts=12, bounds=bounds, constraints=constraints):
        best = None
        best_x = None
        rng = np.random.RandomState(0)
        for seed in range(nstarts):
            x0 = rng.uniform(-2,2,5)
            res = minimize(obj_fn, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                        options={'ftol':1e-8, 'maxiter':1000})
            if res.success and (best is None or res.fun < best):
                best = res.fun; best_x = res.x
        return best_x, best

    # --- find ideal points ---
    x_min_f1, _ = multi_restart_minimize(f1, nstarts=30)
    x_min_f2, _ = multi_restart_minimize(f2, nstarts=30)
    f1_star = f1(x_min_f1)
    f2_star = f2(x_min_f2)
    print("f1* =", f1_star, "  f2* =", f2_star)

    # --- sample feasible points to estimate ranges (light sampling + projection) ---
    def project_to_feasible(x0, maxiter=200):
        # minimize distance to x0 subject to constraints
        def dist(z, x0=x0):
            return np.sum((z-x0)**2)
        res = minimize(lambda z: dist(z), x0, method='SLSQP', bounds=bounds, constraints=constraints,
                    options={'ftol':1e-8, 'maxiter':maxiter})
        if res.success and g_ineq(res.x) >= -1e-6:
            return res.x
        return None

    rng = np.random.RandomState(1)
    feasible_objs = []
    n_sample = 20000
    solutions = []
    for i in range(n_sample):
        x0 = rng.uniform(-5,5,5)
        # print("Trying x0 =", x0)
        x_proj = project_to_feasible(x0)
        # print("Projected x =", x_proj)
        if x_proj is not None:
            x = x_proj
            feasible_objs.append((f1(x_proj), f2(x_proj), x_proj))
            solutions.append({
            # 'alpha': a,
            'x1': x[0], 'x2': x[1], 'x3': x[2], 'x4': x[3], 'x5': x[4],
            'f1': f1(x), 'f2': f2(x), 'h1': h1(x), 'h2': h2(x), 'g': -g_ineq(x)
            })

    df = pd.DataFrame(solutions)
    display(df)




    if len(feasible_objs) < 3:
        # fallback: use the two known feasible points
        feasible_objs = [(f1(x_min_f1), f2(x_min_f1), x_min_f1), (f1(x_min_f2), f2(x_min_f2), x_min_f2)]

    f1_vals = np.array([p[0] for p in feasible_objs])
    f2_vals = np.array([p[1] for p in feasible_objs])
    r1_min, r1_max = f1_vals.min(), f1_vals.max()
    r2_min, r2_max = f2_vals.min(), f2_vals.max()
    print("Estimated f1 range:", r1_min, "->", r1_max)
    print("Estimated f2 range:", r2_min, "->", r2_max)

    # # scaling for normalization (avoid divide by zero)
    # s1 = max(abs(f1_star), 1.0)
    # s2 = max(abs(f2_star), 1.0)

    # # --- sweep reference points and minimize distance to each reference ---
    # n_ref1 = 10   # increase for finer resolution (computational cost ~ n_ref1 * n_ref2 * restarts)
    # n_ref2 = 10
    # r1_grid = np.linspace(r1_min, r1_max, n_ref1)
    # r2_grid = np.linspace(r2_min, r2_max, n_ref2)

    # solutions = []
    # seen = set()
    # for r1 in r1_grid:
    #     for r2 in r2_grid:
    #         def G_r(x, r1=r1, r2=r2):
    #             v1 = (f1(x) - r1)/s1
    #             v2 = (f2(x) - r2)/s2
    #             return np.sqrt(v1*v1 + v2*v2)
    #         x_sol, _ = multi_restart_minimize(G_r, nstarts=18)
    #         if x_sol is None:
    #             continue
    #         f1_s = f1(x_sol); f2_s = f2(x_sol)
    #         key = (round(f1_s,8), round(f2_s,8))
    #         if key in seen:
    #             continue
    #         seen.add(key)
    #         solutions.append({'r1':r1, 'r2':r2,
    #                           'x1':x_sol[0], 'x2':x_sol[1], 'x3':x_sol[2], 'x4':x_sol[3], 'x5':x_sol[4],
    #                           'f1':f1_s, 'f2':f2_s})
    # # organize
    # df = pd.DataFrame(solutions).sort_values('f1').reset_index(drop=True)
    # display(df)
    # df.to_csv("pareto_global_criterion_solutions.csv", index=False)



    # Save generated feasible-point data with the other Example 1 results.
    results_dir = Path(__file__).resolve().parent / 'example1_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / 'projected_feasible_solution.csv'
    df.to_csv(output_file, index=False)
    print(f'Saved projected feasible data to: {output_file}')

    # --- plot ---
    plt.figure(figsize=(7,5))
    plt.scatter(f1_vals, f2_vals, s=8, label='sampled feasible points')
    # if not df.empty:
    #     plt.plot(df['f1'], df['f2'], marker='o', linestyle='-', label='global-criterion solutions (sweep)')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Feasible points')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.show()



    # plt.figure(figsize=(7,5))
    # plt.scatter(f1_vals, f2_vals, s=8, label='sampled feasible points')
    # # if not df.empty:
    # plt.plot(df_weighted_sum['f1'], df_weighted_sum['f2'], marker='o', linestyle='-', color="red", label='Weighted Sum (sweep)')
    # plt.xlabel('f1')
    # plt.ylabel('f2')
    # plt.title('Feasible points)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()



projectedFeasibleRunner()

