import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from common_problem import f1, f2, g_leq, h1, h2, scipy_constraints, diagnostics, BOUNDS, WEIGHTS, SLSQP_OPTIONS

def weightedSumRunner():
    start = time.time()
    rows = []
    x0 = np.zeros(5)
    cons = scipy_constraints()

    for w1 in WEIGHTS:
        obj = lambda x, w=w1: w*f1(x) + (1.0-w)*f2(x)
        sol = minimize(obj, x0, method='SLSQP', bounds=BOUNDS, constraints=cons, options=SLSQP_OPTIONS)
        if not sol.success:
            print(f'Weighted sum failed at w1={w1:.6f}: {sol.message}')
            continue
        x = sol.x
        d = diagnostics(x)
        rows.append({'w1_initial':w1, **{f'x{i+1}':x[i] for i in range(5)},
                     'f1':float(f1(x)), 'f2':float(f2(x)), 'g':d['g'], 'h1':d['h1'], 'h2':d['h2'],
                     'max_abs_h':d['max_abs_h'], 'bound_margin':d['bound_margin'], 'obj':float(sol.fun),
                     'nit':sol.nit, 'success':sol.success})
        x0 = x  # continuation/warm start; does not change the problem

    df = pd.DataFrame(rows)
    df.to_csv('solution_weighted_sum.csv', index=False)
    print(f'Weighted Sum: {len(df)} solutions, {time.time()-start:.2f} s')
    if len(df): print('max equality residual =', df.max_abs_h.max(), '; min bound margin =', df.bound_margin.min())
    plt.figure(figsize=(6,5)); plt.scatter(df.f1, df.f2, s=24, facecolors='none', edgecolors='red')
    plt.xlabel(r'$f_1$'); plt.ylabel(r'$f_2$'); plt.title('Weighted Sum'); plt.tight_layout(); plt.show()
    return df

if __name__ == '__main__': weightedSumRunner()
