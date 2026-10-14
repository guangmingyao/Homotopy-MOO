import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from common_problem import f1, f2, scipy_constraints, diagnostics, BOUNDS, WEIGHTS, SLSQP_OPTIONS, SEED

def multistart(obj, extra=None, n=20):
    rng=np.random.default_rng(SEED); best=None
    starts=[np.zeros(5), np.array([0.4,0.8,0,0,0])]+[rng.uniform(-5,5,5) for _ in range(max(0,n-2))]
    for x0 in starts:
        r=minimize(obj,x0,method='SLSQP',bounds=BOUNDS,constraints=scipy_constraints(extra),options=SLSQP_OPTIONS)
        if r.success and (best is None or r.fun < best.fun): best=r
    return best

def epsilonConstraintRunner():
    start=time.time()
    rmin=multistart(f2,n=20); rmax=multistart(lambda x:-f2(x),n=20)
    if rmin is None or rmax is None: raise RuntimeError('Could not determine feasible f2 range.')
    lo=float(f2(rmin.x)); hi=float(f2(rmax.x))
    # 199 levels, matching the number of weighted-sum/homotopy parameter values.
    eps_values=np.linspace(lo,hi,len(WEIGHTS))
    rows=[]; x0=rmin.x.copy()
    for eps in eps_values:
        extra=[{'type':'ineq','fun':lambda x,e=eps: e-f2(x)}]
        sol=minimize(f1,x0,method='SLSQP',bounds=BOUNDS,constraints=scipy_constraints(extra),options=SLSQP_OPTIONS)
        if not sol.success:
            # one robust fallback rather than silently losing a point
            sol=multistart(f1,extra=extra,n=10)
        if sol is None or not sol.success:
            print(f'Epsilon-constraint failed at eps={eps:.8g}'); continue
        x=sol.x; d=diagnostics(x)
        rows.append({'eps':eps, **{f'x{i+1}':x[i] for i in range(5)},'f1':float(f1(x)),'f2':float(f2(x)),
                     'g':d['g'],'h1':d['h1'],'h2':d['h2'],'max_abs_h':d['max_abs_h'],
                     'bound_margin':d['bound_margin'],'nit':getattr(sol,'nit',np.nan)})
        x0=x
    df=pd.DataFrame(rows); df.to_csv('solution_epsilon_constraint.csv',index=False)
    print(f'Epsilon Constraint: {len(df)} solutions, {time.time()-start:.2f} s; f2 range [{lo:.8g},{hi:.8g}]')
    # Plot only if successful solutions were obtained.
    if df.empty:
        print("No successful epsilon-constraint solutions to plot.")
        return df

    plt.figure(figsize=(6, 5))
    plt.scatter(
        df["f1"],
        df["f2"],
        s=24,
        facecolors="none",
        edgecolors="red",
        linewidths=1.0
    )
    plt.xlabel(r"$f_1$")
    plt.ylabel(r"$f_2$")
    plt.title(r"$\epsilon$-Constraint Method")
    plt.tight_layout()
    plt.show()

    return df

if __name__=='__main__': epsilonConstraintRunner()
