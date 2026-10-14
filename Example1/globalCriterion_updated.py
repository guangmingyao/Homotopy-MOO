import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from common_problem import f1, f2, scipy_constraints, diagnostics, BOUNDS, WEIGHTS, SLSQP_OPTIONS

def solve(obj, x0):
    return minimize(obj, x0, method='SLSQP', bounds=BOUNDS, constraints=scipy_constraints(), options=SLSQP_OPTIONS)

def globalCriterionRunner(p=2):
    start=time.time()
    # Ideal point from the two constrained single-objective minima.
    r1=solve(f1, np.zeros(5)); r2=solve(f2, np.zeros(5))
    if not (r1.success and r2.success): raise RuntimeError('Could not compute ideal point reliably.')
    z=np.array([r1.fun, r2.fun], dtype=float)
    scale=np.maximum(np.abs(np.array([f1(r2.x), f2(r1.x)])-z), 1.0)  # nondimensionalize objectives
    print('ideal point =', z, 'scale =', scale, 'p =', p)

    rows=[]; x0=r1.x.copy()
    for w1 in WEIGHTS:
        w=np.array([w1,1-w1])
        def obj(x):
            r=np.abs((np.array([f1(x),f2(x)])-z)/scale)
            return np.sum(w*r**p)**(1.0/p)
        sol=solve(obj,x0)
        if not sol.success:
            print(f'Global criterion failed at w1={w1:.6f}: {sol.message}'); continue
        x=sol.x; d=diagnostics(x)
        rows.append({'w1_initial':w1, **{f'x{i+1}':x[i] for i in range(5)}, 'f1':float(f1(x)), 'f2':float(f2(x)),
                     'g':d['g'],'h1':d['h1'],'h2':d['h2'],'max_abs_h':d['max_abs_h'],
                     'bound_margin':d['bound_margin'],'obj':float(sol.fun),'p':p,'nit':sol.nit})
        x0=x
    df=pd.DataFrame(rows); df.to_csv('solution_global_criterion.csv',index=False)
    print(f'Global Criterion: {len(df)} solutions, {time.time()-start:.2f} s')
    plt.figure(figsize=(6,5)); plt.scatter(df.f1,df.f2,s=24,facecolors='none',edgecolors='red'); plt.xlabel(r'$f_1$');plt.ylabel(r'$f_2$');plt.title(f'Global Criterion ($p={p}$)');plt.tight_layout();plt.show()
    return df

if __name__=='__main__': globalCriterionRunner(p=2)
