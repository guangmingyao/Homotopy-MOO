import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from common_problem import f1, f2, scipy_constraints, diagnostics, BOUNDS, SLSQP_OPTIONS

def lexicographic_minimize(primary, secondary, x0, lex_tol=1e-8):
    r1=minimize(primary,x0,method='SLSQP',bounds=BOUNDS,constraints=scipy_constraints(),options=SLSQP_OPTIONS)
    if not r1.success: raise RuntimeError(r1.message)
    pstar=float(r1.fun)
    # Smooth two-sided tolerance: pstar-tol <= primary <= pstar+tol.
    extra=[{'type':'ineq','fun':lambda x: lex_tol-(primary(x)-pstar)},
           {'type':'ineq','fun':lambda x: lex_tol+(primary(x)-pstar)}]
    r2=minimize(secondary,r1.x,method='SLSQP',bounds=BOUNDS,constraints=scipy_constraints(extra),options=SLSQP_OPTIONS)
    if not r2.success: raise RuntimeError(r2.message)
    return r2,pstar

def lexicographicRunner():
    start=time.time(); rows=[]
    for order,primary,secondary in [('f1_then_f2',f1,f2),('f2_then_f1',f2,f1)]:
        r,pstar=lexicographic_minimize(primary,secondary,np.zeros(5))
        x=r.x; d=diagnostics(x)
        rows.append({'order':order, **{f'x{i+1}':x[i] for i in range(5)},'f1':float(f1(x)),'f2':float(f2(x)),
                     'g':d['g'],'h1':d['h1'],'h2':d['h2'],'max_abs_h':d['max_abs_h'],
                     'bound_margin':d['bound_margin'],'primary_optimum':pstar})
    df=pd.DataFrame(rows);df.to_csv('solution_lexicographical.csv',index=False)
    print(df.to_string(index=False)); print(f'Lexicographic: {time.time()-start:.2f} s')
    return df

if __name__=='__main__': lexicographicRunner()
