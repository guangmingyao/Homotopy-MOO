import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from common_problem import f1, f2, g_leq, h1, h2, diagnostics, XL, XU, SEED

# NSGA-II cannot normally hit nonlinear equalities exactly by random variation.
# We therefore use a tight equality feasibility band only for the evolutionary
# constraint handler, then report the actual equality residuals explicitly.
EQ_TOL = 1e-2
POP_SIZE = 800
N_GEN = 800

class FiveVarConstrained(Problem):
    def __init__(self):
        super().__init__(n_var=5,n_obj=2,n_ieq_constr=5,xl=XL,xu=XU)
        self.n_points_evaluated=0
    def _evaluate(self,X,out,*args,**kwargs):
        self.n_points_evaluated += len(X)
        H1=h1(X); H2=h2(X)
        out['F']=np.column_stack([f1(X),f2(X)])
        out['G']=np.column_stack([g_leq(X), H1-EQ_TOL, -H1-EQ_TOL, H2-EQ_TOL, -H2-EQ_TOL])

def geneticAlgorithmNSGA2Runner():
    start=time.time(); problem=FiveVarConstrained()
    algorithm=NSGA2(pop_size=POP_SIZE,sampling=FloatRandomSampling(),eliminate_duplicates=True)
    res=minimize(problem,algorithm,get_termination('n_gen',N_GEN),seed=SEED,verbose=False)
    if res.X is None: raise RuntimeError('NSGA-II returned no feasible nondominated points.')
    rows=[]
    for x in np.atleast_2d(res.X):
        d=diagnostics(x)
        rows.append({**{f'x{i+1}':x[i] for i in range(5)},'f1':float(f1(x)),'f2':float(f2(x)),
                     'g':d['g'],'h1':d['h1'],'h2':d['h2'],'max_abs_h':d['max_abs_h'],'bound_margin':d['bound_margin']})
    df=pd.DataFrame(rows).sort_values('f1').reset_index(drop=True)
    df.to_csv('solution_geneticAlgorithmNSGA2.csv',index=False)
    print(f'NSGA-II: {len(df)} nondominated points, {problem.n_points_evaluated} candidate evaluations, {time.time()-start:.2f} s')
    print('max equality residual =',df.max_abs_h.max(),'; EQ_TOL =',EQ_TOL)
    plt.figure(figsize=(6,5));plt.scatter(df.f1,df.f2,s=24,facecolors='none',edgecolors='red');plt.xlabel(r'$f_1$');plt.ylabel(r'$f_2$');plt.title('NSGA-II');plt.tight_layout();plt.show()
    return df

if __name__=='__main__': geneticAlgorithmNSGA2Runner()
