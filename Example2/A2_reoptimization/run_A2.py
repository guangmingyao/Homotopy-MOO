#!/usr/bin/env python3
"""A2: original four-variable Example 2, tolerance-aware evolutionary optimization.

Equality constraints are supplied to pymoo as |h_i(x)| - 1e-4 <= 0,
so the tolerance affects selection during optimization, not merely postprocessing.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

EQ_TOL = 1e-4
INEQ_TOL = 1e-7
METHODS = ('nsga2', 'nsga3', 'ctaea')
COLS = ['x1','x2','x3','x4']
OBJS = ['f1','f2','f3','f4']


def objectives(X):
    a,b = X[:,0],X[:,1]
    return np.column_stack(((a+1)**2+b*b+0.05*np.sin(8*a),
                            (a-1)**2+b*b+0.05*np.sin(8*a),
                            b*b+0.1*(1-np.cos(6*a)),
                            (a*a-1)**2+b*b))


def violations(X):
    a,b,c,d = X.T
    eq = np.maximum(np.abs(d), np.abs(c*c-b*b))
    inequality = np.maximum(0., (a*a-1)**2+b*b-0.08)
    return eq,inequality


class Example2A2(Problem):
    def __init__(self):
        super().__init__(n_var=4,n_obj=4,n_ieq_constr=3,n_eq_constr=0,
                         xl=np.array([-2.,-1.,-1.,-1.]),
                         xu=np.array([2.,1.,1.,1.]))

    def _evaluate(self,X,out,*args,**kwargs):
        a,b,c,d = X.T
        g = (a*a-1)**2+b*b-0.08
        out['F'] = objectives(X)
        # Encode acceptable equality residuals as inequalities for pymoo's
        # native constraint-domination selection during optimization.
        out['G'] = np.column_stack((g-INEQ_TOL,
                                    np.abs(d)-EQ_TOL,
                                    np.abs(c*c-b*b)-EQ_TOL))


class FeasibilityTrace(Callback):
    def __init__(self):
        super().__init__()
        self.rows=[]
    def notify(self,algorithm):
        X=np.atleast_2d(np.asarray(algorithm.pop.get('X'),float))
        eq,ineq=violations(X)
        good=(eq<=EQ_TOL)&(ineq<=INEQ_TOL)
        self.rows.append(dict(generation=int(algorithm.n_gen),
                              n_feasible=int(good.sum()),
                              min_eq_residual=float(eq.min()),
                              min_ineq_violation=float(ineq.min()),
                              min_excess_violation=float(np.maximum(np.maximum(eq-EQ_TOL,ineq-INEQ_TOL),0).min())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=int,default=20)
    p.add_argument('--generations',type=int,default=400)
    p.add_argument('--population',type=int,default=84)
    p.add_argument('--output',default='results')
    args=p.parse_args()
    if args.runs<1 or args.generations<1 or args.population!=84:
        p.error('Use --runs >= 1, --generations >= 1, --population 84.')
    root=Path(__file__).resolve().parent
    out=(root/args.output).resolve()
    if out==root or root not in out.parents:
        p.error('--output must be a subfolder of A2_reoptimization.')
    if out.exists() and any(out.iterdir()):
        p.error(f'Output folder is not empty: {out}. Choose another --output.')
    out.mkdir(parents=True,exist_ok=True)
    dirs=get_reference_directions('das-dennis',4,n_partitions=6)
    populations=[]; per_run=[]; histories=[]
    for method in METHODS:
        for run in range(1,args.runs+1):
            if method=='nsga2':
                algorithm=NSGA2(pop_size=84,eliminate_duplicates=True)
            elif method=='nsga3':
                algorithm=NSGA3(pop_size=84,ref_dirs=dirs,eliminate_duplicates=True)
            else:
                algorithm=CTAEA(ref_dirs=dirs)
            cb=FeasibilityTrace()
            seed=1000+run
            start=time.perf_counter()
            result=minimize(Example2A2(),algorithm,('n_gen',args.generations),
                            seed=seed,callback=cb,verbose=False,
                            return_least_infeasible=True)
            seconds=time.perf_counter()-start
            X=np.atleast_2d(np.asarray(result.pop.get('X'),float))
            eq,ineq=violations(X)
            feasible=(eq<=EQ_TOL)&(ineq<=INEQ_TOL)
            F=objectives(X)
            frame=pd.DataFrame(np.column_stack((X,F)),columns=COLS+OBJS)
            frame.insert(0,'method',method)
            frame.insert(1,'run',run)
            frame['eq_residual']=eq
            frame['ineq_violation']=ineq
            frame['feasible']=feasible
            frame['eligible']=True
            frame['seconds']=seconds
            frame['n_eval']=int(result.algorithm.evaluator.n_eval)
            populations.append(frame)
            accepted=frame.loc[feasible].copy()
            if len(accepted):
                nd=NonDominatedSorting().do(accepted[OBJS].to_numpy(float),
                                             only_non_dominated_front=True)
                accepted=accepted.iloc[nd]
            left=bool((accepted.x1<0).any())
            right=bool((accepted.x1>0).any())
            per_run.append(dict(method=method,run=run,seed=seed,
                                n_candidates=len(frame),n_feasible=int(feasible.sum()),
                                n_nd=len(accepted),left_covered=int(left),
                                right_covered=int(right),both_components=int(left and right),
                                min_eq_residual=float(eq.min()),
                                min_combined_violation=float(np.maximum(eq,ineq).min()),
                                seconds=seconds,n_eval=int(result.algorithm.evaluator.n_eval)))
            histories.extend(dict(method=method,run=run,**r) for r in cb.rows)
            print(f'{method} run {run}/{args.runs}: feasible {feasible.sum()}/{len(X)}, '
                  f'ND {len(accepted)}, both components {left and right}, '
                  f'time {seconds:.2f}s',flush=True)
            pd.concat(populations,ignore_index=True).to_csv(out/'all_final_populations.csv',index=False)
            pd.DataFrame(per_run).to_csv(out/'A2_per_run.csv',index=False)
            pd.DataFrame(histories).to_csv(out/'A2_feasibility_history.csv',index=False)
    per=pd.DataFrame(per_run)
    summary=per.groupby('method',sort=False).agg(
        runs=('run','size'),feasible_runs=('n_feasible',lambda s:int((s>0).sum())),
        mean_feasible_points=('n_feasible','mean'),mean_nondominated=('n_nd','mean'),
        both_components_runs=('both_components','sum'),
        best_combined_violation=('min_combined_violation','min'),
        mean_seconds=('seconds','mean'),mean_evaluations=('n_eval','mean')).reset_index()
    summary['feasible_run_rate']=summary.feasible_runs/summary.runs
    summary.to_csv(out/'A2_summary.csv',index=False)
    (out/'A2_settings.json').write_text(json.dumps(dict(
        experiment='A2',formulation='original four-variable problem',
        equality_tolerance=EQ_TOL,inequality_tolerance=INEQ_TOL,
        constraint_handling='three inequality constraints: g-ineq_tol, abs(h1)-eq_tol, abs(h2)-eq_tol',
        methods=METHODS,runs=args.runs,generations=args.generations,
        population=84,seed_rule='1000+run',
        homotopy='not rerun; compare with original Experiment A baseline',
        note='HV and IGD should be computed with the same reference front and normalization as A1.'),indent=2))
    print('\nA2 complete. Output:',out)
    print(summary.to_string(index=False))

if __name__=='__main__':
    main()
