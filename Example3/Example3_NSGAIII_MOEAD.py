"""IEEE 5-bus three-objective DC-OPF: NSGA-III and MOEA/D (pymoo).

Install: pip install numpy scipy pandas matplotlib pymoo
Run:     python Example3_NSGAIII_MOEAD.py

Uses the exact model data in homotopyIEEE5Bus_figure13_corrected.py.
Equality reduction: P5 = sum(Pd) - sum(P1..P4), theta4=0, and
Bbus(reduced) theta(reduced) = (Pg-Pd)(reduced)/baseMVA.
All 5 nodal power balances then hold up to floating-point precision.

Both algorithms use the SAME exterior penalty for remaining constraints,
because pymoo MOEA/D does not natively support constrained problems.
Reported feasibility is checked against the original, unpenalized model.

Outputs to example3_evolutionary_results/.  HV/IGD use an empirical pooled
reference set (not an exact Pareto front). The reference is constructed from
all feasible nondominated evolutionary points, and optionally from the
homotopy CSV when --homotopy-csv is specified.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError as exc:
    raise SystemExit('Missing pymoo. Install with: pip install pymoo numpy scipy pandas matplotlib') from exc


class IEEE5Bus(Problem):
    def __init__(self):
        self.base = 100.0
        self.Pd = np.array([0., 0., 300., 300., 400.])
        self.cost = np.array([14., 15., 30., 40., 10.])
        self.Pref = np.array([40., 170., 323.4948, 0., 466.5052])
        self.Pmax = np.array([40., 170., 520., 200., 600.])
        self.branch = [(0,1,.0281,400.),(0,3,.0304,9999.),
                       (0,4,.0064,9999.),(1,2,.0108,9999.),
                       (2,3,.0297,9999.),(3,4,.0297,240.)]
        self.Bbus = np.zeros((5,5))
        self.Bf = np.zeros((6,5))
        for k,(i,j,x,_) in enumerate(self.branch):
            b = 1./x
            self.Bbus[i,i] += b; self.Bbus[j,j] += b
            self.Bbus[i,j] -= b; self.Bbus[j,i] -= b
            self.Bf[k,i] = b; self.Bf[k,j] = -b
        self.free = [0,1,2,4]  # reference bus 4 (index 3)
        self.invB = np.linalg.inv(self.Bbus[np.ix_(self.free,self.free)])
        self.flow_limits = np.array([400.,240.])
        # These fixed positive divisors improve numerical balance in selection.
        # Outputs and all indicators are calculated from RAW objectives.
        self.obj_scale = np.array([20000., 100000., 2.])
        # P1..P4 are independent; P5 is determined by total generation=1000 MW.
        super().__init__(n_var=4, n_obj=3, n_ieq_constr=0,
                         xl=np.zeros(4), xu=self.Pmax[:4])

    def physics(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Pg5 = self.Pd.sum() - X.sum(axis=1)
        Pg = np.column_stack((X,Pg5))
        rhs = (Pg-self.Pd)[:,self.free]/self.base
        theta = np.zeros((len(X),5))
        theta[:,self.free] = rhs @ self.invB.T
        flows = self.base*(theta @ self.Bf.T)
        f1 = Pg @ self.cost
        f2 = np.sum((Pg-self.Pref)**2, axis=1)
        f3 = (flows[:,0]/400.)**2 + (flows[:,5]/240.)**2
        F = np.column_stack((f1,f2,f3))
        # pymoo inequality convention: G <= 0. Scale to dimensionless units.
        G = np.column_stack((
            -Pg5/600., (Pg5-600.)/600.,
            flows[:,0]/400.-1., -flows[:,0]/400.-1.,
            flows[:,5]/240.-1., -flows[:,5]/240.-1.,
            theta/np.pi-1., -theta/np.pi-1.
        ))
        # G has 2+4+10=16 columns (P5, two selected lines, five angles).
        balance = Pg-self.Pd-self.base*(theta @ self.Bbus.T)
        return F,G,Pg,theta,flows,balance

    def _evaluate(self, X, out, *args, **kwargs):
        F,G,*_ = self.physics(X)
        # Identical scalar exterior penalty applied to all three objectives.
        # This is a selection device, NOT a replacement for feasibility checks.
        penalty = 100.0*np.maximum(G,0.).sum(axis=1)
        out['F'] = F/self.obj_scale + penalty[:,None]


def nondominated(F):
    if len(F)==0:
        return np.array([],dtype=int)
    return np.asarray(NonDominatedSorting().do(F, only_non_dominated_front=True),dtype=int)


def load_homotopy(path, problem):
    if not path:
        return np.empty((0,3))
    df = pd.read_csv(path)
    required = ['f1','f2','f3']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Homotopy CSV must contain columns {required}')
    if 'mu' in df:
        df = df.loc[np.abs(df['mu']) < 1e-8]
    if 'eq_max_MW' in df:
        df = df.loc[df['eq_max_MW'] <= 1e-5]
    if 'ineq_max' in df:
        df = df.loc[df['ineq_max'] <= 1e-5]
    return df[required].to_numpy(dtype=float)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=int,default=20)
    parser.add_argument('--n-evals',type=int,default=20000)
    parser.add_argument('--partitions',type=int,default=12,
                        help='3-objective reference directions: 12 gives 91 directions')
    parser.add_argument('--homotopy-csv',default=None,
                        help='Optional homotopy output CSV with f1,f2,f3 columns')
    parser.add_argument('--outdir',default='example3_evolutionary_results')
    args=parser.parse_args()
    if args.runs<1 or args.n_evals<1: parser.error('runs and n-evals must be positive')
    outdir=Path(args.outdir);outdir.mkdir(parents=True,exist_ok=True)
    # Validate optional input before starting any expensive evolutionary runs.
    homotopy_path = None
    if args.homotopy_csv:
        candidate = Path(args.homotopy_csv)
        if not candidate.is_file():
            alternative = Path('results') / candidate.name
            if alternative.is_file():
                candidate = alternative
            else:
                parser.error(f'Homotopy CSV not found: {args.homotopy_csv}')
        homotopy_path = candidate
    problem=IEEE5Bus()
    directions=get_reference_directions('das-dennis',3,n_partitions=args.partitions)
    print(f'Population/reference directions: {len(directions)}')
    print(f'Budget: {args.n_evals} objective evaluations per run; {args.runs} runs/method')

    all_rows=[];run_rows=[]

    def save_checkpoint():
        # Raw fronts and run diagnostics are saved after EVERY completed run.
        # HV/IGD are calculated later using the pooled reference front.
        pd.DataFrame(all_rows).to_csv(outdir/'pareto_solutions_all_runs.csv', index=False)
        pd.DataFrame(run_rows).to_csv(outdir/'per_run_metrics.csv', index=False)

    for method in ('NSGA-III','MOEA/D'):
        for run in range(args.runs):
            seed=20261012+run
            if method=='NSGA-III':
                algorithm=NSGA3(pop_size=len(directions),ref_dirs=directions,eliminate_duplicates=True)
            else:
                algorithm=MOEAD(ref_dirs=directions,n_neighbors=min(15,len(directions)),
                                prob_neighbor_mating=0.7)
            start=time.perf_counter()
            result=minimize(problem,algorithm,('n_eval',args.n_evals),
                            seed=seed,verbose=False,save_history=False)
            elapsed=time.perf_counter()-start
            pop=result.pop
            X=np.asarray(pop.get('X'))
            if X.size==0:
                run_rows.append(dict(method=method,run=run,seed=seed,
                                     runtime_s=elapsed,n_feasible=0,feasible_fraction=0.,
                                     n_nd=0,max_balance_MW=np.nan,max_ineq_violation=np.nan))
                save_checkpoint()
                continue
            F,G,Pg,theta,flows,balance=problem.physics(X)
            cv=np.maximum(G,0.).max(axis=1)
            feas=(cv<=1e-7)&(np.max(np.abs(balance),axis=1)<=1e-5)
            inds=np.flatnonzero(feas)
            nd=inds[nondominated(F[inds])] if len(inds) else np.array([],dtype=int)
            run_rows.append(dict(method=method,run=run,seed=seed,runtime_s=elapsed,
                                 n_feasible=int(feas.sum()),feasible_fraction=float(feas.mean()),
                                 n_nd=len(nd),max_balance_MW=float(np.max(np.abs(balance))),
                                 max_ineq_violation=float(cv.max())))
            for j in nd:
                row=dict(method=method,run=run,seed=seed,
                         f1=F[j,0],f2=F[j,1],f3=F[j,2],
                         max_balance_MW=np.max(np.abs(balance[j])),
                         max_ineq_violation=cv[j])
                row.update({f'P{k+1}':Pg[j,k] for k in range(5)})
                row.update({f'theta{k+1}':theta[j,k] for k in range(5)})
                row.update({f'flow{k+1}':flows[j,k] for k in range(6)})
                all_rows.append(row)
            save_checkpoint()
            print(f'{method:8s} run {run+1:02d}/{args.runs}: '
                  f'feasible={feas.sum():3d}/{len(feas)}, nondominated={len(nd):3d}, '
                  f'time={elapsed:.2f}s',flush=True)

    fronts=pd.DataFrame(all_rows)
    runs=pd.DataFrame(run_rows)
    fronts.to_csv(outdir/'pareto_solutions_all_runs.csv',index=False)
    if fronts.empty:
        runs.to_csv(outdir/'per_run_metrics.csv',index=False)
        print('No feasible solutions; inspect constraints/budget.');return

    hom = load_homotopy(str(homotopy_path) if homotopy_path else None, problem)

    all_F=np.vstack((fronts[['f1','f2','f3']].to_numpy(),hom)) if len(hom) else fronts[['f1','f2','f3']].to_numpy()
    # Normalize in objective space using the same bounds for all algorithms.
    lo=all_F.min(axis=0); hi=all_F.max(axis=0)
    span=np.maximum(hi-lo,1e-12)
    normalized=(all_F-lo)/span
    reference=normalized[nondominated(normalized)]
    # Remove near-duplicates for reference stability.
    reference=np.unique(np.round(reference,decimals=10),axis=0)
    hv_indicator=HV(ref_point=np.array([1.1,1.1,1.1]))
    igd_indicator=IGD(reference)
    for idx,row in runs.iterrows():
        sub=fronts[(fronts.method==row.method)&(fronts.run==row.run)]
        if sub.empty:
            runs.loc[idx,'HV']=0.
            runs.loc[idx,'IGD']=np.nan
        else:
            norm=(sub[['f1','f2','f3']].to_numpy()-lo)/span
            runs.loc[idx,'HV']=float(hv_indicator(norm))
            runs.loc[idx,'IGD']=float(igd_indicator(norm))
    runs.to_csv(outdir/'per_run_metrics.csv',index=False)
    summary=runs.groupby('method').agg(
        HV_mean=('HV','mean'),HV_std=('HV','std'),
        IGD_mean=('IGD','mean'),IGD_std=('IGD','std'),
        runtime_mean_s=('runtime_s','mean'),runtime_std_s=('runtime_s','std'),
        feasible_fraction_mean=('feasible_fraction','mean'),
        n_nd_mean=('n_nd','mean')).reset_index()
    summary.to_csv(outdir/'summary.csv',index=False)
    pd.DataFrame(reference*span+lo,columns=['f1','f2','f3']).to_csv(
        outdir/'empirical_reference_front.csv',index=False)
    print('\nSummary (empirical pooled-reference HV and IGD):')
    print(summary.to_string(index=False))

    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    for method,color,marker in [('NSGA-III','tab:blue','o'),('MOEA/D','tab:orange','^')]:
        sub=fronts[fronts.method==method]
        ax.scatter(sub.f1,sub.f2,sub.f3,s=10,alpha=.25,c=color,
                   marker=marker,label=method)
    if len(hom):
        ax.scatter(hom[:,0],hom[:,1],hom[:,2],s=30,c='black',marker='x',
                   label='Homotopy (fixed weights)')
    ax.set_xlabel('f1: fuel cost');ax.set_ylabel('f2: dispatch deviation')
    ax.set_zlabel('f3: line stress')
    ax.set_title('IEEE 5-bus: objective-space coverage (all runs)')
    ax.legend(loc='best');fig.tight_layout()
    fig.savefig(outdir/'objective_space_comparison.png',dpi=220)
    plt.close(fig)
    with open(outdir/'settings.txt','w',encoding='utf-8') as f:
        f.write(f'runs={args.runs}\nn_evals={args.n_evals}\npartitions={args.partitions}\n'
                f'population={len(directions)}\nseeds=20261012..{20261011+args.runs}\n'
                'feasibility_threshold_scaled=1e-7\n'
                'constraint_handling=identical_exterior_penalty_both_methods\n'
                'penalty_coefficient=100\n'
                'balance_threshold_MW=1e-5\n'
                'HV_reference_point_normalized=(1.1,1.1,1.1)\n'
                'IGD_reference=pooled_feasible_nondominated_union\n'
                f'homotopy_csv={args.homotopy_csv}\n'
                f'objective_scaling_internal={problem.obj_scale.tolist()}\n')
    print(f'\nSaved results in {outdir.resolve()}')


if __name__=='__main__':
    main()
