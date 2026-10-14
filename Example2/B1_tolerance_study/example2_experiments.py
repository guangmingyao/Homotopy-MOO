#!/usr/bin/env python3
"""Example 2 reproducible benchmark: original homotopy, NSGA-II, NSGA-III, MOEA/D.

Two explicitly separate settings:
  original: four variables and the actual equalities/inequalities (NSGA-II/III)
  reduced: exact feasible parameterization x=(x1,0,0,0), x1 in one feasible interval
           (NSGA-II/III/MOEA-D). This is an objective-front comparison, NOT
           a test of equality constraint handling.

All quality indicators are recomputed against a common analytical reduction-based
reference, in a fixed reference normalization. Nothing is silently repaired.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OBJ = [f'f{i}' for i in range(1, 5)]
XCOL = [f'x{i}' for i in range(1, 5)]
TOL = 1e-7


def objectives(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    a,b = X[:,0], X[:,1]
    return np.column_stack(((a+1)**2+b*b+.05*np.sin(8*a),
                            (a-1)**2+b*b+.05*np.sin(8*a),
                            b*b+.1*(1-np.cos(6*a)),
                            (a*a-1)**2+b*b))


def constraints(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    a,b,c,d = X.T
    H = np.column_stack((d,c*c-b*b))
    G = np.column_stack(((a*a-1)**2+b*b-.08, -2-a, a-2,
                         -1-b,b-1,-1-c,c-1,-1-d,d-1))
    return H,G


def feasibility(X, tol=TOL):
    H,G = constraints(X)
    eq = np.max(np.abs(H), axis=1)
    ineq = np.maximum(0,np.max(G,axis=1))
    return eq,ineq,(eq <= tol)&(ineq <= tol)


def intervals():
    r=np.sqrt(.08)
    return [(-np.sqrt(1+r),-np.sqrt(1-r)),
            (np.sqrt(1-r),np.sqrt(1+r))]


def x_from_reduced(Z):
    """Z=(component_index, position), each in [0,1]. All points exactly feasible."""
    Z=np.atleast_2d(np.asarray(Z,float))
    out=np.zeros((len(Z),4))
    idx=np.minimum(1,np.floor(np.clip(Z[:,0],0,1-1e-12)*2).astype(int))
    I=np.asarray(intervals())
    out[:,0]=I[idx,0]+np.clip(Z[:,1],0,1)*(I[idx,1]-I[idx,0])
    return out


def reference(n_per_component=1200):
    I=intervals()
    a=np.concatenate([np.linspace(lo,hi,n_per_component) for lo,hi in I])
    X=np.zeros((len(a),4)); X[:,0]=a
    F=objectives(X)
    mask=nd_mask(F)
    return X[mask],F[mask]


def nd_mask(F):
    """Efficient nondominated filter (minimization); duplicate objective rows removed."""
    F=np.asarray(F,float)
    if len(F)==0: return np.zeros(0,bool)
    order=np.lexsort(tuple(F[:,j] for j in range(F.shape[1]-1,-1,-1)))
    keep=np.zeros(len(F),bool)
    chosen=[]
    for i in order:
        f=F[i]
        if chosen and np.any(np.all(np.asarray(chosen)<=f+1e-12,axis=1)):
            continue
        keep[i]=True
        chosen.append(f)
    return keep


def load_homotopy(path):
    D=pd.read_csv(path)
    needed=XCOL
    if not set(needed).issubset(D):
        raise ValueError(f'Homotopy CSV must contain {needed}')
    eligible=(D.success.astype(str).str.lower().isin(['true','1']).to_numpy()
              if 'success' in D else np.ones(len(D),dtype=bool))
    return D[XCOL].to_numpy(float),eligible


def run_homotopy(args, out):
    import importlib.util
    src=ROOT/'homotopy_original.py'
    spec=importlib.util.spec_from_file_location('homotopy_original',src)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # Original solver is preserved unchanged. Its own plotting/metric pipeline may be slow.
    t0=time.perf_counter()
    D,_=mod.solve_homotopy_manyobj(fast_mode=args.homotopy_fast,
                                    results_dir=out/'homotopy_native',show_plots=False)
    runtime=time.perf_counter()-t0
    # Keep every returned point, including failures, in raw output.
    D.to_csv(out/'homotopy_original_raw.csv',index=False)
    X=D[XCOL].to_numpy(float)
    valid=D.success.astype(bool).to_numpy() if 'success' in D else np.ones(len(X),bool)
    return X,valid,runtime


def project_to_pareto_set(X):
    """Exact Pareto-set projection for a separate, explicitly labeled study.

    Not a solver of the original constrained problem: exploits analytical knowledge.
    """
    X=np.atleast_2d(np.asarray(X,float))
    out=np.zeros((len(X),4))
    a=np.clip(X[:,0],-2,2)
    I=np.asarray(intervals())
    endpoints=np.concatenate([I[:,0],I[:,1]])
    inside=np.zeros(len(a),bool)
    for lo,hi in I:
        inside |= (a>=lo)&(a<=hi)
    closest=endpoints[np.argmin(np.abs(a[:,None]-endpoints[None,:]),axis=1)]
    out[:,0]=np.where(inside,a,closest)
    return out


def run_evolutionary(method, formulation, seed, pop_size, n_gen, tol=TOL):
    try:
        from pymoo.core.problem import Problem
        from pymoo.core.callback import Callback
        from pymoo.optimize import minimize
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.algorithms.moo.ctaea import CTAEA
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as exc:
        raise SystemExit('pymoo is required: python -m pip install pymoo') from exc

    class Ex2(Problem):
        def __init__(self):
            if formulation=='original':
                super().__init__(n_var=4,n_obj=4,n_ieq_constr=1,n_eq_constr=2,
                                 xl=np.array([-2,-1,-1,-1]),xu=np.array([2,1,1,1]))
            else:
                # reduced search over 2 variables; no constraints
                super().__init__(n_var=2,n_obj=4,n_ieq_constr=0,n_eq_constr=0,
                                 xl=np.zeros(2),xu=np.ones(2))
        def _evaluate(self,Z,out,*args,**kwargs):
            if formulation=='original':
                X=Z
                H,G=constraints(X)
                out['G']=G[:,0:1]  # box bounds are encoded by xl/xu
                out['H']=H
            else:
                X=x_from_reduced(Z)
            out['F']=objectives(X)

    dirs=get_reference_directions('das-dennis',4,n_partitions=6) # 84 directions
    if method=='nsga2':
        alg=NSGA2(pop_size=pop_size,eliminate_duplicates=True)
    elif method=='nsga3':
        alg=NSGA3(pop_size=pop_size,ref_dirs=dirs,eliminate_duplicates=True)
    elif method=='ctaea':
        alg=CTAEA(ref_dirs=dirs)
    elif method=='moead':
        if formulation!='reduced':
            raise ValueError('Standard MOEA/D is used only for the reduced problem.')
        alg=MOEAD(ref_dirs=dirs,n_neighbors=15,prob_neighbor_mating=.7)
    else: raise ValueError(method)
    class FeasibilityArchive(Callback):
        def __init__(self):
            super().__init__()
            self.feasible=[]
            self.near=[]
            self.diagnostics=[]

        def notify(self, algorithm):
            pop=algorithm.pop
            if pop is None or len(pop)==0: return
            Z=np.atleast_2d(np.asarray(pop.get('X'),float))
            X=Z if formulation=='original' else x_from_reduced(Z)
            eq,ineq,feas=feasibility(X,tol)
            violation=np.maximum(eq,ineq)
            self.diagnostics.append(dict(generation=int(algorithm.n_gen),
                n_feasible=int(feas.sum()), min_eq=float(eq.min()),
                min_ineq=float(ineq.min()), min_violation=float(violation.min()),
                median_violation=float(np.median(violation))))
            if np.any(feas): self.feasible.append(X[feas].copy())
            # Retain a small near-feasible archive for diagnostic inspection.
            ids=np.argsort(violation)[:min(5,len(X))]
            self.near.append(X[ids].copy())

    prob=Ex2()
    archive=FeasibilityArchive()
    t0=time.perf_counter()
    # save_history=False keeps memory use modest; res.pop contains the final population.
    res=minimize(prob,alg,('n_gen',n_gen),seed=seed,verbose=False,
                 save_history=False,return_least_infeasible=True,callback=archive)
    sec=time.perf_counter()-t0
    Z=np.asarray(res.pop.get('X'),float)
    X=Z if formulation=='original' else x_from_reduced(Z)
    found=np.vstack(archive.feasible) if archive.feasible else np.empty((0,4))
    if len(found):
        found=np.unique(np.round(found,decimals=12),axis=0)
    near=np.vstack(archive.near) if archive.near else np.empty((0,4))
    return X,found,near,archive.diagnostics,sec,int(res.algorithm.evaluator.n_eval)


def make_rows(X,method,formulation,run,seconds=np.nan,n_eval=np.nan,eligible=None):
    X=np.asarray(X,float).reshape(-1,4)
    F=objectives(X); eq,ineq,feas=feasibility(X)
    if eligible is None: eligible=np.ones(len(X),bool)
    eligible=np.asarray(eligible,bool)
    if len(eligible)!=len(X): raise ValueError('eligible length must match X')
    D=pd.DataFrame(np.column_stack((X,F)),columns=XCOL+OBJ)
    D.insert(0,'method',method); D.insert(1,'formulation',formulation)
    D.insert(2,'run',run)
    D['eq_residual']=eq; D['ineq_violation']=ineq
    D['feasible']=feas; D['eligible']=eligible
    D['seconds']=seconds; D['n_eval']=n_eval
    D['component']=np.where(X[:,0]<0,'left','right')
    return D


def metrics(D, ref_F, ideal, scale, hv_ref, ref_hv, hv_samples=0):
    from scipy.spatial import cKDTree
    from pymoo.indicators.hv import HV
    eligible=D[D.feasible & D.eligible]
    F=eligible[OBJ].to_numpy(float)
    F=F[nd_mask(F)]
    R=(ref_F-ideal)/scale
    A=(F-ideal)/scale if len(F) else np.empty((0,4))
    result={'n_candidates':len(D),'n_feasible':int(D.feasible.sum()),
            'n_eligible_feasible':len(eligible),'n_nd':len(F),
            'max_eq_all':float(D.eq_residual.max()) if len(D) else np.nan,
            'max_ineq_all':float(D.ineq_violation.max()) if len(D) else np.nan,
            'max_eq_feasible':float(eligible.eq_residual.max()) if len(eligible) else np.nan,
            'max_ineq_feasible':float(eligible.ineq_violation.max()) if len(eligible) else np.nan,
            'left_covered':int(np.any(eligible.x1<0)),
            'right_covered':int(np.any(eligible.x1>0)),
            'seconds':float(D.seconds.iloc[0]) if len(D) else np.nan,
            'n_eval':float(D.n_eval.iloc[0]) if len(D) else np.nan}
    if len(A):
        da=cKDTree(R).query(A)[0]
        dr=cKDTree(A).query(R)[0]
        hv=float(HV(ref_point=ref_hv)(A))
        result.update(HV=hv,HV_ratio=hv/hv_ref if hv_ref else np.nan,
                      GD=float(da.mean()),IGD=float(dr.mean()),
                      AHD=float(max(da.mean(),dr.mean())))
    else:
        result.update(HV=0.,HV_ratio=0.,GD=np.nan,IGD=np.nan,AHD=np.nan)
    return result


def make_plots(D, R, out):
    plt.rcParams.update({'figure.dpi':120})
    use=D[D.feasible & D.eligible].copy()
    pairs=[('f1','f2'),('f1','f3'),('f1','f4'),('f2','f3'),('f2','f4'),('f3','f4')]
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,(a,b) in zip(axes.flat,pairs):
        ax.plot(R[a],R[b],'.',color='0.7',ms=2,label='reference')
        for name,grp in use.groupby('method'):
            ax.scatter(grp[a],grp[b],s=9,alpha=.7,label=name)
        ax.set_xlabel(a);ax.set_ylabel(b);ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(out/'objective_projections.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for name,grp in use.groupby('method'):
        ax.scatter(grp.x1,np.zeros(len(grp)),s=10,alpha=.6,label=name)
    for lo,hi in intervals():ax.axvspan(lo,hi,color='0.8',alpha=.3)
    ax.set_xlabel('x1');ax.set_yticks([]);ax.legend(fontsize=8,ncol=3)
    fig.tight_layout();fig.savefig(out/'component_coverage.png');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='example2_results')
    p.add_argument('--runs',type=int,default=20)
    p.add_argument('--pop-size',type=int,default=100)
    p.add_argument('--generations',type=int,default=400)
    p.add_argument('--reference-size',type=int,default=500,
                   help='reference points per component before nondominated filtering')
    p.add_argument('--formulations',choices=['reduced','original','both'],default='both')
    p.add_argument('--methods',nargs='+',choices=['nsga2','nsga3','moead','ctaea'],
                   default=['nsga2','nsga3','moead','ctaea'])
    p.add_argument('--project-original',action='store_true',help='Separate oracle-assisted projection study; NEVER merge with raw original results')
    p.add_argument('--homotopy-csv',default=None,
                   help='existing homotopy CSV (x1,x2,x3,x4, optionally success)')
    p.add_argument('--run-homotopy',action='store_true',
                   help='run the uploaded homotopy script (may be slow)')
    p.add_argument('--homotopy-fast',action='store_true')
    p.add_argument('--feas-tol',type=float,default=TOL)
    args=p.parse_args()
    # main uses feasibility default arguments fixed at definition; use explicit tolerance below
    if args.feas_tol!=TOL:
        raise SystemExit('Use the documented common tolerance 1e-7 for this release.')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    Xref,Fref=reference(args.reference_size)
    refD=pd.DataFrame(np.column_stack((Xref,Fref)),columns=XCOL+OBJ)
    refD.to_csv(out/'reference_front.csv',index=False)
    ideal=Fref.min(axis=0);scale=np.maximum(Fref.max(axis=0)-ideal,1e-12)
    ref_hv=np.full(4,1.1)
    try:
        from pymoo.indicators.hv import HV
    except ImportError as exc:
        raise SystemExit('Install pymoo: python -m pip install pymoo') from exc
    hv_ref=float(HV(ref_point=ref_hv)((Fref-ideal)/scale))
    config={'runs':args.runs,'pop_size':args.pop_size,'generations':args.generations,
            'reference_size_per_component':args.reference_size,'feasibility_tol':TOL,
            'normalization_ideal':ideal.tolist(),'normalization_scale':scale.tolist(),
            'hv_reference_point_normalized':ref_hv.tolist(),
            'hv_reference_value':hv_ref,'methods':args.methods,
            'formulations':args.formulations,
            'note':'Reduced runs use exact feasible parameterization; not a constraint-handling comparison.'}
    (out/'settings.json').write_text(json.dumps(config,indent=2))
    all_rows=[];all_stats=[];diagnostics=[]
    def store(D):
        per_run=out/'runs';per_run.mkdir(exist_ok=True)
        D.to_csv(per_run/f'{D.formulation.iloc[0]}_{D.method.iloc[0]}_run{int(D.run.iloc[0]):02d}.csv',index=False)
        all_rows.append(D)
        rec=metrics(D,Fref,ideal,scale,hv_ref,ref_hv)
        rec.update(method=str(D.method.iloc[0]),formulation=str(D.formulation.iloc[0]),
                   run=int(D.run.iloc[0]))
        all_stats.append(rec)
        print(f"{rec['formulation']:8} {rec['method']:10} run {rec['run']:02d}: "
              f"feasible {rec['n_feasible']:3d}, ND {rec['n_nd']:3d}, "
              f"HV {rec['HV']:.4f}, IGD {rec['IGD']:.4f}, "
              f"components {rec['left_covered']}+{rec['right_covered']}",flush=True)
        pd.DataFrame(all_stats).to_csv(out/'run_metrics.csv',index=False)
    if args.homotopy_csv:
        X,eligible=load_homotopy(args.homotopy_csv)
        if len(X):store(make_rows(X,'homotopy','original',0,eligible=eligible))
    if args.run_homotopy:
        X,eligible,sec=run_homotopy(args,out)
        if len(X):store(make_rows(X,'homotopy','original',0,seconds=sec,eligible=eligible))
    formulations=['reduced','original'] if args.formulations=='both' else [args.formulations]
    for formulation in formulations:
        for method in args.methods:
            if formulation=='original' and method=='moead':
                print('SKIP original MOEA/D: standard pymoo implementation has no constraint handling')
                continue
            for run in range(1,args.runs+1):
                X, found, near, history, sec, n_eval = run_evolutionary(
                    method, formulation, 1000 + run, args.pop_size, args.generations)
                # Primary metrics use the final population; archived points are supplementary.
                D = make_rows(X, method, formulation, run, sec, n_eval)
                store(D)
                # Preserve final population and closest-to-feasible candidates for diagnosis.
                detail_dir=out/'diagnostics'; detail_dir.mkdir(exist_ok=True)
                if len(found):
                    make_rows(found,method,formulation,run,sec,n_eval).to_csv(
                        detail_dir/f'{formulation}_{method}_run{run:02d}_archive.csv',index=False)
                make_rows(X,method,formulation,run,sec,n_eval).to_csv(
                    detail_dir/f'{formulation}_{method}_run{run:02d}_final.csv',index=False)
                if len(near):
                    make_rows(near,method,formulation,run,sec,n_eval).to_csv(
                        detail_dir/f'{formulation}_{method}_run{run:02d}_near.csv',index=False)
                for record in history:
                    diagnostics.append(dict(formulation=formulation,method=method,run=run,**record))
                if args.project_original and formulation=='original':
                    # Oracle projection is analytically informed; NEVER count as raw original success.
                    XP=project_to_pareto_set(X)
                    store(make_rows(XP,method,'oracle_projection',run,sec,n_eval))
    if diagnostics: pd.DataFrame(diagnostics).to_csv(out/'feasibility_diagnostics.csv',index=False)
    if not all_rows:raise SystemExit('No algorithms selected or homotopy CSV supplied')
    allD=pd.concat(all_rows,ignore_index=True)
    allD.to_csv(out/'all_solutions.csv',index=False)
    S=pd.DataFrame(all_stats)
    agg=S.groupby(['formulation','method'],dropna=False).agg(
        runs=('run','count'),HV_mean=('HV','mean'),HV_std=('HV','std'),
        IGD_mean=('IGD','mean'),IGD_std=('IGD','std'),
        GD_mean=('GD','mean'),AHD_mean=('AHD','mean'),
        feasible_mean=('n_feasible','mean'),ND_mean=('n_nd','mean'),
        left_success_rate=('left_covered','mean'),right_success_rate=('right_covered','mean'),
        time_mean=('seconds','mean'),evals_mean=('n_eval','mean')).reset_index()
    agg.to_csv(out/'summary_by_method.csv',index=False)
    # Keep original, reduced and oracle-projected plots separate.
    for form in allD.formulation.unique():
        subset=allD[allD.formulation==form]
        folder=out/('plots_'+str(form));folder.mkdir(exist_ok=True)
        make_plots(subset,refD,folder)
    print('\nSaved to',out)
    print(agg.to_string(index=False))

if __name__=='__main__':main()
