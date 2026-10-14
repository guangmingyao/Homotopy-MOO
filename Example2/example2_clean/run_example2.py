#!/usr/bin/env python3
"""Single-command, original-formulation Example 2 benchmark.

Runs the user's unchanged mathematical homotopy algorithm and three evolutionary
algorithms on the SAME four-variable constrained problem. All files under results/.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from example2_experiments import (run_evolutionary, make_rows, objectives,
                                  feasibility, reference, intervals, XCOL, OBJ)

ROOT=Path(__file__).resolve().parent

def evaluate(d, reference_f, ideal, scale, hv, hv_ref):
    accepted=d.loc[d.feasible & d.eligible].copy()
    if len(accepted):
        # Drop objective duplicates; keep a consistent corresponding decision vector.
        f=accepted[OBJ].to_numpy(float)
        _, indices=np.unique(np.round(f,12),axis=0,return_index=True)
        accepted=accepted.iloc[np.sort(indices)].copy()
        nd=NonDominatedSorting().do(accepted[OBJ].to_numpy(float),only_non_dominated_front=True)
        accepted=accepted.iloc[nd].copy()
    if len(accepted):
        a=(accepted[OBJ].to_numpy(float)-ideal)/scale
        r=(reference_f-ideal)/scale
        gd=float(cKDTree(r).query(a)[0].mean())
        igd=float(cKDTree(a).query(r)[0].mean())
        h=float(hv(a))
    else:
        gd=igd=np.nan
        h=np.nan  # Undefined for infeasible runs; do not equate failure with HV=0.
    record=dict(formulation='original',method=str(d.method.iloc[0]),run=int(d.run.iloc[0]),
        n_candidates=len(d),n_feasible=int((d.feasible & d.eligible).sum()),n_nd=len(accepted),
        min_eq=float(d.eq_residual.min()),min_ineq=float(d.ineq_violation.min()),
        min_combined_violation=float(np.maximum(d.eq_residual,d.ineq_violation).min()),
        median_combined_violation=float(np.median(np.maximum(d.eq_residual,d.ineq_violation))),
        max_eq_feasible=float(accepted.eq_residual.max()) if len(accepted) else np.nan,
        left=int((accepted.x1<0).any()),right=int((accepted.x1>0).any()),
        HV=h,HV_ratio=h/hv_ref if np.isfinite(h) else np.nan,
        GD=gd,IGD=igd,AHD=max(gd,igd) if len(accepted) else np.nan,
        seconds=float(d.seconds.iloc[0]),n_eval=float(d.n_eval.iloc[0]))
    return record,accepted

def plot_outputs(ndsets, ref_x, ref_f, diagnostics, out):
    # Homotopy and one representative successful evolutionary run per method.
    pairs=[('f1','f2'),('f1','f3'),('f1','f4'),('f2','f3'),('f2','f4'),('f3','f4')]
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for ax,(a,b) in zip(axs.flat,pairs):
        ax.plot(ref_f[:,OBJ.index(a)],ref_f[:,OBJ.index(b)],'.',ms=2,color='.75',label='Analytical reference')
        for method,g in ndsets.items():
            if len(g): ax.scatter(g[a],g[b],s=11,alpha=.75,label=method)
        ax.set(xlabel=a,ylabel=b);ax.grid(alpha=.2)
    h,l=axs.flat[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=4,fontsize=8)  
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'objective_projections.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for lo,hi in intervals(): ax.axvspan(lo,hi,color='.9')
    for i,(method,g) in enumerate(ndsets.items()):
        if len(g): ax.scatter(g.x1,np.full(len(g),i),s=12,alpha=.7,label=method)
    ax.set(xlabel='x1',yticks=[]);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
    fig.tight_layout();fig.savefig(out/'component_coverage.pdf',bbox_inches='tight');plt.close(fig)
    if diagnostics:
        dd=pd.DataFrame(diagnostics)
        fig,ax=plt.subplots(figsize=(8,4))
        for method,g in dd.groupby('method'):
            q=g.groupby('generation').min_violation.median().sort_index()
            ax.semilogy(q.index,np.maximum(q.to_numpy(float),1e-16),label=method)
        ax.axhline(1e-7,ls='--',color='.4',label='Feasibility tolerance')
        ax.set(xlabel='Generation',ylabel='Median per-run minimum violation')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(out/'feasibility_diagnostics.pdf');plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=int,default=20)
    p.add_argument('--generations',type=int,default=400)
    p.add_argument('--population',type=int,default=84)
    p.add_argument('--reference-size',type=int,default=100,help='points per component; quality indicators use this fixed reference')
    p.add_argument('--output',default='results',help='single output folder; default ./results')
    p.add_argument('--homotopy-fast',action='store_true',help='shorter homotopy run for testing only')
    p.add_argument('--skip-homotopy',action='store_true',help='debugging only')
    p.add_argument('--skip-evolutionary',action='store_true',help='debugging only')
    args=p.parse_args()
    if args.population!=84: raise SystemExit('Use --population 84 (84 Das-Dennis reference directions).')
    if args.runs<1 or args.generations<1 or args.reference_size<10: raise SystemExit('Invalid run/generation/reference settings.')
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Output folder already contains files: {out}\nChoose a new --output name to prevent mixing experiments.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'final_populations').mkdir()
    (out/'feasible_archives').mkdir()
    (out/'near_feasible').mkdir()
    (out/'figures').mkdir()
    ref_x,ref_f=reference(args.reference_size)
    pd.DataFrame(np.column_stack((ref_x,ref_f)),columns=XCOL+OBJ).to_csv(out/'reference_front.csv',index=False)
    ideal=ref_f.min(axis=0);scale=np.maximum(ref_f.max(axis=0)-ideal,1e-12)
    hv=HV(ref_point=np.full(4,1.1));hv_ref=float(hv((ref_f-ideal)/scale))
    (out/'settings.json').write_text(json.dumps(dict(formulation='original',methods=['homotopy','nsga2','nsga3','ctaea'],
        runs=args.runs,generations=args.generations,population=args.population,
        reference_size_per_component=args.reference_size,feasibility_tolerance=1e-7,
        homotopy_fast=args.homotopy_fast,homotopy_skipped=args.skip_homotopy,
        evolutionary_skipped=args.skip_evolutionary,normalization_ideal=ideal.tolist(),
        normalization_scale=scale.tolist(),normalized_hv_reference_point=[1.1]*4,
        reference_hv=hv_ref,seed_rule='1000+run',
        caveat='Homotopy is a single deterministic run; CPU time includes its native output pipeline.'),indent=2))
    records=[]; diagnostics=[]; plotsets={}; allfinal=[]
    def store(d,filename):
        d.to_csv(out/'final_populations'/filename,index=False)
        allfinal.append(d)
        rec,nd=evaluate(d,ref_f,ideal,scale,hv,hv_ref)
        records.append(rec)
        if len(nd) and (rec['method'] not in plotsets or rec['IGD']<plotsets[rec['method']][0]):
            plotsets[rec['method']]=(rec['IGD'],nd)
        pd.DataFrame(records).to_csv(out/'table_per_run.csv',index=False)
        print(f"{rec['method']:9s} run {rec['run']:02d}: feasible {rec['n_feasible']:3d}, "
              f"ND {rec['n_nd']:3d}, min violation {rec['min_combined_violation']:.3g}",flush=True)
    if not args.skip_homotopy:
        import homotopy_solver
        print('Running original homotopy...',flush=True)
        t=time.perf_counter()
        hd,_=homotopy_solver.solve_homotopy_manyobj(fast_mode=args.homotopy_fast,
                     results_dir=out/'homotopy_native',show_plots=False,native_analysis=False)
        sec=time.perf_counter()-t
        hd.to_csv(out/'homotopy_raw.csv',index=False)
        X=hd[XCOL].to_numpy(float)
        elig=hd.success.astype(str).str.lower().isin(['true','1']).to_numpy() if 'success' in hd else np.ones(len(X),bool)
        store(make_rows(X,'homotopy','original',0,seconds=sec,eligible=elig),'homotopy.csv')
    if not args.skip_evolutionary:
        for method in ('nsga2','nsga3','ctaea'):
            for run in range(1,args.runs+1):
                print(f'Running {method} {run}/{args.runs}...',flush=True)
                X,found,near,history,sec,evals=run_evolutionary(method,'original',1000+run,args.population,args.generations)
                store(make_rows(X,method,'original',run,sec,evals),f'{method}_run{run:02d}.csv')
                if len(found): make_rows(found,method,'original',run,sec,evals).to_csv(out/'feasible_archives'/f'{method}_run{run:02d}.csv',index=False)
                if len(near): make_rows(near,method,'original',run,sec,evals).to_csv(out/'near_feasible'/f'{method}_run{run:02d}.csv',index=False)
                diagnostics.extend(dict(method=method,run=run,**item) for item in history)
                if diagnostics: pd.DataFrame(diagnostics).to_csv(out/'feasibility_diagnostics.csv',index=False)
    if not records: raise SystemExit('No experiments selected.')
    pd.concat(allfinal,ignore_index=True).to_csv(out/'all_final_populations.csv',index=False)
    r=pd.DataFrame(records)
    s=r.groupby('method',sort=False).agg(runs=('run','size'),feasible_runs=('n_feasible',lambda v:int((v>0).sum())),
        feasible_points_mean=('n_feasible','mean'),nondominated_mean=('n_nd','mean'),
        min_violation_best=('min_combined_violation','min'),
        min_violation_median=('min_combined_violation','median'),
        HV_mean=('HV','mean'),HV_sd=('HV','std'),IGD_mean=('IGD','mean'),IGD_sd=('IGD','std'),
        GD_mean=('GD','mean'),AHD_mean=('AHD','mean'),left_rate=('left','mean'),right_rate=('right','mean'),
        time_mean=('seconds','mean'),eval_mean=('n_eval','mean')).reset_index()
    s.to_csv(out/'table_summary.csv',index=False)
    plot_outputs({k:v[1] for k,v in plotsets.items()},ref_x,ref_f,diagnostics,out/'figures')
    (out/'methodology.txt').write_text('All solvers use the original 4-variable constrained problem.\n'
        'Evolutionary metrics use the FINAL population; accumulated feasible and near-feasible candidates are supplementary.\n'
        'Feasibility: max absolute equality residual and positive inequality violation <= 1e-7.\n'
        'HV, GD, IGD and AHD are computed only for feasible nondominated points; undefined when none exist.\n'
        'GD/IGD/AHD use normalized objectives with the same fixed analytical reference front.\n'
        'Homotopy native quality indicators use its own reference and should not be substituted for the unified table.\n'
        'Homotopy is one deterministic run; stochastic methods are repeated with independent seeds.\n'
        'CPU time and evaluations are descriptive; homotopy function evaluation count is not available.\n')
    print('\nCOMPLETE. All results saved in:',out,flush=True)
    print(s.to_string(index=False),flush=True)

if __name__=='__main__': main()
