"""Rebuild IEEE 5-bus manuscript tables and figures from existing CSVs.

No optimization is rerun. Input CSVs are stored together in Example3/results.
Run from the Example3 folder:
  python Example3_make_manuscript_results.py
or pass --homotopy, --weighted, --evolutionary, --per-run, --outdir.

Requires: numpy, pandas, scipy, matplotlib, pymoo.
Uses one pooled feasible nondominated reference set, fixed normalization and
reference point for ALL methods. Evolutionary statistics use independent runs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from pymoo.indicators.hv import HV
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError as exc:
    raise SystemExit('Install dependencies: python -m pip install pymoo numpy pandas scipy matplotlib') from exc

OBJ = ['f1', 'f2', 'f3']
COLORS = {'Homotopy':'black', 'Weighted sum':'tab:green', 'NSGA-III':'tab:blue', 'MOEA/D':'tab:orange'}


def locate(explicit, candidates, description):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f'{description}: {p} not found')
        return p
    found = [Path(p) for p in candidates if Path(p).is_file()]
    if not found:
        raise FileNotFoundError(f'{description} CSV not found. Pass its path explicitly; tried: {candidates}')
    return found[0]


def check_objectives(df, path):
    missing = [c for c in OBJ if c not in df]
    if missing:
        raise ValueError(f'{path}: missing objective columns {missing}')
    df = df.copy()
    for c in OBJ:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.loc[np.isfinite(df[OBJ]).all(axis=1)].copy()


def nd_idx(a):
    if not len(a):
        return np.array([], dtype=int)
    return np.asarray(NonDominatedSorting().do(a, only_non_dominated_front=True), dtype=int)


def read_classical(path, method, tol_eq=1e-5, tol_ineq=1e-5):
    d = check_objectives(pd.read_csv(path), path)
    n_initial = len(d)
    if method == 'Homotopy' and 'mu' in d:
        d = d.loc[pd.to_numeric(d['mu'], errors='coerce').abs() <= 1e-8]
    if 'eq_max_MW' in d:
        d = d.loc[pd.to_numeric(d['eq_max_MW'], errors='coerce') <= tol_eq]
    if 'ineq_max' in d:
        d = d.loc[pd.to_numeric(d['ineq_max'], errors='coerce') <= tol_ineq]
    if 'ineq_max_MW' in d:
        d = d.loc[pd.to_numeric(d['ineq_max_MW'], errors='coerce') <= tol_ineq]
    d = d.copy()
    d['method'] = method
    print(f'{method}: {len(d)}/{n_initial} rows retained after available feasibility filters')
    if not len(d):
        raise ValueError(f'No feasible {method} solutions found in {path}')
    return d


def select_representative(runs, fronts):
    # Choose the run whose IGD is closest to the median, avoiding cherry-picking.
    selected = {}
    for method in ['NSGA-III', 'MOEA/D']:
        sub = runs.loc[(runs['method'] == method) & np.isfinite(runs['IGD'])]
        if len(sub):
            median = sub['IGD'].median()
            row = sub.iloc[np.argmin(np.abs(sub['IGD'].to_numpy() - median))]
            selected[method] = int(row['run'])
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--homotopy', help='Homotopy CSV (default: auto-detect)')
    ap.add_argument('--weighted', help='Weighted-sum CSV (default: auto-detect)')
    ap.add_argument('--evolutionary', help='Evolutionary nondominated points CSV')
    ap.add_argument('--per-run', help='Evolutionary per_run_metrics.csv')
    ap.add_argument('--results-dir', default=None, help='Folder containing all input CSVs (default: results beside this script)')
    ap.add_argument('--outdir', default=None, help='Output folder (default: results/manuscript_results)')
    ap.add_argument('--show-weighted', action='store_true', help='Overlay weighted-sum points in figure')
    args = ap.parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.results_dir).resolve() if args.results_dir else script_dir / 'results'
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Results folder not found: {data_dir}')
    out = Path(args.outdir).resolve() if args.outdir else data_dir / 'manuscript_results'
    out.mkdir(parents=True, exist_ok=True)

    def csv_file(explicit, filenames, patterns, description):
        if explicit:
            p = Path(explicit)
            if not p.is_file():
                p = data_dir / explicit
            if not p.is_file():
                raise FileNotFoundError(f'{description}: {explicit} not found')
            return p
        for filename in filenames:
            p = data_dir / filename
            if p.is_file():
                return p
        matches = sorted({p for pattern in patterns for p in data_dir.glob(pattern)
                          if p.is_file() and p.suffix.lower() == '.csv'})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f'Multiple possible {description} CSVs: {[p.name for p in matches]}. '
                             f'Please specify the correct file with its command-line option.')
        raise FileNotFoundError(f'{description} CSV not found in {data_dir}. '
                                f'Expected one of {filenames}.')

    hpath = csv_file(args.homotopy,
                     ['solution_homotopy_ieee5.csv', 'solution_homotopy_ieee5_figure13_corrected.csv'],
                     ['solution_homotopy*.csv', '*homotopy*ieee5*.csv'], 'Homotopy')
    wpath = csv_file(args.weighted,
                     ['ieee5bus_weighted_sum.csv', 'ieee5bus_weighted_sum_slsqp.csv'],
                     ['ieee5bus_weighted*.csv', '*weighted_sum*.csv'], 'Weighted-sum')
    epath = csv_file(args.evolutionary,
                     ['pareto_solutions_all_runs.csv'],
                     ['*pareto_solutions_all_runs*.csv'], 'Evolutionary')
    rpath = csv_file(args.per_run,
                     ['per_run_metrics.csv'],
                     ['*per_run_metrics*.csv'], 'Per-run')
    print(f'Reading CSV files from: {data_dir}')
    for name, path in [('Homotopy', hpath), ('Weighted sum', wpath),
                       ('Evolutionary', epath), ('Per-run metrics', rpath)]:
        print(f'  {name}: {path.name}')

    hom = read_classical(hpath, 'Homotopy')
    weighted = read_classical(wpath, 'Weighted sum')
    evo = check_objectives(pd.read_csv(epath), epath)
    for c in ['method', 'run']:
        if c not in evo:
            raise ValueError(f'{epath} missing {c}')
    # Evolutionary file stores final feasible ND individuals; re-check its saved diagnostics.
    if 'max_balance_MW' in evo:
        evo = evo.loc[pd.to_numeric(evo['max_balance_MW'], errors='coerce') <= 1e-5]
    if 'max_ineq_violation' in evo:
        evo = evo.loc[pd.to_numeric(evo['max_ineq_violation'], errors='coerce') <= 1e-7]
    evo = evo.copy()
    runs = pd.read_csv(rpath)
    if not {'method','run','runtime_s'}.issubset(runs):
        raise ValueError(f'{rpath} must contain method, run, runtime_s')
    methods = ['NSGA-III', 'MOEA/D']
    for method in methods:
        if method not in set(runs.method):
            raise ValueError(f'No runs found for {method} in {rpath}')
    print(f'Evolutionary: {len(evo)} feasible nondominated rows across {len(runs)} runs')

    # Reference construction: pool all available feasible solutions, including classical.
    pooled = pd.concat([hom[OBJ], weighted[OBJ], evo[OBJ]], ignore_index=True)
    lo = pooled.min(axis=0).to_numpy(float)
    hi = pooled.max(axis=0).to_numpy(float)
    span = np.maximum(hi-lo, 1e-12)
    normalized = (pooled.to_numpy(float)-lo)/span
    reference = normalized[nd_idx(normalized)]
    reference = np.unique(np.round(reference, 10), axis=0)
    # All sets use the same bounds/reference; HV ref point is outside observed range.
    hv = HV(ref_point=np.array([1.1,1.1,1.1]))
    tree = cKDTree(reference)

    def indicators(df):
        if df.empty:
            return 0.0, np.nan, 0
        vals = (df[OBJ].to_numpy(float)-lo)/span
        vals = vals[nd_idx(vals)]
        return float(hv(vals)), float(cKDTree(vals).query(reference, k=1)[0].mean()), len(vals)

    rows = []
    for idx, row in runs.iterrows():
        sub = evo.loc[(evo.method == row['method']) & (evo.run == row['run'])]
        H, I, N = indicators(sub)
        rows.append(dict(method=row['method'], run=int(row['run']), seed=row.get('seed', np.nan),
                         HV=H, IGD=I, n_nd=N, runtime_s=row['runtime_s'],
                         feasible_fraction=row.get('feasible_fraction', np.nan),
                         max_balance_MW=row.get('max_balance_MW', np.nan),
                         max_ineq_violation=row.get('max_ineq_violation', np.nan)))
    run_metrics = pd.DataFrame(rows)
    run_metrics.to_csv(out/'per_run_metrics_common_reference.csv', index=False)

    summary_rows = []
    for method in methods:
        d = run_metrics.loc[run_metrics.method == method]
        summary_rows.append(dict(method=method, n_runs=len(d), n_points_mean=d.n_nd.mean(),
            HV_mean=d.HV.mean(), HV_std=d.HV.std(ddof=1),
            IGD_mean=d.IGD.mean(), IGD_std=d.IGD.std(ddof=1),
            runtime_mean_s=d.runtime_s.mean(), runtime_std_s=d.runtime_s.std(ddof=1),
            feasible_fraction_mean=d.feasible_fraction.mean(),
            max_balance_MW=d.max_balance_MW.max(),
            max_ineq_violation=d.max_ineq_violation.max()))
    for name, d, runtime in [('Homotopy',hom,11.57), ('Weighted sum',weighted,1.04)]:
        H,I,N = indicators(d)
        summary_rows.append(dict(method=name, n_runs=1, n_points_mean=N,
            HV_mean=H, HV_std=np.nan, IGD_mean=I, IGD_std=np.nan,
            runtime_mean_s=np.nan, runtime_std_s=np.nan,
            feasible_fraction_mean=np.nan, max_balance_MW=np.nan,
            max_ineq_violation=np.nan))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out/'comparison_table.csv',index=False)
    pd.DataFrame(reference*span+lo, columns=OBJ).to_csv(out/'common_reference_front.csv',index=False)

    # Compare corresponding classical solutions by their weight triples.
    weights = ['L1','L2','L3']
    if all(c in hom for c in weights) and all(c in weighted for c in ['w1','w2','w3']):
        h = hom.copy(); w = weighted.copy()
        for i in range(3):
            h[f'weight{i+1}'] = h[weights[i]].round(7)
            w[f'weight{i+1}'] = w[f'w{i+1}'].round(7)
        keys = ['weight1','weight2','weight3']
        m = h.merge(w, on=keys, suffixes=('_homotopy','_weighted'), how='inner')
        for obj in OBJ:
            m[f'abs_diff_{obj}'] = (m[f'{obj}_homotopy']-m[f'{obj}_weighted']).abs()
        for k in range(1,6):
            hc = f'P{k}'
            if hc in h and hc in w:
                m[f'abs_diff_P{k}'] = (m[f'{hc}_homotopy']-m[f'{hc}_weighted']).abs()
        m.to_csv(out/'homotopy_vs_weighted_matched.csv',index=False)
        print(f'Matched homotopy/weighted weight vectors: {len(m)}')
        for obj in OBJ:
            print(f'  max |delta {obj}| = {m[f"abs_diff_{obj}"].max():.6g}' if len(m) else '')
        if len(m) != len(hom) or len(m) != len(weighted):
            print('WARNING: Some classical weight vectors did not match. Check the input CSVs.')
    else:
        print('Classical per-weight comparison skipped (weight columns not found).')

    selected = select_representative(run_metrics, evo)
    print('Representative evolutionary runs (median IGD):', selected)
    with open(out/'methods_and_metrics.txt','w',encoding='utf-8') as f:
        f.write('Input files:\n'+ '\n'.join(map(str,[hpath,wpath,epath,rpath]))+'\n\n')
        f.write('Reference: nondominated union of all feasible homotopy, weighted-sum, and evolutionary final-front points.\n')
        f.write(f'Normalization lower={lo.tolist()}, upper={hi.tolist()}\n')
        f.write('HV reference point: [1.1, 1.1, 1.1] in normalized coordinates.\n')
        f.write('IGD: mean Euclidean distance from common normalized reference set to nearest method point.\n')
        f.write('Evolutionary statistics: sample SD (ddof=1) over independent runs.\n')
        f.write('Classical HV/IGD: single 49-weight front; no SD. Runtime for classical methods is not read from CSV; supply measured values in manuscript separately.\n')
        f.write('Note: pooled empirical reference is not the exact Pareto front; pooled reference may favor methods contributing more points.\n')
        f.write('Representative run: run with IGD nearest its method median.\n')
        f.write(f'Representative run indices: {selected}\n')

    # Objective-space plot: avoid obscuring homotopy with thousands of pooled points.
    fig=plt.figure(figsize=(9,7.3)); ax=fig.add_subplot(111, projection='3d')
    for method, marker in [('NSGA-III','o'), ('MOEA/D','^')]:
        d = evo.loc[(evo.method==method)&(evo.run==selected.get(method,-999))]
        ax.scatter(d.f1,d.f2,d.f3, s=20, alpha=.58, color=COLORS[method],marker=marker,label=f'{method} (representative run)')
    if args.show_weighted:
        ax.scatter(weighted.f1,weighted.f2,weighted.f3,s=23,marker='+',color=COLORS['Weighted sum'],label='Weighted sum',depthshade=False)
    ax.scatter(hom.f1,hom.f2,hom.f3,s=45,marker='x',color='black',linewidths=1.4,label='Homotopy',depthshade=False)
    ax.set_xlabel(r'$f_1$: fuel cost',labelpad=8)
    ax.set_ylabel(r'$f_2$: dispatch deviation',labelpad=8)
    ax.set_zlabel(r'$f_3$: line stress',labelpad=8)
    ax.view_init(elev=23,azim=-58)
    ax.legend(fontsize=8,loc='upper left')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(out/f'figure_objective_space.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

    # Paired projection figure for print readability.
    fig, axs=plt.subplots(1,3,figsize=(14,4.2))
    pairs=[('f1','f2'),('f1','f3'),('f2','f3')]
    for ax,(xx,yy) in zip(axs,pairs):
        for method, marker in [('NSGA-III','o'),('MOEA/D','^')]:
            d=evo.loc[(evo.method==method)&(evo.run==selected.get(method,-999))]
            ax.scatter(d[xx],d[yy],s=15,alpha=.45,c=COLORS[method],marker=marker)
        if args.show_weighted:
            ax.scatter(weighted[xx],weighted[yy],s=20,marker='+',c=COLORS['Weighted sum'])
        ax.scatter(hom[xx],hom[yy],s=27,marker='x',c='black',linewidths=1.1,zorder=5)
        ax.set_xlabel(r'$'+xx+r'$'); ax.set_ylabel(r'$'+yy+r'$'); ax.grid(alpha=.2)
    legend=[Line2D([0],[0],marker='x',color='black',linestyle='None',label='Homotopy'),
            Line2D([0],[0],marker='o',color=COLORS['NSGA-III'],linestyle='None',label='NSGA-III'),
            Line2D([0],[0],marker='^',color=COLORS['MOEA/D'],linestyle='None',label='MOEA/D')]
    if args.show_weighted:
        legend.append(Line2D([0],[0],marker='+',color=COLORS['Weighted sum'],linestyle='None',label='Weighted sum'))
    fig.legend(handles=legend,loc='upper center',ncol=len(legend),frameon=False)
    fig.tight_layout(rect=(0,0,1,.91))
    for ext in ['png','pdf']:
        fig.savefig(out/f'figure_objective_projections.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

    # Table in LaTeX, avoiding any external LaTeX installation.
    def cell(v,sd=None, fmt='.4f'):
        if not np.isfinite(v): return '--'
        if sd is not None and np.isfinite(sd): return f'${v:{fmt}}\\pm {sd:{fmt}}$'
        return f'${v:{fmt}}$'
    lines=[r'\begin{table}[t]',r'\centering',
           r'\caption{IEEE 5-bus three-objective OPF comparison. HV and IGD use a common empirical reference set and objective normalization. Evolutionary results are mean $\pm$ sample standard deviation across independent runs.}',
           r'\label{tab:ieee5_comparison}',r'\begin{tabular}{lcccc}',r'\hline',
           r'Method & Feasible fraction & HV $\uparrow$ & IGD $\downarrow$ & Runtime (s) \\',r'\hline']
    for method in ['Homotopy','Weighted sum','NSGA-III','MOEA/D']:
        d=summary.loc[summary.method==method].iloc[0]
        feas='--' if method in ['Homotopy','Weighted sum'] else cell(d.feasible_fraction_mean,fmt='.3f')
        hv_cell=cell(d.HV_mean,d.HV_std)
        igd_cell=cell(d.IGD_mean,d.IGD_std)
        rt_cell='--' if method in ['Homotopy','Weighted sum'] else cell(d.runtime_mean_s,d.runtime_std_s,fmt='.2f')
        lines.append(f'{method} & {feas} & {hv_cell} & {igd_cell} & {rt_cell} '+r'\\')
    lines.extend([r'\hline',r'\end{tabular}',r'\end{table}'])
    (out/'table_comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    print('\nUnified comparison:')
    print(summary[['method','n_runs','HV_mean','HV_std','IGD_mean','IGD_std','runtime_mean_s','runtime_std_s']].to_string(index=False))
    print(f'\nSaved all manuscript outputs to: {out.resolve()}')

if __name__=='__main__':
    main()
