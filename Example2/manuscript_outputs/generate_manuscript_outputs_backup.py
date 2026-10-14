#!/usr/bin/env python3
"""Example 2 publication plots and LaTeX tables; read-only on experiments.
Run from Example2/manuscript_outputs, with sibling experiment directories.
Requires numpy, pandas, matplotlib. Uses existing per-run metric CSVs, never recomputes metrics.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FIG = HERE/'figures'
TAB = HERE/'tables'
METHODS = ['homotopy','nsga2','nsga3','ctaea']
LABELS = {'homotopy':'Homotopy','nsga2':'NSGA-II','nsga3':'NSGA-III','ctaea':'C-TAEA'}
COLORS = {'homotopy':'#222222','nsga2':'#2878b5','nsga3':'#d08b28','ctaea':'#38936a'}


def require(path):
    if not path.is_file():
        raise SystemExit(f'Missing required file: {path}\nPlease copy or generate this file first.')
    return pd.read_csv(path)


def tex_escape(s):
    return str(s).replace('_',r'\_').replace('%',r'\%')


def latex_table(name, headers, rows, caption, label, align=None):
    align = align or ('l'+'r'*(len(headers)-1))
    lines = [r'\begin{table}[htbp]',r'\centering',r'\small',
             r'\caption{'+caption+'}',r'\label{'+label+'}',
             r'\begin{tabular}{'+align+'}',r'\hline',
             ' & '.join(headers)+r' \\',r'\hline']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    (TAB/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def fmt(mean, sd, digits=4):
    return f'{mean:.{digits}f}' if pd.isna(sd) else f'{mean:.{digits}f} $\\pm$ {sd:.{digits}f}'


def savefig(name):
    plt.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.savefig(FIG/(name+'.png'),dpi=210,bbox_inches='tight')
    plt.close()


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    TAB.mkdir(parents=True,exist_ok=True)
    sources = {
        'A1':BASE/'A1_reassessment'/'results'/'A1_per_run.csv',
        'A2':BASE/'A2_reoptimization'/'analysis'/'A2_metrics_per_run.csv',
        'B2':BASE/'B2_equality_elimination'/'analysis'/'B2_metrics_per_run.csv'
    }
    # A1 may have been saved in the A1 root instead of results.
    if not sources['A1'].exists():
        sources['A1']=BASE/'A1_reassessment'/'results'/'A1_per_run.csv'
        if not sources['A1'].exists():
            sources['A1']=BASE/'A1_reassessment'/'A1_per_run.csv'
    data=[]
    for exp,path in sources.items():
        d=require(path)
        need={'method','run','HV','IGD','GD','both_components'}
        if not need.issubset(d.columns):
            raise SystemExit(f'{path} missing {sorted(need-set(d.columns))}')
        d['experiment']=exp
        data.append(d)
    allm=pd.concat(data,ignore_index=True)
    allm['method']=allm.method.astype(str).str.lower()
    allm.to_csv(TAB/'all_metrics_per_run.csv',index=False)
    summary=(allm.groupby(['experiment','method'],as_index=False)
             .agg(runs=('run','size'),feasible_runs=('HV','count'),
                  HV_mean=('HV','mean'),HV_sd=('HV','std'),
                  IGD_mean=('IGD','mean'),IGD_sd=('IGD','std'),
                  GD_mean=('GD','mean'),GD_sd=('GD','std'),
                  both_components=('both_components','sum')))
    summary.to_csv(TAB/'consolidated_metrics.csv',index=False)

    # Table 1: complete comparison, baseline once, stochastic statistics from 20 runs.
    rows=[]
    hom=summary[(summary.experiment=='B2')&(summary.method=='homotopy')]
    if len(hom)!=1: raise SystemExit('Expected exactly one original homotopy baseline in B2 metrics')
    h=hom.iloc[0]
    rows.append(['Original', 'Homotopy', '1',fmt(h.HV_mean,np.nan),fmt(h.IGD_mean,np.nan),fmt(h.GD_mean,np.nan),'1/1'])
    for exp in ['A1','A2','B2']:
        for method in METHODS[1:]:
            d=summary[(summary.experiment==exp)&(summary.method==method)]
            if len(d)!=1: raise SystemExit(f'Missing metrics: {exp}, {method}')
            r=d.iloc[0]           
            rows.append([exp,LABELS[method],str(int(r.runs)),fmt(r.HV_mean,r.HV_sd),
                         fmt(r.IGD_mean,r.IGD_sd),fmt(r.GD_mean,r.GD_sd),
                         f'{int(r.both_components)}/{int(r.runs)}'])
    latex_table('overall_comparison.tex',
                ['Study','Method','$n$','HV $\\uparrow$','IGD $\\downarrow$','GD $\\downarrow$','Both lobes'],rows,
                'Example 2 objective-space indicators. Evolutionary entries are mean $\\pm$ sample standard deviation across runs; the original homotopy baseline is one run. A1 and A2 accept equality residuals up to $10^{-4}$; B2 enforces equalities by construction. All use the same reference front and normalization.',
                'tab:example2_overall', 'llrrrrr')

    # Boxplots: 20 stochastic runs only; single homotopy baseline shown as horizontal line.
    for metric,up in [('HV',True),('IGD',False),('GD',False)]:
        fig,ax=plt.subplots(figsize=(10.5,4.6))
        groups=[]; positions=[]; colors=[]; ticklabels=[]
        for j,exp in enumerate(['A1','A2','B2']):
            for k,method in enumerate(METHODS[1:]):
                vals=allm.loc[(allm.experiment==exp)&(allm.method==method),metric].dropna().values
                groups.append(vals);positions.append(j*4+k+1);colors.append(COLORS[method]);ticklabels.append(LABELS[method])
        b=ax.boxplot(groups,positions=positions,widths=.65,patch_artist=True,showfliers=True)
        for patch,color in zip(b['boxes'],colors):patch.set(facecolor=color,alpha=.68)
        baseline=float(h[metric+'_mean'])
        ax.axhline(baseline,color=COLORS['homotopy'],ls='--',lw=1.6,label='Original homotopy (one run)')
        ax.set_xticks([2,6,10],['A1','A2','B2'])
        ax.set_ylabel(metric+(' (higher is better)' if up else ' (lower is better)'))
        ax.legend(loc='best',fontsize=9)
        ax.grid(axis='y',alpha=.2)
        fig.text(.5,.01,'Within each group, left to right: NSGA-II, NSGA-III, C-TAEA',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.04,1,1]);savefig(metric.lower()+'_boxplots')

    # Feasible region and solution distributions (one selected run, clearly labeled).
    fig,ax=plt.subplots(figsize=(7.3,4.7))
    xx=np.linspace(-1.35,1.35,700)
    cap=.08-(xx**2-1)**2
    ok=cap>=0
    ax.fill_between(xx,-np.sqrt(np.maximum(cap,0)),np.sqrt(np.maximum(cap,0)),where=ok,color='#cbdbe9',alpha=.85,label='Feasible region')
    ax.set(xlabel='$x_1$',ylabel='$x_2$',title='Reduced feasible region: $(x_1^2-1)^2+x_2^2\\leq0.08$')
    ax.set_aspect('equal',adjustable='box');ax.legend();savefig('feasible_region')

    pop_sources={'A2':BASE/'A2_reoptimization'/'results'/'all_final_populations.csv',
                 'B2':BASE/'B2_equality_elimination'/'results'/'all_final_populations.csv'}
    for exp,path in pop_sources.items():
        p=require(path)
        fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharex=True,sharey=True)
        for ax,method in zip(axes,METHODS[1:]):
            q=p[(p.method.astype(str).str.lower()==method)&(p.run==1)]
            ax.scatter(q.x1,q.x2,s=9,c=COLORS[method],alpha=.65)
            ax.set_title(f'{LABELS[method]} (run 1)');ax.set_xlabel('$x_1$')
        axes[0].set_ylabel('$x_2$')
        fig.suptitle(f'{exp}: final populations (illustrative run 1)')
        fig.tight_layout();savefig(exp.lower()+'_decision_space_run1')

    # Histories: mean feasible count by generation, distinguish A2 vs B2.
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),sharey=True)
    for ax,exp in zip(axes,['A2','B2']):
        hist=require(BASE/({'A2':'A2_reoptimization','B2':'B2_equality_elimination'}[exp])/'results'/f'{exp}_feasibility_history.csv')
        for method in METHODS[1:]:
            q=hist[hist.method.astype(str).str.lower()==method].groupby('generation').n_feasible.mean()
            ax.plot(q.index,q.values,color=COLORS[method],label=LABELS[method],lw=1.6)
        ax.set(title=exp,xlabel='Generation',ylim=(-1,85));ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean feasible individuals (of 84)')
    axes[1].legend(loc='lower right',fontsize=9)
    fig.tight_layout();savefig('feasibility_histories')

    # B1 tolerance study, if present. Search only within B1 directory.
    b1=BASE/'B1_tolerance_study'
    candidates=[b1/'results'/'tolerance_per_run.csv',b1/'tolerance_per_run.csv',
                b1/'results'/'B1_per_run.csv',b1/'B1_per_run.csv']
    found=next((p for p in candidates if p.exists()),None)
    if found:
        t=require(found)
        if {'method','run','equality_tolerance','feasible_run'}.issubset(t.columns):
            r=(t.groupby(['method','equality_tolerance'],as_index=False)
               .agg(feasible_runs=('feasible_run','sum'),runs=('run','size')))
            r['rate']=r.feasible_runs/r.runs
            r.to_csv(TAB/'tolerance_rates.csv',index=False)
            fig,ax=plt.subplots(figsize=(7.5,4.4))
            for method in METHODS:
                q=r[r.method.astype(str).str.lower()==method].sort_values('equality_tolerance')
                if len(q):ax.plot(q.equality_tolerance,q.rate,'o-',label=LABELS[method],color=COLORS[method])
            ax.set(xscale='log',xlabel='Equality tolerance',ylabel='Fraction of runs with feasible points',ylim=(-.05,1.05))
            ax.grid(alpha=.2);ax.legend();fig.tight_layout();savefig('b1_tolerance_sensitivity')
            rr=[]
            for tol in sorted(r.equality_tolerance.unique(),reverse=True):
                row=[f'$10^{{{int(round(np.log10(tol)))}}}$']
                for method in METHODS:
                    q=r[(r.equality_tolerance==tol)&(r.method.astype(str).str.lower()==method)]
                    row.append(f'{int(q.iloc[0].feasible_runs)}/{int(q.iloc[0].runs)}' if len(q) else '--')
                rr.append(row)
            latex_table('b1_tolerance_study.tex',['Equality tol.']+[LABELS[m] for m in METHODS],rr,
                        'Retrospective B1 feasibility rates at fixed inequality tolerance $10^{-7}$.',
                        'tab:example2_b1','lrrrr')
    else:
        print('Optional B1 tolerance CSV not found; skipping B1 figure/table.')

    # Experiment design table.
    latex_table('experiment_design.tex',['Study','Decision variables','Equality treatment','Inequality tolerance'],[
        ['Original','4','Original equalities, $10^{-7}$','$10^{-7}$'],
        ['A1','4','Retrospective test, $10^{-4}$','$10^{-7}$'],
        ['A2','4','Relaxed during search, $10^{-4}$','$10^{-7}$'],
        ['B1','4','Retrospective tolerance sweep','$10^{-7}$'],
        ['B2','2','Eliminated: $x_3=x_2$, $x_4=0$','$10^{-7}$']],
        'Design of Example 2 comparisons. Evolutionary optimization uses population 84, 400 generations and 20 independent runs; A1 and B1 are retrospective analyses of original populations.',
        'tab:example2_design','llll')

    (HERE/'README_generated.txt').write_text(
        'Figures are vector PDFs plus preview PNGs. Tables are LaTeX fragments.\n'
        'Indicators were imported from each experiment analysis; no optimization or indicator recomputation occurred.\n'
        'B2 equality residual is zero by construction, not a claim of algorithmic convergence.\n'
        'Reference front and normalization are inherited from the existing experiment analyses.\n'
        'Single homotopy run is plotted as a fixed baseline, not a distribution.\n',encoding='utf-8')
    print('Finished. Figures:',FIG,'\nTables:',TAB)
    print(summary[['experiment','method','runs','HV_mean','IGD_mean']].to_string(index=False))

if __name__=='__main__':main()

