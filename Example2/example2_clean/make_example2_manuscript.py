"""Publication-oriented Example 2 plots/tables from existing run_example2.py outputs.
Run: python make_example2_manuscript.py --results results
Does not run solvers or alter existing experiment files.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS=['homotopy','nsga2','nsga3','ctaea']
NAMES={'homotopy':'Homotopy','nsga2':'NSGA-II','nsga3':'NSGA-III','ctaea':'C-TAEA'}
COLORS={'homotopy':'#2166ac','nsga2':'#e08214','nsga3':'#1b9e77','ctaea':'#b2182b'}
MARKERS={'homotopy':'o','nsga2':'^','nsga3':'s','ctaea':'D'}

def save(fig,where,name):
    fig.savefig(where/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(where/(name+'.png'),dpi=250,bbox_inches='tight')
    plt.close(fig)

def geom(ax, fill=True):
    x=np.linspace(-1.35,1.35,650); y=np.linspace(-.32,.32,260)
    X,Y=np.meshgrid(x,y); g=(X*X-1)**2+Y*Y-.08
    if fill: ax.contourf(X,Y,g,levels=[-1,0],colors=['#dcebf6'],alpha=.6)
    ax.contour(X,Y,g,levels=[0],colors=['#6989aa'],linewidths=1.1)
    ax.set_xlim(-1.38,1.38);ax.set_ylim(-.32,.32)
    ax.set_xlabel(r'$x_1$');ax.set_ylabel(r'$x_2$')
    ax.set_aspect('equal',adjustable='box')

def feasible_mask(z,tol):
    if 'feasible' in z:
        v=z['feasible']
        if pd.api.types.is_bool_dtype(v): return v.fillna(False).to_numpy(bool)
        return v.astype(str).str.lower().isin(['true','1','yes']).to_numpy()
    for c in ('combined_violation','constraint_violation','violation','cv','CV'):
        if c in z: return (pd.to_numeric(z[c],errors='coerce').to_numpy()<=tol)
    # Recalculate from the actual constraints, independent of CSV convention.
    if all(k in z for k in ('x1','x2','x3','x4')):
        a=z[['x1','x2','x3','x4']].to_numpy(float)
        x1,x2,x3,x4=a.T
        eq=np.maximum(np.abs(x4),np.abs(x3*x3-x2*x2))
        ineq=np.maximum.reduce([np.zeros(len(z)),(x1*x1-1)**2+x2*x2-.08,
            x1-2,-2-x1,x2-1,-1-x2,x3-1,-1-x3,x4-1,-1-x4])
        return (np.maximum(eq,ineq)<=tol)
    return np.zeros(len(z),dtype=bool)

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');p.add_argument('--tol',type=float,default=1e-7)
    args=p.parse_args(); root=Path(args.results).resolve(); out=root/'manuscript';out.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(root/'table_summary.csv').set_index('method'); runs=pd.read_csv(root/'table_per_run.csv')
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42,'savefig.facecolor':'white'})
    # Single, compact manuscript table.
    rows=[]
    for m in METHODS:
        if m not in summary.index:continue
        s=summary.loc[m]
        rows.append({'Method':NAMES[m],'Feasible runs':f'{int(s.feasible_runs)}/{int(s.runs)}',
          'Feasible points':s.get('feasible_points_mean',np.nan),'Best violation':s.min_violation_best,
          'HV':s.get('HV_mean',np.nan),'IGD':s.get('IGD_mean',np.nan),'Mean time (s)':s.time_mean})
    tab=pd.DataFrame(rows);tab.to_csv(out/'comparison_table.csv',index=False)
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{Comparison for Example 2 at feasibility tolerance $10^{-7}$. Quality indicators are computed only for feasible nondominated solutions; dashes denote undefined values.}',r'\label{tab:example2_comparison}',r'\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Method & Feasible runs & Feasible points & Best violation & HV & IGD & Time (s) \\',r'\midrule']
    for _,r in tab.iterrows():
        fmt=lambda v: '--' if pd.isna(v) else f'{v:.4f}'
        lines.append(f'{r["Method"]} & {r["Feasible runs"]} & {r["Feasible points"]:.0f} & ${r["Best violation"]:.2e}$ & {fmt(r["HV"])} & {fmt(r["IGD"])} & {r["Mean time (s)"]:.1f} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (out/'comparison_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Optional secondary experiment. Original data are never modified.
    reduced=root/'reduced'/'table_summary.csv'
    if reduced.exists():
        rs=pd.read_csv(reduced).set_index('method')
        combined=[]
        for label,frame,methods in [('Original formulation',summary,METHODS),
                                     ('Equality eliminated',rs,METHODS[1:])]:
            for m in methods:
                if m not in frame.index: continue
                a=frame.loc[m]
                combined.append(dict(Formulation=label,Method=NAMES[m],
                  Feasible_runs=f'{int(a.feasible_runs)}/{int(a.runs)}',
                  Feasible_points=a.get('feasible_points_mean',np.nan),
                  Best_violation=a.min_violation_best,HV=a.get('HV_mean',np.nan),
                  IGD=a.get('IGD_mean',np.nan),Time_s=a.time_mean,
                  Evaluations=a.get('eval_mean',np.nan)))
        combined=pd.DataFrame(combined)
        combined.to_csv(out/'combined_comparison_table.csv',index=False)
        latex=[r'\begin{table}[htbp]',r'\centering',
          r'\caption{Original and equality-eliminated formulations of Example 2. The reduced evolutionary methods solve both equality branches separately; their reported runtime and evaluations include both branches. Dashes indicate undefined quality indicators.}',
          r'\label{tab:example2_combined}',r'\small',r'\begin{tabular}{llrrrrr}',r'\toprule',
          r'Formulation & Method & Feasible runs & Feasible points & Best violation & HV & IGD \\',r'\midrule']
        for _,a in combined.iterrows():
            fm=lambda v:'--' if pd.isna(v) else f'{v:.4f}'
            latex.append(f'{a.Formulation} & {a.Method} & {a.Feasible_runs} & {a.Feasible_points:.0f} & ${a.Best_violation:.2e}$ & {fm(a.HV)} & {fm(a.IGD)} '+r'\\')
        latex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        (out/'combined_comparison_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf8')
         rp=root/'reduced'/'all_final_populations.csv'
        if rp.exists():
            reduced_pop=pd.read_csv(rp)
            if {'method','x1','x2'}.issubset(reduced_pop):
                fig,axs=plt.subplots(1,3,figsize=(12,3.3),sharex=True,sharey=True)
                for ax,m in zip(axs,METHODS[1:]):
                    geom(ax)
                    z=reduced_pop[reduced_pop.method==m]
                    good=feasible_mask(z,args.tol)
                    ax.scatter(z.loc[~good,'x1'],z.loc[~good,'x2'],s=4,alpha=.12,c=COLORS[m],rasterized=True)
                    ax.scatter(z.loc[good,'x1'],z.loc[good,'x2'],s=6,alpha=.35,c=COLORS[m],rasterized=True)
                    ax.set_title(f'{NAMES[m]}: {good.sum():,}/{len(z):,} feasible',fontsize=10)
                fig.suptitle(r'Analytical equality elimination: $x_3=\pm x_2$, $x_4=0$',fontsize=11)
                fig.tight_layout();save(fig,out,'06_reduced_decision_space')
        print('Combined original/reduced comparison saved.')
    popfile=root/'all_final_populations.csv'
    pop=pd.read_csv(popfile) if popfile.exists() else None
    if pop is not None and {'method','x1','x2'}.issubset(pop.columns):
        # Figure 1: exact geometry + homotopy only (not a spurious connecting band).
        fig,ax=plt.subplots(figsize=(8,3.2));geom(ax)
        z=pop[pop.method=='homotopy'];f=feasible_mask(z,args.tol)
        ax.scatter(z.loc[f,'x1'],z.loc[f,'x2'],s=19,c=COLORS['homotopy'],marker='o',edgecolors='white',linewidths=.3,label='Homotopy feasible points',zorder=3)
        ax.legend(loc='upper center',fontsize=9);save(fig,out,'01_geometry_homotopy')
        # Figure 2: small multiples, separate methods, same axes and same underlying geometry.
        fig,axs=plt.subplots(2,2,figsize=(11,6.1),sharex=True,sharey=True)
        for ax,m in zip(axs.flat,METHODS):
            geom(ax,fill=True);z=pop[pop.method==m];f=feasible_mask(z,args.tol)
            bad=z.loc[~f];good=z.loc[f]
            if len(bad):ax.scatter(bad.x1,bad.x2,s=5,c=COLORS[m],alpha=.12,marker=MARKERS[m],rasterized=True,label='Infeasible candidates')
            if len(good):ax.scatter(good.x1,good.x2,s=14,c=COLORS[m],alpha=.85,marker=MARKERS[m],edgecolors='none',label='Feasible candidates')
            ax.set_title(f'{NAMES[m]} (feasible {len(good):,}/{len(z):,})',fontsize=11)
        fig.suptitle('Final decision vectors; inequality-feasible region shaded',fontsize=12)
        fig.tight_layout();save(fig,out,'02_decision_space_four_panels')
        # Objective plots: only 3 informative pairs; no cross-method overlay.
        if {'f1','f2','f3','f4'}.issubset(pop.columns):
            pairs=[('f1','f2'),('f1','f3'),('f3','f4')]
            fig,axs=plt.subplots(4,3,figsize=(11,11),sharex='col',sharey='col')
            for i,m in enumerate(METHODS):
                z=pop[pop.method==m];f=feasible_mask(z,args.tol)
                for j,(a,b) in enumerate(pairs):
                    ax=axs[i,j];bad=z.loc[~f];good=z.loc[f]
                    if len(bad):ax.scatter(bad[a],bad[b],s=3,alpha=.09,color=COLORS[m],rasterized=True)
                    if len(good):ax.scatter(good[a],good[b],s=10,alpha=.7,color=COLORS[m],rasterized=True)
                    if i==0:ax.set_title(r'$'+a+'$ vs $'+b+'$')
                    if i==3:ax.set_xlabel(r'$'+a+'$')
                    if j==0:ax.set_ylabel(NAMES[m]+'\n'+r'$'+b+'$')
            fig.suptitle('Objective-space candidates (faint = infeasible, solid = feasible)',fontsize=12)
            fig.tight_layout(rect=[0,0,1,.975]);save(fig,out,'04_objective_space_separate_methods')
    else:
        fig,ax=plt.subplots(figsize=(8,3.2));geom(ax);save(fig,out,'01_geometry_homotopy')
        print('Note: all_final_populations.csv absent; decision/objective plots skipped.')
    # Figure 3: compact two-panel final violations, avoid >20-decade empty space.
    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(10.5,4),gridspec_kw={'width_ratios':[.85,1.35]})
    for i,m in enumerate(METHODS):
        vals=runs.loc[runs.method==m,'min_combined_violation'].dropna().to_numpy(float)
        if not len(vals):continue
        jitter=np.linspace(-.18,.18,len(vals)) if len(vals)>1 else np.array([0.])
        ax0.scatter(i+jitter,np.maximum(vals,1e-30),s=24,color=COLORS[m],marker=MARKERS[m],alpha=.75)
        if m!='homotopy':
            ax1.scatter(i-1+jitter,vals,s=30,color=COLORS[m],marker=MARKERS[m],alpha=.75)
    for ax in (ax0,ax1):ax.axhline(args.tol,color='black',ls='--',lw=1,label=r'Tolerance $10^{-7}$');ax.set_yscale('log')
    ax0.set_xticks(range(4),[NAMES[m] for m in METHODS],rotation=22);ax0.set_ylabel('Minimum constraint violation per run');ax0.set_title('All methods')
    ax1.set_xticks(range(3),[NAMES[m] for m in METHODS[1:]]);ax1.set_title('Evolutionary methods (zoom)')
    ev=runs.loc[runs.method.isin(METHODS[1:]),'min_combined_violation'].dropna()
    if len(ev):ax1.set_ylim(min(args.tol*.6,max(ev.min()*.5,1e-9)),ev.max()*2)
    ax1.legend(loc='lower right',fontsize=9)
    fig.tight_layout();save(fig,out,'03_constraint_violation_comparison')
    diag=root/'feasibility_diagnostics.csv'
    if diag.exists():
        d=pd.read_csv(diag)
        if {'method','generation','min_violation'}.issubset(d.columns):
            fig,ax=plt.subplots(figsize=(8,4.2))
            for m in METHODS[1:]:
                z=d[d.method==m]
                if len(z):
                    med=z.groupby('generation')['min_violation'].median().sort_index()
                    ax.semilogy(med.index,np.maximum(med.values,1e-30),color=COLORS[m],lw=1.5,label=NAMES[m])
            ax.axhline(args.tol,color='black',ls='--',lw=1,label=r'Tolerance $10^{-7}$')
            ax.set_xlabel('Generation');ax.set_ylabel('Median minimum constraint violation');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=4,frameon=False)
            fig.tight_layout();save(fig,out,'05_evolutionary_feasibility_history')
    (out/'README.txt').write_text('Generated solely from existing experiment CSV files.\n'
       'Figure 01 shades only the nonlinear inequality-feasible region.\n'
       'Figures 02 and 04 use separate panels; infeasible EA candidates are not Pareto-optimal solutions.\n'
       'Feasibility masks are taken from the CSV feasible field if present, otherwise calculated from the constraints.\n'
       'Homotopy is a single run, evolutionary algorithms are repeated runs; timings are not equal-budget.\n',encoding='utf8')
    print('Created:',out)
    print('\n'.join('  '+p.name for p in sorted(out.iterdir())))
if __name__=='__main__':main()
