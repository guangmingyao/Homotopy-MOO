#!/usr/bin/env python3
"""A2 postprocessing; read only local files, never rerun optimization."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis'
OBJ = ['f1', 'f2', 'f3', 'f4']
XCOL = ['x1', 'x2', 'x3', 'x4']
EQ_TOL, INEQ_TOL = 1e-4, 1e-7

def read_csv(path):
    if not path.is_file():
        raise SystemExit(f'Missing required input: {path}')
    return pd.read_csv(path)

def assess(frame):
    d = frame.copy()
    x1, x2, x3, x4 = d[XCOL].to_numpy(float).T
    eq = np.maximum(np.abs(x4), np.abs(x3*x3-x2*x2))
    g = (x1*x1-1)**2+x2*x2-0.08
    ineq = np.maximum.reduce([np.zeros(len(d)),g,-2-x1,x1-2,
                               -1-x2,x2-1,-1-x3,x3-1,-1-x4,x4-1])
    eligible = (d['eligible'].astype(str).str.lower().isin(['true','1','yes']).to_numpy()
                if 'eligible' in d else np.ones(len(d), dtype=bool))
    d['A2_eq_residual'] = eq
    d['A2_ineq_violation'] = ineq
    d['A2_feasible'] = eligible & (eq <= EQ_TOL) & (ineq <= INEQ_TOL)
    return d

def main():
    a2 = read_csv(ROOT/'results'/'all_final_populations.csv')
    original = read_csv(ROOT/'all_final_populations.csv')
    ref = read_csv(ROOT/'reference_front.csv')
    settings_path = ROOT/'settings.json'
    if not settings_path.is_file():
        raise SystemExit(f'Missing required input: {settings_path}')
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    for name, data in [('A2',a2),('original',original)]:
        missing = set(['method','run']+XCOL+OBJ)-set(data.columns)
        if missing: raise SystemExit(f'{name} missing columns: {sorted(missing)}')
    if not set(OBJ).issubset(ref): raise SystemExit('Reference missing objective columns')
    hom = original[original.method.astype(str).str.lower() == 'homotopy'].copy()
    if hom.empty: raise SystemExit('Original all_final_populations.csv has no homotopy rows')
    if (a2.method.astype(str).str.lower() == 'homotopy').any():
        raise SystemExit('A2 results unexpectedly include homotopy')
    a2['source'] = 'A2_reoptimization'
    hom['source'] = 'Original_homotopy'
    combined = assess(pd.concat([a2,hom], ignore_index=True))

    R = ref[OBJ].to_numpy(float)
    ideal = np.asarray(settings.get('normalization_ideal',R.min(axis=0)),float)
    scale = np.asarray(settings.get('normalization_scale',np.maximum(R.max(axis=0)-ideal,1e-12)),float)
    point = np.asarray(settings.get('normalized_hv_reference_point',[1.1]*4),float)
    if ideal.shape != (4,) or scale.shape != (4,) or (scale<=0).any() or point.shape != (4,):
        raise SystemExit('Invalid reference normalization settings')
    RN = (R-ideal)/scale
    ref_tree = cKDTree(RN)
    hv = HV(ref_point=point)
    ref_hv = float(settings.get('reference_hv',hv(RN)))
    records, nd_rows = [], []
    for (source,method,run), group in combined.groupby(['source','method','run'],sort=True):
        good = group[group.A2_feasible].copy()
        n_feasible = len(good)
        if n_feasible:
            F0 = good[OBJ].to_numpy(float)
            _, idx = np.unique(np.round(F0,12),axis=0,return_index=True)
            good = good.iloc[np.sort(idx)].copy()
            idx_nd = NonDominatedSorting().do(good[OBJ].to_numpy(float),only_non_dominated_front=True)
            good = good.iloc[idx_nd].copy()
            F = (good[OBJ].to_numpy(float)-ideal)/scale
            gd = float(ref_tree.query(F)[0].mean())
            igd = float(cKDTree(F).query(RN)[0].mean())
            hypervolume = float(hv(F))
            nd_rows.append(good)
        else:
            gd = igd = hypervolume = np.nan
        records.append(dict(source=source,method=method,run=run,
            n_candidates=len(group),n_feasible=n_feasible,feasible_run=int(n_feasible>0),
            n_nd=len(good),left_covered=int((good.x1<0).any()),
            right_covered=int((good.x1>0).any()),
            both_components=int((good.x1<0).any() and (good.x1>0).any()),
            min_eq_residual=float(group.A2_eq_residual.min()),
            max_eq_accepted=float(good.A2_eq_residual.max()) if len(good) else np.nan,
            HV=hypervolume,HV_ratio=hypervolume/ref_hv if np.isfinite(hypervolume) and ref_hv>0 else np.nan,
            GD=gd,IGD=igd,AHD=max(gd,igd) if n_feasible else np.nan))
    per = pd.DataFrame(records)
    summary = per.groupby(['source','method'],sort=False).agg(
        runs=('run','size'),feasible_runs=('feasible_run','sum'),
        feasible_points_mean=('n_feasible','mean'),nondominated_mean=('n_nd','mean'),
        both_components_runs=('both_components','sum'),
        HV_mean=('HV','mean'),HV_sd=('HV','std'),
        IGD_mean=('IGD','mean'),IGD_sd=('IGD','std'),
        GD_mean=('GD','mean'),AHD_mean=('AHD','mean'),
        min_eq_residual=('min_eq_residual','min'),
        max_eq_accepted=('max_eq_accepted','max')).reset_index()
    summary['feasible_run_rate']=summary.feasible_runs/summary.runs
    OUT.mkdir(exist_ok=True)
    per.to_csv(OUT/'A2_metrics_per_run.csv',index=False)
    summary.to_csv(OUT/'A2_metrics_summary.csv',index=False)
    pd.concat(nd_rows,ignore_index=True).to_csv(OUT/'A2_nondominated.csv',index=False)
    (OUT/'A2_analysis_settings.json').write_text(json.dumps({
        'experiment':'A2','equality_tolerance':EQ_TOL,'inequality_tolerance':INEQ_TOL,
        'reference':'local reference_front.csv', 'normalization':'local original settings.json',
        'A2_populations':'results/all_final_populations.csv',
        'homotopy_baseline':'original all_final_populations.csv',
        'metrics':'HV, GD, IGD, AHD on feasible nondominated objective vectors; original normalization'
    },indent=2),encoding='utf-8')
    print('\nA2 metrics summary:')
    print(summary[['method','runs','feasible_runs','both_components_runs','HV_mean','IGD_mean','GD_mean']].to_string(index=False))
    print('\nSaved in:',OUT)

if __name__ == '__main__':
    main()
