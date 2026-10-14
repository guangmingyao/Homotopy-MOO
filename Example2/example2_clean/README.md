# Example 2 — clean original constrained benchmark

This folder is self-contained: the homotopy solver is included as `homotopy_solver.py` (the original mathematical solver), and `run_example2.py` runs it together with NSGA-II, NSGA-III, and C-TAEA on the **original four-variable constrained problem**.

## Install once

```powershell
python -m pip install numpy pandas scipy matplotlib pymoo
```

## Test first (quick)

```powershell
python run_example2.py --runs 1 --generations 5 --homotopy-fast --output results_test
```

## Manuscript run (20 runs, 400 generations, population 84)

```powershell
python run_example2.py
```

The default run creates a **single `results` folder** beside the scripts. It refuses to overwrite a nonempty output directory; choose `--output results_v2` for another run. No prior CSVs are needed.

## Files under `results/`

- `table_summary.csv`: method-level manuscript summary
- `table_per_run.csv`: individual-run results
- `all_final_populations.csv`: all final-population candidates, including infeasible ones
- `final_populations/`: one CSV per algorithm/run, including homotopy
- `feasibility_diagnostics.csv`: constraint residual history for evolutionary methods
- `feasible_archives/`: accumulated feasible evolutionary candidates (supplementary only)
- `near_feasible/`: small per-generation near-feasible samples (supplementary only)
- `homotopy_raw.csv`: native homotopy solutions, including success flags
- `homotopy_native/`: original homotopy solver's native data and figures
- `reference_front.csv`, `settings.json`, `methodology.txt`
- `figures/`: unified objective projections, component coverage, feasibility diagnostics

## Important scientific interpretation

The primary comparison uses original constraints for every method. No reduced/oracle-assisted algorithm is included. All quality indicators use the same fixed analytical reference. A failed-feasibility run has **undefined** HV, IGD, GD and AHD (NaN), rather than a misleading zero. The integrated homotopy solver writes raw results but skips its separate, computationally expensive native indicator and plotting pipeline; all methods instead use the unified metrics and figures. The mathematical homotopy and path-tracking functions are unchanged. Homotopy is a single deterministic result, not 20 independent runs. CPU time includes homotopy's native analysis and plots; treat timings as descriptive. The test run uses `--homotopy-fast` and is not manuscript-quality.

## Solver provenance

`homotopy_solver.py` is based on the uploaded `homotopy_manyobj_old_algorithm2(3).py`. Only output management and an optional `native_analysis` switch were changed. The benchmark disables native analysis to avoid duplicate calculations and saves the unified results under one folder.
