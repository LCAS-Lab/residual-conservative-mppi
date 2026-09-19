# P4 LTI Canonical Simulation Checkpoint

Canonical historical run used by the current P4 manuscript:

- Date: 2026-06-10
- SERVO_TAU: 0.9 s
- n_trials: 50
- K: 2048
- T: 40
- steps: 300
- seeds: 0--49
- representative seed: 21

## Canonical aggregate results

### Vanilla MPPI

- success: 0.64
- time-to-goal: 232.78 +/- 22.71 steps
- min clearance: 0.05 +/- 0.10 m
- violation steps: 4.16 +/- 6.32
- path length: 15.85 +/- 0.90 m

### RC-MPPI

- success: 0.94
- time-to-goal: 249.00 +/- 12.62 steps
- min clearance: 0.13 +/- 0.09 m
- violation steps: 0.62 +/- 2.63
- path length: 16.42 +/- 0.74 m

## Representative seed 21

Vanilla MPPI:

- success: 0
- time-to-goal: 218 steps
- min clearance: -0.08171117305755615 m
- violation steps: 21
- max penetration: 0.08171117305755615 m
- path length: 13.026369094848633 m

RC-MPPI:

- success: 1
- time-to-goal: 243 steps
- min clearance: 0.193353533744812 m
- violation steps: 0
- max penetration: 0.0 m
- path length: 16.28592872619629 m

The archived run exactly matches the LTI aggregate and representative-seed values currently reported in the P4 manuscript.

## Local archive

Archive name:

`P4_LTI_canonical_K2048_20260610.tar.gz`

SHA256:

`4e18fdae8a02428626b74788bff6ce28b36d3ff887d16ab338bdd50229a3f4e5`

The archive is intentionally not committed here because it is a local binary artifact. The corresponding historical `run_summary.json` and `mc_trial_metrics.csv` should be copied into this directory from the preserved local results directory before this checkpoint is merged to `main`.

## Reproducibility audit on 2026-09-16

The current remote `main` source is commit:

`fb7bd7e01c7089b65dbee3d5c3bf3683286889f4` (`fixed simulation`)

The current checked-in `sim1_lti.py` is unmodified relative to that commit. In the current software environment, repeated runs with identical seeds are deterministic, but the current code does not exactly reproduce the historical June 10 Monte Carlo realization.

Current-code K=2048 audit (n=50, T=40, steps=300):

### Vanilla MPPI

- success: 0.54
- time-to-goal: 229.20 +/- 26.43 steps
- min clearance: 0.00 +/- 0.12 m
- violation steps: 6.46 +/- 7.94
- path length: 16.14 +/- 0.73 m

### RC-MPPI

- success: 0.80
- time-to-goal: 248.02 +/- 18.71 steps
- min clearance: 0.13 +/- 0.11 m
- violation steps: 1.32 +/- 3.46
- path length: 16.55 +/- 0.75 m

The qualitative RC-MPPI improvement remains present, but the exact historical Monte Carlo numbers are not reproduced by the current source/environment combination.

The historical K=2048 run was generated between commits `a6a27eb` and `fb7bd7e`, so an intermediate uncommitted working-tree state and/or software-environment change may explain the exact numerical difference.

Do not tune parameters merely to recover the historical numbers. Final submission experiments should be regenerated only after the code-paper consistency audit is complete.
