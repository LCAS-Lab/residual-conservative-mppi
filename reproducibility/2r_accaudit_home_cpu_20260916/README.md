# P4 2R ACC Consistency Audit — Home CPU Run

Date: 2026-09-16

Source:
- branch: p4-sim-consistency-audit-v1
- script: sim2_2links_acc_audit.py
- K=4096
- n=50
- T=35
- steps=200
- seeds=0--49

This run was executed on CPU because CUDA initialization failed with
driver/runtime error 803 on the home machine.

The run is an implementation-consistency audit, not the final manuscript
numerical source.

Key aggregate results:

Vanilla MPPI:
- success: 0.28
- time-to-goal: 150.18 +/- 81.65
- min link clearance: -0.01 +/- 0.02 m
- violation steps: 4.04 +/- 4.02
- EE path length: 5.41 +/- 0.25 m
- commanded control energy: 3377.28 +/- 280.25
- executed control energy: 1240.73 +/- 104.71

RC-MPPI:
- success: 0.94
- time-to-goal: 28.82 +/- 43.50
- min link clearance: 0.02 +/- 0.01 m
- violation steps: 0.16 +/- 0.70
- EE path length: 3.90 +/- 0.18 m
- commanded control energy: 1384.16 +/- 116.87
- executed control energy: 574.48 +/- 50.83

Representative seed: 41

This candidate includes:
- clipped sampled rollout torques,
- effective clipped perturbations in the MPPI update,
- true noiseless plant state for physical performance metrics,
- noisy measurements retained for feedback/residual estimation,
- corrected representative-seed selection,
- no controller retuning.

Final manuscript results should be regenerated on the school GPU using
the identical source commit and experiment settings.
