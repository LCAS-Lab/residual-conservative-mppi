# P4 LTI ACC Consistency Audit — Home GPU Run

Date: 2026-09-16

Source:
- branch: p4-sim-consistency-audit-v1
- script: sim1_lti_acc_audit.py
- K=2048
- n=50
- T=40
- steps=300
- seeds=0--49

Environment:
- GPU: NVIDIA GeForce RTX 4060
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Python: 3.10.12

Consistency corrections:
- sampled rollout inputs clipped to actuator bounds
- MPPI update uses effective perturbations after clipping
- practical residual-dependent radius tightening is not claimed to equal
  the theorem-certified sufficient margin
- no controller retuning

Aggregate results:

Vanilla MPPI:
- success: 0.64
- time-to-goal: 218.28 +/- 22.18
- min clearance: 0.04 +/- 0.09 m
- violation steps: 3.82 +/- 5.53
- path length: 16.15 +/- 0.57 m

RC-MPPI:
- success: 0.98
- time-to-goal: 238.42 +/- 14.55
- min clearance: 0.13 +/- 0.07 m
- violation steps: 0.06 +/- 0.42
- path length: 16.59 +/- 0.69 m

Representative seed: 44
- Vanilla: clearance -0.149 m, 18 violations, failure
- RC-MPPI: clearance +0.116 m, 0 violations, success

This is the current consistency-audit numerical candidate. A second run on
the school GPU should be performed using the identical source commit/settings
before freezing final manuscript values.
