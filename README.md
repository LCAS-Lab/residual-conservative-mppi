# Residual-Conservative Model Predictive Path Integral Control

This repository contains the simulation and reproducibility code for the paper

**Residual-Conservative Model Predictive Path Integral Control**  
Hyung-Jin Yoon and Hunmin Kim

RC-MPPI uses the measured prediction--execution residual to adapt constraint tightening, safety-cost scaling, MPPI sampling spread, and temperature when the nominal rollout model becomes less reliable. The implementation separates physical model--plant mismatch from the Gaussian perturbations used internally by MPPI for Monte Carlo exploration.

## Repository structure

- `sim1_lti.py` — LTI point-mass simulation.
- `sim2_2links.py` — planar 2R manipulator simulation.
- `sim1_lti_acc_audit.py` — audited LTI experiment used for the ACC manuscript.
- `sim2_2links_acc_audit.py` — audited 2R experiment used for the ACC manuscript.
- `scripts/generate_final_p4_figures.py` — regenerates the final manuscript figures from frozen audit artifacts without rerunning Monte Carlo simulations.
- `reproducibility/` — frozen audit records, metadata, per-trial metrics, logs, and representative trajectories.

## Manuscript numerical source

The numerical values reported in the ACC manuscript are frozen from the school-GPU audit performed on **2026-09-17**:

- `reproducibility/lti_accaudit_school_gpu_20260917/`
- `reproducibility/2r_accaudit_school_gpu_20260917/`

These runs used the same consistency-corrected implementation and were performed without controller retuning. The stored metadata records the source commit, experiment settings, execution environment, and notes on the distinction between practical saturated tightening and the sufficient theoretical tightening margin.

Independent cross-machine audit records are retained in:

- `reproducibility/lti_accaudit_home_gpu_20260916/`
- `reproducibility/2r_accaudit_home_cpu_20260916/`

The older LTI reference results are preserved in:

- `reproducibility/lti_canonical_20260610/`

## Audited implementation conventions

The ACC audit versions use the following implementation conventions.

1. Raw MPPI control perturbations are Gaussian.
2. Sampled rollout commands are clipped to the admissible input bounds before nominal propagation.
3. The MPPI mean update uses the corresponding **effective clipped perturbations**, i.e., the actual sampled-command displacement from the mean command.
4. The simulation obstacle inflation is the practical saturated rule
   ```
   clip(kappa_r * s_bar, 0, Delta_r_max)
   ```
   and is not asserted to equal the sufficient theoretical tightening margin in the paper.
5. In the 2R study, feedback and residual estimation use noisy measurements, while success, clearance, violation, and path-length metrics are evaluated from the noiseless true plant state.

## Reproducing the audited experiments

From the repository root:

```bash
python3 sim1_lti_acc_audit.py
python3 sim2_2links_acc_audit.py
```

The scripts create run-specific output directories under `results/`, which is intentionally excluded from version control. Frozen manuscript-source outputs are already stored under `reproducibility/`.

## Regenerating the manuscript figures

The final figures are generated from the frozen representative-trajectory files:

```bash
python3 scripts/generate_final_p4_figures.py
```

This produces:

- `fig1_trajectory_seed16.pdf` — LTI representative trajectory.
- `fig1_traj.pdf` — 2R representative trajectory.
- `fig2_clearance.pdf` — 2R link-clearance history.

The figure script uses embedded TrueType PDF fonts (`pdf.fonttype = 42`).

## Reproduced software/hardware environment

The frozen 2026-09-17 manuscript-source audit recorded:

- OS: Linux 6.8 / x86_64
- Python: 3.10.12
- NumPy: 1.24.4
- PyTorch: 2.9.1+cu126
- CUDA runtime reported by PyTorch: 12.6
- cuDNN: 91002
- GPU: NVIDIA GeForce RTX 4080

The code can also run on CPU, although exact Monte Carlo outcomes can vary slightly across hardware/backends. The independent audit records are retained to document this reproducibility check.

## Installation

A minimal Python environment can be installed with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the PyTorch build appropriate for the local CUDA/driver environment if the default `pip` package is not suitable.

## Citation

If you use this repository, please cite the accompanying paper. A formal citation will be added after publication.
