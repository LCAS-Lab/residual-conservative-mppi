# Residual-Conservative MPPI (RC-MPPI)

Official implementation of **RC-MPPI**, a sampling-based Model Predictive Control framework that modulates safety conservatism online using execution prediction residuals.

> **Residual-Conservative Model Predictive Path Integral Control**  
> Hyung-Jin Yoon, Ashik Rasul, Humaira Tasnim, and Hunmin Kim  
> Department of Mechanical and Nuclear Engineering, Tennessee Technological University  
> [[Paper]](#citation) [[Code]](https://github.com/LCAS-Lab/residual-conservative-mppi)

---

## Overview

RC-MPPI addresses the problem of online conservatism adaptation under execution mismatch. When a high-level planner relies on a simplified nominal model, actuation lag, saturation, and unmodeled dynamics create a persistent prediction–execution discrepancy that can lead to constraint violations.

RC-MPPI computes a filtered residual statistic from the discrepancy between predicted and realized state transitions, and embeds it into the MPPI optimization via:
- **Residual-dependent constraint tightening** — obstacle radius inflation scales with observed mismatch
- **Adaptive safety-cost shaping** — penalty weight increases under larger residuals
- **Adaptive sampling parameters** — noise std and temperature modulated by residual

As model mismatch increases, conservatism increases automatically. As residuals diminish, the controller recovers nominal MPPI behavior.

| Trajectories | Clearance vs. Time | MC Scatter |
|:---:|:---:|:---:|
| ![Trajectories](rc_mppi_fig1_trajectories.png) | ![Clearance](rc_mppi_fig2_clearance.png) | ![MC Scatter](rc_mppi_fig3_mc_scatter.png) |

---

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+. GPU execution (CUDA) is supported and recommended for the full `K=8192` rollout count; the code falls back to CPU automatically with a reduced rollout budget.

---

## Usage

Run the full Monte Carlo evaluation and generate figures:

```bash
python rc_mppi.py
```

This will:
1. Run `N=50` paired-seed Monte Carlo trials (Vanilla MPPI vs. RC-MPPI)
2. Print a summary table of performance metrics
3. Replay the representative seed and save three figures:
   - `rc_mppi_fig1_trajectories.png` — trajectory overlay
   - `rc_mppi_fig2_clearance.png` — clearance vs. time
   - `rc_mppi_fig3_mc_scatter.png` — paired MC scatter plot

### Key configuration options

All hyperparameters are set via constants at the top of `rc_mppi.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EXECUTION_MODEL` | `"lag"` | Plant model: `"exact"` or `"lag"` |
| `SERVO_TAU` | `0.60` | First-order lag time constant (s) |
| `K_ROLLOUTS` | `8192` | Number of MPPI rollout samples |
| `T_HORIZON` | `40` | Planning horizon (steps) |
| `USE_RISK_ADAPTATION` | `True` | Enable RC-MPPI modulation |
| `KAPPA_R` | `0.40` | Radius inflation gain |
| `RISK_FILTER_RHO` | `0.20` | Exponential filter rate ρ |
| `MC_N_TRIALS` | `50` | Number of Monte Carlo trials |

To run vanilla MPPI only, set `USE_RISK_ADAPTATION = False`.

---

## Results

Monte Carlo evaluation under servo-lag execution mismatch (`τ = 0.60 s`, `n = 50` paired-seed trials, `K = 8192` rollouts, CUDA):

| Metric | Vanilla MPPI | RC-MPPI |
|--------|-------------|---------|
| Success rate | 0.62 | **0.90** |
| Time-to-goal (steps) | 198.9 ± 27.7 | **176.0 ± 18.9** |
| Min clearance (m) | 0.014 ± 0.271 | **0.196 ± 0.202** |
| Violation steps | 5.68 ± 8.39 | **1.62 ± 5.47** |
| Path length (m) | 15.88 ± 2.20 | **14.19 ± 1.60** |

RC-MPPI achieves a 45% relative improvement in success rate, reduces violation steps by 71%, and increases mean minimum clearance by 13×, without incurring longer paths or slower goal arrival. In the representative trial (seed 25), Vanilla MPPI penetrates the obstacle (−0.611 m, 31 violation steps) while RC-MPPI maintains 0.239 m clearance and reaches the goal successfully.

---

## Repository Structure

```
residual-conservative-mppi/
├── rc_mppi.py          # Main simulation: RC-MPPI and Vanilla MPPI
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # This file
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yoon2025rcmppi,
  title     = {Residual-Conservative Model Predictive Path Integral Control},
  author    = {Yoon, Hyung-Jin and Rasul, Ashik and Tasnim, Humaira and Kim, Hunmin},
  year      = {2025},
  note      = {Code available at \url{https://github.com/LCAS-Lab/residual-conservative-mppi}}
}

@software{yoon2025rcmppi_code,
  title     = {{RC-MPPI}: Residual-Conservative Model Predictive Path Integral Control},
  author    = {Yoon, Hyung-Jin and Rasul, Ashik and Tasnim, Humaira and Kim, Hunmin},
  year      = {2025},
  url       = {https://github.com/LCAS-Lab/residual-conservative-mppi},
  license   = {MIT}
}
```

---

## License

This project is released under the [MIT License](LICENSE).