"""
P4 RC-MPPI LTI consistency-audit candidate
===========================================

This script intentionally leaves sim1_lti.py unchanged and reuses its model,
metrics, plotting utilities, and RC-MPPI modulation. It changes only the
experiment/reproducibility items currently under audit:

1. The paper configuration K=2048 is used explicitly on both CPU and GPU.
2. Every sampled rollout command is clipped to the same actuator bounds used
   for executed commands, and the MPPI update uses the corresponding effective
   perturbation after clipping.
3. The practical residual-dependent obstacle-radius increment is described as
   a simulation heuristic; this script does not claim that it implements the
   certified sufficient tightening margin from the theorem.
4. Environment, git state, and exact settings are saved with each run.

No controller gains are retuned relative to sim1_lti.py.
"""

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import sim1_lti as base


# -----------------------------------------------------------------------------
# Audited MPPI core
# -----------------------------------------------------------------------------
@torch.no_grad()
def mppi_step_clipped(x0_gpu: torch.Tensor,
                      u_bar: torch.Tensor,
                      radius_eff: torch.Tensor,
                      alpha_k: float,
                      K: int,
                      T: int,
                      sigma: float,
                      beta: float):
    """One MPPI update with bounded sampled rollout commands.

    Raw Gaussian perturbations are sampled exactly as in sim1_lti.py. For each
    rollout step, the perturbed command is projected to [-U_CLIP, U_CLIP]. The
    MPPI control update then uses the *effective* perturbation

        eps_eff = clip(u_bar + eps) - u_bar,

    so the weighted update is consistent with the commands actually propagated
    through the nominal rollout model.
    """
    eps = sigma * torch.randn(K, T, 2, device=base.GPU, dtype=base.DTYPE)
    eps_eff = torch.empty_like(eps)

    x = x0_gpu[None, :].expand(K, -1).clone()
    cost = torch.zeros(K, device=base.GPU, dtype=base.DTYPE)

    for t in range(T):
        u_raw = u_bar[t][None, :] + eps[:, t, :]
        u = torch.clamp(u_raw, -base.U_CLIP, base.U_CLIP)
        eps_eff[:, t, :] = u - u_bar[t][None, :]

        x = x @ base.A_gpu.T + u @ base.B_gpu.T
        cost += base.running_cost_batch(x, u, radius_eff, alpha_k)

    cost += base.terminal_cost_batch(x)

    cost -= cost.min()
    w = torch.exp(-cost / beta)
    w /= w.sum() + 1e-12

    du = torch.sum(w[:, None, None] * eps_eff, dim=0)
    u_bar_new = torch.clamp(u_bar + du, -base.U_CLIP, base.U_CLIP)
    u0 = u_bar_new[0]

    u_bar_shifted = torch.cat(
        [u_bar_new[1:],
         torch.zeros(1, 2, device=base.GPU, dtype=base.DTYPE)],
        dim=0,
    )
    return u0, u_bar_shifted


# Patch only the MPPI rollout/update used by base.simulate().
base.mppi_step = mppi_step_clipped


# -----------------------------------------------------------------------------
# Reproducibility metadata
# -----------------------------------------------------------------------------
def _git(cmd):
    try:
        p = subprocess.run(
            ["git", *cmd],
            check=True,
            capture_output=True,
            text=True,
        )
        return p.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def save_audit_metadata(out_dir: Path, settings: dict):
    metadata = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "experiment_role": "LTI code-paper consistency audit candidate",
        "controller_retuned": False,
        "sampled_rollout_input_clipping": True,
        "mppi_update_uses_effective_clipped_perturbation": True,
        "practical_tightening_note": (
            "The simulation uses clip(kappa_r*s_bar, 0, Delta_r_max) as a "
            "practical obstacle-radius increment. It is not asserted here to "
            "equal the theorem's certified sufficient tightening margin."
        ),
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "status": _git(["status", "--short"]),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device": str(base.GPU),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
        },
        "settings": settings,
    }

    path = out_dir / "audit_metadata.json"
    with path.open("w") as f:
        json.dump(metadata, f, indent=2)
    return path


def make_output_dir(n_trials, K, T, steps):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tau = base.safe_float_tag(base.SERVO_TAU)
    name = f"lti_accaudit_tau_{tau}_n{n_trials}_K{K}_T{T}_steps{steps}_{ts}"
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=False)
    return out


# -----------------------------------------------------------------------------
# Main paper-configuration audit run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    K = 2048
    n_trials = 50
    steps_mc = 300
    steps_rep = 300
    T_horizon = 40
    sigma0 = 1.0
    beta0 = 1.0
    goal_tol = 0.25
    base_seed = 0

    settings = {
        "SERVO_TAU": base.SERVO_TAU,
        "alpha_servo": base.alpha_servo,
        "dt": base.dt,
        "n_trials": n_trials,
        "base_seed": base_seed,
        "seed_range": [base_seed, base_seed + n_trials - 1],
        "K": K,
        "T": T_horizon,
        "steps_mc": steps_mc,
        "steps_rep": steps_rep,
        "sigma0": sigma0,
        "beta0": beta0,
        "goal_tol": goal_tol,
        "U_CLIP": base.U_CLIP,
        "KAPPA_R": base.KAPPA_R,
        "DELTA_R_MAX": base.DELTA_R_MAX,
        "KAPPA_SIGMA": base.KAPPA_SIGMA,
        "KAPPA_BETA": base.KAPPA_BETA,
        "beta_convention": "w proportional to exp(-Z/beta); larger beta is softer",
    }

    print("P4 LTI consistency-audit candidate")
    print(f"Device    : {base.GPU}")
    print(f"SERVO_TAU : {base.SERVO_TAU:.3f} s (alpha={base.alpha_servo:.4f})")
    print("Weights   : exp(-cost/beta), larger beta = softer")
    print("Rollouts  : sampled commands CLIPPED to actuator bounds")
    print("Update    : uses effective perturbation after clipping")
    print("Tightening: practical residual-dependent radius increment")
    print(f"\nMC: n={n_trials}, K={K}, steps={steps_mc}, T={T_horizon}")
    print(f"    sigma0={sigma0}, beta0={beta0}, goal_tol={goal_tol}")
    print(
        f"    kappa_r={base.KAPPA_R}, "
        f"kappa_sigma={base.KAPPA_SIGMA}, "
        f"kappa_beta={base.KAPPA_BETA}"
    )

    out_dir = make_output_dir(n_trials, K, T_horizon, steps_mc)
    print(f"\nOutput: {out_dir}")
    save_audit_metadata(out_dir, settings)

    t0 = time.time()
    res, seeds = base.mc(
        n_trials=n_trials,
        base_seed=base_seed,
        steps=steps_mc,
        K=K,
        T=T_horizon,
        sigma0=sigma0,
        beta0=beta0,
        goal_tol=goal_tol,
    )
    print(f"MC elapsed: {time.time()-t0:.1f}s")
    base.summarize(res)

    base.save_metrics_csv(res, seeds, out_dir)

    seed_star, idx_star, reason = base.select_seed(res, seeds)
    print(f"\nRepresentative seed: {seed_star} ({reason})")
    base.save_run_summary(res, seeds, seed_star, idx_star, reason, out_dir)

    base.set_seed(seed_star)
    traj_v, met_v, _ = base.simulate(
        steps=steps_rep,
        K=K,
        T=T_horizon,
        sigma0=sigma0,
        beta0=beta0,
        goal_tol=goal_tol,
        adaptive=False,
        log=False,
    )
    base.set_seed(seed_star)
    traj_a, met_a, _ = base.simulate(
        steps=steps_rep,
        K=K,
        T=T_horizon,
        sigma0=sigma0,
        beta0=beta0,
        goal_tol=goal_tol,
        adaptive=True,
        log=False,
    )

    print(f"\nRepresentative trial (seed={seed_star}):")
    for lab, met in [("Vanilla", met_v), ("RC-MPPI", met_a)]:
        print(
            f"  {lab}: clr={met['min_clearance']:.3f}m, "
            f"viols={met['viols']}, success={met['success']}, "
            f"TTG={met['t_goal']}, path={met['path_len']:.3f}m"
        )

    # Save representative trajectories numerically for cross-machine checks.
    np.savez_compressed(
        out_dir / f"representative_seed{seed_star}_trajectories.npz",
        seed=np.asarray(seed_star),
        vanilla=traj_v,
        adaptive=traj_a,
    )

    print("\nAudit run complete.")
