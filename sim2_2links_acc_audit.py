#!/usr/bin/env python3
"""
P4 RC-MPPI 2R-arm consistency-audit candidate.

The original sim2_2links.py is intentionally left unchanged. This candidate
makes only consistency/reproducibility corrections needed before the ACC
numerical results are frozen:

1. Sampled rollout torques are saturated to the plant actuator bounds and the
   MPPI update uses the corresponding effective perturbation after saturation.
2. Safety/trajectory metrics are computed from the noiseless true plant state;
   noisy measurements remain in the feedback/residual-estimation loop.
3. Representative-seed selection is repaired so the preferred
   "vanilla violates / RC clean" condition is actually reachable.
4. The practical residual-dependent radius increment is explicitly treated as
   a simulation heuristic, not as the theorem's certified sufficient margin.
5. Gaussian measurement noise is retained as a stochastic stress test; it is
   not presented as satisfying the deterministic bounded-defect theorem.
6. Environment, git state, settings, raw Monte Carlo metrics, and the selected
   representative trajectories are saved for reproducibility.

No controller or plant parameters are retuned relative to sim2_2links.py.
"""

import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import sim2_2links as base


@torch.no_grad()
def mppi_clipped(x0, u_bar, r_eff, obs_c, alpha_k,
                 sigma_eff, beta_eff, K=4096, T=35):
    """MPPI update consistent with saturated rollout torques.

    Raw perturbations are Gaussian, but each sampled torque is projected to
    [-TAU_MAX, TAU_MAX] before propagation. The weighted control update uses

        eps_eff = clip(u_bar + eps) - u_bar,

    i.e. the perturbation actually realized by the nominal rollout.
    """
    eps = sigma_eff * torch.randn(K, T, 2, device=base.DEV, dtype=base.F32)
    eps_eff = torch.empty_like(eps)

    x = x0[None, :].expand(K, -1).clone()
    cost = torch.zeros(K, device=base.DEV, dtype=base.F32)

    for t in range(T):
        u_raw = u_bar[t][None, :] + eps[:, t, :]
        u = torch.clamp(u_raw, -base.TAU_MAX, base.TAU_MAX)
        eps_eff[:, t, :] = u - u_bar[t][None, :]

        x = base.step_nominal(x, u)
        cost += base.running_cost(x, u, r_eff, obs_c, alpha_k)

    cost += base.term_cost(x)
    cost -= cost.min()
    w = torch.exp(-cost / beta_eff)
    w /= w.sum() + 1e-12

    du = torch.einsum("k,kti->ti", w, eps_eff)
    u_new = torch.clamp(u_bar + du, -base.TAU_MAX, base.TAU_MAX)
    u0 = u_new[0]
    u_shift = torch.cat(
        [u_new[1:], torch.zeros(1, 2, device=base.DEV, dtype=base.F32)], dim=0)
    return u0, u_shift


def simulate(adaptive=False, steps=200, K=4096, T=35, tol=0.05,
             debug=False, ret_ctrl=False):
    """Closed-loop trial with true-state metrics and noisy feedback.

    The controller receives y_k = x_k + measurement noise exactly as in the
    original experiment. The true noiseless x_k is used only for reported
    clearance, goal, path-length, and safety metrics.
    """
    x = torch.zeros(4, device=base.CPU, dtype=base.F32)
    y_k = x + base.noise_cpu()
    y_prev = y_k.clone()
    u_prev = torch.zeros(2, device=base.CPU, dtype=base.F32)
    tau_exec = torch.zeros(2, device=base.CPU, dtype=base.F32)
    u_bar = torch.zeros(T, 2, device=base.DEV, dtype=base.F32)

    q_true_log = [x[:2].numpy().copy()]
    q_meas_log = [y_k[:2].numpy().copy()]
    ee_true_log = [base.fk_ee_np(*x[:2].tolist())]
    tau_cmd_log = []
    tau_exec_log = []

    s_bar = 0.0
    dbg = ({k: [] for k in ["s_bar", "r_eff", "alpha_k",
                              "sigma_eff", "beta_eff"]}
           if debug else None)

    for k in range(steps):
        if adaptive and k > 0:
            _, _, s_bar = base.update_sbar(y_k, y_prev, u_prev, s_bar)
            margin, sigma_eff, beta_eff, alpha_k = base.modulate(
                s_bar, base.SIGMA_0, base.BETA_0)
        else:
            margin = 0.0
            sigma_eff = base.SIGMA_0
            beta_eff = base.BETA_0
            alpha_k = base.W_OBS_BASE

        r_eff = base.OBS_R + torch.tensor(
            margin, device=base.DEV, dtype=base.F32)
        alpha_t = torch.tensor(alpha_k, device=base.DEV, dtype=base.F32)

        u0, u_bar = mppi_clipped(
            y_k.to(base.DEV), u_bar, r_eff, base.OBS_C,
            alpha_t, sigma_eff, beta_eff, K=K, T=T)
        u_k = u0.to(base.CPU)

        y_prev = y_k.clone()
        u_prev = u_k.clone()
        y_k, x, tau_exec = base.plant_step(x, u_k, tau_exec)

        q_true_log.append(x[:2].numpy().copy())
        q_meas_log.append(y_k[:2].numpy().copy())
        ee_true_log.append(base.fk_ee_np(*x[:2].tolist()))

        if ret_ctrl:
            tau_cmd_log.append(u_k.numpy().copy())
            tau_exec_log.append(tau_exec.numpy().copy())

        if debug:
            dbg["s_bar"].append(s_bar)
            dbg["r_eff"].append(float(r_eff.cpu()))
            dbg["alpha_k"].append(float(alpha_k))
            dbg["sigma_eff"].append(sigma_eff)
            dbg["beta_eff"].append(beta_eff)

    q_true = np.asarray(q_true_log)
    q_meas = np.asarray(q_meas_log)
    ee_true = np.asarray(ee_true_log)
    tau_cmd = np.asarray(tau_cmd_log) if ret_ctrl else None
    tau_ex = np.asarray(tau_exec_log) if ret_ctrl else None

    clr = base.clearance_series(q_true)
    first_goal = base.ttg(ee_true, tol)
    viols = int(np.sum(clr < 0.0))
    success = int(first_goal >= 0 and viols == 0)

    # Preserve the original paper-facing TTG convention: failed trials are
    # coded as steps+1. Also save the uncensored first-goal step separately.
    t_goal = first_goal if success else steps + 1

    metrics = {
        "success": success,
        "t_goal": int(t_goal),
        "first_goal_step": int(first_goal),
        "min_clr": float(np.min(clr)),
        "viols": viols,
        "plen": base.plen(ee_true),
        "energy_cmd": (float(np.sum(tau_cmd**2))
                       if tau_cmd is not None else np.nan),
        "energy_exec": (float(np.sum(tau_ex**2))
                        if tau_ex is not None else np.nan),
    }

    if debug:
        dbg = {k: np.asarray(v) for k, v in dbg.items()}

    return q_true, q_meas, ee_true, tau_cmd, tau_ex, metrics, dbg


def mc_eval(n=50, seed0=0, steps=200, K=4096, T=35, tol=0.05):
    keys = ["success", "t_goal", "first_goal_step", "min_clr", "viols",
            "plen", "energy_cmd", "energy_exec"]
    res = {m: {k: [] for k in keys} for m in ["van", "rc"]}
    seeds = []

    for i in range(n):
        seed = seed0 + i
        seeds.append(seed)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"MC trial {i+1:>2}/{n}", flush=True)

        for adaptive, tag in [(False, "van"), (True, "rc")]:
            base.set_seed(seed)
            *_, met, _ = simulate(
                adaptive=adaptive, steps=steps, K=K, T=T,
                tol=tol, ret_ctrl=True)
            for k in keys:
                res[tag][k].append(float(met[k]))

    for m in ["van", "rc"]:
        for k in keys:
            res[m][k] = np.asarray(res[m][k])
    return res, np.asarray(seeds, dtype=int)


def select_seed(res, seeds):
    """Select a representative safety contrast without impossible conditions."""
    vv = res["van"]["viols"]
    rv = res["rc"]["viols"]
    vc = res["van"]["min_clr"]
    rc = res["rc"]["min_clr"]
    rs = res["rc"]["success"].astype(int)

    cand = np.where((vv > 0) & (rv == 0) & (rs == 1))[0]
    if cand.size:
        i = cand[np.argmax(vv[cand])]
        return int(seeds[i]), int(i), \
            f"vanilla violated ({vv[i]:.0f} steps), RC-MPPI clean"

    cand = np.where(rs == 1)[0]
    if cand.size:
        delta = rc[cand] - vc[cand]
        i = cand[np.argmax(delta)]
        return int(seeds[i]), int(i), \
            f"RC success; max clearance delta={rc[i]-vc[i]:.3f} m"

    i = int(np.argmax(rc - vc))
    return int(seeds[i]), int(i), \
        f"fallback: max clearance delta={rc[i]-vc[i]:.3f} m"


def summarize(res):
    print("\n" + "=" * 68)
    print(f"{'Metric':<28} {'Vanilla MPPI':>18} {'RC-MPPI':>18}")
    print("=" * 68)
    print(f"{'Success rate':<28} "
          f"{np.mean(res['van']['success']):>18.2f} "
          f"{np.mean(res['rc']['success']):>18.2f}")

    rows = [
        ("Penalized TTG (steps)", "t_goal"),
        ("Min link clearance (m)", "min_clr"),
        ("Violation steps", "viols"),
        ("EE path length (m)", "plen"),
        ("Commanded quadratic control metric", "energy_cmd"),
        ("Executed quadratic torque metric", "energy_exec"),
    ]
    for label, key in rows:
        vm, vs = np.mean(res["van"][key]), np.std(res["van"][key])
        rm, rs = np.mean(res["rc"][key]), np.std(res["rc"][key])
        print(f"{label:<28} {vm:>8.2f} ± {vs:<7.2f} {rm:>8.2f} ± {rs:<7.2f}")
    print("=" * 68)


def git_text(*args):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def make_output_dir(n, K, T, steps):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / (
        f"2r_accaudit_n{n}_K{K}_T{T}_steps{steps}_{ts}")
    out.mkdir(parents=True, exist_ok=False)
    return out


def save_csv(res, seeds, out):
    keys = ["success", "t_goal", "first_goal_step", "min_clr", "viols",
            "plen", "energy_cmd", "energy_exec"]
    with (out / "mc_trial_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "method", *keys])
        for i, seed in enumerate(seeds):
            for method in ["van", "rc"]:
                w.writerow([int(seed), method,
                            *[res[method][k][i] for k in keys]])


def save_summary(res, seed_star, idx_star, reason, out):
    summary = {
        "representative_seed": int(seed_star),
        "representative_index": int(idx_star),
        "representative_reason": reason,
        "aggregate": {},
    }
    keys = ["success", "t_goal", "min_clr", "viols", "plen",
            "energy_cmd", "energy_exec"]
    for method in ["van", "rc"]:
        summary["aggregate"][method] = {
            k: {"mean": float(np.mean(res[method][k])),
                "std": float(np.std(res[method][k]))}
            for k in keys
        }
    with (out / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def save_metadata(out, n, K, T, steps, tol):
    metadata = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "script": "sim2_2links_acc_audit.py",
        "experiment_role": "2R code-paper consistency audit candidate",
        "controller_retuned": False,
        "sampled_rollout_torque_clipping": True,
        "mppi_update_uses_effective_clipped_perturbation": True,
        "reported_metrics_use_true_noiseless_plant_state": True,
        "feedback_and_residual_use_noisy_measurements": True,
        "measurement_noise_note": (
            "Gaussian measurement noise is retained as a stochastic stress test; "
            "it is not asserted to satisfy the deterministic bounded-defect theorem."),
        "practical_tightening_note": (
            "The simulation uses clip(kappa_r*s_bar, 0, Delta_r_max) as a "
            "practical obstacle-radius increment. It is not asserted to equal "
            "the theorem's certified sufficient tightening margin."),
        "git": {
            "commit": git_text("rev-parse", "HEAD"),
            "branch": git_text("rev-parse", "--abbrev-ref", "HEAD"),
            "status": git_text("status", "--short"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device": str(base.DEV),
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "CPU"),
        },
        "settings": {
            "DT": base.DT,
            "SERVO_TAU": base.SERVO_TAU,
            "TAU_MAX": base.TAU_MAX,
            "true_masses": [base.M1, base.M2],
            "nominal_masses": [base.M1n, base.M2n],
            "B_DAMP": base.B_DAMP,
            "Q_STD": base.Q_STD,
            "QD_STD": base.QD_STD,
            "n_trials": n,
            "base_seed": 0,
            "seed_range": [0, n - 1],
            "K": K,
            "T": T,
            "steps": steps,
            "goal_tol": tol,
            "SIGMA_0": base.SIGMA_0,
            "BETA_0": base.BETA_0,
            "KAPPA_R": base.KAPPA_R,
            "DELTA_R_MAX": base.DELTA_R_MAX,
            "KAPPA_SIGMA": base.KAPPA_SIGMA,
            "KAPPA_BETA": base.KAPPA_BETA,
            "beta_convention": "w proportional to exp(-Z/beta); larger beta is softer",
        },
    }
    with (out / "audit_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    K = 4096
    n = 50
    steps = 200
    T = 35
    tol = 0.05

    print("P4 2R consistency-audit candidate")
    print(f"Device    : {base.DEV}")
    print(f"SERVO_TAU : {base.SERVO_TAU:.3f} s")
    print(f"TAU_MAX   : {base.TAU_MAX:.1f} N m")
    print("Rollouts  : sampled torques CLIPPED to actuator bounds")
    print("Update    : uses effective perturbation after clipping")
    print("Metrics   : noiseless true plant state")
    print("Feedback  : noisy measured state retained")
    print("Theory    : measurement noise treated as stress test, not theorem assumption")
    print("Tightening: practical residual-dependent radius increment")
    print(f"\nMC: n={n}, K={K}, steps={steps}, T={T}, tol={tol}")

    out = make_output_dir(n, K, T, steps)
    print(f"Output: {out}")
    save_metadata(out, n, K, T, steps, tol)

    t0 = time.time()
    res, seeds = mc_eval(n=n, seed0=0, steps=steps, K=K, T=T, tol=tol)
    print(f"MC elapsed: {time.time() - t0:.1f}s")
    summarize(res)

    seed_star, idx_star, reason = select_seed(res, seeds)
    print(f"\nRepresentative seed: {seed_star} ({reason})")

    save_csv(res, seeds, out)
    save_summary(res, seed_star, idx_star, reason, out)

    replays = {}
    for adaptive, tag in [(False, "van"), (True, "rc")]:
        base.set_seed(seed_star)
        q_true, q_meas, ee_true, tau_cmd, tau_ex, met, dbg = simulate(
            adaptive=adaptive, steps=steps, K=K, T=T, tol=tol,
            debug=True, ret_ctrl=True)
        replays[tag] = (q_true, q_meas, ee_true, tau_cmd, tau_ex, met, dbg)
        print(
            f"  {tag}: success={met['success']}, penalized_TTG={met['t_goal']}, "
            f"clr={met['min_clr']:.4f} m, viols={met['viols']}, "
            f"path={met['plen']:.3f} m")

    np.savez_compressed(
        out / f"representative_seed{seed_star}_trajectories.npz",
        van_q_true=replays["van"][0],
        van_q_meas=replays["van"][1],
        van_ee_true=replays["van"][2],
        van_tau_cmd=replays["van"][3],
        van_tau_exec=replays["van"][4],
        rc_q_true=replays["rc"][0],
        rc_q_meas=replays["rc"][1],
        rc_ee_true=replays["rc"][2],
        rc_tau_cmd=replays["rc"][3],
        rc_tau_exec=replays["rc"][4],
    )

    print("\nAudit run complete.")
