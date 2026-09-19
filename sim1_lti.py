"""
RC-MPPI: LTI Point-Mass Study with Servo-Lag Execution Mismatch
================================================================

This script evaluates Residual-Conservative MPPI (RC-MPPI) on the
discrete-time LTI point-mass system with first-order servo-lag execution
mismatch.

The implementation follows the paper formulation:

  Residual estimation:
    r_k     = y_k - f_theta(y_{k-1}, u_{k-1})
    s_k     = sqrt(wp*||r_pos||^2 + wv*||r_vel||^2)
    s_bar_k = (1-rho)*s_bar_{k-1} + rho*s_k

  Residual-conservative modulation:
    margin    = clip(kappa_r*s_bar, 0, Delta_r_max)
    alpha_k   = alpha_0*(1 + gamma*s_bar)
    sigma_k   = clip(sigma_0/(1 + kappa_sigma*s_bar), sigma_min, sigma_max)
    beta_k    = clip(beta_0*(1 + kappa_beta*s_bar), beta_min, beta_max)

  MPPI weighting convention:
    w_i = exp(-Z_i/beta_k) / sum_j exp(-Z_j/beta_k)

Thus larger beta produces softer importance weights. In RC-MPPI, beta
increases under residual mismatch to reduce overcommitment to unreliable
nominal rollout costs, while obstacle tightening and adaptive barrier
scaling increase safety conservatism.
"""

import time
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# 0) Device
# ─────────────────────────────────────────────────────────────
GPU   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU   = torch.device("cpu")
DTYPE = torch.float32

# ─────────────────────────────────────────────────────────────
# 1) Nominal planning dynamics [eq:lti, paper Sec. VII-A]
#    x = [px, py, vx, vy],  dt = 0.1 s
# ─────────────────────────────────────────────────────────────
dt = 0.1

A_gpu = torch.tensor(
    [[1, 0, dt, 0 ],
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]],
    device=GPU, dtype=DTYPE)
B_gpu = torch.tensor(
    [[0,  0 ],
     [0,  0 ],
     [dt, 0 ],
     [0,  dt]],
    device=GPU, dtype=DTYPE)
A_cpu = A_gpu.to(CPU)
B_cpu = B_gpu.to(CPU)

# ─────────────────────────────────────────────────────────────
# 2) Task geometry [paper Sec. VII-A]
# ─────────────────────────────────────────────────────────────
goal_gpu        = torch.tensor([5.0, 0.0], device=GPU, dtype=DTYPE)
obs_center_true = torch.tensor([2.5, 0.0], device=GPU, dtype=DTYPE)
obs_radius      = torch.tensor(1.5,        device=GPU, dtype=DTYPE)

# ─────────────────────────────────────────────────────────────
# 3) Cost weights [paper Sec. VII-A, eq. cost function]
# ─────────────────────────────────────────────────────────────
W_GOAL    = 5.0
W_VEL     = 0.1
W_CTRL    = 0.01
W_TERM    = 50.0
W_OBS_BASE = 1e4     # alpha_0 in eq:penalty_scaling
GAMMA_OBS  = 1.0     # gamma  in eq:penalty_scaling
U_CLIP    = 4.0

# ─────────────────────────────────────────────────────────────
# 4) Plant: first-order servo-lag [paper Sec. VII-A, eq. plant]
#    tau = 0.9 s  =>  alpha = 1 - exp(-0.1/0.9) ~ 0.105
# ─────────────────────────────────────────────────────────────
SERVO_TAU   = 0.9                                      # [s] paper Sec. VII-A
alpha_servo = 1.0 - float(np.exp(-dt / SERVO_TAU))    # ~0.105

def step_plant_cpu(x_cpu: torch.Tensor,
                   u_cpu: torch.Tensor,
                   v_exec: torch.Tensor):
    """
    First-order servo-lag plant step (CPU). Paper plant model:
      v_ref_{k+1}  = v_k + dt * u_k
      v_{k+1}      = (1 - alpha) * v_k + alpha * v_ref_{k+1}
      p_{k+1}      = p_k + dt * v_{k+1}
    Returns: (x_next, v_exec_next)
    """
    p     = x_cpu[:2]
    v     = x_cpu[2:]
    u_sat = torch.clamp(u_cpu, -U_CLIP, U_CLIP)
    v_ref = v + dt * u_sat
    v_exec_next = (1.0 - alpha_servo) * v_exec + alpha_servo * v_ref
    p_next = p + dt * v_exec_next
    return torch.cat([p_next, v_exec_next]), v_exec_next

# ─────────────────────────────────────────────────────────────
# 5) Residual estimation [eq:residual_def, eqn:fil_statistics]
# ─────────────────────────────────────────────────────────────
# Residual score weights [paper Sec. VII-A: wp=1.0, wv'=0.5]
E_POS_W = 1.0
E_VEL_W = 0.5

# Exponential filter [paper Sec. VII-A: rho=0.2]
RHO = 0.20

def compute_residual(y_k: torch.Tensor,
                     y_prev: torch.Tensor,
                     u_prev: torch.Tensor) -> torch.Tensor:
    """
    One-step prediction residual [eq:residual_def]:
      r_k = y_k - f_theta(y_{k-1}, u_{k-1})
    where f_theta is the nominal LTI predictor.
    Sign convention: measurement minus prediction (matches paper).
    """
    x_pred = A_cpu @ y_prev + B_cpu @ u_prev
    return y_k - x_pred

def mismatch_score(r_cpu: torch.Tensor) -> float:
    """
    Scalar residual indicator s_k [paper Sec. VII-A]:
      s_k = sqrt(wp*||r_pos||^2 + wv'*||r_vel||^2)
    """
    s2 = E_POS_W * torch.sum(r_cpu[:2]**2) \
       + E_VEL_W * torch.sum(r_cpu[2:]**2)
    return float(torch.sqrt(s2).item())

# ─────────────────────────────────────────────────────────────
# 6) RC-MPPI modulation [eq:tightening_def, eq:penalty_scaling,
#                        eq:sampling_modulation]
# ─────────────────────────────────────────────────────────────
# Modulation gains [paper Sec. VII-A]
KAPPA_R    = 1.0    # obstacle radius inflation gain  (kappa_r)
DELTA_R_MAX = 1.0    # max radius inflation (m)        (Delta_r_max)

KAPPA_SIGMA = 0.50   # sigma contraction gain          (kappa_sigma)
SIGMA_MIN   = 0.10
SIGMA_MAX   = 2.50

# beta INCREASES with s_bar [eq:sampling_modulation, Proposition 4]
# beta_k = clip(beta0*(1 + kappa_beta*s_bar), beta_min, beta_max)
# Higher mismatch -> larger beta -> softer weights -> conservative averaging
# Safety maintained: alpha_k*phi(m(s_bar)) ~ O(s_bar^2) >> beta_k ~ O(s_bar)
KAPPA_BETA = 5.00    # temperature relaxation gain     (kappa_beta)
BETA_MIN   = 0.10
BETA_MAX   = 5.00

def modulate(s_bar: float, sigma0: float, beta0: float):
    """
    Compute residual-adaptive modulation parameters.

    Returns: (margin, sigma_eff, beta_eff, alpha_k)

    margin    = clip(kappa_r * s_bar, 0, Delta_r_max)
      Simulation form of eq:tightening_def.

    sigma_eff = clip(sigma0 / (1 + kappa_sigma * s_bar), sigma_min, sigma_max)
      Perturbation std DECREASES: contracts exploration [eq:sampling_modulation].

    beta_eff  = clip(beta0 * (1 + kappa_beta * s_bar), beta_min, beta_max)
      Temperature INCREASES: reflects reduced confidence in cost evaluations
      under model mismatch [eq:sampling_modulation, Proposition 4].
      Epistemic interpretation: higher s_bar -> costs are less trustworthy
      -> raise beta -> average more broadly over the pre-conditioned safe
      rollout distribution.
      Safety proof: alpha_k*phi(m) ~ O(s_bar^2), beta_k ~ O(s_bar), so
      the unsafe weight ratio exp(-alpha_k*phi(m)/beta_k) -> 0 [Lemma 2(ii)].

    alpha_k   = W_OBS_BASE * (1 + GAMMA_OBS * s_bar)
      Barrier penalty INCREASES with s_bar [eq:penalty_scaling].
      Quadratic growth of alpha_k*phi(m(s_bar)) guarantees unsafe rollouts
      get suppressed even under rising temperature [Proposition 3].
    """
    # Constraint tightening (radius inflation)
    margin = float(np.clip(KAPPA_R * s_bar, 0.0, DELTA_R_MAX))

    # Perturbation variance: contracts under mismatch [eq:sampling_modulation]
    sigma_eff = float(np.clip(
        sigma0 / (1.0 + KAPPA_SIGMA * s_bar), SIGMA_MIN, SIGMA_MAX))

    # Temperature: INCREASES (softens) under mismatch [eq:sampling_modulation]
    beta_eff = float(np.clip(
        beta0 * (1.0 + KAPPA_BETA * s_bar), BETA_MIN, BETA_MAX))

    # Barrier penalty scaling [eq:penalty_scaling]
    alpha_k = W_OBS_BASE * (1.0 + GAMMA_OBS * s_bar)

    return margin, sigma_eff, beta_eff, alpha_k

# ─────────────────────────────────────────────────────────────
# 7) Cost functions [paper Sec. VII-A, eq:barrier_cost]
# ─────────────────────────────────────────────────────────────
def running_cost_batch(x_k4: torch.Tensor,
                       u_k2: torch.Tensor,
                       radius_eff: torch.Tensor,
                       alpha_k: float) -> torch.Tensor:
    """
    Vectorized running cost for K rollouts at one step. Returns (K,).

    ell_safe(x; s_bar) = alpha_k * phi(h(x) + m(s_bar))   [eq:barrier_cost]
    where h(x) = r - d(x), phi(z) = max(0,z)^2,
    and r_eff = r + m(s_bar) absorbs the tightening.
    alpha_k from eq:penalty_scaling.
    """
    pos = x_k4[:, :2]
    vel = x_k4[:, 2:]

    goal_cost = W_GOAL * torch.sum((pos - goal_gpu[None,:])**2, dim=1)
    vel_cost  = W_VEL  * torch.sum(vel**2, dim=1)
    ctrl_cost = W_CTRL * torch.sum(u_k2**2, dim=1)

    d        = torch.linalg.norm(pos - obs_center_true[None,:], dim=1)
    obs_cost = alpha_k * torch.clamp(radius_eff - d, min=0.0)**2

    return goal_cost + vel_cost + ctrl_cost + obs_cost

def terminal_cost_batch(x_k4: torch.Tensor) -> torch.Tensor:
    pos = x_k4[:, :2]
    return W_TERM * torch.sum((pos - goal_gpu[None,:])**2, dim=1)

# ─────────────────────────────────────────────────────────────
# 8) MPPI core (GPU) [eq:mppi_weights, eq:mppi_update]
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def mppi_step(x0_gpu:     torch.Tensor,
              u_bar:      torch.Tensor,
              radius_eff: torch.Tensor,
              alpha_k:    float,
              K:          int,
              T:          int,
              sigma:      float,
              beta:       float):
    """
    One MPPI update step [eq:mppi_weights, eq:mppi_update].

    Weights: w^(i) = exp(-Z^(i) / beta) / sum_j exp(-Z^(j) / beta)

    beta is the temperature from eq:sampling_modulation.
    Larger beta -> softer, more uniform weights.
    Smaller beta -> sharper weights committed to apparent minimum.

    Note: cost -= cost.min() is standard numerical stabilization;
    it does not change the weights (shifts partition function only).
    """
    eps  = sigma * torch.randn(K, T, 2, device=GPU, dtype=DTYPE)
    x    = x0_gpu[None,:].expand(K,-1).clone()
    cost = torch.zeros(K, device=GPU, dtype=DTYPE)

    for t in range(T):
        u     = u_bar[t][None,:] + eps[:,t,:]
        x     = x @ A_gpu.T + u @ B_gpu.T
        cost += running_cost_batch(x, u, radius_eff, alpha_k)
    cost += terminal_cost_batch(x)

    cost -= cost.min()   # numerical stabilization (invariant to weights)
    w     = torch.exp(-cost / beta)
    w    /= w.sum() + 1e-12

    du       = torch.sum(w[:,None,None] * eps, dim=0)   # (T,2)
    u_bar_new = torch.clamp(u_bar + du, -U_CLIP, U_CLIP)
    u0       = u_bar_new[0]

    # Receding-horizon shift
    u_bar_shifted = torch.cat(
        [u_bar_new[1:], torch.zeros(1, 2, device=GPU, dtype=DTYPE)], dim=0)
    return u0, u_bar_shifted

# ─────────────────────────────────────────────────────────────
# 9) Metric utilities
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _c() -> np.ndarray:
    return obs_center_true.cpu().numpy()

def _r() -> float:
    return float(obs_radius.cpu().item())

def clearance_series(pos_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pos_xy - _c()[None,:], axis=1) - _r()

def clearance_true(pos_xy: np.ndarray) -> float:
    return float(np.min(clearance_series(pos_xy)))

def violation_steps(pos_xy: np.ndarray) -> int:
    """Number of timesteps with negative clearance (paper metric)."""
    return int(np.sum(clearance_series(pos_xy) < 0.0))

def max_penetration(pos_xy: np.ndarray) -> float:
    return float(np.max(np.maximum(0.0, _r() -
        np.linalg.norm(pos_xy - _c()[None,:], axis=1))))

def time_to_goal(pos_xy: np.ndarray, tol: float) -> int:
    dist = np.linalg.norm(pos_xy - goal_gpu.cpu().numpy()[None,:], axis=1)
    hit  = np.where(dist <= tol)[0]
    return int(hit[0]) if hit.size > 0 else -1

def path_length(pos_xy: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pos_xy, axis=0), axis=1)))

# ─────────────────────────────────────────────────────────────
# 10) Single simulation rollout
# ─────────────────────────────────────────────────────────────
def simulate(steps:    int   = 300,
             K:        int   = 8192,
             T:        int   = 40,
             sigma0:   float = 1.0,
             beta0:    float = 1.0,
             goal_tol: float = 0.25,
             adaptive: bool  = False,
             log:      bool  = False):
    """
    Run one closed-loop trial.
    Success = goal reached AND zero obstacle violations (paper definition).
    """
    x      = torch.zeros(4, device=CPU, dtype=DTYPE)
    v_exec = torch.zeros(2, device=CPU, dtype=DTYPE)  # executed velocity (lag state)
    y_prev = x.clone()
    u_prev = torch.zeros(2, device=CPU, dtype=DTYPE)
    u_bar  = torch.zeros(T, 2, device=GPU, dtype=DTYPE)

    pos_log = [x[:2].cpu().numpy().copy()]
    s_bar   = 0.0
    dbg = ({k: [] for k in ["s_bar","r_eff","alpha_k","sigma_eff","beta_eff"]}
           if log else None)

    for k in range(steps):
        # ── Residual estimation and modulation ────────────────────────────
        if adaptive and k > 0:
            r_k   = compute_residual(x, y_prev, u_prev)   # [eq:residual_def]
            s_k   = mismatch_score(r_k)
            s_bar = (1.0 - RHO) * s_bar + RHO * s_k      # [eqn:fil_statistics]
            margin, sigma_eff, beta_eff, alpha_k = modulate(s_bar, sigma0, beta0)
        else:
            # Vanilla MPPI: no adaptation, fixed parameters
            margin, sigma_eff, beta_eff, alpha_k = 0.0, sigma0, beta0, W_OBS_BASE

        radius_eff = obs_radius + torch.tensor(margin, device=GPU, dtype=DTYPE)

        # ── MPPI optimization (GPU) ────────────────────────────────────────
        u0, u_bar = mppi_step(
            x.to(GPU), u_bar, radius_eff, alpha_k,
            K=K, T=T, sigma=sigma_eff, beta=beta_eff)

        # ── Plant execution (CPU) ──────────────────────────────────────────
        y_prev = x.clone()
        u_prev = u0.to(CPU).clone()
        x, v_exec = step_plant_cpu(x, u_prev, v_exec)

        pos_log.append(x[:2].cpu().numpy().copy())
        if log:
            dbg["s_bar"].append(s_bar)
            dbg["r_eff"].append(float(radius_eff.cpu()))
            dbg["alpha_k"].append(float(alpha_k))
            dbg["sigma_eff"].append(sigma_eff)
            dbg["beta_eff"].append(beta_eff)

    pos    = np.asarray(pos_log)
    t_goal = time_to_goal(pos, goal_tol)
    v_steps = violation_steps(pos)   # paper metric: steps with neg clearance

    metrics = dict(
        success         = 1 if (t_goal >= 0 and v_steps == 0) else 0,
        t_goal          = t_goal if t_goal >= 0 else steps + 1,
        min_clearance   = clearance_true(pos),
        viols           = v_steps,
        max_penetration = max_penetration(pos),
        path_len        = path_length(pos),
    )
    if log:
        dbg = {k: np.array(v) for k, v in dbg.items()}
    return pos, metrics, dbg

# ─────────────────────────────────────────────────────────────
# 11) Paired Monte Carlo
# ─────────────────────────────────────────────────────────────
def mc(n_trials: int = 50, base_seed: int = 0, **sim_kwargs):
    """
    Paired-seed MC: vanilla and RC-MPPI share identical noise realizations
    so differences are attributable to adaptation only (paper protocol).
    """
    res   = {"vanilla": [], "adaptive": []}
    seeds = []
    for i in tqdm(range(n_trials), desc="MC trials", ncols=80):
        seed = base_seed + i
        seeds.append(seed)
        set_seed(seed); _, mv, _ = simulate(adaptive=False, **sim_kwargs)
        set_seed(seed); _, ma, _ = simulate(adaptive=True,  **sim_kwargs)
        res["vanilla"].append(mv)
        res["adaptive"].append(ma)
    return res, np.asarray(seeds, dtype=int)

def summarize(res: dict):
    """Print Monte Carlo summary statistics."""
    def arr(m, k): return np.asarray([d[k] for d in res[m]], dtype=float)

    hdr = f"{'Metric':<26} {'Vanilla MPPI':>20} {'RC-MPPI':>20}"
    print("\n" + "="*len(hdr))
    print(hdr)
    print("="*len(hdr))
    rows = [
        ("Success rate",         "success",         False),
        ("Time-to-goal (steps)", "t_goal",          True),
        ("Min clearance (m)",    "min_clearance",   True),
        ("Violation steps",      "viols",           True),
        ("Path length (m)",      "path_len",        True),
    ]
    for label, key, with_std in rows:
        vv = arr("vanilla", key)
        av = arr("adaptive", key)
        if with_std:
            vs  = f"{np.mean(vv):.2f} ± {np.std(vv):.2f}"
            as_ = f"{np.mean(av):.2f} ± {np.std(av):.2f}"
        else:
            vs  = f"{np.mean(vv):.2f}"
            as_ = f"{np.mean(av):.2f}"
        print(f"{label:<26} {vs:>20} {as_:>20}")
    print("="*len(hdr))

# ─────────────────────────────────────────────────────────────
# 12) Representative seed selection
# ─────────────────────────────────────────────────────────────
def select_seed(res: dict, seeds: np.ndarray):
    """
    Select representative seed. Paper criterion (Sec. VII-A):
    prefer seed where vanilla violates and RC-MPPI maintains clearance.
    """
    v_viols = np.asarray([m["viols"]         for m in res["vanilla"]],  dtype=float)
    a_viols = np.asarray([m["viols"]         for m in res["adaptive"]], dtype=float)
    v_clr   = np.asarray([m["min_clearance"] for m in res["vanilla"]],  dtype=float)
    a_clr   = np.asarray([m["min_clearance"] for m in res["adaptive"]], dtype=float)
    a_succ  = np.asarray([m["success"]       for m in res["adaptive"]], dtype=float)
    delta   = a_clr - v_clr

    # Priority 1: vanilla violated, RC-MPPI clean and succeeded
    cand1 = np.where((v_viols > 0) & (a_viols == 0) & (a_succ > 0))[0]
    if cand1.size:
        idx = cand1[np.argmax(v_viols[cand1])]   # most vanilla violations
        return int(seeds[idx]), int(idx), \
               f"vanilla violated ({v_viols[idx]:.0f} steps), RC-MPPI clean"

    # Priority 2: vanilla near-miss (<0.05m), RC-MPPI wider clearance
    cand2 = np.where((a_succ > 0) & (v_clr < 0.05))[0]
    if cand2.size:
        idx = cand2[np.argmax(delta[cand2])]
        return int(seeds[idx]), int(idx), \
               f"vanilla near-miss, max clearance delta={delta[idx]:.3f}m"

    # Fallback
    idx = int(np.argmax(delta))
    return int(seeds[idx]), int(idx), \
           f"fallback: max clearance delta={delta[idx]:.3f}m"

# ─────────────────────────────────────────────────────────────
# 13) Output utilities
# ─────────────────────────────────────────────────────────────
def safe_float_tag(v): return f"{v:.2f}".replace(".", "p")

def make_output_dir(n_trials, K, T, steps):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"lti_tau_{safe_float_tag(SERVO_TAU)}"
            f"_n{n_trials}_K{K}_T{T}_steps{steps}_{ts}")
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=False)
    return out

def save_metrics_csv(res, seeds, out_dir):
    path = out_dir / "mc_trial_metrics.csv"
    keys = ["success","t_goal","min_clearance","viols","max_penetration","path_len"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed","method",*keys])
        for i, seed in enumerate(seeds):
            for method in ["vanilla","adaptive"]:
                w.writerow([int(seed), method,
                             *[res[method][i][k] for k in keys]])
    return path

def save_run_summary(res, seeds, seed_star, idx_star, reason, out_dir):
    def arr(m, k): return np.asarray([d[k] for d in res[m]], dtype=float)
    summary = {
        "settings": {
            "SERVO_TAU": SERVO_TAU,
            "alpha_servo": round(alpha_servo, 4),
            "dt": dt,
            "n_trials": int(len(seeds)),
            "beta_convention": "beta INCREASES with s_bar (softer under mismatch)",
            "representative_seed": int(seed_star),
            "representative_index": int(idx_star),
            "representative_reason": reason,
        },
        "aggregate": {}
    }
    for method in ["vanilla","adaptive"]:
        summary["aggregate"][method] = {
            k: {
                "mean": float(np.mean(arr(method, k))),
                "std":  float(np.std( arr(method, k))),
            }
            for k in ["success","t_goal","min_clearance","viols","path_len"]
        }
    path = out_dir / "run_summary.json"
    with path.open("w") as f:
        json.dump(summary, f, indent=2)
    return path

# ─────────────────────────────────────────────────────────────
# 14) Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device    : {GPU}")
    print(f"SERVO_TAU : {SERVO_TAU:.3f} s  (alpha = {alpha_servo:.4f})")
    print(f"beta conv.: exp(-cost/beta), larger beta = softer weights")
    print(f"Modulation: margin up, sigma down, beta UP (softer), alpha up")

    # ── Simulation settings [paper Sec. VII-A] ───────────────────────────
    K         = 8192 if GPU.type == "cuda" else 2048
    n_trials  = 50
    steps_mc  = 300   # paper: 300 steps per trial
    steps_rep = 300   # same for representative replay
    T_horizon = 40
    sigma0    = 1.0
    beta0     = 1.0   # paper nominal temperature; vanilla sharp, RC-MPPI softens upward
    goal_tol  = 0.25

    print(f"\nMC: n={n_trials}, K={K}, steps={steps_mc}, T={T_horizon}")
    print(f"    sigma0={sigma0}, beta0={beta0}, goal_tol={goal_tol}")
    print(f"    kappa_r={KAPPA_R}, kappa_sigma={KAPPA_SIGMA}, kappa_beta={KAPPA_BETA}")

    out_dir = make_output_dir(n_trials, K, T_horizon, steps_mc)
    print(f"\nOutput: {out_dir}")

    # ── Monte Carlo ──────────────────────────────────────────────────────
    t0 = time.time()
    res, seeds = mc(n_trials=n_trials, base_seed=0,
                    steps=steps_mc, K=K, T=T_horizon,
                    sigma0=sigma0, beta0=beta0, goal_tol=goal_tol)
    print(f"MC elapsed: {time.time()-t0:.1f}s")
    summarize(res)

    save_metrics_csv(res, seeds, out_dir)

    # ── Representative seed ──────────────────────────────────────────────
    seed_star, idx_star, reason = select_seed(res, seeds)
    print(f"\nRepresentative seed: {seed_star}  ({reason})")
    save_run_summary(res, seeds, seed_star, idx_star, reason, out_dir)

    set_seed(seed_star)
    traj_v, met_v, dbg_v = simulate(steps=steps_rep, K=K, T=T_horizon,
                                     sigma0=sigma0, beta0=beta0,
                                     goal_tol=goal_tol, adaptive=False, log=True)
    set_seed(seed_star)
    traj_a, met_a, dbg_a = simulate(steps=steps_rep, K=K, T=T_horizon,
                                     sigma0=sigma0, beta0=beta0,
                                     goal_tol=goal_tol, adaptive=True, log=True)

    print(f"\nRepresentative trial (seed={seed_star}):")
    for lab, met in [("Vanilla", met_v), ("RC-MPPI", met_a)]:
        print(f"  {lab}: clr={met['min_clearance']:.3f}m, "
              f"viols={met['viols']}, success={met['success']}, "
              f"TTG={met['t_goal']}")

    # ── Figure 1: Trajectory overlay (paper Fig. 1) ──────────────────────
    g = goal_gpu.cpu().numpy()
    def first_goal_step(traj, tol=0.5):
        d   = np.linalg.norm(traj - g[None,:], axis=1)
        hit = np.where(d <= tol)[0]
        return int(hit[0]) + 15 if hit.size else len(traj)

    clip = min(first_goal_step(traj_v), first_goal_step(traj_a), steps_rep)
    tv, ta = traj_v[:clip], traj_a[:clip]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.add_patch(plt.Circle(_c(), _r(), color="lightgray", zorder=1))
    ax.add_patch(plt.Circle(_c(), _r(), fill=False, edgecolor="dimgray",
                             linewidth=1.5, zorder=2))
    ax.text(_c()[0], _c()[1], "Obstacle", ha="center", va="center",
            fontsize=8, color="dimgray", zorder=3)
    ax.plot(tv[:,0], tv[:,1], "r--", lw=2, label="Vanilla MPPI", zorder=4)
    ax.plot(ta[:,0], ta[:,1], "b-",  lw=2, label="RC-MPPI",      zorder=4)
    ax.scatter(0, 0, s=60, color="black", marker="o", zorder=5, label="Start")
    ax.scatter(*g, s=100, color="green", marker="*", zorder=5, label="Goal")
    # Annotate vanilla minimum clearance point
    clr_v = clearance_series(tv)
    ci = int(np.argmin(clr_v))
    ax.scatter(tv[ci,0], tv[ci,1], s=80, color="red", marker="x",
               linewidths=2, zorder=6)
    off = (0.12, 0.12) if tv[ci,1] > 0 else (0.12, -0.18)
    ax.annotate(f"clr={clr_v[ci]:+.3f} m",
                xy=(tv[ci,0], tv[ci,1]),
                xytext=(tv[ci,0]+off[0], tv[ci,1]+off[1]),
                fontsize=7.5, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.0))
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectories — seed {seed_star}")
    fig.tight_layout()
    p = out_dir / f"fig1_trajectory_seed{seed_star}.pdf"
    fig.savefig(p, bbox_inches="tight"); print(f"Saved: {p}")

    # ── Figure 2: Clearance vs time ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(clearance_series(traj_v), "r--", lw=2, label="Vanilla MPPI")
    ax.plot(clearance_series(traj_a), "b-",  lw=2, label="RC-MPPI")
    ax.axhline(0.0, color="k", lw=1, ls=":")
    ax.set_xlabel("Step"); ax.set_ylabel("Clearance (m)")
    ax.set_title(f"Clearance vs Time — seed {seed_star}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / f"fig2_clearance_seed{seed_star}.pdf"
    fig.savefig(p, bbox_inches="tight"); print(f"Saved: {p}")

    # ── Figure 3: Adaptive parameter profiles ────────────────────────────
    if dbg_a:
        fig, axes = plt.subplots(4, 1, figsize=(6,7), sharex=True)
        axes[0].plot(dbg_a["s_bar"],    "g-", lw=2)
        axes[0].set_ylabel(r"Residual $\bar{s}_k$")
        axes[1].plot(dbg_a["r_eff"],    "b-", lw=2)
        axes[1].axhline(float(obs_radius), color="k", ls="--", lw=1)
        axes[1].set_ylabel(r"$r_\mathrm{eff}$ (m)")
        axes[2].plot(dbg_a["sigma_eff"],"m-", lw=2)
        axes[2].axhline(sigma0, color="k", ls="--", lw=1, label=r"$\sigma_0$")
        axes[2].set_ylabel(r"$\sigma_k$ (contracts $\downarrow$)")
        axes[2].legend(fontsize=8)
        # beta INCREASES — axis label reflects this
        axes[3].plot(dbg_a["beta_eff"], "c-", lw=2)
        axes[3].axhline(beta0, color="k", ls="--", lw=1, label=r"$\beta_0$")
        axes[3].set_ylabel(r"$\beta_k$ (relaxes $\uparrow$)")
        axes[3].set_xlabel("Step")
        axes[3].legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / f"fig3_modulation_seed{seed_star}.pdf"
        fig.savefig(p, bbox_inches="tight"); print(f"Saved: {p}")

    # ── Figure 4: Paired MC scatter ──────────────────────────────────────
    def arr(m, k): return np.asarray([d[k] for d in res[m]], dtype=float)
    xsc = arr("vanilla",  "min_clearance")
    ysc = arr("adaptive", "min_clearance")
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.scatter(xsc, ysc, alpha=0.7, s=40)
    ax.scatter(xsc[idx_star], ysc[idx_star], s=140, marker="x",
               linewidths=2.5, color="red", zorder=5, label="Rep. seed")
    lim = [min(xsc.min(), ysc.min())-0.02, max(xsc.max(), ysc.max())+0.02]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Vanilla MPPI min clearance (m)")
    ax.set_ylabel("RC-MPPI min clearance (m)")
    ax.set_title("Paired MC: Min Clearance")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "fig4_paired_clearance_scatter.pdf"
    fig.savefig(p, bbox_inches="tight"); print(f"Saved: {p}")

    print("\nDone.")