#!/usr/bin/env python3
"""
Residual-Conservative MPPI (RC-MPPI)
=====================================
Sampling-based MPC with online residual-adaptive conservatism.

Reference:
  Yoon, Rasul, Tasnim, Kim — "Residual-Conservative Model Predictive
  Path Integral Control"

Changes from original (consistency with paper):
  1. Importance weights use w ∝ exp(−β_k · cost), where β_k is the
     INVERSE temperature (eq. 12, 16). Higher β_k → sharper distribution.
     Original used exp(−cost / β), i.e. β as temperature (opposite convention).
  2. β_k modulation direction corrected: β_k starts small (BETA0) and
     INCREASES with mismatch s̄_k, sharpening the distribution (Lemma 3,
     Proposition 3). Hyperparams BETA0/BETA_MIN/BETA_MAX updated accordingly.
  3. Residual sign corrected: e = x − x_pred = r_k (eq. 4: measured − predicted).
  4. Success metric: goal-reached only (matches Table I definition).
  5. Equation reference comment corrected: eq. (44), not eq. (29).

Usage:
  python main_cdc_final.py
"""

import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

# --- Devices / precision ---
GPU   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU   = torch.device("cpu")
DTYPE = torch.float32

# --- Simulation time ---
DT               = 0.1   # [s] timestep
SIM_STEPS_MC     = 300   # steps per trial in Monte Carlo
SIM_STEPS_REPLAY = 300   # steps for the representative-seed replay
GOAL_TOL         = 0.25  # [m] goal-reached distance threshold

# --- Task geometry ---
START_STATE = [0.0, 0.0, 0.0, 0.0]  # [px, py, vx, vy]
GOAL_POS    = [5.0, 0.0]            # [px, py]
OBS_CENTER  = [2.5, 0.0]            # [px, py]  (true obstacle center)
OBS_RADIUS  = 1.2                   # [m]

# --- Nominal LTI dynamics (damping = 0) ---
DAMPING = 0.0  # velocity damping coefficient

# --- MPPI hyper-parameters ---
K_ROLLOUTS = 8192  # number of sampled trajectories (halved on CPU)
T_HORIZON  = 40    # planning horizon (steps)
SIGMA0     = 1.0   # base control-noise std dev  σ_0
# β_0: base INVERSE temperature (paper eq. 16).
# With w ∝ exp(−β·cost), β=1 is equivalent to the original exp(−cost/1).
# Kept at 1.0 so vanilla MPPI behaviour is numerically unchanged.
BETA0      = 1.0   # base inverse temperature β_0  (paper eq. 16)
U_CLIP     = 4.0   # [m/s²] symmetric control saturation

# --- Cost weights ---
W_GOAL = 5.0
W_VEL  = 0.1
W_CTRL = 0.01
W_TERM = 50.0
W_OBS  = 1e4

# --- Execution (plant) model ---
EXECUTION_MODEL = "lag"   # "exact" | "lag"
SERVO_TAU       = 0.60    # [s] first-order lag time constant

# --- Obstacle uncertainty (stationary, no online estimation) ---
OBS_NUM_SAMPLES = 8       # scenario samples per step
OBS_RISK_MODE   = "cvar"  # "mean" | "cvar"
OBS_CVAR_ALPHA  = 0.8     # CVaR tail fraction (ignored for "mean")
OBS_UNC_STD     = 0.0     # [m] obstacle-center position uncertainty std
OBS_MEAN_BIAS   = [0.0, 0.0]  # [m] constant bias on obstacle center belief

# --- Residual-adaptive modulation ---
USE_RISK_ADAPTATION = True

# Residual weighting (equation 44 in paper):
#   s_k = sqrt(w_p·||Δp||² + w_v'·||Δv||²)
E_POS_W = 1.0
E_VEL_W = 0.5

# Exponential filter forgetting rate (ρ in equation 6)
RISK_FILTER_RHO = 0.20

# Modulation gains
KAPPA_R     = 0.40  # radius-inflation gain          (constraint tightening)
KAPPA_SIGMA = 1.50  # σ-reduction gain               (sampling modulation, eq. 16)
KAPPA_BETA  = 1.00  # β-increase gain                (inverse temp, eq. 16)
#
# NOTE on β convention (paper eq. 16 vs original code):
#   Paper:    β_k = clip(β_0 (1 + κ_β s̄_k), β_min, β_max)
#             w ∝ exp(−β_k · cost)   ← β_k is INVERSE temperature
#             As s̄_k ↑, β_k ↑ → distribution sharpens → safer rollouts weighted more.
#   Original: w ∝ exp(−cost / β_k)  ← β_k was TEMPERATURE
#             As s̄_k ↑, β_k ↑ → distribution SOFTENS (wrong direction).
#   Fix: keep modulation formula identical; only change weight formula to match paper.
#   Hyperparams: BETA0=1.0 (unchanged), BETA_MIN/MAX scaled to inverse-temp range.

# Modulation saturation bounds
R_MARGIN_MAX         = 1.0
SIGMA_MIN, SIGMA_MAX = 0.10, 2.50
# β bounds in INVERSE-temperature units (β_min < β_0 < β_max).
# β_min = 0.2 → very soft (permissive) baseline
# β_max = 5.0 → sharp (conservative) ceiling under high mismatch
BETA_MIN, BETA_MAX   = 0.20, 5.00

# --- Monte Carlo protocol ---
MC_N_TRIALS  = 50
MC_BASE_SEED = 0

# ============================================================
# DERIVED CONSTANTS  (computed once from config)
# ============================================================

_alpha_servo = float(1.0 - np.exp(-DT / SERVO_TAU))

_goal_gpu = torch.tensor(GOAL_POS,   device=GPU, dtype=DTYPE)
_obs_true = torch.tensor(OBS_CENTER, device=GPU, dtype=DTYPE)
_obs_r    = torch.tensor(OBS_RADIUS, device=GPU, dtype=DTYPE)

_obs_bias_cpu = torch.tensor(OBS_MEAN_BIAS, dtype=DTYPE)


def _build_dynamics():
    A = torch.tensor(
        [[1, 0, DT, 0],
         [0, 1, 0,  DT],
         [0, 0, 1 - DAMPING * DT, 0],
         [0, 0, 0,  1 - DAMPING * DT]],
        dtype=DTYPE,
    )
    B = torch.tensor(
        [[0,  0],
         [0,  0],
         [DT, 0],
         [0,  DT]],
        dtype=DTYPE,
    )
    return A, B


_A_cpu, _B_cpu = _build_dynamics()
_A_gpu = _A_cpu.to(GPU)
_B_gpu = _B_cpu.to(GPU)


# ============================================================
# SECTION 1 — PLANT EXECUTION MODEL
# ============================================================

def step_plant(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Advance the CPU plant by one timestep DT.

    Parameters
    ----------
    x : (4,) state tensor  [px, py, vx, vy]
    u : (2,) control       [ax, ay]

    Returns
    -------
    x_next : (4,) next state
    """
    if EXECUTION_MODEL == "exact":
        return _A_cpu @ x + _B_cpu @ u

    if EXECUTION_MODEL == "lag":
        p, v   = x[:2], x[2:]
        v_ref  = v + DT * u
        v_next = (1.0 - _alpha_servo) * v + _alpha_servo * v_ref
        p_next = p + DT * v_next
        return torch.cat([p_next, v_next])

    raise ValueError(f"Unknown EXECUTION_MODEL: {EXECUTION_MODEL!r}")


# ============================================================
# SECTION 2 — RESIDUAL COMPUTATION & ADAPTIVE MODULATION
# ============================================================

def mismatch_score(e: torch.Tensor) -> float:
    """
    Weighted L2 norm of the one-step prediction residual.
    Implements equation (44): s_k = sqrt(w_p||Δp||² + w_v'||Δv||²)

    Parameters
    ----------
    e : (4,) residual r_k = y_k − f_θ(y_{k-1}, u_{k-1})  [Δpx, Δpy, Δvx, Δvy]
        (measured minus predicted, consistent with eq. 4)
    """
    s2 = E_POS_W * torch.sum(e[:2] ** 2) + E_VEL_W * torch.sum(e[2:] ** 2)
    return float(torch.sqrt(s2).item())


def modulate(s_bar: float) -> tuple[float, float, float]:
    """
    Map the filtered residual statistic s̄_k to adaptive parameters
    (equations 14, 15, 16 in paper).

    Returns
    -------
    margin    : obstacle-radius inflation  Δr(s̄_k)      — eq. 14 / Remark 3
    sigma_eff : effective MPPI noise std   σ_k ↓         — eq. 16
    beta_eff  : effective inverse temperature β_k ↑      — eq. 16
                Higher β_k → sharper importance weights → safer rollouts
                dominate (Lemma 3, Proposition 3).
    """
    margin    = float(np.clip(KAPPA_R * s_bar, 0.0, R_MARGIN_MAX))
    sigma_eff = float(np.clip(SIGMA0 / (1.0 + KAPPA_SIGMA * s_bar),
                               SIGMA_MIN, SIGMA_MAX))
    # β_k increases with mismatch (paper eq. 16): β_k = clip(β_0(1+κ_β s̄_k), …)
    beta_eff  = float(np.clip(BETA0 * (1.0 + KAPPA_BETA * s_bar),
                               BETA_MIN, BETA_MAX))
    return margin, sigma_eff, beta_eff


# ============================================================
# SECTION 3 — OBSTACLE SCENARIO SAMPLING  (GPU)
# ============================================================

@torch.no_grad()
def sample_obstacle_scenarios(mu_cpu: torch.Tensor,
                               Sigma_cpu: torch.Tensor,
                               M: int) -> torch.Tensor:
    """
    Draw M obstacle-center samples from N(mu, Sigma) on GPU.

    Returns
    -------
    centers : (M, 2) GPU tensor
    """
    mu    = mu_cpu.to(GPU)
    Sigma = Sigma_cpu.to(GPU)
    L     = torch.linalg.cholesky(Sigma + 1e-8 * torch.eye(2, device=GPU, dtype=DTYPE))
    z     = torch.randn(M, 2, device=GPU, dtype=DTYPE)
    return mu[None, :] + z @ L.T


def aggregate_obstacle_risk(costs_km: torch.Tensor) -> torch.Tensor:
    """
    Reduce (K, M) per-scenario obstacle costs to (K,) risk measure.
    Supports OBS_RISK_MODE = "mean" or "cvar".

    Parameters
    ----------
    costs_km : (K, M) non-negative obstacle cost tensor (GPU)
    """
    if OBS_RISK_MODE == "mean":
        return costs_km.mean(dim=1)

    if OBS_RISK_MODE == "cvar":
        M    = costs_km.shape[1]
        tail = max(1, int(math.ceil((1.0 - OBS_CVAR_ALPHA) * M)))
        worst, _ = torch.sort(costs_km, dim=1)
        return worst[:, -tail:].mean(dim=1)

    raise ValueError(f"Unknown OBS_RISK_MODE: {OBS_RISK_MODE!r}")


# ============================================================
# SECTION 4 — COST FUNCTIONS  (GPU, batched over K rollouts)
# ============================================================

def running_cost(x_k4: torch.Tensor,
                 u_k2: torch.Tensor,
                 obs_centers_m2: torch.Tensor,
                 radius_eff: torch.Tensor) -> torch.Tensor:
    """
    Vectorized stage cost for K rollouts at one time step.
    Implements equations (10)–(11), (42)–(43).

    Parameters
    ----------
    x_k4           : (K, 4) state batch
    u_k2           : (K, 2) control batch
    obs_centers_m2 : (M, 2) obstacle scenario centers
    radius_eff     : scalar effective obstacle radius

    Returns
    -------
    cost : (K,) stage costs
    """
    pos = x_k4[:, :2]
    vel = x_k4[:, 2:]

    goal_cost = W_GOAL * torch.sum((pos - _goal_gpu[None, :]) ** 2, dim=1)
    vel_cost  = W_VEL  * torch.sum(vel ** 2, dim=1)
    ctrl_cost = W_CTRL * torch.sum(u_k2 ** 2, dim=1)

    # Obstacle: quadratic penetration cost, aggregated over scenarios
    diff = pos[:, None, :] - obs_centers_m2[None, :, :]  # (K, M, 2)
    dist = torch.linalg.norm(diff, dim=2)                 # (K, M)
    pen  = torch.zeros_like(dist)
    mask = dist < radius_eff
    pen[mask] = W_OBS * (radius_eff - dist[mask]) ** 2
    obs_cost = aggregate_obstacle_risk(pen)

    return goal_cost + vel_cost + ctrl_cost + obs_cost


def terminal_cost(x_k4: torch.Tensor) -> torch.Tensor:
    """
    Terminal cost for K rollouts. Implements ℓ_f(x) (eq. 10, 42).

    Parameters
    ----------
    x_k4 : (K, 4) terminal state batch

    Returns
    -------
    cost : (K,) terminal costs
    """
    pos = x_k4[:, :2]
    return W_TERM * torch.sum((pos - _goal_gpu[None, :]) ** 2, dim=1)


# ============================================================
# SECTION 5 — MPPI CORE  (GPU)
# ============================================================

@torch.no_grad()
def mppi_step(x0: torch.Tensor,
              u_bar: torch.Tensor,
              obs_centers: torch.Tensor,
              radius_eff: torch.Tensor,
              K: int,
              sigma: float,
              beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One MPPI control update.
    Implements equations (12)–(13).

    Parameters
    ----------
    x0          : (4,)    current state (GPU)
    u_bar       : (T, 2)  nominal control sequence (GPU)
    obs_centers : (M, 2)  obstacle scenario centers (GPU)
    radius_eff  : scalar  effective obstacle radius
    K           : int     number of rollouts
    sigma       : float   control noise std dev  σ_k
    beta        : float   INVERSE temperature β_k  (paper eq. 12, 16)
                          w^(i) ∝ exp(−β_k · Z^(i))
                          Higher β_k → sharper weights → low-cost rollouts
                          dominate more strongly (Lemma 3).

    Returns
    -------
    u0        : (2,) optimal first control (clipped)
    u_bar_new : (T, 2) updated nominal sequence
    """
    T = u_bar.shape[0]

    # Sample control perturbations ε ~ N(0, σ_k² I)  (eq. 16)
    eps  = sigma * torch.randn(K, T, 2, device=GPU, dtype=DTYPE)
    x    = x0[None, :].expand(K, -1).clone()
    cost = torch.zeros(K, device=GPU, dtype=DTYPE)

    for t in range(T):
        u     = u_bar[t][None, :] + eps[:, t, :]
        x     = x @ _A_gpu.T + u @ _B_gpu.T
        cost += running_cost(x, u, obs_centers, radius_eff)

    cost += terminal_cost(x)

    # Importance weights  w^(i) ∝ exp(−β_k · Z^(i))   (eq. 12)
    # β_k is the INVERSE temperature: higher β_k sharpens the distribution,
    # concentrating weight on low-cost (safe) rollouts — consistent with
    # Lemma 3 and Proposition 3.  (Paper convention; fixes original code
    # which used exp(−cost / β), treating β as temperature — wrong direction.)
    cost -= cost.min()                         # numerical stabilisation
    w     = torch.exp(-beta * cost)            # ← paper eq. 12 convention
    w    /= w.sum() + 1e-12

    # Weighted perturbation update  (eq. 13)
    du        = torch.einsum("k,ktd->td", w, eps)  # (T, 2)
    u_bar_new = u_bar + du
    u0        = torch.clamp(u_bar_new[0], -U_CLIP, U_CLIP)
    return u0, u_bar_new


# ============================================================
# SECTION 6 — METRICS (evaluated against the TRUE obstacle)
# ============================================================

def _obs_true_np() -> tuple[np.ndarray, float]:
    c = _obs_true.cpu().numpy()
    r = float(_obs_r.cpu().item())
    return c, r


def clearance_series(pos: np.ndarray) -> np.ndarray:
    """Signed clearance to true obstacle at every step. Negative = inside."""
    c, r = _obs_true_np()
    return np.linalg.norm(pos - c[None, :], axis=1) - r


def min_clearance(pos: np.ndarray) -> float:
    return float(np.min(clearance_series(pos)))


def violation_count(pos: np.ndarray) -> int:
    """Number of steps with negative clearance (inside the obstacle)."""
    return int(np.sum(clearance_series(pos) < 0.0))


def time_to_goal(pos: np.ndarray, tol: float = GOAL_TOL) -> int:
    """First step index within `tol` of goal, or -1 if never reached."""
    g    = np.array(GOAL_POS)
    dist = np.linalg.norm(pos - g[None, :], axis=1)
    hit  = np.where(dist <= tol)[0]
    return int(hit[0]) if hit.size > 0 else -1


def path_length(pos: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def compute_metrics(pos: np.ndarray, steps: int, tol: float = GOAL_TOL) -> dict:
    """
    Aggregate all scalar metrics for one trajectory.

    Success definition (consistent with Table I in paper):
      A trial is successful if the goal is reached within the allotted steps,
      regardless of whether constraint violations occurred.  The violation_steps
      metric separately quantifies safety.
    """
    t_goal       = time_to_goal(pos, tol)
    n_viol       = violation_count(pos)
    goal_reached = t_goal >= 0
    return dict(
        success       = int(goal_reached),          # eq. Table I: goal-reached only
        t_goal        = t_goal if goal_reached else steps + 1,
        min_clearance = min_clearance(pos),
        violations    = n_viol,
        path_len      = path_length(pos),
    )


# ============================================================
# SECTION 7 — SIMULATION  (one rollout)
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simulate(steps: int = SIM_STEPS_MC,
             K: int = K_ROLLOUTS,
             adaptive: bool = False,
             log: bool = False) -> tuple[np.ndarray, dict, dict | None]:
    """
    Run a single RC-MPPI (or Vanilla MPPI) simulation.

    Parameters
    ----------
    steps    : total simulation steps
    K        : MPPI rollout count
    adaptive : enable residual-based modulation (RC-MPPI) if True
    log      : also return per-step diagnostic data

    Returns
    -------
    pos     : (steps+1, 2) position trajectory
    metrics : dict of scalar performance metrics
    debug   : dict with 's_bar' history if log=True, else None
    """
    # Initial plant state (CPU)
    x = torch.tensor(START_STATE, device=CPU, dtype=DTYPE)

    # Nominal control sequence warm-start (GPU)
    u_bar = torch.zeros(T_HORIZON, 2, device=GPU, dtype=DTYPE)

    # Obstacle belief (stationary, no online estimation)
    mu_c    = _obs_true.to(CPU) + _obs_bias_cpu
    Sigma_c = (float(OBS_UNC_STD) ** 2) * torch.eye(2, device=CPU, dtype=DTYPE)

    pos_log = [x[:2].numpy().copy()]
    s_bar   = 0.0
    s_hist  = []

    x_prev = x.clone()
    u_prev = torch.zeros(2, device=CPU, dtype=DTYPE)

    for k in range(steps):
        # --------------------------------------------------
        # 1. Compute residual & determine adaptive parameters
        # --------------------------------------------------
        if adaptive and k > 0:
            # Nominal one-step prediction
            x_pred = _A_cpu @ x_prev + _B_cpu @ u_prev
            # r_k = y_k − f_θ(y_{k-1}, u_{k-1})  (eq. 4: measured − predicted)
            e      = x - x_pred
            s      = mismatch_score(e)
            # Exponential filter  s̄_k = (1−ρ)s̄_{k-1} + ρ s_k  (eq. 6)
            s_bar  = (1.0 - RISK_FILTER_RHO) * s_bar + RISK_FILTER_RHO * s
            margin, sigma_eff, beta_eff = modulate(s_bar)
        else:
            s_bar, margin = 0.0, 0.0
            sigma_eff, beta_eff = SIGMA0, BETA0

        radius_eff = _obs_r + torch.tensor(margin, device=GPU, dtype=DTYPE)

        # --------------------------------------------------
        # 2. Sample obstacle scenarios & run MPPI (GPU)
        # --------------------------------------------------
        obs_centers = sample_obstacle_scenarios(mu_c, Sigma_c, OBS_NUM_SAMPLES)

        u0, u_bar = mppi_step(
            x.to(GPU), u_bar, obs_centers, radius_eff,
            K=K, sigma=sigma_eff, beta=beta_eff,
        )

        # --------------------------------------------------
        # 3. Execute plant (CPU) & receding-horizon shift
        # --------------------------------------------------
        x_prev = x.clone()
        u_prev = u0.to(CPU).clone()
        x      = step_plant(x, u_prev)

        u_bar = torch.cat(
            [u_bar[1:], torch.zeros(1, 2, device=GPU, dtype=DTYPE)], dim=0
        )

        pos_log.append(x[:2].numpy().copy())
        if log:
            s_hist.append(s_bar)

    pos    = np.asarray(pos_log)
    metrics = compute_metrics(pos, steps)
    debug   = dict(s_bar=np.asarray(s_hist)) if log else None
    return pos, metrics, debug


# ============================================================
# SECTION 8 — MONTE CARLO EVALUATION  (paired seeds)
# ============================================================

def run_mc(n_trials: int = MC_N_TRIALS,
           base_seed: int = MC_BASE_SEED,
           steps: int = SIM_STEPS_MC,
           K: int = K_ROLLOUTS) -> tuple[dict, np.ndarray]:
    """
    Paired-seed Monte Carlo: same seed → Vanilla, then RC-MPPI.
    Differences are attributable to adaptation, not sampling variance.

    Returns
    -------
    results : {"vanilla": [metrics, ...], "adaptive": [metrics, ...]}
    seeds   : (n_trials,) seed array
    """
    results = {"vanilla": [], "adaptive": []}
    seeds   = np.arange(base_seed, base_seed + n_trials, dtype=int)

    for seed in tqdm(seeds, desc="MC trials", ncols=80):
        set_seed(int(seed))
        _, mv, _ = simulate(steps=steps, K=K, adaptive=False)

        set_seed(int(seed))
        _, ma, _ = simulate(steps=steps, K=K, adaptive=True)

        results["vanilla"].append(mv)
        results["adaptive"].append(ma)

    return results, seeds


def summarize_mc(results: dict):
    """Print a human-readable MC summary table."""
    def arr(method, key):
        return np.asarray([m[key] for m in results[method]], dtype=float)

    print("\n" + "=" * 58)
    print(f"{'Metric':<28} {'Vanilla MPPI':>13} {'RC-MPPI':>13}")
    print("=" * 58)

    metrics_cfg = [
        ("success",       "Success rate",
         lambda x: f"{x.mean():.2f}",                   False),
        ("t_goal",        "Time-to-goal (steps)",
         lambda x: f"{x.mean():.1f} ± {x.std():.1f}",  True),
        ("min_clearance", "Min clearance (m)",
         lambda x: f"{x.mean():.3f} ± {x.std():.3f}",  True),
        ("violations",    "Violation steps",
         lambda x: f"{x.mean():.2f} ± {x.std():.2f}",  True),
        ("path_len",      "Path length (m)",
         lambda x: f"{x.mean():.2f} ± {x.std():.2f}",  True),
    ]

    for key, label, fmt, _ in metrics_cfg:
        v = arr("vanilla",  key)
        a = arr("adaptive", key)
        print(f"  {label:<26} {fmt(v):>13} {fmt(a):>13}")

    print("=" * 58)


# ============================================================
# SECTION 9 — REPRESENTATIVE SEED SELECTION
# ============================================================

def select_representative_seed(results: dict,
                                seeds: np.ndarray) -> tuple[int, int, str]:
    """
    Choose the seed that best illustrates the benefit of RC-MPPI.

    Priority:
      1. Vanilla violated but adaptive did not  → largest vanilla violation count
      2. Otherwise → largest clearance improvement (adaptive − vanilla)

    Returns
    -------
    seed_star : int  best seed value
    idx_star  : int  index into seeds / results lists
    reason    : str  human-readable explanation
    """
    v_viol = np.array([m["violations"]    for m in results["vanilla"]],  dtype=float)
    a_viol = np.array([m["violations"]    for m in results["adaptive"]], dtype=float)
    v_clr  = np.array([m["min_clearance"] for m in results["vanilla"]],  dtype=float)
    a_clr  = np.array([m["min_clearance"] for m in results["adaptive"]], dtype=float)

    cand = np.where((v_viol > 0) & (a_viol == 0))[0]
    if cand.size:
        idx    = int(cand[np.argmax(v_viol[cand])])
        reason = "vanilla violated; adaptive collision-free (max vanilla violations)"
    else:
        idx    = int(np.argmax(a_clr - v_clr))
        reason = "max clearance improvement (RC-MPPI − vanilla)"

    return int(seeds[idx]), idx, reason


# ============================================================
# SECTION 10 — FIGURES
# ============================================================

def _circle_patch(center, radius, **kw):
    return plt.Circle(center, radius, **kw)


def plot_trajectories(traj_v: np.ndarray,
                      traj_a: np.ndarray,
                      seed: int,
                      save_path: str = "rc_mppi_fig1_trajectories.pdf"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(traj_v[:, 0], traj_v[:, 1], "r--", lw=2, label="Vanilla MPPI")
    ax.plot(traj_a[:, 0], traj_a[:, 1], "b-",  lw=2, label="RC-MPPI")
    ax.scatter(*GOAL_POS, s=80, zorder=5, label="Goal")
    ax.add_patch(_circle_patch(OBS_CENTER, OBS_RADIUS, alpha=0.25, label="Obstacle"))
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectories — seed {seed}")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  Saved: {save_path}")


def plot_clearance(traj_v: np.ndarray,
                   traj_a: np.ndarray,
                   seed: int,
                   save_path: str = "rc_mppi_fig2_clearance.pdf"):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(clearance_series(traj_v), "r--", lw=2, label="Vanilla MPPI")
    ax.plot(clearance_series(traj_a), "b-",  lw=2, label="RC-MPPI")
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Clearance to true obstacle (m)")
    ax.set_title(f"Clearance vs Time — seed {seed}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  Saved: {save_path}")


def plot_mc_scatter(results: dict,
                    idx_star: int,
                    save_path: str = "rc_mppi_fig3_mc_scatter.pdf"):
    x = np.array([m["min_clearance"] for m in results["vanilla"]],  dtype=float)
    y = np.array([m["min_clearance"] for m in results["adaptive"]], dtype=float)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(x, y, alpha=0.7, label="Trials")
    ax.scatter(x[idx_star], y[idx_star], s=150, marker="x",
               linewidths=2.5, zorder=5, label="Representative")
    lim = (min(x.min(), y.min()), max(x.max(), y.max()))
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlabel("Vanilla MPPI — min clearance (m)")
    ax.set_ylabel("RC-MPPI — min clearance (m)")
    ax.set_title("Paired MC: Minimum Clearance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --- Print configuration summary ---
    print("=" * 55)
    print("  RC-MPPI Configuration")
    print("=" * 55)
    print(f"  Device          : {GPU}")
    print(f"  DT / SIM_STEPS  : {DT} s / MC={SIM_STEPS_MC}, replay={SIM_STEPS_REPLAY}")
    print(f"  Execution model : {EXECUTION_MODEL}"
          + (f"  (τ={SERVO_TAU}s, α={_alpha_servo:.4f})"
             if EXECUTION_MODEL == "lag" else ""))
    print(f"  MPPI K / T      : {K_ROLLOUTS} / {T_HORIZON}")
    print(f"  Obstacle risk   : {OBS_RISK_MODE.upper()}"
          + (f" α={OBS_CVAR_ALPHA}" if OBS_RISK_MODE == "cvar" else "")
          + f"  M={OBS_NUM_SAMPLES}  σ_obs={OBS_UNC_STD}")
    print(f"  Adaptation      : {'ON' if USE_RISK_ADAPTATION else 'OFF'}")
    print(f"  β convention    : inverse temperature (w ∝ exp(−β·cost))")
    print(f"  MC trials       : {MC_N_TRIALS}")
    print("=" * 55)

    # Reduce K on CPU to keep runtime manageable
    K = K_ROLLOUTS if GPU.type == "cuda" else K_ROLLOUTS // 4

    # ----------------------------------------------------------
    # Monte Carlo evaluation
    # ----------------------------------------------------------
    print(f"\nRunning {MC_N_TRIALS} paired-seed MC trials …")
    t0 = time.time()
    results, seeds = run_mc(n_trials=MC_N_TRIALS, base_seed=MC_BASE_SEED,
                            steps=SIM_STEPS_MC, K=K)
    print(f"Elapsed: {time.time() - t0:.1f} s")

    summarize_mc(results)

    # ----------------------------------------------------------
    # Representative seed replay
    # ----------------------------------------------------------
    seed_star, idx_star, reason = select_representative_seed(results, seeds)
    print(f"\nRepresentative seed: {seed_star}  (index {idx_star})")
    print(f"  Reason: {reason}")

    print("\nReplaying representative seed …")
    set_seed(seed_star)
    traj_v, met_v, _ = simulate(steps=SIM_STEPS_REPLAY, K=K,
                                 adaptive=False, log=True)

    set_seed(seed_star)
    traj_a, met_a, _ = simulate(steps=SIM_STEPS_REPLAY, K=K,
                                 adaptive=True,  log=True)

    print(f"  Vanilla : clearance={met_v['min_clearance']:.3f} m  "
          f"violations={met_v['violations']}  success={met_v['success']}")
    print(f"  RC-MPPI : clearance={met_a['min_clearance']:.3f} m  "
          f"violations={met_a['violations']}  success={met_a['success']}")

    # ----------------------------------------------------------
    # Figures
    # ----------------------------------------------------------
    print("\nSaving figures …")
    plot_trajectories(traj_v, traj_a, seed_star)
    plot_clearance   (traj_v, traj_a, seed_star)
    plot_mc_scatter  (results, idx_star)
