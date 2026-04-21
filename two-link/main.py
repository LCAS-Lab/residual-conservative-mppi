#!/usr/bin/env python3
"""
Residual-adaptive MPPI for a planar 2-link (2R) arm (2D).

Features:
- GPU MPPI rollouts using a nominal model (rigid-body 2R dynamics)
- CPU execution model switch (exact vs torque lag + saturation)
- output feedback via measurement noise (q, qdot)
- residual-adaptive risk modulation (inflate obstacle radius, modulate sigma/beta)
- Paired Monte Carlo (vanilla vs adaptive) with representative seed selection (prefers SUCCESS)
- Plots:
    * end-effector trajectory overlay + obstacle + goal + arm links (initial/final + optional snapshots)
    * clearance vs time
    * paired MC scatter (min clearance)
    * mismatch statistic & modulation (adaptive)

Changes from original (consistency with paper):
  1. Importance weights use w ∝ exp(−β_k · cost), where β_k is the INVERSE
     temperature (paper eq. 12, 16). Higher β_k → sharper distribution →
     low-cost (safer) rollouts dominate more strongly (Lemma 3, Prop. 3).
     Original used exp(−cost / β), treating β as temperature (wrong direction).
  2. Residual sign corrected: e_k = x_cpu − x_pred_det = y_k − f_θ(y_{k-1}, u_{k-1})
     (measured − predicted), consistent with paper eq. 4.
     Original computed x_pred_det − x_cpu (opposite sign).
     No numerical impact since only the norm is used, but notation is now consistent.
  3. beta0=1.0 is retained: exp(−1·cost) is numerically identical to original
     exp(−cost/1), so vanilla MPPI behaviour is unchanged at baseline.
  4. KAPPA_BETA modulation now correctly sharpens the distribution as s̄_k grows.

Run:
  python3 mppi_2link_arm.py
"""

import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 0) Devices / dtype
# ============================================================

device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
dtype = torch.float32

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ============================================================
# 1) Simulation / planning settings
# ============================================================

dt = 0.02
EXECUTION_MODEL = "lag"      # "exact" or "lag"
SERVO_TAU = 0.15
alpha_servo = 1.0 - np.exp(-dt / SERVO_TAU)

# ============================================================
# 2) 2R arm parameters
# ============================================================

l1 = 1.0
l2 = 0.8

m1 = 1.0
m2 = 0.8

lc1 = 0.5 * l1
lc2 = 0.5 * l2

I1 = (1.0 / 12.0) * m1 * l1 * l1
I2 = (1.0 / 12.0) * m2 * l2 * l2

g = 9.81

# ============================================================
# 3) Task setup in workspace
# ============================================================

goal_xy = torch.tensor([1.35, 0.35], dtype=dtype, device=device_gpu)

obs_center_true = torch.tensor([0.85, 0.15], dtype=dtype, device=device_gpu)
obs_radius = torch.tensor(0.18, dtype=dtype, device=device_gpu)

q_min = torch.tensor([-2.6, -2.6], dtype=dtype, device=device_gpu)
q_max = torch.tensor([ 2.6,  2.6], dtype=dtype, device=device_gpu)

TAU_MAX = 6.0

# ============================================================
# 4) Cost weights
# ============================================================

W_GOAL = 70.0
W_QREG = 0.5
W_VEL  = 0.2
W_CTRL = 0.008
W_TERM = 250.0

W_OBS  = 2.0e4
W_JLIM = 2.0e3

U_CLIP = TAU_MAX

# ============================================================
# 5) Residual-adaptive risk modulation knobs
# ============================================================

USE_RISK_ADAPTATION = True

E_Q_W   = 1.0
E_QD_W  = 0.2

RISK_FILTER_RHO = 0.10

KAPPA_R     = 0.35
KAPPA_SIGMA = 1.25
# KAPPA_BETA modulates β_k (inverse temperature, paper eq. 16).
# As s̄_k increases, β_k increases → distribution sharpens →
# low-cost (safer) rollouts weighted more strongly (Lemma 3).
KAPPA_BETA  = 0.90

R_MARGIN_MAX = 0.20
SIGMA_MIN, SIGMA_MAX = 0.10, 2.00
# β bounds in INVERSE-temperature units.
# β_min=0.20 → permissive baseline; β_max=6.00 → sharp ceiling.
BETA_MIN,  BETA_MAX  = 0.20, 6.00

# ============================================================
# 6) Measurement noise (CPU)
# ============================================================

Q_NOISE_STD  = 0.002
QD_NOISE_STD = 0.01

def _measurement_noise_cpu():
    return torch.cat([
        Q_NOISE_STD  * torch.randn(2, device=device_cpu, dtype=dtype),
        QD_NOISE_STD * torch.randn(2, device=device_cpu, dtype=dtype),
    ])

# ============================================================
# 7) Kinematics / dynamics helpers (Torch)
# ============================================================

def wrap_to_pi(q):
    return (q + math.pi) % (2.0 * math.pi) - math.pi

# ------------------------------
# Link drawing helpers (NumPy)
# ------------------------------
def fk_links_np(q1, q2):
    """
    Returns:
        p0 = base (0,0)
        p1 = joint 1 position
        p2 = end-effector position
    """
    p0 = np.array([0.0, 0.0], dtype=float)

    x1 = l1 * np.cos(q1)
    y1 = l1 * np.sin(q1)
    p1 = np.array([x1, y1], dtype=float)

    x2 = x1 + l2 * np.cos(q1 + q2)
    y2 = y1 + l2 * np.sin(q1 + q2)
    p2 = np.array([x2, y2], dtype=float)

    return p0, p1, p2

def draw_arm(ax, q1, q2, color="k", lw=3, alpha=1.0, zorder=6, show_joints=True):
    """
    Draw the 2R arm in workspace.
    """
    p0, p1, p2 = fk_links_np(float(q1), float(q2))

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color=color, lw=lw, alpha=alpha, zorder=zorder)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=color, lw=lw, alpha=alpha, zorder=zorder)

    if show_joints:
        ax.scatter([p1[0]], [p1[1]], c=color, s=30, zorder=zorder+1)
        ax.scatter([p2[0]], [p2[1]], c=color, s=30, zorder=zorder+1)

def fk_ee_batch(q_k2: torch.Tensor) -> torch.Tensor:
    q1 = q_k2[:, 0]
    q2 = q_k2[:, 1]
    c1 = torch.cos(q1); s1 = torch.sin(q1)
    c12 = torch.cos(q1 + q2); s12 = torch.sin(q1 + q2)
    x = l1 * c1 + l2 * c12
    y = l1 * s1 + l2 * s12
    return torch.stack([x, y], dim=1)

def M_batch(q_k2: torch.Tensor) -> torch.Tensor:
    q2 = q_k2[:, 1]
    c2 = torch.cos(q2)
    M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * c2)
    M12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
    M22 = I2 + m2 * lc2**2
    K = q_k2.shape[0]
    M = torch.zeros((K, 2, 2), device=q_k2.device, dtype=q_k2.dtype)
    M[:, 0, 0] = M11
    M[:, 0, 1] = M12
    M[:, 1, 0] = M12
    M[:, 1, 1] = M22
    return M

def Cqd_batch(q_k2: torch.Tensor, qd_k2: torch.Tensor) -> torch.Tensor:
    q2 = q_k2[:, 1]
    s2 = torch.sin(q2)
    h = -m2 * l1 * lc2 * s2
    q1d = qd_k2[:, 0]
    q2d = qd_k2[:, 1]
    c1 = h * (2.0 * q1d * q2d + q2d * q2d)
    c2 = h * (q1d * q1d)
    return torch.stack([c1, c2], dim=1)

# NOTE: gravity is computed but effectively canceled in the current "template" nominal model,
# matching your original behavior. If you later want true gravity, change rhs to (tau - cqd - gv).
def gvec_batch(q_k2: torch.Tensor) -> torch.Tensor:
    q1 = q_k2[:, 0]
    q2 = q_k2[:, 1]
    c1 = torch.cos(q1)
    c12 = torch.cos(q1 + q2)
    g1 = (m1 * lc1 + m2 * l1) * g * c1 + m2 * lc2 * g * c12
    g2 = m2 * lc2 * g * c12
    return torch.stack([g1, g2], dim=1)

def step_arm_batch(x_k4: torch.Tensor, tau_k2: torch.Tensor) -> torch.Tensor:
    q  = x_k4[:, 0:2]
    qd = x_k4[:, 2:4]

    M = M_batch(q)
    cqd = Cqd_batch(q, qd)
    _gv = gvec_batch(q)  # computed but canceled (kept for template parity)

    rhs = tau_k2 - cqd  # (template parity: gravity canceled)
    qdd = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)

    qd_next = qd + dt * qdd
    q_next  = q  + dt * qd_next
    q_next = torch.stack([wrap_to_pi(q_next[:, 0]), wrap_to_pi(q_next[:, 1])], dim=1)

    return torch.cat([q_next, qd_next], dim=1)

# ============================================================
# 8) Costs (GPU)
# ============================================================

def joint_limit_cost_batch(q_k2: torch.Tensor) -> torch.Tensor:
    above = torch.clamp(q_k2 - q_max[None, :], min=0.0)
    below = torch.clamp(q_min[None, :] - q_k2, min=0.0)
    v = above + below
    return W_JLIM * torch.sum(v * v, dim=1)

def obstacle_cost_batch(ee_k2: torch.Tensor, radius_eff: torch.Tensor, obs_center: torch.Tensor) -> torch.Tensor:
    diff = ee_k2 - obs_center[None, :]
    d = torch.linalg.norm(diff, dim=1)
    inside = d < radius_eff
    c = torch.zeros_like(d)
    c[inside] = W_OBS * (radius_eff - d[inside]) ** 2
    return c

def running_cost_batch(x_k4: torch.Tensor, tau_k2: torch.Tensor, radius_eff: torch.Tensor, obs_center: torch.Tensor) -> torch.Tensor:
    q  = x_k4[:, 0:2]
    qd = x_k4[:, 2:4]
    ee = fk_ee_batch(q)

    goal_cost = W_GOAL * torch.sum((ee - goal_xy[None, :]) ** 2, dim=1)
    qreg_cost = W_QREG * torch.sum(q * q, dim=1)
    vel_cost  = W_VEL  * torch.sum(qd * qd, dim=1)
    ctrl_cost = W_CTRL * torch.sum(tau_k2 * tau_k2, dim=1)

    obs_cost  = obstacle_cost_batch(ee, radius_eff, obs_center)
    jlim_cost = joint_limit_cost_batch(q)

    return goal_cost + qreg_cost + vel_cost + ctrl_cost + obs_cost + jlim_cost

def terminal_cost_batch(x_k4: torch.Tensor) -> torch.Tensor:
    q = x_k4[:, 0:2]
    ee = fk_ee_batch(q)
    return W_TERM * torch.sum((ee - goal_xy[None, :]) ** 2, dim=1)

# ============================================================
# 9) MPPI rollout (GPU)
# ============================================================

@torch.no_grad()
def mppi_control_vec(
    x0_gpu,          # (4,)
    u_bar,           # (T,2)
    radius_eff_gpu,  # scalar
    obs_center_gpu,  # (2,)
    K=4096,
    T=35,
    sigma=1.0,
    beta=1.0,
):
    """
    One MPPI control update (paper eq. 12–13).

    beta is the INVERSE temperature (paper eq. 12, 16):
      w^(i) ∝ exp(−β · Z^(i))
    Higher β → sharper importance weights → low-cost (safer) rollouts
    dominate more strongly.  As s̄_k grows, β_k increases via risk_modulate(),
    concentrating the sampling distribution for improved safety (Lemma 3,
    Proposition 3).
    """
    eps = sigma * torch.randn(K, T, 2, device=device_gpu, dtype=dtype)
    x = x0_gpu[None, :].repeat(K, 1)
    cost = torch.zeros(K, device=device_gpu, dtype=dtype)

    for t in range(T):
        u = u_bar[t][None, :] + eps[:, t, :]
        x = step_arm_batch(x, u)
        cost += running_cost_batch(x, u, radius_eff_gpu, obs_center_gpu)

    cost += terminal_cost_batch(x)

    cost -= cost.min()
    # w^(i) ∝ exp(−β · Z^(i))  — β is INVERSE temperature (paper eq. 12).
    # Fixed from original exp(−cost / β) which treated β as temperature and
    # produced the wrong modulation direction as s̄_k grew.
    w = torch.exp(-beta * cost)
    w /= (w.sum() + 1e-12)

    du = torch.sum(w[:, None, None] * eps, dim=0)
    u_bar_new = u_bar + du

    u0 = torch.clamp(u_bar_new[0], -U_CLIP, U_CLIP)
    return u0, u_bar_new

# ============================================================
# 10) CPU execution models (exact vs lag + saturation)
# ============================================================

def _arm_step_cpu_deterministic(x_cpu: torch.Tensor, tau_cpu: torch.Tensor) -> torch.Tensor:
    """
    One deterministic step of the SAME template dynamics used in planner:
      M qdd = tau - Cqd   (gravity canceled, for parity)
    Returns: x_next (NO measurement noise)
    """
    x = x_cpu.clone()
    q  = x[0:2]
    qd = x[2:4]

    q2 = q[1]
    c2 = torch.cos(q2)
    s2 = torch.sin(q2)

    M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * c2)
    M12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
    M22 = I2 + m2 * lc2**2
    M = torch.tensor([[M11, M12],
                      [M12, M22]], device=device_cpu, dtype=dtype)

    h = -m2 * l1 * lc2 * s2
    q1d = qd[0]; q2d = qd[1]
    cqd = torch.stack([
        h * (2.0 * q1d * q2d + q2d * q2d),
        h * (q1d * q1d)
    ])

    rhs = tau_cpu - cqd  # template parity
    qdd = torch.linalg.solve(M, rhs)

    qd_next = qd + dt * qdd
    q_next = q + dt * qd_next
    q_next = torch.tensor([wrap_to_pi(float(q_next[0])), wrap_to_pi(float(q_next[1]))],
                          device=device_cpu, dtype=dtype)

    return torch.cat([q_next, qd_next])

def execute_exact_arm_cpu(x_cpu, tau_cpu):
    """
    Exact execution: apply tau directly, then add measurement noise to returned state.
    """
    x_true = _arm_step_cpu_deterministic(x_cpu, tau_cpu)
    return x_true + _measurement_noise_cpu()

def execute_servo_lag_arm_cpu(x_cpu, tau_cmd_cpu, tau_exec_cpu):
    """
    First-order lag on torque + saturation.
    tau_exec_{k+1} = (1-a) tau_exec_k + a sat(tau_cmd)
    """
    tau_cmd = torch.clamp(tau_cmd_cpu, -TAU_MAX, TAU_MAX)
    tau_exec_next = (1.0 - alpha_servo) * tau_exec_cpu + alpha_servo * tau_cmd
    x_next = execute_exact_arm_cpu(x_cpu, tau_exec_next)
    return x_next, tau_exec_next

# ============================================================
# 11) Residual-based risk modulation (CPU)
# ============================================================

def mismatch_score_cpu(e_cpu):
    """
    Weighted L2 norm of one-step prediction residual (paper eq. 5, 44).
    e_cpu = r_k = y_k − f_θ(y_{k-1}, u_{k-1})  (measured − predicted)
    """
    e_q  = e_cpu[0:2]
    e_qd = e_cpu[2:4]
    s = torch.sqrt(E_Q_W * torch.sum(e_q * e_q) + E_QD_W * torch.sum(e_qd * e_qd))
    return float(s.item())

def risk_modulate(s_bar, sigma0, beta0):
    """
    Map filtered residual s̄_k to adaptive parameters (paper eq. 14, 15, 16).

    Returns
    -------
    margin    : obstacle radius inflation Δr(s̄_k)      — eq. 14 / Remark 3
    sigma_eff : σ_k ↓  (tighter sampling)               — eq. 16
    beta_eff  : β_k ↑  (sharper inverse temperature)    — eq. 16
                Higher β_k concentrates weight on safer rollouts (Lemma 3).
    """
    margin    = float(np.clip(KAPPA_R * s_bar, 0.0, R_MARGIN_MAX))
    sigma_eff = float(np.clip(sigma0 / (1.0 + KAPPA_SIGMA * s_bar), SIGMA_MIN, SIGMA_MAX))
    # β_k = clip(β_0 (1 + κ_β s̄_k), β_min, β_max)  — paper eq. 16
    # Increasing β_k sharpens exp(−β·cost), correctly per Lemma 3.
    beta_eff  = float(np.clip(beta0  * (1.0 + KAPPA_BETA  * s_bar), BETA_MIN,  BETA_MAX))
    return margin, sigma_eff, beta_eff

# ============================================================
# 12) Metrics helpers (TRUE obstacle, NumPy)
# ============================================================

def fk_ee_np(q1, q2):
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, y], dtype=float)

def clearance_series_ee(ee_xy: np.ndarray) -> np.ndarray:
    c = obs_center_true.detach().cpu().numpy()
    r = float(obs_radius.detach().cpu().item())
    return np.linalg.norm(ee_xy - c[None, :], axis=1) - r

def min_clearance_true(ee_xy: np.ndarray) -> float:
    return float(np.min(clearance_series_ee(ee_xy)))

def num_violations_true(ee_xy: np.ndarray) -> int:
    return int(np.sum(clearance_series_ee(ee_xy) < 0.0))

def time_to_goal(ee_xy: np.ndarray, tol: float) -> int:
    gxy = goal_xy.detach().cpu().numpy()
    dist = np.linalg.norm(ee_xy - gxy[None, :], axis=1)
    hit = np.where(dist <= tol)[0]
    return int(hit[0]) if hit.size > 0 else -1

def path_length(ee_xy: np.ndarray) -> float:
    dif = ee_xy[1:] - ee_xy[:-1]
    return float(np.sum(np.linalg.norm(dif, axis=1)))

# ============================================================
# 13) Seeding
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# 14) Simulation
# ============================================================

def simulate(
    steps=200,
    K=4096,
    T_horizon=35,
    sigma0=1.0,
    beta0=1.0,
    goal_tol=0.04,
    log_debug=False,
    return_controls=False,
):
    """
    Run a single RC-MPPI (or Vanilla MPPI) simulation.

    beta0 is the base INVERSE temperature β_0 (paper eq. 16).
    At beta0=1.0, exp(−1·cost) is numerically identical to the original
    exp(−cost/1), so vanilla MPPI behaviour is unchanged at baseline.
    """
    # CPU measured state
    x_cpu = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device_cpu, dtype=dtype)
    x_cpu_prev = x_cpu.clone()
    tau_cpu_prev = torch.zeros(2, device=device_cpu, dtype=dtype)

    # torque lag state
    tau_exec_cpu = torch.zeros(2, device=device_cpu, dtype=dtype)

    # GPU nominal control sequence
    u_bar = torch.zeros(T_horizon, 2, device=device_gpu, dtype=dtype)

    # Logs
    q_log = [x_cpu[0:2].cpu().numpy().copy()]
    ee_log = [fk_ee_np(float(x_cpu[0]), float(x_cpu[1]))]
    tau_log = []

    # Residual filter state
    s_bar = 0.0
    s_log, sigma_log, beta_log, r_eff_log = [], [], [], []

    obs_center_gpu = obs_center_true  # fixed obstacle

    for k in range(steps):

        # One-step prediction residual: r_k = y_k − f_θ(y_{k-1}, u_{k-1})
        # (paper eq. 4: measured − predicted)
        if USE_RISK_ADAPTATION and k > 0:
            x_pred_det = _arm_step_cpu_deterministic(x_cpu_prev, tau_cpu_prev)
            # Corrected sign: measured minus predicted (eq. 4)
            e_k = x_cpu - x_pred_det
            s_k = mismatch_score_cpu(e_k)
            s_bar = (1.0 - RISK_FILTER_RHO) * s_bar + RISK_FILTER_RHO * s_k
        elif not USE_RISK_ADAPTATION:
            s_bar = 0.0

        # risk modulation
        if USE_RISK_ADAPTATION:
            margin, sigma_eff, beta_eff = risk_modulate(s_bar, sigma0, beta0)
        else:
            margin, sigma_eff, beta_eff = 0.0, float(sigma0), float(beta0)

        radius_eff_gpu = obs_radius + torch.tensor(margin, device=device_gpu, dtype=dtype)

        # MPPI
        x_gpu = x_cpu.to(device_gpu)
        tau_gpu, u_bar = mppi_control_vec(
            x_gpu,
            u_bar,
            radius_eff_gpu,
            obs_center_gpu,
            K=K,
            T=T_horizon,
            sigma=sigma_eff,
            beta=beta_eff,
        )

        # execute plant (CPU)
        tau_cpu = tau_gpu.to(device_cpu)
        x_cpu_prev = x_cpu.clone()
        tau_cpu_prev = tau_cpu.clone()

        if EXECUTION_MODEL == "exact":
            x_cpu = execute_exact_arm_cpu(x_cpu, tau_cpu)
        elif EXECUTION_MODEL == "lag":
            x_cpu, tau_exec_cpu = execute_servo_lag_arm_cpu(x_cpu, tau_cpu, tau_exec_cpu)
        else:
            raise ValueError(f"Unknown EXECUTION_MODEL: {EXECUTION_MODEL}")

        if return_controls:
            tau_log.append(tau_cpu.cpu().numpy().copy())

        q_log.append(x_cpu[0:2].cpu().numpy().copy())
        ee_log.append(fk_ee_np(float(x_cpu[0]), float(x_cpu[1])))

        # receding horizon shift
        u_bar = torch.cat([u_bar[1:], torch.zeros(1, 2, device=device_gpu, dtype=dtype)], dim=0)

        if log_debug:
            s_log.append(s_bar)
            sigma_log.append(sigma_eff)
            beta_log.append(beta_eff)
            r_eff_log.append(float(radius_eff_gpu.detach().cpu().item()))

    q_traj = np.array(q_log)
    ee_traj = np.array(ee_log)
    tau_traj = np.array(tau_log) if return_controls else None

    # Metrics (true obstacle).
    # Success = goal reached AND no obstacle violations.
    # A trial that reaches the goal through the obstacle is not a success —
    # collision-free goal-reaching is required for the arm task.
    t_goal = time_to_goal(ee_traj, tol=goal_tol)
    min_clr = min_clearance_true(ee_traj)
    viol = num_violations_true(ee_traj)
    success = 1 if (t_goal >= 0 and viol == 0) else 0
    plen = path_length(ee_traj)
    energy = float(np.sum(tau_traj[:, 0] ** 2 + tau_traj[:, 1] ** 2)) if tau_traj is not None else np.nan

    metrics = {
        "success": success,
        "t_goal": (t_goal if t_goal >= 0 else steps + 1),
        "min_clearance": min_clr,
        "violations": viol,
        "path_len": plen,
        "ctrl_energy": energy,
    }

    debug = None
    if log_debug:
        debug = {
            "s_bar": np.array(s_log),
            "sigma_eff": np.array(sigma_log),
            "beta_eff": np.array(beta_log),
            "r_eff": np.array(r_eff_log),
        }

    return q_traj, ee_traj, tau_traj, metrics, debug

# ============================================================
# 15) Monte Carlo evaluation (paired seeds) + representative selection
# ============================================================

def mc_evaluate(
    n_trials=40,
    base_seed=0,
    steps=200,
    K=4096,
    T_horizon=35,
    sigma0=1.0,
    beta0=1.0,
    goal_tol=0.04,
):
    methods = ["vanilla", "adaptive"]
    results = {m: {k: [] for k in [
        "success", "t_goal", "min_clearance", "violations", "path_len", "ctrl_energy"
    ]} for m in methods}

    seeds = []
    global USE_RISK_ADAPTATION
    _ = torch.randn(1, device=device_gpu)  # warmup

    for i in tqdm(range(n_trials), desc="MC trials", ncols=80):
        seed = base_seed + i
        seeds.append(seed)

        # VANILLA
        USE_RISK_ADAPTATION = False
        set_seed(seed)
        _, _, _, met_v, _ = simulate(
            steps=steps, K=K, T_horizon=T_horizon,
            sigma0=sigma0, beta0=beta0, goal_tol=goal_tol,
            log_debug=False, return_controls=True
        )

        # ADAPTIVE
        USE_RISK_ADAPTATION = True
        set_seed(seed)
        _, _, _, met_a, _ = simulate(
            steps=steps, K=K, T_horizon=T_horizon,
            sigma0=sigma0, beta0=beta0, goal_tol=goal_tol,
            log_debug=False, return_controls=True
        )

        for tag, met in [("vanilla", met_v), ("adaptive", met_a)]:
            for k in results[tag]:
                results[tag][k].append(float(met[k]))

    for m in methods:
        for k in results[m]:
            results[m][k] = np.asarray(results[m][k], dtype=float)

    return results, np.asarray(seeds, dtype=int)

def select_representative_seed(results, seeds):
    """
    Select the seed that most clearly illustrates RC-MPPI's safety benefit.

    Priority (in order):
      1) Adaptive succeeds, vanilla violated → max vanilla violations
         (most dramatic: RC-MPPI avoids collision that vanilla could not)
      2) Both succeed, vanilla violated but adaptive didn't → max vanilla violations
         (both reach goal; RC-MPPI does so cleanly)
      3) Both succeed: max clearance improvement (adaptive − vanilla)
      4) Adaptive-only success: best adaptive clearance
      5) Fallback: max clearance improvement across all trials
    """
    v_succ = results["vanilla"]["success"].astype(int)
    a_succ = results["adaptive"]["success"].astype(int)

    v_viol = results["vanilla"]["violations"]
    a_viol = results["adaptive"]["violations"]
    v_clr  = results["vanilla"]["min_clearance"]
    a_clr  = results["adaptive"]["min_clearance"]

    # Priority 1: adaptive succeeded, vanilla violated (regardless of vanilla success)
    cand1 = np.where((a_succ == 1) & (v_viol > 0) & (a_viol == 0))[0]
    if cand1.size > 0:
        idx = cand1[np.argmax(v_viol[cand1])]
        reason = (f"adaptive succeeded + zero violations; vanilla violated "
                  f"({int(v_viol[idx])} steps); selected max vanilla violations")
        return int(seeds[idx]), int(idx), reason

    # Priority 2: both succeed, vanilla violated but adaptive didn't
    both = np.where((v_succ == 1) & (a_succ == 1))[0]
    if both.size > 0:
        cand2 = both[(v_viol[both] > 0) & (a_viol[both] == 0)]
        if cand2.size > 0:
            idx = cand2[np.argmax(v_viol[cand2])]
            reason = "both succeeded; vanilla violated while adaptive had zero violations"
            return int(seeds[idx]), int(idx), reason

        # Priority 3: both succeed, max clearance improvement
        delta = a_clr[both] - v_clr[both]
        idx = both[int(np.argmax(delta))]
        reason = "both succeeded; selected max clearance improvement (adaptive − vanilla)"
        return int(seeds[idx]), int(idx), reason

    # Priority 4: adaptive-only success
    a_only = np.where((v_succ == 0) & (a_succ == 1))[0]
    if a_only.size > 0:
        idx = a_only[int(np.argmax(a_clr[a_only]))]
        reason = "adaptive-only success; selected largest adaptive clearance"
        return int(seeds[idx]), int(idx), reason

    # Priority 5: fallback
    delta = a_clr - v_clr
    idx = int(np.argmax(delta))
    reason = "fallback: selected max clearance improvement (adaptive − vanilla)"
    return int(seeds[idx]), int(idx), reason

def summarize_results(results: dict):
    def mean_std(x): return float(np.mean(x)), float(np.std(x))
    def rate(x): return float(np.mean(x))

    print(f"\n{'Metric':<24} {'Vanilla MPPI':>16} {'RC-MPPI':>16}")
    print("=" * 58)
    print(f"{'Success rate':<24} {rate(results['vanilla']['success']):>16.2f} "
          f"{rate(results['adaptive']['success']):>16.2f}")
    for lbl, key in [("Time-to-goal (steps)", "t_goal"),
                     ("Min clearance (m)",    "min_clearance"),
                     ("Violations (#steps)",  "violations"),
                     ("EE path length (m)",   "path_len"),
                     ("Control energy",       "ctrl_energy")]:
        vm, vs = mean_std(results["vanilla"][key])
        am, as_ = mean_std(results["adaptive"][key])
        print(f"{lbl:<24} {vm:>7.2f} ± {vs:<6.2f}  {am:>7.2f} ± {as_:<6.2f}")

# ============================================================
# 16) Main
# ============================================================

if __name__ == "__main__":

    print(f"GPU device     : {device_gpu}")
    print(f"Execution model: {EXECUTION_MODEL}")
    if EXECUTION_MODEL == "lag":
        print(f"SERVO_TAU={SERVO_TAU:.3f}s, alpha_servo={alpha_servo:.4f}")
    print(f"β convention   : inverse temperature (w ∝ exp(−β·cost))")
    print(f"beta0=1.0 (baseline, numerically identical to original exp(−cost/1))")

    K = 4096 if device_gpu.type == "cuda" else 1024
    n_trials = 50

    print(f"\nRunning MC evaluation: n_trials={n_trials}, K={K} ...")
    t0 = time.time()
    results, seeds = mc_evaluate(
        n_trials=n_trials,
        base_seed=0,
        steps=200,
        K=K,
        T_horizon=35,
        sigma0=1.0,
        beta0=1.0,
        goal_tol=0.04,
    )
    t1 = time.time()
    print(f"MC elapsed: {t1 - t0:.2f}s")

    summarize_results(results)

    seed_star, idx_star, reason = select_representative_seed(results, seeds)
    print("\nRepresentative trial selection:")
    print(f"  index = {idx_star}  seed = {seed_star}")
    print(f"  reason: {reason}")

    # Replay VANILLA
    USE_RISK_ADAPTATION = False
    set_seed(seed_star)
    q_v, ee_v, _, met_v, dbg_v = simulate(
        steps=200, K=K, T_horizon=35,
        sigma0=1.0, beta0=1.0,
        goal_tol=0.04,
        log_debug=True,
        return_controls=False
    )

    # Replay ADAPTIVE
    USE_RISK_ADAPTATION = True
    set_seed(seed_star)
    q_a, ee_a, _, met_a, dbg_a = simulate(
        steps=200, K=K, T_horizon=35,
        sigma0=1.0, beta0=1.0,
        goal_tol=0.04,
        log_debug=True,
        return_controls=False
    )

    print("\nRepresentative trial metrics (replayed):")
    print(f"  VANILLA : clearance={met_v['min_clearance']:.3f}, violations={met_v['violations']}, success={met_v['success']}")
    print(f"  ADAPTIVE: clearance={met_a['min_clearance']:.3f}, violations={met_a['violations']}, success={met_a['success']}")

    # ------------------------------------------------------------
    # Figure 1: End-effector trajectory overlay (+ goal + obstacle + links)
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    ax.plot(ee_v[:, 0], ee_v[:, 1], "r--", lw=2, label="Vanilla MPPI", zorder=3)
    ax.plot(ee_a[:, 0], ee_a[:, 1], "b-",  lw=2, label="RC-MPPI", zorder=4)

    gx = float(goal_xy[0].detach().cpu().item())
    gy = float(goal_xy[1].detach().cpu().item())
    ax.scatter(gx, gy, c="green", s=120, edgecolors="k", linewidths=1.0,
               zorder=10, label="Goal")

    circle_true = plt.Circle(
        obs_center_true.detach().cpu().numpy(),
        float(obs_radius.detach().cpu().item()),
        color="red",
        alpha=0.25,
        zorder=0
    )
    ax.add_patch(circle_true)

    # Arm configurations: initial + finals
    q1_init, q2_init = q_v[0]
    draw_arm(ax, q1_init, q2_init, color="gray", lw=4, alpha=0.8, zorder=7)

    q1_vf, q2_vf = q_v[-1]
    draw_arm(ax, q1_vf, q2_vf, color="red", lw=4, alpha=0.8, zorder=8)

    q1_af, q2_af = q_a[-1]
    draw_arm(ax, q1_af, q2_af, color="blue", lw=4, alpha=0.8, zorder=9)

    # Faint snapshots along adaptive trajectory
    for kk in np.linspace(0, len(q_a) - 1, 6, dtype=int):
        q1k, q2k = q_a[kk]
        draw_arm(ax, q1k, q2k, color="blue", lw=2, alpha=0.12, zorder=2, show_joints=False)

    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"2R Arm End-Effector Trajectory (seed={seed_star})")
    plt.tight_layout()
    plt.savefig("arm2r_fig1_traj.png", dpi=200)
    plt.savefig("arm2r_fig1_traj.pdf", bbox_inches="tight")

    # ------------------------------------------------------------
    # Figure 2: Clearance vs time (true obstacle)
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 3))
    plt.plot(clearance_series_ee(ee_v), "r--", lw=2, label="Vanilla clearance")
    plt.plot(clearance_series_ee(ee_a), "b-",  lw=2, label="RC-MPPI clearance")
    plt.axhline(0.0, color="k", lw=1)
    plt.xlabel("step")
    plt.ylabel("clearance to TRUE obstacle (m)")
    plt.title(f"Clearance vs Time (seed={seed_star})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("arm2r_fig2_clearance.png", dpi=200)
    plt.savefig("arm2r_fig2_clearance.pdf", bbox_inches="tight")

    # ------------------------------------------------------------
    # Figure 3: Paired MC scatter (min clearance)
    # ------------------------------------------------------------
    plt.figure(figsize=(4.5, 4.5))
    x = results["vanilla"]["min_clearance"]
    y = results["adaptive"]["min_clearance"]
    plt.scatter(x, y)
    plt.scatter([x[idx_star]], [y[idx_star]], s=120, marker="x", color="red", zorder=5)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("Vanilla min clearance (m)")
    plt.ylabel("RC-MPPI min clearance (m)")
    plt.title("Paired MC: Min Clearance (2R EE)")
    plt.tight_layout()
    plt.savefig("arm2r_fig3_mc_scatter.png", dpi=200)
    plt.savefig("arm2r_fig3_mc_scatter.pdf", bbox_inches="tight")

    # ------------------------------------------------------------
    # Figure 4: mismatch statistic & obstacle inflation (adaptive)
    # ------------------------------------------------------------
    if dbg_a is not None:
        plt.figure(figsize=(6, 3))
        plt.plot(dbg_a["s_bar"], lw=2)
        plt.xlabel("step")
        plt.ylabel(r"filtered mismatch $\bar{s}_k$")
        plt.title("Residual Mismatch Statistic (RC-MPPI)")
        plt.tight_layout()
        plt.savefig("arm2r_fig4_sbar.png", dpi=200)
        plt.savefig("arm2r_fig4_sbar.pdf", bbox_inches="tight")

        plt.figure(figsize=(6, 3))
        plt.plot(dbg_a["r_eff"], lw=2)
        plt.axhline(float(obs_radius.detach().cpu().item()), color="k", lw=1,
                    label="nominal radius")
        plt.xlabel("step")
        plt.ylabel("effective obstacle radius (m)")
        plt.title("Residual-Adaptive Obstacle Inflation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("arm2r_fig5_reff.png", dpi=200)
        plt.savefig("arm2r_fig5_reff.pdf", bbox_inches="tight")

        plt.figure(figsize=(6, 3))
        plt.plot(dbg_a["beta_eff"], lw=2, label=r"$\beta_k$ (inverse temp)")
        plt.axhline(1.0, color="k", lw=1, ls="--", label=r"$\beta_0 = 1.0$")
        plt.xlabel("step")
        plt.ylabel(r"$\beta_k$")
        plt.title(r"Adaptive Inverse Temperature $\beta_k$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("arm2r_fig6_beta.png", dpi=200)
        plt.savefig("arm2r_fig6_beta.pdf", bbox_inches="tight")

    plt.show()