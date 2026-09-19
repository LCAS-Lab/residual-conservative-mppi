#!/usr/bin/env python3
"""
Residual-Conservative Model Predictive Path Integral Control (RC-MPPI)
Planar 2R Arm Simulation — v6, consistent with RC-MPPI paper (v6).

Mathematical formulations (equation numbers match paper Section IV-V):

  Residual estimation (Sec. III):
    r_k     = y_k - f_theta(y_{k-1}, u_{k-1})          [eq:residual_def]
    s_k     = ||W_r * r_k||                              [inline after eq:residual_def]
    s_bar_k = (1 - rho) * s_bar_{k-1} + rho * s_k       [eqn:fil_statistics]

  Residual-conservative barrier modulation (Sec. IV-B):
    margin    = clip(kappa_r * s_bar, 0, Delta_r_max)    [simulation form of eq:tightening_def]
    alpha_k   = alpha_0 * (1 + gamma * s_bar)            [eq:penalty_scaling]

  Residual-adaptive sampling modulation (Sec. IV-B):
    sigma_k = clip(sigma_0 / (1 + kappa_sigma * s_bar),
                   sigma_min, sigma_max)                  [eq:sampling_modulation]
    beta_k  = clip(beta_0 * (1 + kappa_beta * s_bar),
                   beta_min, beta_max)                    [eq:sampling_modulation]

  MPPI importance weights (Sec. IV-A):
    w^(i) = exp(-Z^(i) / beta_k) / sum_j exp(-Z^(j) / beta_k)
                                                          [eq:mppi_weights]

  Temperature convention (Sec. IV-A, Proposition 4):
    beta encodes CONFIDENCE in rollout cost evaluations.
    beta_k INCREASES with s_bar (larger mismatch -> softer weights).
    This is NOT sharpening: it reflects that costs computed under an
    inaccurate model should be trusted less. Safety is maintained
    because alpha_k * phi(m(s_bar)) grows as O(s_bar^2) while
    beta_k grows only as O(s_bar), so unsafe rollouts receive
    asymptotically zero weight despite the rising temperature
    (Lemma 2(ii), Proposition 3). Proposition 4 proves that
    beta-up achieves strictly lower violation probability than
    beta-down above the explicit threshold s_bar* = mu*||U_bar - u*|| / C_Delta.
"""

import time, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Device & precision ─────────────────────────────────────────
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")
F32 = torch.float32
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ── Simulation timing ──────────────────────────────────────────
DT        = 0.02          # control timestep (s)
SERVO_TAU = 0.15          # first-order torque lag time constant (s)
ALPHA     = 1.0 - np.exp(-DT / SERVO_TAU)  # lag filter coefficient

# ── True plant parameters ──────────────────────────────────────
L1,  L2  = 1.0, 0.8       # link lengths (m)
M1,  M2  = 1.0, 0.8       # link masses (kg)
LC1, LC2 = 0.5*L1, 0.5*L2 # CoM distances (m)
I1 = M1 * L1**2 / 12.0
I2 = M2 * L2**2 / 12.0

# ── Nominal (planner) model parameters — intentionally mismatched ──
# M1n != M1, M2n != M2: this is the model error driving s_bar > 0
L1n, L2n   = 1.0, 0.8
M1n, M2n   = 1.1, 0.5     # mismatched masses (paper Sec. VII-B)
LC1n, LC2n = 0.5*L1n, 0.5*L2n
I1n = M1n * L1n**2 / 12.0
I2n = M2n * L2n**2 / 12.0

# ── Task geometry ──────────────────────────────────────────────
GOAL  = torch.tensor([1.35, 0.35], dtype=F32, device=DEV)
OBS_C = torch.tensor([0.85, 0.20], dtype=F32, device=DEV)
OBS_R = torch.tensor(0.15,         dtype=F32, device=DEV)
Q_MIN = torch.tensor([-2.6, -2.6], dtype=F32, device=DEV)
Q_MAX = torch.tensor([ 2.6,  2.6], dtype=F32, device=DEV)
TAU_MAX = 6.0              # torque saturation bound (N·m)

# ── Cost weights (paper Sec. VII-B) ───────────────────────────
W_GOAL     = 70.0
W_QREG     = 0.5
W_VEL      = 0.2
W_CTRL     = 0.008         # control regularization weight W_u; also lower-bounds mu
W_TERM     = 250.0
W_OBS_BASE = 2.0e4         # alpha_0 in eq:penalty_scaling
W_JLIM     = 2.0e3
GAMMA_OBS  = 1.0           # gamma in eq:penalty_scaling
W_VEL_NEAR = 0.0
D_NEAR     = 0.0

# ── RC-MPPI adaptive flags and residual filter ─────────────────
USE_ADAPT = True
W_R_DIAG  = torch.tensor([1.0, 1.0, math.sqrt(0.2), math.sqrt(0.2)], dtype=F32)
RHO       = 0.10           # rho in eqn:fil_statistics (paper Sec. VII-B)

# ── Modulation gains (paper Sec. VII-B, eq:sampling_modulation) ─
KAPPA_R    = 0.15          # obstacle radius inflation gain
DELTA_R_MAX = 0.10         # max radius inflation (m)

KAPPA_SIGMA = 0.50         # sigma contraction gain
SIGMA_MIN   = 0.40
SIGMA_MAX   = 2.00

# beta_k = clip(beta_0*(1 + kappa_beta*s_bar), beta_min, beta_max)
# beta INCREASES with s_bar: higher mismatch -> softer weights
# (see temperature convention note in module docstring)
KAPPA_BETA  = 5.00         # temperature relaxation gain
BETA_MIN    = 0.50
BETA_MAX    = 6.00

# ── Nominal MPPI baseline hyperparameters ──────────────────────
SIGMA_0 = 1.0
BETA_0  = 2.0              # nominal temperature (used when s_bar = 0)

# ── Plant noise parameters ─────────────────────────────────────
B_DAMP      = 0.10         # joint damping (N·m·s/rad)
Q_STD       = 0.002        # position measurement noise (rad)
QD_STD      = 0.010        # velocity measurement noise (rad/s)

# ── Utility functions ──────────────────────────────────────────
def wrap(q):
    return (q + math.pi) % (2*math.pi) - math.pi

def noise_cpu():
    return torch.cat([Q_STD  * torch.randn(2, device=CPU, dtype=F32),
                      QD_STD * torch.randn(2, device=CPU, dtype=F32)])

def fk_ee(q):
    """Forward kinematics to end-effector (nominal model, batched)."""
    return torch.stack([
        L1n*torch.cos(q[:,0]) + L2n*torch.cos(q[:,0]+q[:,1]),
        L1n*torch.sin(q[:,0]) + L2n*torch.sin(q[:,0]+q[:,1])
    ], dim=1)

def inertia_nominal(q):
    """Nominal inertia matrix M(q) for K rollouts."""
    c2  = torch.cos(q[:,1])
    M11 = I1n+I2n + M1n*LC1n**2 + M2n*(L1n**2+LC2n**2+2*L1n*LC2n*c2)
    M12 = I2n + M2n*(LC2n**2 + L1n*LC2n*c2)
    M22 = torch.full_like(M11, I2n + M2n*LC2n**2)
    K   = q.shape[0]
    Mt  = torch.zeros(K, 2, 2, device=q.device, dtype=q.dtype)
    Mt[:,0,0]=M11; Mt[:,0,1]=M12; Mt[:,1,0]=M12; Mt[:,1,1]=M22
    return Mt

def coriolis_nominal(q, qd):
    """Nominal Coriolis/centripetal vector C(q,qd)*qd for K rollouts."""
    h = -M2n*L1n*LC2n*torch.sin(q[:,1])
    return torch.stack([h*(2*qd[:,0]*qd[:,1]+qd[:,1]**2),
                        h*qd[:,0]**2], dim=1)

def step_nominal(x, tau):
    """One-step nominal dynamics rollout (GPU, batched K rollouts)."""
    q, qd = x[:,0:2], x[:,2:4]
    qdd   = torch.linalg.solve(
                inertia_nominal(q),
                (tau - coriolis_nominal(q,qd)).unsqueeze(-1)
            ).squeeze(-1)
    qd_n = qd + DT*qdd
    q_n  = torch.stack([
        torch.remainder(q[:,0]+DT*qd_n[:,0]+math.pi, 2*math.pi)-math.pi,
        torch.remainder(q[:,1]+DT*qd_n[:,1]+math.pi, 2*math.pi)-math.pi
    ], dim=1)
    return torch.cat([q_n, qd_n], dim=1)

def _step_cpu_nominal(x, tau):
    """One-step nominal dynamics on CPU (used for residual prediction)."""
    q, qd = x[0:2], x[2:4]
    c2 = float(torch.cos(q[1]))
    s2 = float(torch.sin(q[1]))
    M11 = I1n+I2n + M1n*LC1n**2 + M2n*(L1n**2+LC2n**2+2*L1n*LC2n*c2)
    M12 = float(I2n + M2n*(LC2n**2 + L1n*LC2n*c2))
    M22 = float(I2n + M2n*LC2n**2)
    Mt  = torch.tensor([[float(M11),M12],[M12,M22]], device=CPU, dtype=F32)
    h   = -M2n*L1n*LC2n*s2
    Cqd = torch.stack([h*(2*qd[0]*qd[1]+qd[1]**2), h*qd[0]**2])
    qdd = torch.linalg.solve(Mt, tau - Cqd)
    qd_n = qd + DT*qdd
    q_n  = torch.tensor([wrap(float(q[0]+DT*qd_n[0])),
                          wrap(float(q[1]+DT*qd_n[1]))], device=CPU, dtype=F32)
    return torch.cat([q_n, qd_n])

def _step_cpu_true(x, tau, damping=0.0):
    """One-step true plant dynamics on CPU (used for closed-loop execution)."""
    q, qd = x[0:2], x[2:4]
    c2 = float(torch.cos(q[1]))
    s2 = float(torch.sin(q[1]))
    M11 = I1+I2 + M1*LC1**2 + M2*(L1**2+LC2**2+2*L1*LC2*c2)
    M12 = float(I2 + M2*(LC2**2 + L1*LC2*c2))
    M22 = float(I2 + M2*LC2**2)
    Mt  = torch.tensor([[float(M11),M12],[M12,M22]], device=CPU, dtype=F32)
    h   = -M2*L1*LC2*s2
    Cqd = torch.stack([h*(2*qd[0]*qd[1]+qd[1]**2), h*qd[0]**2])
    qdd = torch.linalg.solve(Mt, tau - Cqd - damping*qd)
    qd_n = qd + DT*qdd
    q_n  = torch.tensor([wrap(float(q[0]+DT*qd_n[0])),
                          wrap(float(q[1]+DT*qd_n[1]))], device=CPU, dtype=F32)
    return torch.cat([q_n, qd_n])

def predict_nominal(y, u):
    """Nominal one-step prediction f_theta(y, u) used for residual r_k."""
    return _step_cpu_nominal(y, u)

def plant_step(x, tau_cmd, tau_exec):
    """
    True plant step with:
      - torque saturation at TAU_MAX
      - first-order torque lag (servo model, time constant SERVO_TAU)
      - joint damping B_DAMP
      - additive output noise (Q_STD, QD_STD)
    Returns: (y_next, x_next_noiseless, tau_exec_updated)
    """
    tau_sat  = torch.clamp(tau_cmd, -TAU_MAX, TAU_MAX)
    tau_exec = (1-ALPHA)*tau_exec + ALPHA*tau_sat
    x_next   = _step_cpu_true(x, tau_exec, damping=B_DAMP)
    return x_next + noise_cpu(), x_next, tau_exec

# ── Obstacle and joint-limit costs ─────────────────────────────
def fk_links_torch(q):
    """Forward kinematics for both links (nominal model, batched)."""
    p0 = torch.zeros(q.shape[0], 2, device=q.device, dtype=q.dtype)
    p1 = torch.stack([L1n*torch.cos(q[:,0]),
                      L1n*torch.sin(q[:,0])], dim=1)
    p2 = torch.stack([p1[:,0] + L2n*torch.cos(q[:,0]+q[:,1]),
                      p1[:,1] + L2n*torch.sin(q[:,0]+q[:,1])], dim=1)
    return p0, p1, p2

def point_segment_distance_torch(p, a, b):
    """Minimum distance from point p to line segment [a,b] (batched)."""
    ab = b - a
    ap = p[None,:] - a
    t  = torch.sum(ap*ab, dim=1) / (torch.sum(ab*ab, dim=1) + 1e-12)
    t  = torch.clamp(t, 0.0, 1.0)
    return torch.linalg.norm(p[None,:] - (a + t[:,None]*ab), dim=1)

def link_obstacle_distance_torch(q, obs_c):
    """
    Link-aware minimum distance from obstacle center to either link segment.
    This implements the d(q) term in ell_obs(q) = W_obs*max(0, r_eff - d(q))^2.
    """
    p0, p1, p2 = fk_links_torch(q)
    d1 = point_segment_distance_torch(obs_c, p0, p1)
    d2 = point_segment_distance_torch(obs_c, p1, p2)
    return torch.minimum(d1, d2)

def obs_cost(q, r_eff, obs_c, alpha_k):
    """
    Residual-adaptive obstacle barrier cost [eq:barrier_cost]:
      ell_safe(x; s_bar) = alpha_k * phi(h(x) + m(s_bar))
    where h(x) = r - d(q), phi(z) = max(0,z)^2, and r_eff = r + m(s_bar).
    alpha_k = alpha_0*(1 + gamma*s_bar) [eq:penalty_scaling].
    """
    d = link_obstacle_distance_torch(q, obs_c)
    return alpha_k * torch.clamp(r_eff - d, min=0.0)**2

def jlim_cost(q):
    """Quadratic joint-limit penalty."""
    v = (torch.clamp(q - Q_MAX[None,:], min=0)
       + torch.clamp(Q_MIN[None,:] - q, min=0))
    return W_JLIM * torch.sum(v**2, dim=1)

def running_cost(x, tau, r_eff, obs_c, alpha_k):
    """Full running cost Z(U) per step [eq:traj_cost]."""
    q, qd = x[:,0:2], x[:,2:4]
    ee    = fk_ee(q)
    d_g   = torch.linalg.norm(ee - GOAL[None,:], dim=1)
    near  = (d_g < D_NEAR).float()
    return (W_GOAL * d_g**2
          + W_QREG * torch.sum(q**2, dim=1)
          + (W_VEL + W_VEL_NEAR*near) * torch.sum(qd**2, dim=1)
          + W_CTRL * torch.sum(tau**2, dim=1)
          + obs_cost(q, r_eff, obs_c, alpha_k)
          + jlim_cost(q))

def term_cost(x):
    """Terminal cost ell_f(x_{k+N}) [eq:traj_cost]."""
    return W_TERM * torch.sum((fk_ee(x[:,0:2]) - GOAL[None,:])**2, dim=1)

# ── MPPI optimizer ─────────────────────────────────────────────
@torch.no_grad()
def mppi(x0, u_bar, r_eff, obs_c, alpha_k, sigma_eff, beta_eff, K=4096, T=35):
    """
    MPPI importance-sampling update [eq:mppi_weights, eq:mppi_update].

    Weights: w^(i) = exp(-Z^(i) / beta_eff) / sum_j exp(-Z^(j) / beta_eff)

    beta_eff is the temperature from eq:sampling_modulation.
    Larger beta_eff -> softer weights -> conservative averaging over rollouts.
    Smaller beta_eff -> sharper weights -> commit to apparent cost minimum.

    Numerical note: cost -= cost.min() before computing exp() is standard
    MPPI stabilization (shifts partition function, does not change weights).
    """
    eps  = sigma_eff * torch.randn(K, T, 2, device=DEV, dtype=F32)
    x    = x0[None,:].expand(K,-1).clone()
    cost = torch.zeros(K, device=DEV, dtype=F32)
    for t in range(T):
        u    = torch.clamp(u_bar[t][None,:] + eps[:,t,:], -TAU_MAX, TAU_MAX)
        x    = step_nominal(x, u)
        cost += running_cost(x, u, r_eff, obs_c, alpha_k)
    cost += term_cost(x)
    cost -= cost.min()   # numerical stabilization (invariant to weights)
    w     = torch.exp(-cost / beta_eff)
    w    /= w.sum() + 1e-12
    u_new = torch.clamp(u_bar + torch.einsum("k,kti->ti", w, eps), -TAU_MAX, TAU_MAX)
    u0    = u_new[0]
    u_bar_shifted = torch.cat(
        [u_new[1:], torch.zeros(1, 2, device=DEV, dtype=F32)], dim=0)
    return u0, u_bar_shifted

# ── Residual estimation [eq:residual_def, eqn:fil_statistics] ──
def update_sbar(y_k, y_prev, u_prev, s_bar):
    """
    Compute one-step prediction residual and update filtered statistic.
      r_k     = y_k - f_theta(y_{k-1}, u_{k-1})   [eq:residual_def]
      s_k     = ||W_r * r_k||
      s_bar_k = (1-rho)*s_bar_{k-1} + rho*s_k      [eqn:fil_statistics]
    """
    r   = y_k - predict_nominal(y_prev, u_prev)
    s_k = float(torch.linalg.norm(W_R_DIAG.to(y_k.device) * r))
    return r, s_k, (1-RHO)*s_bar + RHO*s_k

# ── RC-MPPI modulation [eq:tightening_def, eq:penalty_scaling,
#                        eq:sampling_modulation] ─────────────────
def modulate(s_bar, sigma0, beta0):
    """
    Compute all residual-adaptive parameters from current s_bar.

    Returns: (margin, sigma_eff, beta_eff, alpha_k)

    margin    = clip(kappa_r * s_bar, 0, Delta_r_max)
      Simulation-level form of eq:tightening_def:
      r_eff = r + margin implements h(x) + m(s_bar) = (r + m(s_bar)) - d(x).

    sigma_eff = clip(sigma_0 / (1 + kappa_sigma * s_bar), sigma_min, sigma_max)
      Perturbation std DECREASES with s_bar: contracts exploration [eq:sampling_modulation].

    beta_eff  = clip(beta_0 * (1 + kappa_beta * s_bar), beta_min, beta_max)
      Temperature INCREASES with s_bar: reflects reduced confidence in
      cost evaluations under model mismatch [eq:sampling_modulation].
      Epistemic interpretation: beta encodes trust in rollout costs.
      Safety maintained because alpha_k*phi(m(s_bar)) ~ O(s_bar^2)
      dominates beta_k ~ O(s_bar) in the unsafe weight ratio bound
      [Lemma 2(ii), Proposition 3, Proposition 4].

    alpha_k   = alpha_0 * (1 + gamma * s_bar)
      Barrier penalty weight INCREASES with s_bar [eq:penalty_scaling].
      Works with beta_eff: even as weights soften, unsafe rollouts incur
      quadratically larger cost, so their total weight -> 0 as s_bar -> inf.
    """
    # Constraint tightening (obstacle radius inflation)
    margin = float(np.clip(KAPPA_R * s_bar, 0.0, DELTA_R_MAX))

    # Perturbation variance: contracts under mismatch [eq:sampling_modulation]
    sigma_eff = float(np.clip(
        sigma0 / (1.0 + KAPPA_SIGMA * s_bar), SIGMA_MIN, SIGMA_MAX))

    # Temperature: INCREASES (softens) under mismatch [eq:sampling_modulation]
    # Higher s_bar -> larger beta -> softer importance weights
    # This is the uncertainty-aware temperature relaxation of Proposition 4
    beta_eff = float(np.clip(
        beta0 * (1.0 + KAPPA_BETA * s_bar), BETA_MIN, BETA_MAX))

    # Barrier penalty scaling [eq:penalty_scaling]
    alpha_k = W_OBS_BASE * (1.0 + GAMMA_OBS * s_bar)

    return margin, sigma_eff, beta_eff, alpha_k

# ── Kinematic utilities (numpy, for logging/plotting) ──────────
def fk_ee_np(q1, q2):
    return np.array([L1*np.cos(q1)+L2*np.cos(q1+q2),
                     L1*np.sin(q1)+L2*np.sin(q1+q2)])

def fk_links_np(q1, q2):
    p0 = np.array([0., 0.])
    p1 = np.array([L1*np.cos(q1), L1*np.sin(q1)])
    p2 = np.array([p1[0]+L2*np.cos(q1+q2), p1[1]+L2*np.sin(q1+q2)])
    return p0, p1, p2

def point_segment_distance_np(p, a, b):
    ab = b - a
    t  = np.dot(p-a, ab) / (np.dot(ab,ab) + 1e-12)
    t  = np.clip(t, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t*ab)))

def link_clearance_np(q1, q2):
    """True link clearance c(q) = d(q) - r (positive = safe)."""
    p0, p1, p2 = fk_links_np(q1, q2)
    c  = OBS_C.cpu().numpy()
    d1 = point_segment_distance_np(c, p0, p1)
    d2 = point_segment_distance_np(c, p1, p2)
    return min(d1, d2) - float(OBS_R.cpu())

def clearance_series(q_traj):
    return np.array([link_clearance_np(q1, q2) for q1, q2 in q_traj])

def ttg(ee, tol):
    d = np.linalg.norm(ee - GOAL.cpu().numpy()[None,:], axis=1)
    h = np.where(d <= tol)[0]
    return int(h[0]) if h.size else -1

def plen(ee):
    return float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ── Closed-loop simulation ─────────────────────────────────────
def simulate(steps=200, K=4096, T=35, tol=0.05, debug=False, ret_ctrl=False):
    x   = torch.zeros(4, device=CPU, dtype=F32)
    y_k = x + noise_cpu()
    y_p = y_k.clone()
    u_p = torch.zeros(2, device=CPU, dtype=F32)
    tex = torch.zeros(2, device=CPU, dtype=F32)
    ub  = torch.zeros(T, 2, device=DEV, dtype=F32)

    q_log   = [y_k[:2].numpy().copy()]
    ee_log  = [fk_ee_np(*y_k[:2].tolist())]
    tau_log = []
    s_bar   = 0.0
    dbg = ({k: [] for k in ["s_bar","r_eff","alpha_k","sigma_eff","beta_eff"]}
           if debug else None)

    for k in range(steps):
        # --- Residual estimation and modulation ---
        if USE_ADAPT and k > 0:
            _, _, s_bar = update_sbar(y_k, y_p, u_p, s_bar)
            margin, sigma_eff, beta_eff, alpha_k = modulate(s_bar, SIGMA_0, BETA_0)
        else:
            # Vanilla MPPI: no adaptation
            margin, sigma_eff, beta_eff, alpha_k = 0.0, SIGMA_0, BETA_0, W_OBS_BASE

        r_eff   = OBS_R + torch.tensor(margin, device=DEV, dtype=F32)
        alpha_t = torch.tensor(alpha_k, device=DEV, dtype=F32)

        # --- MPPI optimization ---
        u0, ub = mppi(y_k.to(DEV), ub, r_eff, OBS_C,
                      alpha_t, sigma_eff, beta_eff, K=K, T=T)
        u_k = u0.to(CPU)

        # --- Plant execution ---
        y_p, u_p = y_k.clone(), u_k.clone()
        y_k, x, tex = plant_step(x, u_k, tex)

        if ret_ctrl:
            tau_log.append(u_k.numpy().copy())
        q_log.append(y_k[:2].numpy().copy())
        ee_log.append(fk_ee_np(*y_k[:2].tolist()))

        if debug:
            dbg["s_bar"].append(s_bar)
            dbg["r_eff"].append(float(r_eff.cpu()))
            dbg["alpha_k"].append(float(alpha_k))
            dbg["sigma_eff"].append(sigma_eff)
            dbg["beta_eff"].append(beta_eff)

    q_t   = np.array(q_log)
    ee_t  = np.array(ee_log)
    tau_t = np.array(tau_log) if ret_ctrl else None
    clr   = clearance_series(q_t)

    t_g          = ttg(ee_t, tol)
    obstacle_hit = int((clr < 0).sum()) > 0
    met = {
        "success": int(t_g >= 0) and not obstacle_hit,
        "t_goal":  t_g if (t_g >= 0 and not obstacle_hit) else steps+1,
        "min_clr": float(clr.min()),
        "viols":   int((clr < 0).sum()),
        "plen":    plen(ee_t),
        "energy":  float(np.sum(tau_t**2)) if tau_t is not None else np.nan,
    }
    if debug:
        dbg = {k: np.array(v) for k, v in dbg.items()}
    return q_t, ee_t, tau_t, met, dbg

# ── Monte Carlo evaluation ─────────────────────────────────────
def mc_eval(n=50, seed0=0, steps=200, K=4096, T=35, tol=0.05):
    """
    Paired-seed Monte Carlo: runs vanilla MPPI and RC-MPPI with identical
    noise realizations so differences are attributable to adaptation.
    """
    global USE_ADAPT
    torch.randn(1, device=DEV)   # warm up RNG
    keys = ["success","t_goal","min_clr","viols","plen","energy"]
    res  = {m: {k: [] for k in keys} for m in ["van","rc"]}
    seeds = []
    for i in tqdm(range(n), desc="MC trials", ncols=75):
        seed = seed0 + i
        seeds.append(seed)
        for flag, tag in [(False,"van"), (True,"rc")]:
            USE_ADAPT = flag
            set_seed(seed)
            *_, met, _ = simulate(steps, K, T, tol, ret_ctrl=True)
            for k in keys:
                res[tag][k].append(float(met[k]))
    for m in ["van","rc"]:
        for k in keys:
            res[m][k] = np.array(res[m][k])
    return res, np.array(seeds)

def best_seed(res, seeds):
    """
    Select representative seed per paper criterion (Sec. VII-B):
    prefer seed where vanilla violates and RC maintains clearance.
    """
    vs  = res["van"]["success"].astype(int)
    rs  = res["rc"]["success"].astype(int)
    vv  = res["van"]["viols"]
    rv  = res["rc"]["viols"]
    vc  = res["van"]["min_clr"]
    rc  = res["rc"]["min_clr"]
    both = np.where((vs==1) & (rs==1))[0]
    if both.size:
        cand = both[(vv[both]>0) & (rv[both]==0)]
        if cand.size:
            i = cand[np.argmax(vv[cand])]
            return int(seeds[i]), int(i), "vanilla violated, RC maintained clearance"
        i = both[np.argmax(rc[both] - vc[both])]
        return int(seeds[i]), int(i), "max clearance margin improvement"
    only = np.where((vs==0) & (rs==1))[0]
    if only.size:
        i = only[np.argmax(rc[only])]
        return int(seeds[i]), int(i), "RC-only success"
    i = int(np.argmax(rc - vc))
    return int(seeds[i]), int(i), "fallback: max clearance delta"

def summarize(res):
    def ms(x): return float(np.mean(x)), float(np.std(x))
    print(f"\n{'Metric':<26} {'Vanilla MPPI':>16} {'RC-MPPI':>16}")
    print("="*60)
    print(f"{'Success rate':<26} "
          f"{np.mean(res['van']['success']):>16.2f} "
          f"{np.mean(res['rc']['success']):>16.2f}")
    for lbl, key in [
        ("Time-to-goal (steps)", "t_goal"),
        ("Min link clearance (m)", "min_clr"),
        ("Violation steps",        "viols"),
        ("EE path length (m)",     "plen"),
        ("Control energy",         "energy"),
    ]:
        vm, vs = ms(res["van"][key])
        am, as_ = ms(res["rc"][key])
        print(f"{lbl:<26} {vm:>7.2f}±{vs:<7.2f}  {am:>7.2f}±{as_:<7.2f}")

def draw_arm(ax, q1, q2, color="k", lw=3, alpha=1.0, zorder=6):
    p0, p1, p2 = fk_links_np(q1, q2)
    for a, b in [(p0,p1),(p1,p2)]:
        ax.plot([a[0],b[0]], [a[1],b[1]],
                color=color, lw=lw, alpha=alpha, zorder=zorder)
    ax.scatter([p1[0],p2[0]], [p1[1],p2[1]],
               c=color, s=25, zorder=zorder+1)

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEV} | tau_max = {TAU_MAX} N·m")
    K = 4096 if DEV.type == "cuda" else 1024

    res, seeds = mc_eval(n=50, K=K)
    summarize(res)

    s_star, i_star, reason = best_seed(res, seeds)
    print(f"\nRepresentative seed: {s_star}  ({reason})")

    replays = {}
    for flag, tag in [(False,"van"), (True,"rc")]:
        USE_ADAPT = flag
        set_seed(s_star)
        q, ee, _, met, dbg = simulate(K=K, debug=True, ret_ctrl=False)
        replays[tag] = dict(q=q, ee=ee, met=met, dbg=dbg)

    # Figure 1: end-effector trajectories (paper Fig. 2)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(*replays["van"]["ee"].T, "r--", lw=2, label="Vanilla MPPI")
    ax.plot(*replays["rc"]["ee"].T,  "b-",  lw=2, label="RC-MPPI")
    ax.scatter(*GOAL.cpu(), c="green", s=120, edgecolors="k",
               zorder=10, label="Goal")
    ax.add_patch(plt.Circle(OBS_C.cpu().numpy(), float(OBS_R),
                            color="red", alpha=0.25))
    draw_arm(ax, *replays["van"]["q"][0],  color="gray", lw=3, alpha=0.7)
    draw_arm(ax, *replays["van"]["q"][-1], color="red",  lw=3, alpha=0.8)
    draw_arm(ax, *replays["rc"]["q"][-1],  color="blue", lw=3, alpha=0.8)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"2R Arm Trajectory + Link Geometry (seed={s_star})")
    plt.tight_layout()
    plt.savefig("fig1_traj.pdf", bbox_inches="tight")

    # Figure 2: link clearance over time
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(clearance_series(replays["van"]["q"]), "r--", lw=2,
            label="Vanilla MPPI")
    ax.plot(clearance_series(replays["rc"]["q"]),  "b-",  lw=2,
            label="RC-MPPI")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Link clearance (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig2_clearance.pdf", bbox_inches="tight")

    # Figure 3: adaptive parameter profiles
    dbg = replays["rc"]["dbg"]
    fig, axes = plt.subplots(4, 1, figsize=(6,7), sharex=True)
    a1, a2, a3, a4 = axes

    a1.plot(dbg["s_bar"], "g-", lw=2)
    a1.set_ylabel(r"Residual $\bar{s}_k$")

    a2.plot(dbg["r_eff"], "b-", lw=2)
    a2.axhline(float(OBS_R), color="k", ls="--", lw=1,
               label=r"$r$ (nominal)")
    a2.set_ylabel(r"$r_\mathrm{eff}$ (m)")
    a2.legend(fontsize=8)

    a3.plot(dbg["sigma_eff"], "m-", lw=2)
    a3.axhline(SIGMA_0, color="k", ls="--", lw=1,
               label=r"$\sigma_0$")
    a3.set_ylabel(r"$\sigma_k$ (contracts $\downarrow$)")
    a3.legend(fontsize=8)

    # beta INCREASES with s_bar — label reflects this
    a4.plot(dbg["beta_eff"], "c-", lw=2)
    a4.axhline(BETA_0, color="k", ls="--", lw=1,
               label=r"$\beta_0$")
    a4.set_ylabel(r"$\beta_k$ (relaxes $\uparrow$)")
    a4.set_xlabel("Step")
    a4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("fig3_modulation_profile.pdf", bbox_inches="tight")