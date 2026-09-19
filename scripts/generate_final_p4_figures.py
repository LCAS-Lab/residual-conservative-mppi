#!/usr/bin/env python3
"""
Generate final P4 manuscript figures from frozen school-GPU audit artifacts.

Expected inputs (default locations under repo root):
  reproducibility/lti_accaudit_school_gpu_20260917/
  reproducibility/2r_accaudit_school_gpu_20260917/

Outputs (default in repo root):
  fig1_trajectory_seed16.pdf
  fig1_traj.pdf
  fig2_clearance.pdf

The script is intentionally standalone so figures can be regenerated from the
stored reproducibility records without rerunning simulations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


def _load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _first_present(d: dict, candidates: Sequence[str]):
    for key in candidates:
        if key in d:
            return d[key]
    return None


def _find_array_by_label(
    d: dict,
    label_tokens: Sequence[str],
    ndim: int | None = None,
    min_last_dim: int | None = None,
):
    """Best-effort match for saved arrays with flexible key names."""
    scored = []
    for k, arr in d.items():
        if not isinstance(arr, np.ndarray):
            continue
        if ndim is not None and arr.ndim != ndim:
            continue
        if min_last_dim is not None:
            if arr.ndim < 1 or arr.shape[-1] < min_last_dim:
                continue
        lk = k.lower()
        score = sum(tok in lk for tok in label_tokens)
        if score:
            scored.append((score, k, arr))
    if not scored:
        return None, None
    scored.sort(key=lambda x: (-x[0], x[1]))
    _, key, arr = scored[0]
    return key, arr


def _ensure_2d_state(arr: np.ndarray, name: str, min_dim: int):
    if arr is None:
        raise KeyError(f"Could not locate array for {name}.")
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < min_dim:
        raise ValueError(
            f"Array '{name}' has shape {arr.shape}, expected (T,{min_dim}+)."
        )
    return arr


def load_lti_states(npz_path: Path):
    d = _load_npz(npz_path)

    van = _first_present(
        d,
        [
            "vanilla",
            "x_van",
            "x_vanilla",
            "van_x",
            "vanilla_x",
            "states_van",
            "states_vanilla",
            "traj_van",
            "traj_vanilla",
            "x_true_van",
            "state_van",
        ],
    )
    rc = _first_present(
        d,
        [
            "adaptive",
            "x_rc",
            "x_adaptive",
            "rc_x",
            "adaptive_x",
            "states_rc",
            "states_adaptive",
            "traj_rc",
            "traj_adaptive",
            "x_true_rc",
            "x_true_adaptive",
            "state_rc",
            "state_adaptive",
        ],
    )

    if van is None:
        _, van = _find_array_by_label(d, ["van"], ndim=2, min_last_dim=4)
    if rc is None:
        _, rc = _find_array_by_label(d, ["adaptive", "rc"], ndim=2, min_last_dim=4)

    van = _ensure_2d_state(van, "LTI vanilla position trajectory", 2)
    rc = _ensure_2d_state(rc, "LTI RC position trajectory", 2)
    return van, rc


def plot_lti(van: np.ndarray, rc: np.ndarray, outpath: Path):
    obs_c = np.array([2.5, 0.0])
    obs_r = 1.5
    goal = np.array([5.0, 0.0])
    start = van[0, :2]

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    circle = plt.Circle(obs_c, obs_r, fill=False, linestyle="--", linewidth=1.5)
    ax.add_patch(circle)
    ax.plot(van[:, 0], van[:, 1], "r--", linewidth=2, label="Vanilla MPPI")
    ax.plot(rc[:, 0], rc[:, 1], "b-", linewidth=2, label="RC-MPPI")
    ax.plot(start[0], start[1], marker="o", markersize=6, label="Start")
    ax.plot(goal[0], goal[1], marker="*", markersize=10, label="Goal")

    ax.set_xlabel("$p_x$ (m)")
    ax.set_ylabel("$p_y$ (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    xs = np.concatenate(
        [
            van[:, 0],
            rc[:, 0],
            [obs_c[0] - obs_r, obs_c[0] + obs_r, goal[0]],
        ]
    )
    ys = np.concatenate(
        [
            van[:, 1],
            rc[:, 1],
            [obs_c[1] - obs_r, obs_c[1] + obs_r, goal[1]],
        ]
    )
    padx = 0.3 * max(1.0, xs.max() - xs.min()) / 5.0
    pady = 0.3 * max(1.0, ys.max() - ys.min()) / 5.0
    ax.set_xlim(xs.min() - padx, xs.max() + padx)
    ax.set_ylim(ys.min() - pady, ys.max() + pady)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def fk_2r(q: np.ndarray, L1: float = 1.0, L2: float = 0.8):
    q = np.asarray(q)
    q1 = q[..., 0]
    q2 = q[..., 1]
    elbow = np.stack([L1 * np.cos(q1), L1 * np.sin(q1)], axis=-1)
    ee = np.stack(
        [
            L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
            L1 * np.sin(q1) + L2 * np.sin(q1 + q2),
        ],
        axis=-1,
    )
    return elbow, ee


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 1e-12:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def link_clearance_series(
    q_hist: np.ndarray,
    obs_center=(0.85, 0.20),
    obs_radius=0.15,
):
    obs = np.asarray(obs_center, dtype=float)
    elbow, ee = fk_2r(np.asarray(q_hist))
    base = np.zeros(2)
    vals = []
    for e, tip in zip(elbow, ee):
        d1 = point_segment_distance(obs, base, e)
        d2 = point_segment_distance(obs, e, tip)
        vals.append(min(d1, d2) - obs_radius)
    return np.asarray(vals)


def load_2r_q(npz_path: Path):
    d = _load_npz(npz_path)
    van = _first_present(
        d,
        [
            "van_q_true",
            "q_true_van",
            "q_van",
            "q_vanilla",
            "van_q",
            "vanilla_q",
            "qhist_van",
            "qhist_vanilla",
            "q_true_vanilla",
        ],
    )
    rc = _first_present(
        d,
        [
            "rc_q_true",
            "q_true_rc",
            "q_true_adaptive",
            "q_rc",
            "q_adaptive",
            "rc_q",
            "adaptive_q",
            "qhist_rc",
            "qhist_adaptive",
        ],
    )

    if van is None:
        _, van = _find_array_by_label(d, ["q", "van"], ndim=2, min_last_dim=2)
    if rc is None:
        _, rc = _find_array_by_label(d, ["q", "rc"], ndim=2, min_last_dim=2)

    van = _ensure_2d_state(van, "2R vanilla joint history", 2)
    rc = _ensure_2d_state(rc, "2R RC joint history", 2)
    return van, rc


def plot_2r_traj(q_van: np.ndarray, q_rc: np.ndarray, outpath: Path):
    _, ee_van = fk_2r(q_van)
    _, ee_rc = fk_2r(q_rc)

    obs_c = np.array([0.85, 0.20])
    obs_r = 0.15
    goal = np.array([1.35, 0.35])
    start = ee_van[0]

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    circle = plt.Circle(obs_c, obs_r, fill=True, alpha=0.15)
    ax.add_patch(circle)
    ax.plot(ee_van[:, 0], ee_van[:, 1], "r--", linewidth=2, label="Vanilla MPPI")
    ax.plot(ee_rc[:, 0], ee_rc[:, 1], "b-", linewidth=2, label="RC-MPPI")
    ax.plot(start[0], start[1], marker="o", markersize=6, label="Start")
    ax.plot(goal[0], goal[1], marker="*", markersize=10, label="Goal")

    for q_hist, style in [(q_van, "r--"), (q_rc, "b-")]:
        idxs = np.linspace(0, len(q_hist) - 1, 4, dtype=int)
        for idx in idxs:
            elbow, ee = fk_2r(q_hist[idx : idx + 1])
            elbow = elbow[0]
            ee = ee[0]
            ax.plot(
                [0, elbow[0], ee[0]],
                [0, elbow[1], ee[1]],
                style,
                linewidth=0.8,
                alpha=0.25,
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    xs = np.concatenate(
        [
            ee_van[:, 0],
            ee_rc[:, 0],
            [obs_c[0] - obs_r, obs_c[0] + obs_r, goal[0], 0],
        ]
    )
    ys = np.concatenate(
        [
            ee_van[:, 1],
            ee_rc[:, 1],
            [obs_c[1] - obs_r, obs_c[1] + obs_r, goal[1], 0],
        ]
    )
    ax.set_xlim(xs.min() - 0.1, xs.max() + 0.1)
    ax.set_ylim(ys.min() - 0.1, ys.max() + 0.1)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_2r_clearance(q_van: np.ndarray, q_rc: np.ndarray, outpath: Path):
    clr_van = link_clearance_series(q_van)
    clr_rc = link_clearance_series(q_rc)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(np.arange(len(clr_van)), clr_van, "r--", linewidth=2, label="Vanilla MPPI")
    ax.plot(np.arange(len(clr_rc)), clr_rc, "b-", linewidth=2, label="RC-MPPI")
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Link clearance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--lti-dir",
        type=Path,
        default=None,
        help="Override LTI reproducibility directory",
    )
    parser.add_argument(
        "--arm-dir",
        type=Path,
        default=None,
        help="Override 2R reproducibility directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output PDFs (default: repo root)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    lti_dir = (
        args.lti_dir.resolve()
        if args.lti_dir
        else repo_root / "reproducibility" / "lti_accaudit_school_gpu_20260917"
    )
    arm_dir = (
        args.arm_dir.resolve()
        if args.arm_dir
        else repo_root / "reproducibility" / "2r_accaudit_school_gpu_20260917"
    )
    out_dir = args.out_dir.resolve() if args.out_dir else repo_root
    out_dir.mkdir(parents=True, exist_ok=True)

    lti_npz = next(
        iter(sorted(lti_dir.glob("representative_seed*_trajectories.npz"))),
        None,
    )
    arm_npz = next(
        iter(sorted(arm_dir.glob("representative_seed*_trajectories.npz"))),
        None,
    )
    if lti_npz is None:
        raise FileNotFoundError(
            f"No representative trajectory NPZ found in {lti_dir}"
        )
    if arm_npz is None:
        raise FileNotFoundError(
            f"No representative trajectory NPZ found in {arm_dir}"
        )

    x_van, x_rc = load_lti_states(lti_npz)
    q_van, q_rc = load_2r_q(arm_npz)

    outputs = [
        out_dir / "fig1_trajectory_seed16.pdf",
        out_dir / "fig1_traj.pdf",
        out_dir / "fig2_clearance.pdf",
    ]

    plot_lti(x_van, x_rc, outputs[0])
    plot_2r_traj(q_van, q_rc, outputs[1])
    plot_2r_clearance(q_van, q_rc, outputs[2])

    print("Generated:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
