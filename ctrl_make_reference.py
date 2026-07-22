# -*- coding: utf-8 -*-
"""RUN R: generate offline swing-up reference trajectories.

The control scripts load references but do not generate them online. This script
solves the reference OCP from the identified parameter JSON and writes parameter
SHA256 provenance into the NPZ file so stale references are rejected after a
device or parameter change. It requires CasADi/IPOPT only; no acados or hardware
session is needed.

Examples:
  python ctrl_make_reference.py
  python ctrl_make_reference.py --plant pendubot
  python ctrl_make_reference.py --params my_params.json
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import time
from pathlib import Path

import numpy as np
import casadi as ca

from sysid_common import load_config, resolve_data, data_dir
from ctrl_plants import (HERE, PLANTS, REF_T, REF_N, REF_DQ_LIMIT, ref_horizon,
                         coarse_rk4, load_plant_params, make_rhs,
                         save_reference, swingup_ocp)

# Reference OCPs that ride the dq limit (aggressive swing-up) can stall IPOPT at
# its acceptable tolerance (~1e-4 constraint violation) rather than the full 1e-6.
# A defect of ~1e-4 rad is physically negligible (<0.01 deg / 1e-4 rad/s over a
# 20 ms step) and the NMPC re-solves online anyway; gross model errors are orders
# of magnitude larger (>=1e-2). Gate at 5e-4 to accept acceptable-tol solutions
# while still rejecting a genuinely wrong model.
DEFECT_ACCEPT_TOL = 5e-4


def _stretch_guess(Xold, Uold, Told, T, N):
    # --------------------------------------------------------------------------
    tnew = np.linspace(0.0, T, N + 1)
    old_q = tnew * (Told / T)
    grid = np.linspace(0.0, Told, Xold.shape[1])
    Xg = np.vstack([np.interp(old_q, grid, Xold[r]) for r in range(4)])
    Xg[2:] *= Told / T
    ugrid = np.linspace(0.0, Told, Uold.shape[1], endpoint=False)
    Ug = np.vstack([np.interp(tnew[:-1] * (Told / T), ugrid, Uold[r]) for r in range(2)])
    return Xg, Ug


def _load_warm(p):
    """Load a warm-start trajectory, zero-padding a single-input (passive-joint) U."""
    d = np.load(p)
    U = np.atleast_2d(d['U'])
    if U.shape[0] == 1:
        U = np.vstack([U, np.zeros_like(U)])   # single input -> pad passive row with 0
    return d['X'], U, float(d['T'])


def find_warm_start(plant, cfg):
    """Return an old reference to seed the OCP (IPOPT initial guess only; the final
    trajectory is always re-solved with the device's real parameters).

    Search order, most-specific first:
      1. this device's own reference (data dir / compensation_controller),
         EXCEPT the exact output file we are about to overwrite;
      2. pendubot: main-project single-input shoulder references for this cell;
      3. the canonical device-agnostic seed shipped with the pipeline
         (swingup_reference_seed_{plant}.npz).

    The cold two-stage solve is not reliable for the underactuated swing-up (the
    solution sits in a narrow feasible basin), so the canonical seed is what makes
    generation work on a fresh device with no prior reference. Acrobot and pendubot
    seeds are never exchanged because their actuated joints differ.

    The output file itself is never used as its own seed: a stale/foreign-parameter
    reference sitting at that path would poison regeneration (IPOPT stalls warm-
    starting from it) and mask the reliable canonical seed. Regeneration must be
    deterministic regardless of what leftover output happens to be on disk.
    """
    out_path = resolve_data(cfg, PLANTS[plant]['ref_file'].format(cell=cfg['cell_id'])).resolve()
    names = [PLANTS[plant]['ref_file'].format(cell=cfg['cell_id'])]
    cands = [resolve_data(cfg, n) for n in names]
    cands += [HERE.parent / 'compensation_controller' / n for n in names]
    for p in cands:
        if p.resolve() == out_path:
            continue  # never seed from the file being (re)generated
        if p.exists() and np.atleast_2d(np.load(p)['U']).shape[0] == 2:
            print(f'  warm start <- {p}')
            return _load_warm(p)
    if plant == 'pendubot':
        for name in (f'swingup_reference_{cfg["cell_id"]}_pendubot_cap13.npz',
                     f'swingup_reference_{cfg["cell_id"]}_pendubot.npz',
                     f'swingup_reference_{cfg["cell_id"]}_pendubot_gentle.npz'):
            for p in (resolve_data(cfg, name), HERE.parent / name):
                if p.exists() and np.atleast_2d(np.load(p)['U']).shape[0] == 1:
                    print(f'  warm start <- {p} (single input, zero-padded elbow row)')
                    return _load_warm(p)
    # Canonical device-agnostic seed (ships with the pipeline). Checked last so a
    # device-specific reference always wins, but present so a fresh device can bootstrap.
    seed_name = f'swingup_reference_seed_{plant}.npz'
    for p in (resolve_data(cfg, seed_name), HERE / seed_name, HERE.parent / seed_name):
        if p.exists():
            print(f'  warm start <- {p} (canonical {plant} seed)')
            return _load_warm(p)
    return None


def accept(tag, X, U, spec, rk4_np):
    """Validate reference dynamics, bounds, terminal state, and saturation."""
    N = U.shape[1]
    spec_T, spec_N = ref_horizon(spec)
    dt = spec_T / spec_N
    defect = 0.0
    for k in range(N):
        defect = max(defect, float(np.max(np.abs(rk4_np(X[:, k], U[:, k], dt) - X[:, k + 1]))))

    peak_tau1_mnm = np.max(np.abs(U[0])) * 1e3
    peak_tau2_mnm = np.max(np.abs(U[1])) * 1e3
    peak_dq = np.max(np.abs(X[2:]))
    terminal_q1_deg = np.degrees(abs((X[0, -1] - np.pi + np.pi) % (2 * np.pi) - np.pi))

    active = U[spec['active_joint']]
    cap = (spec['TAU1_MAX'], spec['TAU_MAX'])[spec['active_joint']]
    pinned = np.abs(active) >= 0.98 * cap
    longest = current = 0
    for is_pinned in pinned:
        current = current + 1 if is_pinned else 0
        longest = max(longest, current)
    flat_top_ms = longest * dt * 1e3

    ok = (
        defect < DEFECT_ACCEPT_TOL
        and peak_tau1_mnm <= spec['TAU1_MAX'] * 1e3 + 1e-6
        and peak_tau2_mnm <= spec['TAU_MAX'] * 1e3 + 1e-6
        and peak_dq <= REF_DQ_LIMIT + 1e-6
        and terminal_q1_deg < 0.01
    )
    note = '' if defect < 1e-8 else ' (> 1e-8: IPOPT stopped at acceptable tolerance)'
    print(f'  [{tag}] defect={defect:.1e}{note}  '
          f'peak_tau1={peak_tau1_mnm:.1f}mNm peak_tau2={peak_tau2_mnm:.1f}mNm  '
          f'peak|dq|={peak_dq:.2f}  terminal|q1-pi|={terminal_q1_deg:.4f}deg')
    print(f'  [{tag}] active-joint saturation {pinned.sum()}/{N}={100 * pinned.mean():.1f}%  '
          f'longest flat-top {flat_top_ms:.0f}ms')
    if not ok:
        print(f'  [{tag}] gates: defect<{DEFECT_ACCEPT_TOL:.1e}, '
              f'tau1<={spec["TAU1_MAX"] * 1e3:.1f}mNm, '
              f'tau2<={spec["TAU_MAX"] * 1e3:.1f}mNm, '
              f'|dq|<={REF_DQ_LIMIT:.1f}, terminal<0.01deg')
    return ok
def main():
    ap = argparse.ArgumentParser(description='RUN R: generate offline swing-up references')
    ap.add_argument('--plant', choices=['pendubot', 'acrobot', 'both'], default='both')
    ap.add_argument('--params', default=None,
                    help='identified-parameter JSON; defaults to latest identified_params_{cell}_fit_*.json')
    ap.add_argument('--force', action='store_true', help='overwrite existing reference files')
    args = ap.parse_args()

    cfg = load_config(need_token=False)
    pfile = args.params
    if pfile is None:
        ddir = data_dir(cfg)
        candidates = sorted(ddir.glob(f"identified_params_{cfg['cell_id']}_fit_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            candidates = sorted(ddir.glob(f"identified_params_{cfg['cell_id']}_*.json"),
                                key=lambda p: p.stat().st_mtime)
        if candidates:
            pfile = str(candidates[-1])
        else:
            pfile = cfg.get('ctrl_params') or cfg.get('design_params')
    if not pfile:
        print(f"ERROR: no identified parameter JSON found for cell_id={cfg['cell_id']}."
              "Run sysid_workflow.ipynb first.")
        return 1
    P = load_plant_params(resolve_data(cfg, pfile))
    print(f'params: {P["_file"]}  sha256={P["_sha256"][:12]}...')
    _, rk4_step, rk4_np = make_rhs(P)

    plants = ['pendubot', 'acrobot'] if args.plant == 'both' else [args.plant]
    stamp = time.strftime('%Y%m%d-%H%M%S')
    failures = 0
    for plant in plants:
        spec = PLANTS[plant]
        out = resolve_data(cfg, spec['ref_file'].format(cell=cfg['cell_id']))
        print(f'\n== {spec["title"]} -> {out.name} ==')
        if out.exists() and not args.force:
            print('  already exists; skipping. Use --force to regenerate.')
            continue

        T, N = ref_horizon(spec)
        try:
            ws = find_warm_start(plant, cfg)
            X = U = None
            if ws is not None:
                Xg, Ug = _stretch_guess(ws[0], ws[1], ws[2], T, N)
                try:
                    t0 = time.perf_counter()
                    X, U = swingup_ocp(rk4_step, spec, T=T, N=N, Xg=Xg, Ug=Ug, eps_vel=P['eps'])
                    print(f'  warm-start solve converged in {time.perf_counter() - t0:.1f}s')
                except Exception as e:
                    print(f'  warm start failed ({type(e).__name__}); falling back to two-stage solve...')
            if X is None:
                t0 = time.perf_counter()
                Xg, Ug = swingup_ocp(lambda x, u, dt: coarse_rk4(x, u, dt, P['eps']), spec,
                                     T=T, N=N, eps_vel=P['eps'])
                X, U = swingup_ocp(rk4_step, spec, T=T, N=N, Xg=Xg, Ug=Ug, eps_vel=P['eps'])
                print(f'  two-stage solve converged in {time.perf_counter() - t0:.1f}s')
        except Exception as e:
            print(f'  FAIL {plant} solve failed: {type(e).__name__}: {e}')
            failures += 1
            continue

        if not accept(plant, X, U, spec, rk4_np):
            print(f'  FAIL acceptance check failed; not saving {out.name}')
            failures += 1
            continue
        save_reference(out, X, U, spec, P, stamp)
        print(f'  saved {out} (notebook-compatible keys plus params_sha256 provenance)')

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

