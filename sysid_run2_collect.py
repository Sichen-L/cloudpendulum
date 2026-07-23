# -*- coding: utf-8 -*-
"""RUN 2: collect hardware data for system identification.

Stages:
  free: five passive free-swing releases
  oid : D-optimal excitation, with A for training and B for validation
  hi  : segmented high-speed burst-chirp excitation with velocity guards

This script starts CloudPendulum experiments and requires cloudpendulumclient.
Each trajectory is saved incrementally so interrupted runs keep collected data.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import casadi as ca

from sysid_common import (PARAM_NOMINAL, PARAM_NAMES, FIT_NAMES, data_dir,
                          load_config, mask_token, param_vec, resolve_data, rk4_p)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def make_collector(cfg):
    try:
        from cloudpendulumclient.client import Client
    except ImportError:
        print("ERROR: cloudpendulumclient is not installed. RUN 2 must be run "
              "inside CloudPendulum/JupyterHub or an environment with the client.")
        sys.exit(2)

    USER_TOKEN = cfg["user_token"]
    EXPERIMENT_TYPE = cfg["experiment_type"]
    CELL_ID = int(cfg["cell_id"])
    TAU_MAX_ID = float(cfg["tau_max_id"])
    DQ_ABORT = float(cfg["dq_abort"])
    dt_ctrl = 1.0 / float(cfg["ctrl_hz"])
    INTER_RUN_COOLDOWN = float(cfg["inter_run_cooldown"])
    START_RETRY_MAX = int(cfg["start_retry_max"])

    def start_experiment_with_retry(client, *, duration, initial_position,
                                    record, prep, label):
        # --------------------------------------------------------------------------
        for attempt in range(1, START_RETRY_MAX + 1):
            try:
                return client.start_experiment(
                    user_token=USER_TOKEN, experiment_type=EXPERIMENT_TYPE,
                    experiment_time=duration + 1.0, preparation_time=prep,
                    record=record, initial_position=initial_position, cell_id=CELL_ID)
            except RuntimeError as exc:
                msg = str(exc)
                limited = 'Too many requests' in msg or 'try again in' in msg
                if not limited or attempt >= START_RETRY_MAX:
                    raise
                match = re.search(r'try again in\s+(\d+(?:\.\d+)?)\s+seconds', msg)
                wait = float(match.group(1)) if match else 30.0
                wait = max(wait, INTER_RUN_COOLDOWN) + 2.0
                print(f'[{label}] start_experiment rate-limited, retry '
                      f'{attempt}/{START_RETRY_MAX}; waiting {wait:.0f}s.')
                time.sleep(wait)

    def collect_trajectory(torque_fn, duration, initial_position=None,
                           record=False, label='', prep=3.0):
        assert USER_TOKEN, 'token is empty; cannot collect hardware data.'
        client = Client()
        session, live = start_experiment_with_retry(
            client, duration=duration, initial_position=initial_position,
            record=record, prep=prep, label=label)
        print(f'[{label}] cell_id={CELL_ID}  session={session}')
        if live:
            print('  live:', live)
        client.set_impedance_controller_params(0.0, 0.0, session)

        n = int(duration / dt_ctrl)
        T = np.zeros(n); Q = np.zeros((n, 2)); DQ = np.zeros((n, 2)); U = np.zeros(n)
        aborted = False; k = 0
        t0 = time.perf_counter(); next_step = t0
        try:
            for k in range(n):
                while time.perf_counter() < next_step:
                    pass
                try:
                    q = np.array(client.get_position(session)).flatten()
                    dq = np.array(client.get_velocity(session)).flatten()
                except RuntimeError as exc:
                    client_error = str(exc)
                    velocity_limited = (
                        'Velocity Limit Violation' in client_error
                        or 'outside the limits' in client_error
                    )
                    if not velocity_limited:
                        raise
                    aborted = True
                    print(f'  !! ABORT step {k}: hardware velocity limit ({client_error})')
                    break
                t = time.perf_counter() - t0
                if (not np.all(np.isfinite(q))) or (not np.all(np.isfinite(dq))) \
                   or np.any(np.abs(dq) > DQ_ABORT):
                    try:
                        client.set_torque([0.0, 0.0], session)
                    except Exception as exc:
                        print(f'  zero-torque after abort failed: {exc}')
                    aborted = True
                    print(f'  !! ABORT step {k}: dq={np.round(dq, 1)}'); break
                u = float(np.clip(torque_fn(t, q, dq), -TAU_MAX_ID, TAU_MAX_ID))
                client.set_torque([0.0, u], session)
                T[k] = t; Q[k] = q; DQ[k] = dq; U[k] = u
                next_step = max(next_step + dt_ctrl, time.perf_counter())
            else:
                k = n
        finally:
            try:
                client.set_torque([0.0, 0.0], session)
            except Exception as e:
                print('  zero-torque on stop failed:', e)
            try:
                url = client.stop_experiment(session)
                print('  stopped.', ('data: ' + url) if url else '')
            except Exception as e:
                print('  stop_experiment failed:', e)
        sl = slice(0, k); Ts = T[sl]
        dt_meas = np.diff(Ts)
        ds = {'t': Ts, 'q': Q[sl], 'dq': DQ[sl], 'u': U[sl], 'dt': dt_meas,
              'label': label, 'aborted': aborted}
        dt_msg = (f', dt={1e3 * np.median(dt_meas):.2f}+/-{1e3 * np.std(dt_meas):.2f}ms'
                  if dt_meas.size else '')
        print(f'  collected {k} steps, aborted={aborted}, '
              f'peak|dq|={np.max(np.abs(DQ[sl])) if k else 0:.1f}{dt_msg}')
        return ds

    return collect_trajectory


def save_groups(free_sets, exc_sets, path):
    """Save free/excitation datasets in the notebook-compatible npz layout."""
    payload = {}; manifest = []
    for group, sets in [('free', free_sets), ('exc', exc_sets)]:
        for i, ds in enumerate(sets):
            key = f'{group}_{i:02d}'
            for name in ('t', 'q', 'dq', 'u'):
                payload[f'{key}_{name}'] = np.asarray(ds[name])
            raw_dt = np.asarray(ds['dt'])
            payload[f'{key}_dt'] = (np.diff(np.asarray(ds['t'])) if raw_dt.ndim == 0 else raw_dt)
            manifest.append({'key': key, 'group': group, 'label': str(ds.get('label', key)),
                             'aborted': bool(ds.get('aborted', False))})
    assert manifest, 'no datasets to save'
    payload['manifest_json'] = np.array(json.dumps(manifest, ensure_ascii=False))
    np.savez_compressed(path, **payload)
    return path


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def stage_free(collect, cooldown, out_path, exc_sets):
    free_sets = []
    for q1_0 in [0.6, 1.0, 1.4, 1.9, 2.4]:
        ds = collect(lambda t, q, dq: 0.0, duration=6.0,
                     initial_position=[q1_0, 0.0], record=False,
                     label=f'free_q1={q1_0}')
        free_sets.append(ds)
        save_groups(free_sets, exc_sets, out_path)
        print(f'  incremental save: {len(free_sets)} free + {len(exc_sets)} exc -> {out_path.name}')
        time.sleep(cooldown)
    print(f'free-swing collection complete: {len(free_sets)} datasets.')
    return free_sets


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def design_oid(cfg, tau_max_id):
    dp = resolve_data(cfg, cfg.get("design_params", ""))
    if dp.is_file():
        OID_BASE = dict(PARAM_NOMINAL)
        OID_BASE.update(json.loads(dp.read_text(encoding="utf-8")))
        _bootstrap = True
        print(f'OID base model: {dp.name} (bootstrap with relaxed planning budget)')
    else:
        OID_BASE = dict(PARAM_NOMINAL)
        _bootstrap = False
        print('OID base model: PARAM_NOMINAL (first pass, conservative abort guard)')

    OID_T = 4.0; OID_HZ = 50.0; OID_DT = 1.0 / OID_HZ; OID_N = int(OID_T * OID_HZ)
    OID_F0 = 0.25; OID_K = 6
    DQ_PLAN = 12.0 if _bootstrap else 6.0
    N_ITER = 120
    TAU_MAX_ID = tau_max_id

    theta_names = list(FIT_NAMES)
    theta_nom = np.array([OID_BASE[n] for n in theta_names])

    _coef = ca.MX.sym('coef', 2 * OID_K)
    _th = ca.MX.sym('th', len(theta_names))
    _pl = [ca.MX(v) for v in param_vec(OID_BASE)]
    for j, n in enumerate(theta_names):
        _pl[PARAM_NAMES.index(n)] = _th[j]
    _pv = ca.vertcat(*_pl)

    def _fourier(tk):
        v = 0
        for k in range(OID_K):
            w = 2 * np.pi * OID_F0 * (k + 1)
            v = v + _coef[k] * ca.sin(w * tk) + _coef[OID_K + k] * ca.cos(w * tk)
        return v

    _x = ca.MX.zeros(4)
    _qs = []; _dqs = []; _us = []
    for kk in range(OID_N):
        tk = kk * OID_DT
        u = _fourier(tk)
        _qs.append(_x[0:2]); _dqs.append(_x[2:4]); _us.append(u)
        _x = rk4_p(_x, u, _pv, OID_DT)
    _S = ca.jacobian(ca.vertcat(*_qs), _th)
    F_oid = ca.Function('F_oid', [_coef, _th], [_S, ca.vertcat(*_dqs), ca.vertcat(*_us)])
    print('Sensitivity function built; designing excitation...')

    def eval_design(coef):
        S_, dq_, u_ = F_oid(np.asarray(coef, float), theta_nom)
        S_ = np.array(S_); dq_ = np.array(dq_); u_ = np.array(u_)
        umax = float(np.max(np.abs(u_))) if np.all(np.isfinite(u_)) else np.inf
        if not (np.all(np.isfinite(S_)) and np.all(np.isfinite(dq_))):
            return -1e18, np.inf, umax, np.inf
        dqmax = float(np.max(np.abs(dq_)))
        Ss = S_ * theta_nom[None, :]
        FIM = Ss.T @ Ss
        try:
            sign, logdet = np.linalg.slogdet(FIM + 1e-9 * np.eye(FIM.shape[0]))
            if sign <= 0 or not np.isfinite(logdet):
                logdet = -1e18
        except np.linalg.LinAlgError:
            logdet = -1e18
        try:
            cond = float(np.linalg.cond(FIM))
            if not np.isfinite(cond):
                cond = np.inf
        except np.linalg.LinAlgError:
            cond = np.inf
        return logdet, cond, umax, dqmax

    def scale_design(coef):
        _, _, umax1, _ = eval_design(coef)
        if umax1 <= 1e-9:
            return None, None
        alpha = TAU_MAX_ID / umax1
        m = eval_design(alpha * np.asarray(coef, float))
        for _ in range(6):
            if m[3] <= DQ_PLAN:
                break
            alpha *= 0.8 * DQ_PLAN / m[3]
            m = eval_design(alpha * np.asarray(coef, float))
        return alpha, m

    rng_oid = np.random.default_rng(7)
    base = np.zeros(2 * OID_K); base[:OID_K] = 1.0
    a_b, m_b = scale_design(base)
    print(f'Baseline excitation: logdet={m_b[0]:+.2f}  cond={m_b[1]:.1e}  '
          f'max|tau|={m_b[2] * 1e3:.0f}mNm  max|dq|={m_b[3]:.1f}')

    c = base.copy(); best_c = a_b * base; best = m_b; best_val = m_b[0]; step = 0.6
    for it in range(N_ITER):
        cand = c + rng_oid.normal(0, step, c.shape)
        a, m = scale_design(cand)
        if m is None:
            continue
        if m[0] > best_val:
            best_val = m[0]; best = m; best_c = a * cand; c = cand
        step *= 0.99
    print(f'Optimized excitation: logdet={best[0]:+.2f}  cond={best[1]:.1e}  '
          f'max|tau|={best[2] * 1e3:.0f}mNm  max|dq|={best[3]:.1f}')
    print(f'Information gain: delta logdet={best[0] - m_b[0]:+.2f}  condition {m_b[1]:.1e} -> {best[1]:.1e}')

    # --------------------------------------------------------------------------
    OID_PHASE_SHIFT = 0.73
    shift_c = np.zeros_like(best_c)
    for k in range(OID_K):
        phi = 2 * np.pi * OID_F0 * (k + 1) * OID_PHASE_SHIFT
        a, b = best_c[k], best_c[OID_K + k]
        shift_c[k] = a * np.cos(phi) - b * np.sin(phi)
        shift_c[OID_K + k] = a * np.sin(phi) + b * np.cos(phi)
    alpha_b, metrics_b = scale_design(shift_c)
    assert alpha_b is not None and np.isfinite(metrics_b[3]), 'OID-B safe scaling failed.'
    best_c_b = alpha_b * shift_c
    print(f'OID-B prediction check: max|tau|={metrics_b[2] * 1e3:.0f}mNm, max|dq|={metrics_b[3]:.1f}rad/s')

    def torque_A(t, q=None, dq=None):
        v = 0.0
        for k in range(OID_K):
            w = 2 * np.pi * OID_F0 * (k + 1)
            v += best_c[k] * np.sin(w * t) + best_c[OID_K + k] * np.cos(w * t)
        return float(v)

    def torque_B(t, q=None, dq=None):
        v = 0.0
        for k in range(OID_K):
            w = 2 * np.pi * OID_F0 * (k + 1)
            v += best_c_b[k] * np.sin(w * t) + best_c_b[OID_K + k] * np.cos(w * t)
        return float(v)

    return torque_A, torque_B, OID_T


def stage_oid(cfg, collect, cooldown, out_path, free_sets, exc_sets):
    torque_A, torque_B, OID_T = design_oid(cfg, float(cfg["tau_max_id"]))
    for label, torque_fn in [('oid_train', torque_A), ('oid_valid', torque_B)]:
        exc_sets.append(collect(torque_fn, duration=OID_T,
                                initial_position=[0.0, 0.0], record=False, label=label))
        save_groups(free_sets, exc_sets, out_path)
        print(f'  incremental save: {len(free_sets)} free + {len(exc_sets)} exc -> {out_path.name}')
        time.sleep(cooldown)
    print(f'OID collection complete: {len(exc_sets)} excitation datasets.')
    return exc_sets


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def stage_hi(cfg, collect, cooldown, out_hi_path):
    TAU_MAX_ID = float(cfg["tau_max_id"])
    LONG_HI_RUNS = 10
    LONG_HI_SEG_T = 3.0
    # Conservative high-speed coverage. The older 16 rad/s target repeatedly
    # overshot to 40-50 rad/s on hardware before software could react.
    LONG_HI_TARGET_DQ = 10.0
    LONG_HI_DESIGN_HZ = 80.0
    LONG_HI_RAMP_START = 12.0
    LONG_HI_SOFT_LIMIT = 18.0
    LONG_HI_BRAKE_GAIN = 0.020
    LONG_HI_ABORT_STREAK_LIMIT = 1

    dp = resolve_data(cfg, cfg.get("design_params", ""))
    params = dict(PARAM_NOMINAL)
    if dp.is_file():
        params.update(json.loads(dp.read_text(encoding="utf-8")))
        print(f'Design model: {dp.name}')
    else:
        print('Design model: PARAM_NOMINAL (design_params file does not exist)')

    def make_burst_chirp(amp, f0, f1, phase, side_phase, duration=LONG_HI_SEG_T):
        def torque(t, q=None, dq=None):
            tt = float(np.clip(t, 0.0, duration))
            envelope = np.sin(np.pi * tt / duration) ** 2
            chirp_phase = 2 * np.pi * (f0 * tt + (f1 - f0) * tt * tt / (2 * duration)) + phase
            side = 0.28 * np.sin(2 * np.pi * 0.55 * tt + side_phase)
            return float(amp * envelope * (np.sin(chirp_phase) + side))
        return torque

    def with_velocity_guard(raw_torque, runtime_scale=1.0):
        def guarded(t, q=None, dq=None):
            u = runtime_scale * raw_torque(t, q, dq)
            if dq is not None:
                speed = float(np.max(np.abs(dq)))
                if speed >= LONG_HI_RAMP_START:
                    span = max(1e-6, LONG_HI_SOFT_LIMIT - LONG_HI_RAMP_START)
                    u *= float(np.clip((LONG_HI_SOFT_LIMIT - speed) / span, 0.0, 1.0))
                if speed >= LONG_HI_SOFT_LIMIT:
                    u += -LONG_HI_BRAKE_GAIN * np.tanh(float(dq[1]) / 8.0)
            return float(np.clip(u, -TAU_MAX_ID, TAU_MAX_ID))
        return guarded

    def predict_peak_dq(torque_fn, params, duration=LONG_HI_SEG_T):
        pv = ca.DM(param_vec(params))
        dt = 1.0 / LONG_HI_DESIGN_HZ
        n = int(round(duration / dt))
        x = ca.DM.zeros(4)
        peak = 0.0
        for k in range(n):
            u = float(np.clip(torque_fn(k * dt), -TAU_MAX_ID, TAU_MAX_ID))
            x = rk4_p(x, u, pv, dt)
            peak = max(peak, float(ca.mmax(ca.fabs(x[2:4]))))
        return peak

    def calibrate_amplitude(f0, f1, phase, side_phase, params, target=LONG_HI_TARGET_DQ):
        lo, hi = 0.0, TAU_MAX_ID
        for _ in range(26):
            amp = 0.5 * (lo + hi)
            trial = make_burst_chirp(amp, f0, f1, phase, side_phase)
            if predict_peak_dq(trial, params) > target:
                hi = amp
            else:
                lo = amp
        return lo

    bands = [
        (1.15, 2.45), (2.45, 1.15),
        (1.35, 2.85), (2.85, 1.35),
        (1.65, 3.10), (3.10, 1.65),
        (1.25, 2.65), (2.65, 1.25),
        (1.50, 2.35), (2.35, 1.50),
    ]
    rng = np.random.default_rng(203)
    plan = []
    for i in range(LONG_HI_RUNS):
        f0, f1 = bands[i % len(bands)]
        phase = float(rng.uniform(0, 2 * np.pi))
        side_phase = float(rng.uniform(0, 2 * np.pi))
        amp = calibrate_amplitude(f0, f1, phase, side_phase, params)
        raw = make_burst_chirp(amp, f0, f1, phase, side_phase)
        predicted = predict_peak_dq(raw, params)
        plan.append(dict(label=f'oid_hi_long_{i:02d}', f0=f0, f1=f1,
                         phase=phase, side_phase=side_phase, amp=amp,
                         predicted_peak_dq=predicted, raw_torque=raw))

    total_time = LONG_HI_RUNS * LONG_HI_SEG_T
    print(f'Long high-speed plan: {LONG_HI_RUNS} segments x {LONG_HI_SEG_T:.1f}s = {total_time:.1f}s')
    print(f'Target predicted peak|dq|={LONG_HI_TARGET_DQ:.1f} rad/s; '
          f'soft guard {LONG_HI_RAMP_START:.1f}->{LONG_HI_SOFT_LIMIT:.1f} rad/s')
    for item in plan:
        print(f"  {item['label']}: {item['f0']:.2f}->{item['f1']:.2f} Hz, "
              f"amp={1e3 * item['amp']:.1f} mNm, pred peak|dq|={item['predicted_peak_dq']:.1f} rad/s")

    hi_long_sets = []
    runtime_scale = 1.0
    abort_streak = 0
    for i, item in enumerate(plan):
        torque_fn = with_velocity_guard(item['raw_torque'], runtime_scale=runtime_scale)
        print(f"\nStart {item['label']} ({i + 1}/{len(plan)}), runtime_scale={runtime_scale:.2f}")
        ds = collect(torque_fn, duration=LONG_HI_SEG_T,
                     initial_position=[0.0, 0.0], record=False, label=item['label'])
        hi_long_sets.append(ds)

        peak = float(np.max(np.abs(ds['dq']))) if len(ds['dq']) else 0.0
        if ds.get('aborted', False):
            abort_streak += 1
            runtime_scale *= 0.55
            print(f'  Segment aborted; next runtime_scale={runtime_scale:.2f}')
        else:
            abort_streak = 0
            if peak < 0.75 * LONG_HI_TARGET_DQ:
                runtime_scale = min(1.10, runtime_scale * 1.05)

        save_groups([], hi_long_sets, out_hi_path)
        print(f'  Incremental save: {len(hi_long_sets)} segments -> {out_hi_path.name}')

        if abort_streak >= LONG_HI_ABORT_STREAK_LIMIT:
            print(f'{abort_streak} consecutive aborted segments; stopping early.')
            break
        if i < len(plan) - 1:
            time.sleep(cooldown)
    return hi_long_sets


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RUN 2: hardware collection; starts experiments and is rate-limited")
    ap.add_argument("--stage", choices=["all", "free", "oid", "hi"], default="all")
    ap.add_argument("--out-prefix", default=None,
                    help="output filename prefix; default system_id_data_{cell}_{timestamp}")
    args = ap.parse_args()

    cfg = load_config(need_token=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = args.out_prefix or f"system_id_data_{cfg['cell_id']}_{stamp}"
    ddir = data_dir(cfg)
    out_main = ddir / f"{prefix}.npz"
    out_hi = ddir / f"{prefix}_hi.npz"
    for p in ((out_main,) if args.stage in ("all", "free", "oid") else ()) + \
             ((out_hi,) if args.stage in ("all", "hi") else ()):
        if p.exists():
            print(f"ERROR: {p.name} already exists; refusing to overwrite. Use --out-prefix.")
            return 1

    cooldown = float(cfg["inter_run_cooldown"])
    print(f"token source: {cfg['_token_source']}  token: {mask_token(cfg['user_token'])}")
    print(f"device cell_id={cfg['cell_id']}  tau_max_id={cfg['tau_max_id']}  "
          f"dq_abort={cfg['dq_abort']}  ctrl {cfg['ctrl_hz']:.0f}Hz")
    print(f"stage: {args.stage}   output: {out_main.name}" +
          (f" + {out_hi.name}" if args.stage in ("all", "hi") else ""))

    collect = make_collector(cfg)
    free_sets, exc_sets = [], []
    if args.stage in ("all", "free"):
        free_sets = stage_free(collect, cooldown, out_main, exc_sets)
    if args.stage in ("all", "oid"):
        exc_sets = stage_oid(cfg, collect, cooldown, out_main, free_sets, exc_sets)
    if args.stage in ("all", "hi"):
        stage_hi(cfg, collect, cooldown, out_hi)

    print("\n=== collection finished ===")
    made = [p.name for p in (out_main, out_hi) if p.exists()]
    print("generated files:", ", ".join(made) if made else "(none)")
    print("next: put these filenames into sysid_config.json fit_data, or pass them directly:")
    print(f"  python sysid_run3_fit.py " + " ".join(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
