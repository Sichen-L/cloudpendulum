# -*- coding: utf-8 -*-
"""Shared CloudPendulum system-identification module.

Ported from system_identification_acrobot.ipynb: base-parameter dynamics,
forward simulation, the multi-stage identify() solver, and identifiability
analysis. Data files and parameter JSON paths are resolved relative to
sysid_config.json:data_dir, which defaults to this sysid_pipeline directory.
This module is hardware-free and can be imported offline.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import os
import time
from pathlib import Path

import numpy as np
import casadi as ca

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "sysid_config.json"

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
_CONFIG_REQUIRED = ("user_token", "experiment_type", "cell_id", "tau_max_id",
                    "dq_abort", "ctrl_hz", "inter_run_cooldown", "start_retry_max")


def load_config(path=CONFIG_FILE, need_token=False):
    """Load config; CLOUDPENDULUM_TOKEN overrides the file token.

    need_token=True requires a nonempty token for hardware collection.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file {path.name} was not found. Run: python sysid_make_config.py")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in _CONFIG_REQUIRED if k not in cfg]
    if missing:
        raise KeyError(f"{path.name} is missing fields: {missing}. Rerun sysid_make_config.py")
    env_tok = os.environ.get("CLOUDPENDULUM_TOKEN", "").strip()
    if env_tok:
        cfg["user_token"] = env_tok
        cfg["_token_source"] = "env:CLOUDPENDULUM_TOKEN"
    else:
        cfg["_token_source"] = f"file:{path.name}"
    if need_token and not cfg["user_token"]:
        raise RuntimeError("token is empty: fill user_token in sysid_config.json,"
                           "or set CLOUDPENDULUM_TOKEN.")
    return cfg


def mask_token(tok):
    tok = str(tok)
    return f"{tok[:3]}...{tok[-3:]}(len={len(tok)})" if len(tok) >= 8 else "<too short>"


def data_dir(cfg):
    """Resolve config data_dir relative to this folder; default is sysid_pipeline."""
    return (HERE / str(cfg.get("data_dir", "."))).resolve()


def resolve_data(cfg, name):
    """Resolve a data filename to an absolute path under data_dir unless already absolute."""
    p = Path(str(name))
    return p if p.is_absolute() else data_dir(cfg) / p


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
PHYS = {
    'm1': 0.10985804, 'm2': 0.07216499, 'l1': 0.05, 'r1': 0.05, 'r2': 0.04103148,
    'I1': 2.2984637e-4, 'I2': 1.8893780e-4, 'g': 9.81,
    'b1': 2.0690483e-3, 'b2': 5.0788180e-4, 'cf1': 8.6826365e-4, 'cf2': 1.4269109e-3,
    'eps': 0.05,
}


def phys_to_base(P):
    # --------------------------------------------------------------------------
    J1 = P['I1'] + P['m1'] * P['r1'] ** 2
    J2 = P['I2'] + P['m2'] * P['r2'] ** 2
    return {
        'P1': J1 + J2 + P['m2'] * P['l1'] ** 2,
        'P2': P['m2'] * P['l1'] * P['r2'],
        'P3': J2,
        'P4': P['g'] * (P['m1'] * P['r1'] + P['m2'] * P['l1']),
        'P5': P['g'] * P['m2'] * P['r2'],
        'b1': P['b1'], 'b2': P['b2'], 'cf1': P['cf1'], 'cf2': P['cf2'], 'eps': P['eps'],
    }


PARAM_NAMES = ['P1', 'P2', 'P3', 'P6', 'P4', 'P5', 'b1', 'b2', 'cf1', 'cf2', 'd1', 'd2', 'eps']
PARAM_NOMINAL = phys_to_base(PHYS)
PARAM_NOMINAL['P6'] = PARAM_NOMINAL['P3']
PARAM_NOMINAL['d1'] = 1e-4
PARAM_NOMINAL['d2'] = 1e-4


def _pget(p, name):
    return p[PARAM_NAMES.index(name)]


def acrobot_rhs_p(x, u, p):
    # --------------------------------------------------------------------------
    P1 = _pget(p, 'P1'); P2 = _pget(p, 'P2'); P3 = _pget(p, 'P3'); P6 = _pget(p, 'P6')
    P4 = _pget(p, 'P4'); P5 = _pget(p, 'P5')
    b1 = _pget(p, 'b1'); b2 = _pget(p, 'b2'); cf1 = _pget(p, 'cf1'); cf2 = _pget(p, 'cf2')
    eps = _pget(p, 'eps')
    d1 = _pget(p, 'd1'); d2 = _pget(p, 'd2')
    q1, q2, dq1, dq2 = x[0], x[1], x[2], x[3]
    cq2 = ca.cos(q2); sq2 = ca.sin(q2)
    M11 = P1 + 2 * P2 * cq2
    M12 = P3 + P2 * cq2
    M22 = P6
    h = P2 * sq2
    c1 = -2 * h * dq2 * dq1 - h * dq2 ** 2
    c2 = h * dq1 ** 2
    G1 = P4 * ca.sin(q1) + P5 * ca.sin(q1 + q2)
    G2 = P5 * ca.sin(q1 + q2)
    rhs1 = 0.0 - c1 - G1 - b1 * dq1 - cf1 * ca.tanh(dq1 / eps) - d1 * dq1 * ca.sqrt(dq1 ** 2 + 0.25)
    rhs2 = u - c2 - G2 - b2 * dq2 - cf2 * ca.tanh(dq2 / eps) - d2 * dq2 * ca.sqrt(dq2 ** 2 + 0.25)
    det = M11 * M22 - M12 ** 2
    ddq1 = (M22 * rhs1 - M12 * rhs2) / det
    ddq2 = (M11 * rhs2 - M12 * rhs1) / det
    return ca.vertcat(dq1, dq2, ddq1, ddq2)


def rk4_p(x, u, p, dt):
    k1 = acrobot_rhs_p(x, u, p); k2 = acrobot_rhs_p(x + dt / 2 * k1, u, p)
    k3 = acrobot_rhs_p(x + dt / 2 * k2, u, p); k4 = acrobot_rhs_p(x + dt * k3, u, p)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def param_vec(params):
    # --------------------------------------------------------------------------
    vals = []
    for n in PARAM_NAMES:
        if n == 'P6':
            vals.append(float(params.get('P6', params.get('P3', PARAM_NOMINAL['P6']))))
        else:
            vals.append(float(params.get(n, PARAM_NOMINAL[n])))
    return vals


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def step_dt(ds, k):
    dt = np.asarray(ds['dt'])
    return float(dt) if dt.ndim == 0 else float(dt[k])


def simulate(ds, params):
    # --------------------------------------------------------------------------
    pv = ca.DM(param_vec(params))
    x = ca.DM([ds['q'][0, 0], ds['q'][0, 1], ds['dq'][0, 0], ds['dq'][0, 1]])
    n = len(ds['t'])
    Qs = np.zeros((n, 2)); DQs = np.zeros((n, 2))
    for k in range(n):
        Qs[k] = [float(x[0]), float(x[1])]; DQs[k] = [float(x[2]), float(x[3])]
        if k < n - 1:
            x = rk4_p(x, float(ds['u'][k]), pv, step_dt(ds, k))
    return Qs, DQs


def rmse_deg(ds, params):
    Qs, _ = simulate(ds, params)
    e = np.degrees(np.sqrt(np.mean((Qs - ds['q']) ** 2, axis=0)))
    return e


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
FIT_NAMES = ['P1', 'P2', 'P3', 'P6', 'P4', 'P5', 'b1', 'b2', 'cf1', 'cf2', 'd1', 'd2']
BOUNDS = {
    'P1': (3e-4, 3e-3), 'P2': (3e-5, 6e-4), 'P3': (8e-5, 1e-3), 'P6': (8e-5, 1e-3),
    'P4': (0.03, 0.25), 'P5': (0.008, 0.09),
    'b1': (0.0, 1e-2), 'b2': (0.0, 1e-2), 'cf1': (0.0, 5e-3), 'cf2': (0.0, 5e-3),
    'd1': (0.0, 5e-3), 'd2': (0.0, 5e-3),
}
WIN = 80
COST_STRIDE = 2


def identify(datasets, fit_names=FIT_NAMES, w_dq=0.1, win=WIN, max_iter=500, verbose=False):
    t_start = time.perf_counter()
    fit_idx = [PARAM_NAMES.index(n) for n in fit_names]
    npar = len(fit_idx)
    scale = np.array([abs(PARAM_NOMINAL[n]) or 1.0 for n in fit_names])

    x0_cols = []; u_cols = []; dt_cols = []; y_blocks = []; mask_cols = []
    ncost = 0; nsteps = 0
    for ds in datasets:
        q = ds['q']; dq = ds['dq']; u = ds['u']; dt = np.asarray(ds['dt']); N = len(ds['t'])
        if N < 5:
            continue
        k = 0
        while k < N - 1:
            end = min(k + win, N - 1); nv = end - k
            x0_cols.append(np.r_[q[k], dq[k]])
            uc = np.zeros(win); uc[:nv] = u[k:end]; u_cols.append(uc)
            dc = np.zeros(win); dc[:nv] = float(dt) if dt.ndim == 0 else dt[k:end]; dt_cols.append(dc)
            y = np.zeros((4, win)); y[:, :nv] = np.vstack((q[k:end].T, dq[k:end].T)); y_blocks.append(y)
            mask = np.zeros(win); mask[:nv:COST_STRIDE] = 1.0; mask_cols.append(mask)
            ncost += int(mask.sum()); nsteps += nv; k = end
    assert x0_cols and ncost, 'not enough training data.'
    nwin = len(x0_cols)

    zw = ca.MX.sym('zw', npar); x0 = ca.MX.sym('x0w', 4)
    uw = ca.MX.sym('uw', win); dtw = ca.MX.sym('dtw', win)
    yw = ca.MX.sym('yw', 4, win); mw = ca.MX.sym('mw', win)
    p = [ca.MX(float(PARAM_NOMINAL[n])) for n in PARAM_NAMES]
    for j, idx in enumerate(fit_idx):
        p[idx] = zw[j] * scale[j]
    pvec = ca.vertcat(*p); x = x0; Jw = 0; Rw = []
    for kk in range(win):
        r = x - yw[:, kk]
        Jw += mw[kk] * (r[0] ** 2 + r[1] ** 2 + w_dq * (r[2] ** 2 + r[3] ** 2))
        if kk % COST_STRIDE == 0:
            Rw.append(mw[kk] * ca.vertcat(r[0], r[1], np.sqrt(w_dq) * r[2], np.sqrt(w_dq) * r[3]))
        x = rk4_p(x, uw[kk], pvec, dtw[kk])
    window_cost = ca.Function('id_window_cost', [zw, x0, uw, dtw, yw, mw], [Jw]).expand()
    mapped_cost = window_cost.map(nwin, 'serial')

    X0 = np.column_stack(x0_cols); U = np.column_stack(u_cols); DT = np.column_stack(dt_cols)
    Y = np.hstack(y_blocks); MASK = np.column_stack(mask_cols)
    z = ca.MX.sym('z_id', npar)
    J = ca.sum2(mapped_cost(ca.repmat(z, 1, nwin), X0, U, DT, Y, MASK)) / ncost
    lb = []; ub = []
    for j, n in enumerate(fit_names):
        lo, hi = BOUNDS[n]; lb.append(lo / scale[j]); ub.append(hi / scale[j])

    def run_solver(name, x_init, niter, limited_memory=False):
        opts = {'ipopt.print_level': 5 if verbose else 0, 'print_time': int(verbose),
                'ipopt.max_iter': niter, 'ipopt.tol': 1e-6,
                'ipopt.acceptable_tol': 1e-4, 'ipopt.acceptable_iter': 8,
                'ipopt.acceptable_obj_change_tol': 1e-7, 'error_on_fail': False}
        if limited_memory:
            opts['ipopt.hessian_approximation'] = 'limited-memory'
        solver = ca.nlpsol(name, 'ipopt', {'x': z, 'f': J}, opts)
        sol = solver(x0=x_init, lbx=lb, ubx=ub)
        return np.asarray(sol['x']).reshape(-1), solver.stats()

    def run_least_squares(x_init):
        from scipy.optimize import least_squares
        window_res = ca.Function('id_window_res', [zw, x0, uw, dtw, yw, mw], [ca.vertcat(*Rw)]).expand()
        mapped_res = window_res.map(nwin, 'serial')
        Rmat = mapped_res(ca.repmat(z, 1, nwin), X0, U, DT, Y, MASK)
        R = ca.reshape(Rmat, Rmat.numel(), 1) / np.sqrt(ncost)
        residual_jac = ca.Function('id_residual_jac', [z], [R, ca.jacobian(R, z)])
        cache = {}

        def evaluate(v):
            key = np.asarray(v, dtype=float).tobytes()
            if cache.get('key') != key:
                rv, jv = residual_jac(v)
                cache.update(key=key, r=np.asarray(rv).reshape(-1), j=np.asarray(jv))
            return cache['r'], cache['j']

        lbv = np.asarray(lb); ubv = np.asarray(ub)
        start = np.minimum(np.maximum(np.asarray(x_init), lbv + 1e-10), ubv - 1e-10)
        result = least_squares(lambda v: evaluate(v)[0], start, jac=lambda v: evaluate(v)[1],
                               bounds=(lbv, ubv), method='trf', x_scale='jac', loss='linear',
                               ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=max_iter,
                               verbose=2 if verbose else 0)
        return result.x, result

    zval, stats = run_solver('id_solver_exact', np.ones(npar), min(max_iter, 80))
    method = 'exact Hessian'
    total_iter = int(stats.get('iter_count', 0))
    success = bool(stats.get('success', False)); status = stats.get('return_status', 'unknown')
    if not success:
        print(f"Exact Hessian did not converge ({status}); switching to trust-region least-squares...")
        z0 = np.clip(zval, lb, ub) if np.all(np.isfinite(zval)) else np.ones(npar)
        try:
            zval, lsq = run_least_squares(z0)
            success = bool(lsq.success); status = lsq.message
            method = 'trust-region least-squares'; total_iter += int(lsq.nfev)
        except Exception as e:
            success = False; status = f'least-squares unavailable/failed: {e}'
            print(status)
    if not success:
        print(f"Trust-region did not converge ({status}); switching to limited-memory...")
        z0 = np.clip(zval, lb, ub) if np.all(np.isfinite(zval)) else np.ones(npar)
        zval, stats = run_solver('id_solver_lbfgs', z0, max_iter, limited_memory=True)
        success = bool(stats.get('success', False)); status = stats.get('return_status', 'unknown')
        method = 'limited-memory'; total_iter += int(stats.get('iter_count', 0))
    if not success:
        raise RuntimeError(f"Identification did not converge ({status}); parameters were not accepted. Check data and identifiability report.")
    out = {n: float(zval[j] * scale[j]) for j, n in enumerate(fit_names)}
    print(f"Identification complete: {nsteps} steps/{nwin} windows, {method}, {total_iter} total iterations, "
          f"{time.perf_counter() - t_start:.2f} s")

    for j, n in enumerate(fit_names):
        lo, hi = BOUNDS[n]; v = out[n]; span = (hi - lo) or 1.0
        if v <= lo + 0.02 * span or v >= hi - 0.02 * span:
            print(f'  WARNING {n} near bound ({v:.3e} in [{lo:.1e},{hi:.1e}]); '
                  f'likely weakly identifiable. Consider removing it from FIT_NAMES after checking the report.')
    return out


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def sensitivity_fim_on_data(datasets, fit_names, win=WIN):
    fit_idx = [PARAM_NAMES.index(n) for n in fit_names]
    npar = len(fit_idx)
    scale = np.array([PARAM_NOMINAL[n] for n in fit_names], dtype=float)
    xs = ca.SX.sym('xs', 4); us = ca.SX.sym('us'); dts = ca.SX.sym('dts')
    z = ca.SX.sym('z', npar)
    pl = [ca.SX(float(PARAM_NOMINAL[n])) for n in PARAM_NAMES]
    for j, idx in enumerate(fit_idx):
        pl[idx] = z[j] * scale[j]
    pv = ca.vertcat(*pl)
    xn = rk4_p(xs, us, pv, dts)
    step_sens = ca.Function('step_sens', [xs, us, z, dts],
                            [xn, ca.jacobian(xn, xs), ca.jacobian(xn, z)])

    fim = np.zeros((npar, npar), dtype=float)
    z_nom = np.ones(npar)
    n_outputs = 0
    for ds in datasets:
        q = ds['q']; dq = ds['dq']; u = ds['u']; dt = np.asarray(ds['dt']); N = len(ds['t'])
        k = 0
        while k < N - 1:
            x = np.array([q[k, 0], q[k, 1], dq[k, 0], dq[k, 1]], dtype=float)
            dx_dz = np.zeros((4, npar), dtype=float)
            end = min(k + win, N - 1)
            for kk in range(k, end):
                sq = dx_dz[:2]
                fim += sq.T @ sq
                n_outputs += 2
                dt_k = float(dt if dt.ndim == 0 else dt[kk])
                x_next, A, B = step_sens(x, float(u[kk]), z_nom, dt_k)
                dx_dz = np.asarray(A) @ dx_dz + np.asarray(B)
                x = np.asarray(x_next).reshape(4)
            k = end
    assert n_outputs, 'no samples are available for sensitivity analysis.'
    return 0.5 * (fim + fim.T), n_outputs


def identifiability_report(datasets, fit_names=FIT_NAMES):
    t0 = time.perf_counter()
    FIM_rel, n_outputs = sensitivity_fim_on_data(datasets, fit_names)
    eigval, eigvec = np.linalg.eigh(FIM_rel)
    order = np.argsort(eigval)[::-1]
    sv = np.sqrt(np.maximum(eigval[order], 0.0))
    Vt = eigvec[:, order].T
    colnorm = np.sqrt(np.maximum(np.diag(FIM_rel), 0.0))

    print(f'Processed {n_outputs // 2} time samples in {time.perf_counter() - t0:.2f} s (one-step AD + recursion)')
    print(f'Singular values (large->small): {np.array2string(sv, precision=2, max_line_width=200)}')
    print(f'Condition number sigma_max/sigma_min = {sv[0] / max(sv[-1], 1e-30):.1e}\n')
    print('Relative parameter sensitivities (column norm; smaller means harder to identify):')
    weak = []
    for j in np.argsort(colnorm):
        is_weak = colnorm[j] < 0.05 * colnorm.max()
        if is_weak:
            weak.append(fit_names[j])
        print(f'  {fit_names[j]:>4}: {colnorm[j]:.2e}{"  <== weak; consider removing" if is_weak else ""}')
    print('\nParameter weights in the least-identifiable direction (smallest singular value):')
    w = Vt[-1]
    for j in np.argsort(-np.abs(w)):
        print(f'  {fit_names[j]:>4}: {w[j]:+.2f}')
    return weak


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
sys.path.insert(0, str(HERE))
from system_id_data_io import load_datasets, merge_dataset_files  # noqa: E402


def load_fit_groups(paths):
    """Load one NPZ directly or merge several files. Return {'free': [...], 'exc': [...]}."""
    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"data file does not exist: {p}")
    if len(paths) == 1:
        return load_datasets(paths[0])
    return merge_dataset_files(paths, label_prefix=True)


def usable_sets(groups):
    """Cell22 filter: keep trajectories with >20 steps and not aborted. Return (free_ok, exc_ok)."""
    free_ok = [d for d in groups.get('free', []) if len(d['t']) > 20 and not d['aborted']]
    exc_ok = [d for d in groups.get('exc', []) if len(d['t']) > 20 and not d['aborted']]
    return free_ok, exc_ok
