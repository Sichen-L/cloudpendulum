# -*- coding: utf-8 -*-
"""Controller runner (server-side script; invoked by ctrl_run_pendubot.py / ctrl_run_acrobot.py).

Ported from the compensation_controller/ notebooks: Cell 1 (acados loading), Cell 5 (NMPC/IPOPT
fallback/LQR), Cell 7 (device discovery), Cell 8 (real-time loop, verbatim; per-plant differences
injected via the PLANTS spec), Cell 10 (video download). Cell 9's six plots are not ported --
instead all log arrays are stored in ctrlrun_{plant}_{cell}_{stamp}.npz so any plot can be made
offline afterwards.

Three deliberate behavioral differences from the notebook:
  1. The reference trajectory is [loaded only, not generated]: run ctrl_make_reference.py first.
     On load it checks N/T/dq/tau bounds + parameter sha256 -- the notebook's cache check does not
     inspect the parameter file, so switching devices would silently reuse an old trajectory; this
     closes that gap.
  2. token / cell_id come from sysid_config.json (user_token; overridable via the
     CLOUDPENDULUM_TOKEN environment variable), not hard-coded.
  3. Full logs are stored as npz (same style as the main project's hwrun_*.npz), no pop-up plots.

Requires acados + cloudpendulumclient (i.e., the CloudPendulum JupyterHub environment).
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import sys
import glob as _glob
import ctypes
import time
import urllib.request
from pathlib import Path

import numpy as np
import casadi as ca

from sysid_common import load_config, mask_token, resolve_data, data_dir
from ctrl_plants import (PLANTS, DQ_LIMIT, DQ_ABORT, CTRL_Hz, dt_ctrl, BUSYWAIT_MS,
                         T_SIM, RECORD, x_goal, REF_T, REF_N, REF_DQ_LIMIT,
                         N_MPC, Q_DIAG, QF_DIAG,
                         USE_BASELINE_WRAPPED_NMPC_STATE, USE_STATE_PHASE_SWINGUP,
                         REF_MAX_ADVANCE, REF_MAX_LAG_STEPS, REF_MAX_LEAD_STEPS,
                         REF_TIME_BIAS, REF_VEL_WEIGHT,
                         USE_LQR_TOP, LQR_ENTER_Q1, LQR_ENTER_Q2, LQR_ENTER_DQ,
                         LQR_EXIT_Q1, LQR_EXIT_Q2, LQR_EXIT_DQ, LQR_Q_DIAG,
                         USE_TOP_TAU_SLEW_LIMIT, USE_SWINGUP_SPEED_BRAKE,
                         DQ_BRAKE, DQ_BRAKE_RELEASE,
                         StateEstimator, precise_sleep_until, wrap_pi,
                         check_reference, append_hold, load_plant_params, make_rhs)


# ══════════════════════════════════════════════════════════════════════════
# Cell 1 -- acados runtime loading  [verbatim port]
# ══════════════════════════════════════════════════════════════════════════
def _candidate_acados_roots(cfg=None):
    candidates = []
    for value in (
        os.environ.get('ACADOS_SOURCE_DIR'),
        os.environ.get('ACADOS_ROOT'),
        (cfg or {}).get('acados_root') if cfg else None,
        '~/acados',
        '/opt/acados',
        '/usr/local/acados',
    ):
        if value:
            path = os.path.abspath(os.path.expanduser(str(value)))
            if path not in candidates:
                candidates.append(path)
    return candidates


def setup_acados(cfg=None):
    roots = _candidate_acados_roots(cfg)
    ACADOS_ROOT = next(
        (root for root in roots
         if os.path.isdir(os.path.join(root, 'lib'))
         or os.path.isdir(os.path.join(root, 'interfaces', 'acados_template'))),
        roots[0],
    )
    os.environ['ACADOS_SOURCE_DIR'] = ACADOS_ROOT
    os.environ['ACADOS_ROOT'] = ACADOS_ROOT
    os.environ['LD_LIBRARY_PATH'] = (
        os.path.join(ACADOS_ROOT, 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    template_path = os.path.join(ACADOS_ROOT, 'interfaces', 'acados_template')
    if os.path.isdir(template_path) and template_path not in sys.path:
        sys.path.insert(0, template_path)

    libdir = os.path.join(ACADOS_ROOT, 'lib')
    if os.path.isdir(libdir):
        shared_objects = sorted(_glob.glob(os.path.join(libdir, '*.so')))
        shared_objects = [p for p in shared_objects if not p.endswith('libacados.so')] + [
            p for p in shared_objects if p.endswith('libacados.so')]
        loaded = set(); failed = {}
        for _ in range(3):
            progress = False
            for library in shared_objects:
                if library in loaded:
                    continue
                try:
                    ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
                    loaded.add(library); failed.pop(library, None); progress = True
                except OSError as exc:
                    failed[library] = str(exc)
            if not progress:
                break
        print(f'  preloaded {len(loaded)}/{len(shared_objects)} acados libraries from {libdir}')
        for library, error in failed.items():
            print(f'  warning: {os.path.basename(library)} -> {error}')
    else:
        print(f'  acados library directory not found: {libdir}')
        tried = ', '.join(roots)
        print(f'  tried acados roots: {tried}')

    try:
        from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
        print(f'acados   OK ({AcadosOcp.__module__})')
        return AcadosOcp, AcadosOcpSolver, AcadosModel
    except Exception as exc:
        print('acados NOT available:', repr(exc))
        print('  Control scripts must run in an environment with acados (JupyterHub); the IPOPT fallback is not real-time,'
              'it is for functional checks only (--allow-ipopt-fallback).')
        return None


# ══════════════════════════════════════════════════════════════════════════
# Main run (ports Cell 5/7/8/10; all plant differences injected via spec)
# ══════════════════════════════════════════════════════════════════════════
def main(plant, argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=f'{plant} NMPC swing-up control (hardware run! requires acados+client)')
    ap.add_argument('--t-sim', type=float, default=T_SIM, help=f'experiment duration in s (default {T_SIM})')
    ap.add_argument('--no-record', action='store_true', help='do not request server-side recording')
    ap.add_argument('--allow-ipopt-fallback', action='store_true',
                    help='allow IPOPT fallback when acados is unavailable (not real-time, functional check only)')
    ap.add_argument('--allow-legacy-ref', action='store_true',
                    help='allow an old reference cache without parameter provenance (notebook-generated)')
    ap.add_argument('--params', default=None,
                    help='identified-parameter json (default: latest identified_params_{cell}_fit_*.json)')
    args = ap.parse_args(argv)

    spec = PLANTS[plant]
    cfg = load_config(need_token=False)
    token = (os.environ.get('CLOUDPENDULUM_TOKEN', '').strip()
             or cfg.get('ctrl_user_token', '')
             or cfg.get('user_token', ''))
    if not token:
        print('ERROR: config is missing user_token,'
              'or set the CLOUDPENDULUM_TOKEN environment variable.')
        return 1
    CELL_ID = int(cfg['cell_id'])
    t_sim = float(args.t_sim)
    record = RECORD and not args.no_record

    print(f'== {spec["title"]} | cell_id={CELL_ID} | token {mask_token(token)} ==')

    # -- Parameters and dynamics --
    pfile = args.params
    if pfile is None:
        ddir = data_dir(cfg)
        candidates = sorted(ddir.glob(f"identified_params_{CELL_ID}_fit_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            candidates = sorted(ddir.glob(f"identified_params_{CELL_ID}_*.json"),
                                key=lambda p: p.stat().st_mtime)
        if candidates:
            pfile = str(candidates[-1])
        else:
            pfile = cfg.get('ctrl_params') or cfg.get('design_params')
    if not pfile:
        print(f'ERROR: no identified-parameter json found for cell_id={CELL_ID}.'
              'Run sysid_workflow.ipynb first to complete system identification.')
        return 1
    P = load_plant_params(resolve_data(cfg, pfile))
    print(f'params: {P["_file"]}  sha256={P["_sha256"][:12]}...')
    TAU1_MAX, TAU_MAX = spec['TAU1_MAX'], spec['TAU_MAX']
    R_DIAG = spec['R_DIAG']; LQR_R = spec['LQR_R']
    TOP_TAU_SLEW_RATE = spec['TOP_TAU_SLEW_RATE']
    USE_SWINGUP_TAU_SLEW_LIMIT = spec['USE_SWINGUP_TAU_SLEW_LIMIT']
    SWINGUP_TAU_SLEW_RATE = spec['SWINGUP_TAU_SLEW_RATE']
    _, rk4_step, rk4_np = make_rhs(P)

    # -- Reference (load only; generation is ctrl_make_reference.py's job) --
    ref_path = resolve_data(cfg, spec['ref_file'].format(cell=CELL_ID))
    ok, Xref_su, Uref_su, dt_ref, msgs = check_reference(ref_path, spec, P)
    for m in msgs:
        print('  ' + m)
    if not ok:
        print(f'ERROR: reference unavailable. Run first:  python ctrl_make_reference.py --plant {plant}')
        return 1
    if msgs and not args.allow_legacy_ref:      # ok but warnings = legacy format without provenance
        print('Refusing a reference without provenance (--allow-legacy-ref overrides; parameter consistency is your responsibility).')
        return 1
    Xref, Uref, N_SWINGUP, NTOT = append_hold(Xref_su, Uref_su, dt_ref)
    print(f'  reference: {N_SWINGUP} swing-up + {NTOT - N_SWINGUP} hold steps, '
          f'dt_ref={dt_ref * 1e3:.0f} ms')

    # -- acados / client --
    acados = setup_acados(cfg)
    if acados is None and not args.allow_ipopt_fallback:
        return 2
    try:
        from cloudpendulumclient.client import Client
    except ImportError:
        print('ERROR: cloudpendulumclient is not installed -- this script must run on CloudPendulum JupyterHub.')
        return 2

    # -- Cell 5 -- Controller (verbatim port; plant constants injected) --
    def _top_error(x):
        return np.array([wrap_pi(x[0] - np.pi), wrap_pi(x[1]), x[2], x[3]], dtype=float)

    def _normalize_state_for_ref(x, idx):
        x = np.array(x, dtype=float).copy()
        if USE_BASELINE_WRAPPED_NMPC_STATE:
            return x
        if int(idx) >= N_SWINGUP:
            ref = Xref[:, min(int(idx), NTOT)]
            x[0] = ref[0] + wrap_pi(x[0] - ref[0])
            x[1] = ref[1] + wrap_pi(x[1] - ref[1])
        return x

    def select_reference_index(x_meas, idx_prev, meas_time=None):
        if meas_time is None:
            time_idx = int(idx_prev)
        else:
            time_idx = int(np.clip(np.floor(meas_time / dt_ref), 0, N_SWINGUP))
        if idx_prev >= N_SWINGUP:
            return N_SWINGUP, time_idx, 0.0, np.inf
        lo = max(int(idx_prev), time_idx - REF_MAX_LAG_STEPS, 0)
        hi = min(int(idx_prev) + REF_MAX_ADVANCE, time_idx + REF_MAX_LEAD_STEPS, N_SWINGUP)
        if hi < lo:
            hi = lo
        cand = np.arange(lo, hi + 1, dtype=int)
        refs = Xref[:, cand]
        e0 = x_meas[0] - refs[0]
        e1 = x_meas[1] - refs[1]
        ev0 = (x_meas[2] - refs[2]) / max(DQ_LIMIT, 1.0)
        ev1 = (x_meas[3] - refs[3]) / max(DQ_LIMIT, 1.0)
        score = 1.8 * e0 ** 2 + 0.9 * e1 ** 2 + REF_VEL_WEIGHT * (ev0 ** 2 + ev1 ** 2)
        score += REF_TIME_BIAS * ((cand - time_idx) / max(REF_MAX_LAG_STEPS, 1)) ** 2
        order = np.argsort(score)
        best = int(cand[order[0]])
        gap = float(score[order[1]] - score[order[0]]) if len(order) > 1 else np.inf
        return best, time_idx, float(score[order[0]]), gap

    def _horizon_refs(idx):
        xr = np.zeros((4, N_MPC + 1)); ur = np.zeros((2, N_MPC))
        for j in range(N_MPC + 1):
            xr[:, j] = Xref[:, min(idx + j, NTOT)]
        for j in range(N_MPC):
            ur[:, j] = Uref[:, min(idx + j, NTOT - 1)]
        return xr, ur

    class AcadosTracker:
        """Real-time nonlinear reference tracker using one SQP-RTI iteration."""

        def __init__(self):
            AcadosOcp, AcadosOcpSolver, AcadosModel = acados
            model = AcadosModel()
            model.name = spec['model_name']
            q1 = ca.SX.sym('q1'); q2 = ca.SX.sym('q2')
            dq1 = ca.SX.sym('dq1'); dq2 = ca.SX.sym('dq2')
            x = ca.vertcat(q1, q2, dq1, dq2)
            u = ca.SX.sym('u', 2)
            xdot = ca.SX.sym('xdot', 4)
            rhs, _, _ = make_rhs(P)
            f_expl = rhs(x, u)
            model.x = x; model.u = u; model.xdot = xdot
            model.f_expl_expr = f_expl
            model.f_impl_expr = xdot - f_expl

            ocp = AcadosOcp()
            ocp.model = model
            nx, nu = 4, 2
            ny, ny_e = nx + nu, nx
            N = N_MPC; Tf = N * dt_ref
            try:
                ocp.solver_options.N_horizon = N
            except Exception:
                ocp.dims.N = N
            ocp.solver_options.tf = Tf
            ocp.cost.cost_type = 'LINEAR_LS'; ocp.cost.cost_type_e = 'LINEAR_LS'
            ocp.cost.W = np.diag(np.concatenate([Q_DIAG, R_DIAG]))
            ocp.cost.W_e = np.diag(QF_DIAG)
            Vx = np.zeros((ny, nx)); Vx[:nx, :nx] = np.eye(nx); ocp.cost.Vx = Vx
            Vu = np.zeros((ny, nu)); Vu[nx:, :] = np.eye(nu); ocp.cost.Vu = Vu
            ocp.cost.Vx_e = np.eye(nx)
            ocp.cost.yref = np.zeros(ny); ocp.cost.yref_e = np.zeros(ny_e)
            ocp.constraints.idxbu = np.array([0, 1])
            ocp.constraints.lbu = np.array([-TAU1_MAX, -TAU_MAX])
            ocp.constraints.ubu = np.array([TAU1_MAX, TAU_MAX])
            ocp.constraints.idxbx = np.array([2, 3])
            ocp.constraints.lbx = np.array([-DQ_LIMIT, -DQ_LIMIT])
            ocp.constraints.ubx = np.array([DQ_LIMIT, DQ_LIMIT])
            ocp.constraints.idxsbx = np.array([0, 1])
            ns = 2
            ocp.cost.Zl = 1e5 * np.ones(ns); ocp.cost.Zu = 1e5 * np.ones(ns)
            ocp.cost.zl = 1e3 * np.ones(ns); ocp.cost.zu = 1e3 * np.ones(ns)
            ocp.constraints.x0 = np.zeros(nx)
            ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
            ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
            ocp.solver_options.integrator_type = 'ERK'
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 1
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
            self.N = N; self.nx = nx; self.nu = nu
            self.solver = AcadosOcpSolver(ocp, json_file=spec['ocp_json'], verbose=False)
            for _ in range(5):
                self.solve(_normalize_state_for_ref(Xref[:, 0], 0), 0)

        def solve(self, x_meas, idx, tau_prev=None, dt_apply=None, slew_rate=None):
            if tau_prev is None:
                tau_prev = np.zeros(self.nu)
            tau_prev = np.atleast_1d(tau_prev).astype(float)
            x_ctrl = _normalize_state_for_ref(x_meas, idx)
            xr, ur = _horizon_refs(idx)
            for j in range(self.N):
                self.solver.set(j, 'yref', np.concatenate([xr[:, j], ur[:, j]]))
            self.solver.set(self.N, 'yref', xr[:, self.N])
            self.solver.set(0, 'lbx', x_ctrl); self.solver.set(0, 'ubx', x_ctrl)

            if slew_rate is not None:
                dt_a = dt_ctrl if dt_apply is None else max(float(dt_apply), 1e-4)
                sr = np.atleast_1d(slew_rate).astype(float)
                du = sr * dt_a
                lbu0 = np.maximum(np.array([-TAU1_MAX, -TAU_MAX]), tau_prev - du)
                ubu0 = np.minimum(np.array([TAU1_MAX, TAU_MAX]), tau_prev + du)
                self.solver.constraints_set(0, 'lbu', lbu0)
                self.solver.constraints_set(0, 'ubu', ubu0)

            t0 = time.perf_counter()
            status = int(self.solver.solve())
            st = time.perf_counter() - t0
            u0 = np.array(self.solver.get(0, 'u'), dtype=float)
            return u0, st, status

    class IpoptTracker:
        def __init__(self):
            N = N_MPC
            Q = np.diag(Q_DIAG); Qf = np.diag(QF_DIAG); R = np.diag(R_DIAG)
            opti = ca.Opti()
            X = opti.variable(4, N + 1); U = opti.variable(2, N)
            Px0 = opti.parameter(4); Pxr = opti.parameter(4, N + 1); Pur = opti.parameter(2, N)
            opti.subject_to(X[:, 0] == Px0); Jc = 0
            for k in range(N):
                opti.subject_to(X[:, k + 1] == rk4_step(X[:, k], U[:, k], dt_ref))
                opti.subject_to(opti.bounded(-TAU1_MAX, U[0, k], TAU1_MAX))
                opti.subject_to(opti.bounded(-TAU_MAX, U[1, k], TAU_MAX))
                opti.subject_to(opti.bounded(-DQ_LIMIT, X[2, k], DQ_LIMIT))
                opti.subject_to(opti.bounded(-DQ_LIMIT, X[3, k], DQ_LIMIT))
                ex = X[:, k] - Pxr[:, k]; eu = U[:, k] - Pur[:, k]
                Jc += ca.mtimes([ex.T, Q, ex]) + ca.mtimes([eu.T, R, eu])
            eN = X[:, N] - Pxr[:, N]; Jc += ca.mtimes([eN.T, Qf, eN])
            opti.minimize(Jc)
            opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0,
                                  'ipopt.max_iter': 40, 'ipopt.tol': 1e-4})
            self.F = opti.to_function('F', [Px0, Pxr, Pur], [U[:, 0]])
            self.solve(Xref[:, 0], 0)

        def solve(self, x_meas, idx, tau_prev=None, dt_apply=None, slew_rate=None):
            xr, ur = _horizon_refs(idx)
            x_ctrl = _normalize_state_for_ref(x_meas, idx)
            t0 = time.perf_counter()
            u0 = np.array(self.F(x_ctrl, xr, ur)).flatten()
            return u0, time.perf_counter() - t0, 0

    class UprightLQR:
        def __init__(self):
            x = ca.SX.sym('x', 4); u = ca.SX.sym('u', 2)
            x_next = rk4_step(x, u, dt_ctrl)
            A_fun = ca.Function('lqr_A', [x, u], [ca.jacobian(x_next, x)])
            B_fun = ca.Function('lqr_B', [x, u], [ca.jacobian(x_next, u)])
            A = np.array(A_fun(x_goal, np.zeros(2)), dtype=float)
            B = np.array(B_fun(x_goal, np.zeros(2)), dtype=float)
            Q = np.diag(LQR_Q_DIAG); R = LQR_R
            try:
                from scipy.linalg import solve_discrete_are
                Pm = solve_discrete_are(A, B, Q, R)
            except Exception:
                Pm = Q.copy()
                for _ in range(1000):
                    S = R + B.T @ Pm @ B
                    K = np.linalg.solve(S, B.T @ Pm @ A)
                    Pn = Q + A.T @ Pm @ (A - B @ K)
                    if np.max(np.abs(Pn - Pm)) < 1e-10:
                        Pm = Pn; break
                    Pm = Pn
            self.K = np.linalg.solve(R + B.T @ Pm @ B, B.T @ Pm @ A)
            eig = np.linalg.eigvals(A - B @ self.K)
            print('Upright LQR K=', np.round(self.K, 3), ' max|eig|=', float(np.max(np.abs(eig))))

        def solve(self, x_meas):
            e = _top_error(x_meas)
            u = -(self.K @ e.reshape(-1, 1)).flatten()
            return np.clip(u, [-TAU1_MAX, -TAU_MAX], [TAU1_MAX, TAU_MAX])

    def lqr_should_enter(x_meas):
        e = _top_error(x_meas)
        return (abs(e[0]) < LQR_ENTER_Q1 and abs(e[1]) < LQR_ENTER_Q2 and
                np.max(np.abs(e[2:])) < LQR_ENTER_DQ)

    def lqr_should_exit(x_meas):
        e = _top_error(x_meas)
        return (abs(e[0]) > LQR_EXIT_Q1 or abs(e[1]) > LQR_EXIT_Q2 or
                np.max(np.abs(e[2:])) > LQR_EXIT_DQ)

    if acados is not None:
        controller = AcadosTracker(); print('Using AcadosTracker (SQP-RTI).')
    else:
        controller = IpoptTracker(); print('Using IpoptTracker (fallback, NOT real-time).')
    lqr_controller = UprightLQR() if USE_LQR_TOP else None
    print(f'MPC horizon = {N_MPC} * {dt_ref * 1e3:.0f} ms = {N_MPC * dt_ref:.2f} s | top LQR={USE_LQR_TOP}')

    # -- Cell 7 -- Device discovery  [verbatim port] --
    probe_client = Client()
    try:
        cells = probe_client.get_cells(token)
        compatible_ids = []
        for cell in cells:
            cell_type = str(cell.cell_type).split('.')[-1]
            if cell_type == 'DOUBLE':
                compatible_ids.append(cell.cell_id)
        print(f'Compatible DoublePendulum cells: {compatible_ids}')
        if CELL_ID not in compatible_ids:
            print(f'WARNING: CELL_ID={CELL_ID} is not a DOUBLE device and may be rejected.')
    except Exception as exc:
        print('Device discovery failed:', exc)

    # -- Cell 8 -- Real-time loop  [verbatim port; plant differences injected] --
    client = Client()
    client.get_user_info(token)
    print('Requested cell_id =', CELL_ID)
    session_token, livestream_url = client.start_experiment(
        user_token=token, experiment_type='DoublePendulum',
        experiment_time=t_sim, preparation_time=5.0, record=record,
        cell_id=CELL_ID)
    print('Session   :', session_token)
    if livestream_url:
        print('Livestream:', livestream_url)
    client.set_impedance_controller_params(0.0, 0.0, session_token)
    client.set_torque([0.0, 0.0], session_token)

    n_steps = int(t_sim / dt_ctrl)
    q_hist = np.zeros((n_steps + 1, 2)); q_unwrapped_hist = np.zeros((n_steps + 1, 2))
    dq_hw_hist = np.zeros((n_steps, 2)); dq_f_hist = np.zeros((n_steps, 2))
    tau_hist = np.zeros((n_steps, 2)); u_cmd_hist = np.zeros((n_steps, 2)); u_ref_hist = np.zeros((n_steps, 2))
    dt_hist = np.zeros(n_steps); t_hist = np.zeros(n_steps)
    sample_time_hist = np.zeros(n_steps); x_meas_hist = np.full((n_steps, 4), np.nan)
    idx_hist = np.zeros(n_steps, dtype=int); idx_time_hist = np.zeros(n_steps, dtype=int)
    idx_state_hist = np.zeros(n_steps, dtype=int)
    idx_score_hist = np.full(n_steps, np.nan); idx_gap_hist = np.full(n_steps, np.nan)
    solve_hist = np.zeros(n_steps); solver_status_hist = np.zeros(n_steps, dtype=int)
    mode_hist = np.zeros(n_steps, dtype=int)
    slew_active_hist = np.zeros(n_steps, dtype=bool); brake_hist = np.zeros(n_steps, dtype=bool)
    get_q_time_hist = np.zeros(n_steps); get_dq_time_hist = np.zeros(n_steps); set_u_time_hist = np.zeros(n_steps)
    x_pred1_hist = np.full((n_steps, 4), np.nan)
    pred1_err_hist = np.full((n_steps, 4), np.nan)
    x_pred1_actualdt_hist = np.full((n_steps, 4), np.nan)
    pred1_actualdt_err_hist = np.full((n_steps, 4), np.nan)

    est = StateEstimator()
    download_url = None
    aborted = False; n_act = 0
    idx_ref = 0
    lqr_active = False
    brake_active = False
    tau_prev = np.zeros(2)
    i = 0
    aj = spec['active_joint']

    try:
        q0 = np.array(client.get_position(session_token)).flatten()
        dq0 = np.array(client.get_velocity(session_token)).flatten()
        dqf = est.update(q0, dq0, dt_ctrl)
        q_hist[0] = q0
        q_cont = q0.copy(); q_raw_prev = q0.copy()
        q_unwrapped_hist[0] = q_cont
        print('start q=', np.round(q0, 3), ' dq=', np.round(dqf, 3))

        meas_time = 0.0
        t_step = time.perf_counter()
        SESSION_STOP_MARGIN = 2.0
        LOOP_STOP_TIME = max(1.0, t_sim - SESSION_STOP_MARGIN)
        for i in range(n_steps):
            if meas_time >= LOOP_STOP_TIME:
                n_act = i
                print(f'  reached control window {LOOP_STOP_TIME:.1f}s of {t_sim:.1f}s session; '
                      f'stopping cleanly so the recording can be downloaded.')
                break
            dt_prev = dt_hist[i - 1] if i > 0 else dt_ctrl

            _ta = time.perf_counter()
            q_raw = np.array(client.get_position(session_token)).flatten()
            get_q_time_hist[i] = time.perf_counter() - _ta
            _ta = time.perf_counter()
            dq_hw = np.array(client.get_velocity(session_token)).flatten()
            get_dq_time_hist[i] = time.perf_counter() - _ta
            dqf = est.update(q_raw, dq_hw, dt_prev)
            q_cont = q_cont + wrap_pi(q_raw - q_raw_prev)
            q_raw_prev = q_raw.copy()
            dq_hw_hist[i] = dq_hw; dq_f_hist[i] = dqf

            if (not np.all(np.isfinite(q_raw))) or (not np.all(np.isfinite(dq_hw))) \
               or np.any(np.abs(dq_hw) > DQ_ABORT):
                client.set_torque([0.0, 0.0], session_token)
                print(f'  !! ABORT at step {i}: raw dq={np.round(dq_hw, 1)} (limit {DQ_ABORT})')
                aborted = True
                q_hist[i + 1] = q_raw; q_unwrapped_hist[i + 1] = q_cont; t_hist[i] = meas_time
                n_act = i
                break

            x_meas = np.concatenate([q_cont, dqf])
            x_control = np.concatenate([q_raw if USE_BASELINE_WRAPPED_NMPC_STATE else q_cont, dqf])
            x_meas_hist[i] = x_meas
            sample_time_hist[i] = meas_time

            idx_time = int(np.clip(np.floor(meas_time / dt_ref), 0, NTOT - 1))
            if USE_STATE_PHASE_SWINGUP:
                idx_state, _, idx_score, idx_gap = select_reference_index(x_meas, idx_ref, meas_time)
                idx_ref = idx_state
                idx_state_hist[i] = idx_state
                idx_score_hist[i] = idx_score
                idx_gap_hist[i] = idx_gap
            else:
                idx_ref = idx_time
                idx_state_hist[i] = idx_ref
            idx_time_hist[i] = idx_time

            if USE_LQR_TOP and lqr_controller is not None:
                if (not lqr_active) and lqr_should_enter(x_control):
                    lqr_active = True
                    idx_ref = NTOT - 1
                    print(f'  -> switch to LQR at t={meas_time:.2f}s, step={i}')
                elif lqr_active and lqr_should_exit(x_control):
                    lqr_active = False
                    idx_ref = min(N_SWINGUP, idx_ref)
                    print(f'  -> back to NMPC at t={meas_time:.2f}s, step={i}')

            idx_hist[i] = idx_ref
            u_ref_hist[i] = Uref[:, min(idx_ref, NTOT - 1)] if not lqr_active else np.zeros(2)
            status = 0
            try:
                if lqr_active:
                    u0 = lqr_controller.solve(x_control); st = 0.0; mode_hist[i] = 1
                else:
                    sw_slew = SWINGUP_TAU_SLEW_RATE if USE_SWINGUP_TAU_SLEW_LIMIT else None
                    u0, st, status = controller.solve(x_control, idx_ref, tau_prev, dt_prev,
                                                      slew_rate=sw_slew)
                    mode_hist[i] = 0
                u0 = np.atleast_1d(u0).astype(float)
                if not np.all(np.isfinite(u0)):
                    u0 = np.zeros(2); status = -1
            except Exception as exc:
                print(f'  solver error step {i}: {exc}'); u0 = np.zeros(2); st = 0.0; status = -2
            if status != 0:
                u0 = np.zeros(2)
            u_cmd_hist[i] = u0
            solver_status_hist[i] = status

            if USE_SWINGUP_SPEED_BRAKE and (not lqr_active):
                if (not brake_active) and abs(dq_hw[aj]) >= DQ_BRAKE:
                    brake_active = True
                elif brake_active and abs(dq_hw[aj]) <= DQ_BRAKE_RELEASE:
                    brake_active = False
            else:
                brake_active = False
            # Braking damps only the active joint, the other goes to zero (plant difference between the two notebooks)
            if brake_active:
                u_des = np.zeros(2)
                cap = (TAU1_MAX, TAU_MAX)[aj]
                u_des[aj] = -np.sign(dq_hw[aj]) * cap
            else:
                u_des = u0
            brake_hist[i] = brake_active

            tau_clip = np.clip(u_des, [-TAU1_MAX, -TAU_MAX], [TAU1_MAX, TAU_MAX])
            tau_cmd = tau_clip.copy()
            if lqr_active and USE_TOP_TAU_SLEW_LIMIT:
                du_max = TOP_TAU_SLEW_RATE * max(float(dt_prev), dt_ctrl)
                tau_cmd = np.clip(tau_clip, tau_prev - du_max, tau_prev + du_max)
            slew_active_hist[i] = bool(np.any(np.abs(tau_cmd - tau_clip) > 1e-10))
            tau_prev = tau_cmd
            solve_hist[i] = st

            _ta = time.perf_counter()
            client.set_torque([float(tau_cmd[0]), float(tau_cmd[1])], session_token)
            set_u_time_hist[i] = time.perf_counter() - _ta

            q_hist[i + 1] = q_raw; q_unwrapped_hist[i + 1] = q_cont; tau_hist[i] = tau_cmd

            deadline = t_step + dt_ctrl
            precise_sleep_until(deadline, BUSYWAIT_MS * 1e-3)
            now = time.perf_counter(); dt_act = now - t_step; t_step = now
            dt_hist[i] = dt_act; meas_time += dt_act; t_hist[i] = meas_time
        else:
            n_act = n_steps

    except KeyboardInterrupt:
        print('\n  !! interrupted by user')
        aborted = True; n_act = max(n_act, i)
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f'\n  !! loop crashed: {exc}')
        aborted = True; n_act = max(n_act, i)
    finally:
        try:
            client.set_torque([0.0, 0.0], session_token)
        except Exception as e:
            print(f'  (set_torque zero on stop failed: {e})')
        try:
            download_url = client.stop_experiment(session_token)
            print('\nSession stopped.')
            if download_url:
                print('Video / data:', download_url)
        except Exception as e:
            print(f'  (stop_experiment failed: {e})')

    # -- Offline diagnostics (tail of Cell 8, verbatim) --
    diagnostic_t0 = time.perf_counter()
    if n_act > 0 and not USE_STATE_PHASE_SWINGUP:
        idx_state_offline = 0
        for k in range(n_act):
            try:
                idx_state_offline, _, score, gap = select_reference_index(
                    x_meas_hist[k], idx_state_offline, sample_time_hist[k])
                idx_state_hist[k] = idx_state_offline
                idx_score_hist[k] = score
                idx_gap_hist[k] = gap
            except Exception:
                continue
    for k in range(max(0, n_act - 1)):
        try:
            x_pred = rk4_np(x_meas_hist[k], tau_hist[k], dt_ctrl)
            x_pred1_hist[k] = x_pred
            err = x_meas_hist[k + 1] - x_pred
            err[:2] = wrap_pi(err[:2])
            pred1_err_hist[k] = err
            dt_actual = dt_hist[k] if dt_hist[k] > 1e-6 else dt_ctrl
            x_pred_actual = rk4_np(x_meas_hist[k], tau_hist[k], dt_actual)
            x_pred1_actualdt_hist[k] = x_pred_actual
            err_actual = x_meas_hist[k + 1] - x_pred_actual
            err_actual[:2] = wrap_pi(err_actual[:2])
            pred1_actualdt_err_hist[k] = err_actual
        except Exception:
            continue
    diagnostic_post_time = time.perf_counter() - diagnostic_t0

    # -- Summary (verbatim) --
    if n_act == 0:
        print('No steps completed.')
    else:
        q1_final = np.degrees(abs(wrap_pi(q_hist[n_act, 0] - np.pi)))
        nz_solve = solve_hist[:n_act][solve_hist[:n_act] > 0]
        print('\n' + '=' * 56)
        print(f'Done. steps={n_act}  aborted={aborted}')
        print(f'  final q1 err : {q1_final:.2f} deg')
        print(f'  peak |dq|    : {np.max(np.abs(dq_f_hist[:n_act])):.1f} rad/s (filtered)')
        print(f'  peak raw|dq| : {np.max(np.abs(dq_hw_hist[:n_act])):.1f} rad/s')
        print(f'  peak |u1 cmd|: {np.max(np.abs(u_cmd_hist[:n_act, 0])) * 1e3:.1f} mNm (shoulder)')
        print(f'  peak |u2 cmd|: {np.max(np.abs(u_cmd_hist[:n_act, 1])) * 1e3:.1f} mNm (elbow)')
        print(f'  peak |tau1|  : {np.max(np.abs(tau_hist[:n_act, 0])) * 1e3:.1f} mNm (shoulder)')
        print(f'  peak |tau2|  : {np.max(np.abs(tau_hist[:n_act, 1])) * 1e3:.1f} mNm (elbow)')
        print(f'  slew active  : {100 * np.mean(slew_active_hist[:n_act]):.1f}% of steps')
        print(f'  brake steps  : {int(np.sum(brake_hist[:n_act]))} / {n_act}')
        print(f'  solver bad   : {int(np.sum(solver_status_hist[:n_act] != 0))} / {n_act}')
        print(f'  LQR steps    : {int(np.sum(mode_hist[:n_act] == 1))} / {n_act}')
        valid_pred = np.all(np.isfinite(pred1_err_hist[:n_act]), axis=1)
        if np.any(valid_pred):
            e = pred1_err_hist[:n_act][valid_pred]
            e_deg = np.degrees(np.abs(e[:, :2])); e_vel = np.abs(e[:, 2:])
            print(f'  1-step q err : med {np.median(e_deg):.2f} deg  p95 {np.percentile(e_deg, 95):.2f} deg')
            print(f'  1-step dq err: med {np.median(e_vel):.2f} rad/s p95 {np.percentile(e_vel, 95):.2f} rad/s')
        if nz_solve.size:
            print(f'  solve time   : mean {nz_solve.mean() * 1e3:.2f} ms  p99 {np.percentile(nz_solve, 99) * 1e3:.2f} ms')
        print(f'  mean API ms  : get_q {get_q_time_hist[:n_act].mean() * 1e3:.2f}  '
              f'get_dq {get_dq_time_hist[:n_act].mean() * 1e3:.2f}  set_u {set_u_time_hist[:n_act].mean() * 1e3:.2f}')
        print(f'  mean dt      : {dt_hist[:n_act].mean() * 1e3:.2f} ms (target {dt_ctrl * 1e3:.1f})')
        print(f'  diagnostics  : {diagnostic_post_time:.2f} s (offline, after hardware stop)')
        print('=' * 56)

    # -- Full log save (replaces Cell 9's interactive plots; all arrays kept, any plot can be made later) --
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_npz = data_dir(cfg) / f'ctrlrun_{plant}_{CELL_ID}_{stamp}.npz'
    np.savez_compressed(
        out_npz,
        q_hist=q_hist, q_unwrapped_hist=q_unwrapped_hist, dq_hw_hist=dq_hw_hist,
        dq_f_hist=dq_f_hist, tau_hist=tau_hist, u_cmd_hist=u_cmd_hist, u_ref_hist=u_ref_hist,
        dt_hist=dt_hist, t_hist=t_hist, sample_time_hist=sample_time_hist,
        x_meas_hist=x_meas_hist, idx_hist=idx_hist, idx_time_hist=idx_time_hist,
        idx_state_hist=idx_state_hist, idx_score_hist=idx_score_hist, idx_gap_hist=idx_gap_hist,
        solve_hist=solve_hist, solver_status_hist=solver_status_hist, mode_hist=mode_hist,
        slew_active_hist=slew_active_hist, brake_hist=brake_hist,
        get_q_time_hist=get_q_time_hist, get_dq_time_hist=get_dq_time_hist,
        set_u_time_hist=set_u_time_hist,
        x_pred1_hist=x_pred1_hist, pred1_err_hist=pred1_err_hist,
        x_pred1_actualdt_hist=x_pred1_actualdt_hist, pred1_actualdt_err_hist=pred1_actualdt_err_hist,
        Xref=Xref, Uref=Uref, n_act=n_act, aborted=aborted,
        plant=np.array(plant), cell_id=CELL_ID, dt_ctrl=dt_ctrl, dt_ref=dt_ref,
        N_SWINGUP=N_SWINGUP, NTOT=NTOT, N_MPC=N_MPC, T_SIM=t_sim,
        TAU1_MAX=TAU1_MAX, TAU_MAX=TAU_MAX, DQ_LIMIT=DQ_LIMIT, DQ_ABORT=DQ_ABORT,
        Q_DIAG=Q_DIAG, QF_DIAG=QF_DIAG, R_DIAG=R_DIAG,
        USE_SWINGUP_TAU_SLEW_LIMIT=spec['USE_SWINGUP_TAU_SLEW_LIMIT'],
        SWINGUP_TAU_SLEW_RATE=(spec['SWINGUP_TAU_SLEW_RATE']
                               if spec['SWINGUP_TAU_SLEW_RATE'] is not None else np.array([])),
        params_file=np.array(P['_file']), params_sha256=np.array(P['_sha256']),
    )
    print(f'log -> {out_npz.name}')

    # -- Cell 10 -- Video download (urllib, no IPython) --
    if download_url:
        video = data_dir(cfg) / f'ctrlrun_{plant}_{CELL_ID}_{stamp}.mp4'
        try:
            urllib.request.urlretrieve(download_url, video)
            print(f'video -> {video.name}')
        except Exception as e:
            print(f'video download failed ({e}); URL: {download_url}')

    return 3 if aborted else 0
