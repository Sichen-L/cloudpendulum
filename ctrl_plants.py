# -*- coding: utf-8 -*-
"""Shared CloudPendulum controller core.

Ported from the two compensation_controller notebooks. The mathematical plant
model is shared; acrobot and pendubot differ only through the PLANTS spec table.
This module does not import acados or cloudpendulumclient, so reference
generation can run offline with CasADi/IPOPT only.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import json
from pathlib import Path

import numpy as np
import casadi as ca

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
PLANTS = {
    'acrobot': dict(
        title='ACROBOT (elbow-driven)',
        active_joint=1,
        TAU_MAX=0.13,
        TAU1_MAX=0.005,
        tau_cost_weights=np.array([1.0, 1.0]),   # equal weights; [5,1] made the OCP
                                                  # numerically infeasible to solve
        R_DIAG=np.array([0.1, 0.05]),
        LQR_R=np.diag([1.0e6, 3.5e5]),
        TOP_TAU_SLEW_RATE=np.array([0.5, 3.0]),
        USE_SWINGUP_TAU_SLEW_LIMIT=False,
        SWINGUP_TAU_SLEW_RATE=None,
        model_name='acrobot_elbow',
        ocp_json='acrobot_elbow_ocp.json',
        ref_file='acrobot_reference_{cell}.npz',
        video_file='acrobot_recording.mp4',
        REF_T=3.0, REF_N=150,                 # elbow-driven: 3.0 s swing-up is enough
    ),
    'pendubot': dict(
        title='PENDUBOT (shoulder-driven)',
        active_joint=0,
        TAU_MAX=0.005,
        TAU1_MAX=0.13,
        tau_cost_weights=np.array([1.0, 1.0]),
        R_DIAG=np.array([0.05, 0.1]),
        LQR_R=np.diag([3.5e5, 1.0e6]),
        TOP_TAU_SLEW_RATE=np.array([3.0, 0.5]),
        USE_SWINGUP_TAU_SLEW_LIMIT=True,
        SWINGUP_TAU_SLEW_RATE=np.array([20.0, 100.0]),
        model_name='pendubot',
        ocp_json='pendubot_ocp.json',
        ref_file='pendubot_reference_{cell}.npz',
        video_file='pendubot_recording.mp4',
        REF_T=3.5, REF_N=175,                 # shoulder-driven + elbow damping: needs a
                                              # longer swing-up (3.0 s is infeasible); N
                                              # keeps dt_ref=20 ms so control timing is unchanged
    ),
}

# --------------------------------------------------------------------------
DQ_LIMIT = 20.0
DQ_ABORT = 50.0
CTRL_Hz = 400.0
dt_ctrl = 1.0 / CTRL_Hz
BUSYWAIT_MS = 1.0
T_SIM = 20.0
RECORD = True
x_goal = np.array([np.pi, 0.0, 0.0, 0.0])
g_acc = 9.81

REF_T = 3.0
REF_N = 150
REF_DQ_LIMIT = 20.0
HOLD_SECONDS = 3.0

N_MPC = 18
Q_DIAG = np.array([120., 50., 2.0, 2.0])
QF_DIAG = np.array([3000., 1200., 80., 80.])
USE_BASELINE_WRAPPED_NMPC_STATE = True
USE_STATE_PHASE_SWINGUP = False
REF_MAX_ADVANCE = 1
REF_MAX_LAG_STEPS = 10
REF_MAX_LEAD_STEPS = 2
REF_TIME_BIAS = 0.15
REF_VEL_WEIGHT = 0.80
USE_LQR_TOP = False
LQR_ENTER_Q1 = np.deg2rad(3.0)
LQR_ENTER_Q2 = np.deg2rad(5.0)
LQR_ENTER_DQ = 0.8
LQR_EXIT_Q1 = np.deg2rad(8.0)
LQR_EXIT_Q2 = np.deg2rad(12.0)
LQR_EXIT_DQ = 2.0
LQR_Q_DIAG = np.array([90., 32., 4., 3.])
USE_TOP_TAU_SLEW_LIMIT = True
USE_SWINGUP_SPEED_BRAKE = False
DQ_BRAKE = 30.0
DQ_BRAKE_RELEASE = 24.0


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def load_plant_params(param_path):
    param_path = Path(param_path)
    raw = param_path.read_bytes()
    _P = json.loads(raw.decode('utf-8'))
    P = dict(
        P1=_P['P1'], P2=_P['P2'], P3=_P['P3'], P6=_P.get('P6', _P['P3']),
        P4=_P['P4'], P5=_P['P5'],
        b1=_P['b1'], b2=_P['b2'], cf1=_P['cf1'], cf2=_P['cf2'],
        d1=_P.get('d1', 0.0), d2=_P.get('d2', 0.0),
        eps=_P.get('eps', 0.05),
    )
    P['_file'] = param_path.name
    P['_sha256'] = hashlib.sha256(raw).hexdigest()
    return P


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def make_rhs(P):
    P1, P2, P3, P6 = P['P1'], P['P2'], P['P3'], P['P6']
    P4, P5 = P['P4'], P['P5']
    b1, b2, cf1, cf2 = P['b1'], P['b2'], P['cf1'], P['cf2']
    d1, d2 = P['d1'], P['d2']
    EPS_VEL = P['eps']

    def rhs(x, u):
        q1, q2, dq1, dq2 = x[0], x[1], x[2], x[3]
        tau1, tau2 = u[0], u[1]
        cq2 = ca.cos(q2); sq2 = ca.sin(q2)
        M11 = P1 + 2 * P2 * cq2
        M12 = P3 + P2 * cq2
        M22 = P6
        h = P2 * sq2
        c1 = -2 * h * dq2 * dq1 - h * dq2 ** 2
        c2 = h * dq1 ** 2
        G1 = P4 * ca.sin(q1) + P5 * ca.sin(q1 + q2)
        G2 = P5 * ca.sin(q1 + q2)
        rhs1 = tau1 - c1 - G1 - b1 * dq1 - cf1 * ca.tanh(dq1 / EPS_VEL) - d1 * dq1 * ca.sqrt(dq1 ** 2 + 0.25)
        rhs2 = tau2 - c2 - G2 - b2 * dq2 - cf2 * ca.tanh(dq2 / EPS_VEL) - d2 * dq2 * ca.sqrt(dq2 ** 2 + 0.25)
        det = M11 * M22 - M12 ** 2
        ddq1 = (M22 * rhs1 - M12 * rhs2) / det
        ddq2 = (M11 * rhs2 - M12 * rhs1) / det
        return ca.vertcat(dq1, dq2, ddq1, ddq2)

    def rk4_step(x, u, dt):
        k1 = rhs(x, u)
        k2 = rhs(x + dt / 2 * k1, u)
        k3 = rhs(x + dt / 2 * k2, u)
        k4 = rhs(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4_np(x, u, dt):
        return np.array(rk4_step(ca.DM(x), ca.DM(u), dt)).flatten()

    return rhs, rk4_step, rk4_np


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def swingup_ocp(step_fn, spec, T=REF_T, N=REF_N, Xg=None, Ug=None, max_iter=5000,
                eps_vel=0.05):
    TAU1_MAX, TAU_MAX = spec['TAU1_MAX'], spec['TAU_MAX']
    w0, w1 = spec.get('tau_cost_weights', np.array([1.0, 1.0]))
    dt = T / N
    opti = ca.Opti()
    X = opti.variable(4, N + 1)
    U = opti.variable(2, N)     # row 0 = shoulder tau1, row 1 = elbow tau2
    opti.subject_to(X[:, 0] == [0, 0, 0, 0])
    J = 0
    for k in range(N):
        opti.subject_to(X[:, k + 1] == step_fn(X[:, k], U[:, k], dt))
        J += (w0 * U[0, k] ** 2 + w1 * U[1, k] ** 2) * dt
        opti.subject_to(opti.bounded(-TAU1_MAX, U[0, k], TAU1_MAX))
        opti.subject_to(opti.bounded(-TAU_MAX, U[1, k], TAU_MAX))
        opti.subject_to(opti.bounded(-REF_DQ_LIMIT, X[2, k], REF_DQ_LIMIT))
        opti.subject_to(opti.bounded(-REF_DQ_LIMIT, X[3, k], REF_DQ_LIMIT))
    opti.subject_to(ca.cos(X[0, N]) == ca.cos(np.pi))
    opti.subject_to(ca.sin(X[0, N]) == ca.sin(np.pi))
    opti.subject_to(X[1, N] == 0)
    opti.subject_to(X[2, N] == 0)
    opti.subject_to(X[3, N] == 0)
    J += 50.0 * ((X[0, N] - np.pi) ** 2 + X[1, N] ** 2 + X[2, N] ** 2 + X[3, N] ** 2)
    opti.minimize(J)
    opti.set_initial(X, Xg if Xg is not None else np.linspace([0, 0, 0, 0], [np.pi, 0, 0, 0], N + 1).T)
    if Ug is not None:
        opti.set_initial(U, Ug)
    opti.solver('ipopt', {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.max_iter': max_iter,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'ipopt.acceptable_iter': 20,
    })
    sol = opti.solve()
    return sol.value(X), np.atleast_2d(sol.value(U)).reshape(2, N)


def coarse_rk4(x, u, dt, eps_vel=0.05):
    # --------------------------------------------------------------------------
    _m1, _m2, _l1, _r1, _r2 = 0.10985804428893016, 0.07216498924919043, 0.05, 0.05, 0.0410314832240761
    _I1, _I2 = 0.00022984636956203125, 0.00018893779828634916
    _b1, _b2, _cf1, _cf2 = 0.002069048333819205, 0.0005078818014235568, 0.0008682636503908641, 0.0014269108758301003
    _J1 = _I1 + _m1 * _r1 ** 2; _J2 = _I2 + _m2 * _r2 ** 2

    def _rhs(x, u):
        q1, q2, dq1, dq2 = x[0], x[1], x[2], x[3]
        tau1, tau2 = u[0], u[1]
        cq2 = ca.cos(q2); sq2 = ca.sin(q2)
        M11 = _J1 + _J2 + _m2 * (_l1 ** 2 + 2 * _l1 * _r2 * cq2)
        M12 = _J2 + _m2 * _l1 * _r2 * cq2
        M22 = _J2
        h = _m2 * _l1 * _r2 * sq2
        c1 = -2 * h * dq2 * dq1 - h * dq2 ** 2
        c2 = h * dq1 ** 2
        G1 = g_acc * (_m1 * _r1 + _m2 * _l1) * ca.sin(q1) + g_acc * _m2 * _r2 * ca.sin(q1 + q2)
        G2 = g_acc * _m2 * _r2 * ca.sin(q1 + q2)
        rhs1 = tau1 - c1 - G1 - _b1 * dq1 - _cf1 * ca.tanh(dq1 / eps_vel)
        rhs2 = tau2 - c2 - G2 - _b2 * dq2 - _cf2 * ca.tanh(dq2 / eps_vel)
        det = M11 * M22 - M12 ** 2
        return ca.vertcat(dq1, dq2, (M22 * rhs1 - M12 * rhs2) / det, (M11 * rhs2 - M12 * rhs1) / det)

    k1 = _rhs(x, u); k2 = _rhs(x + dt / 2 * k1, u)
    k3 = _rhs(x + dt / 2 * k2, u); k4 = _rhs(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def ref_horizon(spec):
    """Per-plant swing-up horizon (T seconds, N steps), falling back to the globals."""
    return float(spec.get('REF_T', REF_T)), int(spec.get('REF_N', REF_N))


def save_reference(path, X, U, spec, P, created):
    T, N = ref_horizon(spec)
    np.savez(path, X=X, U=U, dt=T / N, T=T, N=N, dq_limit=REF_DQ_LIMIT,
             tau1_max=spec['TAU1_MAX'], tau_max=spec['TAU_MAX'],
             params_file=np.array(P['_file']), params_sha256=np.array(P['_sha256']),
             created=np.array(created))


def check_reference(path, spec, P=None, verbose=True):
    """Notebook-style cache validation plus provenance checks. Return (ok, X, U, dt, msgs)."""
    msgs = []
    path = Path(path)
    if not path.exists():
        return False, None, None, None, [f'{path.name}: does not exist']
    d = np.load(path)
    spec_T, spec_N = ref_horizon(spec)
    Ucache = np.atleast_2d(d['U'])
    cached_tau1_max = float(d.get('tau1_max', -1.0))
    cached_tau_max = float(d.get('tau_max', spec['TAU_MAX']))
    ok = (int(d['N']) == spec_N and abs(float(d['T']) - spec_T) < 1e-9 and
          abs(float(d.get('dq_limit', REF_DQ_LIMIT)) - REF_DQ_LIMIT) < 1e-9 and
          Ucache.shape[0] == 2 and
          abs(cached_tau1_max - spec['TAU1_MAX']) < 1e-12 and
          abs(cached_tau_max - spec['TAU_MAX']) < 1e-12)
    if not ok:
        msgs.append(f'{path.name}: N/T/dq_limit/tau bounds do not match current config')
        return False, None, None, None, msgs
    # --------------------------------------------------------------------------
    if P is not None:
        if 'params_sha256' in d.files:
            if str(d['params_sha256']) != P['_sha256']:
                msgs.append(f'{path.name}: parameter file changed (cached {str(d["params_file"])} '
                            f'!= current {P["_file"]}); regenerate reference')
                return False, None, None, None, msgs
        else:
            msgs.append(f'{path.name}: legacy cache without parameter provenance. '
                        'Confirm it matches current parameters or regenerate with ctrl_make_reference.py.')
    X = d['X']; U = Ucache.reshape(2, spec_N); dt = float(d['dt'])
    if verbose:
        print(f'  reference: loaded {path.name}  peak|dq|={np.max(np.abs(X[2:])):.1f} '
              f'peak|tau1|={np.max(np.abs(U[0])) * 1e3:.0f}mNm peak|tau2|={np.max(np.abs(U[1])) * 1e3:.0f}mNm')
    return True, X, U, dt, msgs


def append_hold(Xref_su, Uref_su, dt_ref):
    HOLD = int(round(HOLD_SECONDS / dt_ref))
    Xref = np.hstack([Xref_su, np.tile(x_goal.reshape(4, 1), HOLD)])
    Uref = np.hstack([Uref_su, np.zeros((2, HOLD))])
    return Xref, Uref, Uref_su.shape[1], Uref.shape[1]


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
from collections import deque
import time as _time


class StateEstimator:
    """Fuse position-derived and hardware velocity for both joints."""

    def __init__(self, n=2, win=3, alpha=0.55, pos_weight=0.70):
        self.alpha = alpha
        self.pos_weight = pos_weight
        self._buffers = [deque(maxlen=win) for _ in range(n)]
        self._ema = np.zeros(n)
        self._previous_q = None
        self._ready = False

    def update(self, q_raw, dq_hw, dt):
        # Wrapped differences prevent false velocity spikes at +/-pi crossings.
        if self._previous_q is not None and dt > 5e-4:
            dq_position = wrap_pi(q_raw - self._previous_q) / dt
        else:
            dq_position = dq_hw.copy()
        self._previous_q = q_raw.copy()

        blended = self.pos_weight * dq_position + (1 - self.pos_weight) * dq_hw

        for joint, value in enumerate(blended):
            self._buffers[joint].append(float(value))
        median = np.array([np.median(buffer) for buffer in self._buffers])

        self._ema = median.copy() if not self._ready else (
            self.alpha * median + (1 - self.alpha) * self._ema
        )
        self._ready = True
        return self._ema.copy()


def precise_sleep_until(deadline, busywait_s=0.001):
    """Sleep efficiently, then busy-wait over the final short interval."""
    coarse_sleep = deadline - _time.perf_counter() - busywait_s
    if coarse_sleep > 0:
        _time.sleep(coarse_sleep)
    while _time.perf_counter() < deadline:
        pass
