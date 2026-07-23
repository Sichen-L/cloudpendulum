# -*- coding: utf-8 -*-
"""RUN 1: offline self-check with synthetic data and no hardware.

The script generates a noisy synthetic trajectory from perturbed nominal
parameters, runs identify(), and checks that the identified model improves the
cross-validation RMSE. Passing this check means the fitting stack is usable.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from sysid_common import (FIT_NAMES, PARAM_NOMINAL, identify, rmse_deg, simulate)


def main():
    rng = np.random.default_rng(1)

    _n = 800; _dt = 1.0 / 200.0
    _t = np.arange(_n) * _dt
    _freqs = np.array([0.3, 0.7, 1.3, 2.1, 3.4]); _ph = rng.uniform(0, 2 * np.pi, _freqs.size)
    _u = 0.04 * np.sum(np.sin(2 * np.pi * _freqs[:, None] * _t[None, :] + _ph[:, None]), axis=0) / _freqs.size

    _fac = {'P1': 1.25, 'P2': 0.80, 'P3': 1.15, 'P6': 1.10, 'P4': 1.08, 'P5': 0.90,
            'b1': 1.4, 'b2': 1.3, 'cf1': 1.2, 'cf2': 1.5, 'd1': 2.0, 'd2': 1.6}
    true_params = {n: PARAM_NOMINAL[n] * _fac[n] for n in FIT_NAMES}
    _q0 = np.zeros((_n, 2)); _q0[0] = [1.8, 0.0]
    _ds_true = {'t': _t, 'u': _u, 'dt': _dt, 'q': _q0, 'dq': np.zeros((_n, 2))}
    _Qs, _DQs = simulate(_ds_true, true_params)
    _Qs_n = _Qs + rng.normal(0, np.radians(0.2), _Qs.shape)
    _DQs_n = _DQs + rng.normal(0, 0.05, _DQs.shape)
    syn = {'t': _t, 'u': _u, 'dt': _dt, 'q': _Qs_n, 'dq': _DQs_n, 'label': 'synthetic'}

    est = identify([syn])
    print(f'{"param":>5} | {"true":>10} | {"nominal":>10} | {"identified":>10}')
    for n in FIT_NAMES:
        print(f'{n:>5} | {true_params[n]:>10.3e} | {PARAM_NOMINAL[n]:>10.3e} | {est[n]:>10.3e}')
    r_nom = rmse_deg(syn, PARAM_NOMINAL)
    r_id = rmse_deg(syn, est)
    print(f'\nSynthetic cross-validation RMSE[deg]  nominal: {np.round(r_nom, 3)}  identified: {np.round(r_id, 3)}')

    ok = np.all(r_id < r_nom) and np.all(r_id < 1.0)
    msg = "Self-check passed: identification engine is working; proceed to RUN2 collection or RUN3 fitting."
    fail = "Self-check failed: identified model did not improve on nominal; check casadi/ipopt/scipy versions."
    print(f'\n{msg if ok else fail}')
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
