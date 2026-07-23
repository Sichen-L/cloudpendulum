# -*- coding: utf-8 -*-
"""Validate identified parameters by actual-dt one-step prediction error.

This is intentionally separate from fitting. It answers the control-side question:
"does the current parameter file still explain this real device under a fresh
excitation?"  If the check fails after switching cell_id, rerun the SysID notebook
or RUN2/RUN3 for that device before running the controller.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
from pathlib import Path

import casadi as ca
import numpy as np

from sysid_common import load_config, load_fit_groups, param_vec, rk4_p, resolve_data


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def dt_at(ds, k):
    dt = np.asarray(ds["dt"])
    return float(dt) if dt.ndim == 0 else float(dt[k])


def one_step_errors(ds, params):
    pv = ca.DM(param_vec(params))
    angle = []
    velocity = []
    n = len(ds["t"])
    for k in range(n - 1):
        x0 = ca.DM([ds["q"][k, 0], ds["q"][k, 1], ds["dq"][k, 0], ds["dq"][k, 1]])
        x1 = rk4_p(x0, float(ds["u"][k]), pv, dt_at(ds, k))
        pred = np.asarray(x1, dtype=float).reshape(4)
        actual = np.r_[ds["q"][k + 1], ds["dq"][k + 1]]
        angle.append(float(np.linalg.norm(np.degrees(wrap_pi(actual[:2] - pred[:2])))))
        velocity.append(float(np.linalg.norm(actual[2:] - pred[2:])))
    return np.asarray(angle), np.asarray(velocity)


def describe_metric(values):
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "n": int(values.size),
    }


def find_latest_params(cfg):
    ddir = resolve_data(cfg, ".")
    cell = cfg.get("cell_id", "*")
    candidates = sorted(ddir.glob(f"identified_params_{cell}_fit_*.json"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    legacy = ddir / f"identified_params_{cell}_long_inertia_allfriction.json"
    return legacy if legacy.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Check one-step prediction error for identified parameters")
    ap.add_argument("data", nargs="+", help="Fresh excitation/system-id npz file(s)")
    ap.add_argument("--params", default=None,
                    help="Parameter json. Default: latest identified_params_{cell}_fit_*.json, then legacy long_inertia file")
    ap.add_argument("--angle-median-deg", type=float, default=1.0,
                    help="Fail if median one-step angle error exceeds this value")
    ap.add_argument("--angle-p95-deg", type=float, default=5.0,
                    help="Fail if p95 one-step angle error exceeds this value")
    ap.add_argument("--velocity-p95", type=float, default=8.0,
                    help="Fail if p95 one-step velocity error [rad/s] exceeds this value")
    ap.add_argument("--include-aborted", action="store_true", help="Include aborted trajectories")
    args = ap.parse_args()

    cfg = load_config(need_token=False)
    param_path = resolve_data(cfg, args.params) if args.params else find_latest_params(cfg)
    if param_path is None or not param_path.exists():
        print("ERROR: no identified parameter file was found for the current device.")
        print("Open sysid_workflow.ipynb and rerun full system identification first.")
        return 2

    params = json.loads(param_path.read_text(encoding="utf-8"))
    data_files = [str(resolve_data(cfg, f)) for f in args.data]
    groups = load_fit_groups(data_files)
    datasets = groups.get("free", []) + groups.get("exc", [])
    datasets = [d for d in datasets if len(d["t"]) > 2 and (args.include_aborted or not d["aborted"])]
    if not datasets:
        print("ERROR: no valid trajectories are available for validation.")
        return 2

    all_angle = []
    all_velocity = []
    print(f"params: {param_path.name}")
    print("data:")
    for f in data_files:
        print(f"  {Path(f).name}")
    print("\nOne-step prediction error:")
    for ds in datasets:
        angle, velocity = one_step_errors(ds, params)
        all_angle.append(angle)
        all_velocity.append(velocity)
        ma = describe_metric(angle)
        mv = describe_metric(velocity)
        print(f"  {ds['label']:<28} angle med/p95/max={ma['median']:.3f}/{ma['p95']:.3f}/{ma['max']:.3f} deg"
              f" | vel p95={mv['p95']:.3f} rad/s | n={ma['n']}")

    angle = np.concatenate(all_angle)
    velocity = np.concatenate(all_velocity)
    ma = describe_metric(angle)
    mv = describe_metric(velocity)
    print("\nAggregate:")
    print(f"  angle median={ma['median']:.3f} deg, p95={ma['p95']:.3f} deg, max={ma['max']:.3f} deg, n={ma['n']}")
    print(f"  velocity p95={mv['p95']:.3f} rad/s")

    failed = (
        ma["median"] > args.angle_median_deg
        or ma["p95"] > args.angle_p95_deg
        or mv["p95"] > args.velocity_p95
    )
    if failed:
        print("\nFAIL: current parameters did not pass the one-step prediction check on fresh excitation data.")
        print("Manually rerun full system identification in sysid_workflow.ipynb before control experiments.")
        return 1

    print("\nPASS: current parameters passed the one-step prediction check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
