# -*- coding: utf-8 -*-
"""RUN 3: offline fitting from NPZ data to identified_params_*.json.

Ported from the notebook identifiability, train/validation split, fitting,
cross-validation, and save cells. Each condition group keeps its final trajectory
for independent validation; all others are used for training.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sysid_common import (FIT_NAMES, PARAM_NAMES, PARAM_NOMINAL, data_dir,
                          identifiability_report, identify, load_config,
                          load_fit_groups, resolve_data, rmse_deg, simulate,
                          usable_sets)


def warn_timing_quality(datasets, expected_dt):
    bad = []
    severe = []
    for ds in datasets:
        dt = np.asarray(ds["dt"], dtype=float)
        if dt.ndim == 0 or dt.size == 0:
            continue
        median = float(np.median(dt))
        p95 = float(np.percentile(dt, 95))
        max_dt = float(np.max(dt))
        short_frac = float(np.mean(dt < 0.5 * expected_dt))
        long_frac = float(np.mean(dt > 4.0 * expected_dt))
        if median < 0.7 * expected_dt or p95 > 4.0 * expected_dt or long_frac > 0.02:
            bad.append((ds["label"], median, p95, max_dt, short_frac, long_frac))
        if median < 0.5 * expected_dt and short_frac > 0.25:
            severe.append(ds["label"])
    if not bad:
        return False
    print("\nWARNING: timing jitter detected in collected data.")
    print(f"  expected dt ~= {1e3 * expected_dt:.2f} ms")
    for label, median, p95, max_dt, short_frac, long_frac in bad[:8]:
        print(f"  {label}: median={1e3 * median:.2f} ms, p95={1e3 * p95:.2f} ms, "
              f"max={1e3 * max_dt:.2f} ms, short={100 * short_frac:.1f}%, "
              f"long={100 * long_frac:.1f}%")
    if len(bad) > 8:
        print(f"  ... {len(bad) - 8} more datasets with timing jitter")
    print("  If full-parameter fitting fails, use a smaller --fit-names set or recollect with the updated RUN2 scheduler.")
    return bool(severe)


def main():
    ap = argparse.ArgumentParser(description="RUN 3: offline identification fitting")
    ap.add_argument("data", nargs="*", help="NPZ data files; defaults to sysid_config.json fit_data")
    ap.add_argument("--fit-names", nargs="+", default=None, metavar="P",
                    help=f"override the default identified parameter set (default {' '.join(FIT_NAMES)})")
    ap.add_argument("--out", default=None, help="output JSON path; defaults to timestamped non-overwriting name")
    ap.add_argument("--skip-identifiability", action="store_true", help="skip the Cell 8c identifiability report")
    ap.add_argument("--no-plot", action="store_true", help="do not save validation PNG")
    ap.add_argument("--allow-bad-timing", action="store_true",
                    help="allow severe timing-jitter data to continue fitting; diagnostic use only")
    args = ap.parse_args()

    cfg = load_config(need_token=False)
    data_files = args.data or cfg.get("fit_data", [])
    if not data_files:
        print("ERROR: no RUN3 data files were provided.")
        print("  New devices must recollect NPZ data first: python run_all.py --collect --stage all")
        print("  For an offline refit, pass existing data explicitly: python sysid_run3_fit.py your_data.npz")
        print("  Recommended: keep sysid_config.json fit_data=[] to avoid stale data reuse.")
        return 1
    data_files = [str(resolve_data(cfg, f)) for f in data_files]
    fit_names = args.fit_names or FIT_NAMES
    bad = [n for n in fit_names if n not in PARAM_NAMES or n == 'eps']
    if bad:
        print(f"ERROR: invalid fit-names {bad} (choices: {[n for n in PARAM_NAMES if n != 'eps']})")
        return 1

    print("Data files:")
    for f in data_files:
        print(f"  {Path(f).name}")
    groups = load_fit_groups(data_files)
    free_ok, exc_ok = usable_sets(groups)
    severe_timing = warn_timing_quality(free_ok + exc_ok, 1.0 / float(cfg["ctrl_hz"]))
    if severe_timing and not args.allow_bad_timing:
        print("\nERROR: severe timing jitter detected; refusing to fit unreliable parameters.")
        print("  Recollect the npz data with the updated RUN2 scheduler.")
        print("  For diagnosis only, rerun with --allow-bad-timing.")
        return 1
    n_free_all = len(groups.get('free', [])); n_exc_all = len(groups.get('exc', []))
    print(f"Usable trajectories: free {len(free_ok)}/{n_free_all}, exc {len(exc_ok)}/{n_exc_all} "
          f"(filter: >20 steps and not aborted)")
    assert len(free_ok) >= 2 and len(exc_ok) >= 2, \
        'Need at least 2 valid free-swing and 2 valid excitation trajectories for train/validation split.'

    # Split: keep the final trajectory from each condition group for independent validation
    train = free_ok[:-1] + exc_ok[:-1]
    valid_sets = [free_ok[-1], exc_ok[-1]]
    valid = exc_ok[-1]
    print(f"Training {len(train)} trajectories / validation 2 trajectories (free: {valid_sets[0]['label']} | exc: {valid['label']})")

    # Identifiability
    if not args.skip_identifiability:
        print("\n-- Identifiability analysis (Cell 8c) --")
        weak = identifiability_report(train, fit_names)
        if weak:
            print(f"\nHint: weak parameters {weak}. If they hit bounds after fitting, rerun with --fit-names excluding them.")

    # Identification
    print("\n-- Identification (Cell 9) --")
    identified = identify(train, fit_names)
    print('Identified parameters:')
    for n in fit_names:
        print(f'  {n:>4}: {PARAM_NOMINAL[n]:>10.3e}  ->  {identified[n]:>10.3e}')

    print('\nIndependent validation RMSE[deg]:')
    rmse_meta = {}
    for ds in valid_sets:
        r_nom = rmse_deg(ds, PARAM_NOMINAL); r_id = rmse_deg(ds, identified)
        rmse_meta[ds['label']] = {"nominal": [round(float(v), 4) for v in r_nom],
                                  "identified": [round(float(v), 4) for v in r_id]}
        print(f"  {ds['label']:<24} nominal {np.round(r_nom, 3)}  identified {np.round(r_id, 3)}")

    # Save with _meta provenance
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = (Path(args.out) if args.out
                else data_dir(cfg) / f"identified_params_{cfg['cell_id']}_fit_{stamp}.json")
    if out_path.exists():
        print(f"ERROR: {out_path.name} already exists; refusing to overwrite. Use --out.")
        return 1
    final_params = dict(PARAM_NOMINAL)
    final_params.update(identified)
    final_params["_meta"] = {
        "created": stamp,
        "script": "sysid_run3_fit.py",
        "data_files": [Path(f).name for f in data_files],
        "fit_names": list(fit_names),
        "train_sets": len(train), "valid_sets": [d['label'] for d in valid_sets],
        "valid_rmse_deg": rmse_meta,
    }
    out_path.write_text(json.dumps(final_params, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f'\nSaved {out_path.name}:')
    for n in PARAM_NAMES:
        tag = '  <== identified' if n in fit_names else ''
        print(f'  {n:>4} = {final_params[n]:.6g}{tag}')

    # Compare against production parameters if present
    prod = resolve_data(cfg, "identified_params_203_long_inertia_allfriction.json")
    if prod.exists():
        prev = json.loads(prod.read_text(encoding="utf-8"))
        print(f'\nCompared with active parameters ({prod.name}):')
        print(f'{"param":>5} | {"active":>11} | {"current":>11} | {"change":>8}')
        for n in fit_names:
            if n in prev:
                a, b = float(prev[n]), final_params[n]
                pct = 100 * (b - a) / a if a else float("inf")
                print(f'{n:>5} | {a:>11.4e} | {b:>11.4e} | {pct:>+7.1f}%')

    # Validation plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            Qs_nom, _ = simulate(valid, PARAM_NOMINAL)
            Qs_id, _ = simulate(valid, identified)
            tt = valid['t']
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            for i, name in enumerate(['q1', 'q2']):
                ax[i].plot(tt, np.degrees(valid['q'][:, i]), 'k', lw=1.2, label=f'{name} meas')
                ax[i].plot(tt, np.degrees(Qs_nom[:, i]), 'r--', lw=1, label=f'{name} sim (nominal)')
                ax[i].plot(tt, np.degrees(Qs_id[:, i]), 'g-', lw=1, label=f'{name} sim (identified)')
                ax[i].set_ylabel(f'{name} [deg]'); ax[i].legend(fontsize=8); ax[i].grid(alpha=.3)
            ax[1].set_xlabel('t [s]')
            plt.suptitle('Cross-validation: measured vs simulated', fontweight='bold')
            plt.tight_layout()
            png = out_path.with_suffix(".png")
            plt.savefig(png, dpi=110)
            print(f'validation plot: {png.name}')
        except ImportError:
            print('(matplotlib unavailable; skipping validation plot)')

    print("\nNext: to make a control notebook use these parameters, set PARAM_FILE to "
          f"'{out_path.name}'. Active parameter files are unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
