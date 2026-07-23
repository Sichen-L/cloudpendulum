# -*- coding: utf-8 -*-
"""Run the system-identification pipeline in order and stop on failure.

Examples:
  python run_all.py                      # Offline: RUN0 -> RUN1 -> RUN3 using configured fit_data
  python run_all.py --collect            # Full: RUN0 -> RUN1 -> RUN2 hardware collection -> RUN3 with new NPZ files
  python run_all.py --collect --stage hi # Collect only the high-speed stage, then refit
  python run_all.py --data a.npz b.npz   # Fit explicitly supplied data files
  python run_all.py --skip-selfcheck     # Skip RUN1

RUN2 starts CloudPendulum hardware experiments and is skipped by default.
Use --collect explicitly when switching devices; each new device must recollect NPZ data.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(tag, script, extra=()):
    sep = "=" * 64
    print(f"\n{sep}\n  {tag}  ->  python {script} {' '.join(extra)}\n{sep}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(HERE / script), *extra], cwd=str(HERE))
    dt = time.perf_counter() - t0
    ok = (r.returncode == 0)
    print(f"  {'OK' if ok else f'FAILED(exit={r.returncode})'}  [{dt:.1f}s]", flush=True)
    return ok


def load_config():
    cfg_path = HERE / "sysid_config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def data_dir(cfg):
    return (HERE / str(cfg.get("data_dir", "."))).resolve()


def list_data_files(cfg):
    ddir = data_dir(cfg)
    cell = cfg.get("cell_id", "*")
    return set(ddir.glob(f"system_id_data_{cell}_*.npz"))


def main():
    ap = argparse.ArgumentParser(description="Run the system-identification pipeline")
    ap.add_argument("--collect", action="store_true",
                    help="Include RUN2 hardware collection. This starts experiments and is skipped by default.")
    ap.add_argument("--stage", choices=["all", "free", "oid", "hi"], default="all",
                    help="RUN2 collection stage, used with --collect")
    ap.add_argument("--data", nargs="+", default=None, metavar="NPZ",
                    help="RUN3 offline refit data files. For a new device, use --collect to recollect NPZ files.")
    ap.add_argument("--fit-names", nargs="+", default=None, metavar="P",
                    help="Override the RUN3 parameter set")
    ap.add_argument("--skip-selfcheck", action="store_true", help="Skip RUN1 self-check")
    args = ap.parse_args()

    steps_done = []

    if not run_step("RUN 0  configuration", "sysid_make_config.py"):
        return 1
    steps_done.append("RUN0")

    if not args.skip_selfcheck:
        if not run_step("RUN 1  offline self-check", "sysid_run1_selfcheck.py"):
            print("\nSelf-check failed; stopping the pipeline. Fix the environment before hardware collection or fitting.")
            return 1
        steps_done.append("RUN1")

    collected_data = []
    if args.collect:
        cfg_before = load_config()
        before_npz = list_data_files(cfg_before)
        if not run_step("RUN 2  hardware collection", "sysid_run2_collect.py", ("--stage", args.stage)):
            print("\nCollection failed; stopping the pipeline. Incrementally saved data files are kept.")
            return 1
        steps_done.append("RUN2")
        cfg_after = load_config()
        after_npz = list_data_files(cfg_after)
        collected_data = sorted(after_npz - before_npz, key=lambda p: p.stat().st_mtime)
        if not collected_data:
            print("\nERROR: RUN2 finished but no newly generated NPZ files were detected. Stopping to avoid stale data.")
            return 1
        print("\nRUN2 newly collected data:")
        for p in collected_data:
            print(f"  {p.name}")
    else:
        print("\n(Skipping RUN2 hardware collection. Add --collect when needed; offline refits do not need it.)")

    extra = []
    if args.data:
        extra += list(args.data)
    elif collected_data:
        extra += [str(p) for p in collected_data]
    else:
        cfg = load_config()
        configured = list(cfg.get("fit_data", []) or [])
        if not configured:
            print("\nNo RUN3 data is available.")
            print("  Full system identification for a new device: python run_all.py --collect --stage all")
            print("  Offline refit of existing data:             python run_all.py --data your_data.npz")
            print("sysid_config.json intentionally keeps fit_data=[] by default to avoid reusing old-device NPZ files.")
            return 2
    if args.fit_names:
        extra += ["--fit-names", *args.fit_names]
    if not run_step("RUN 3  offline fitting", "sysid_run3_fit.py", tuple(extra)):
        return 1
    steps_done.append("RUN3")

    sep = "=" * 64
    print(f"\n{sep}\n  Pipeline complete: {' -> '.join(steps_done)}")
    print("  See RUN3 output for identified_params_*_fit_*.json, including _meta provenance.")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
