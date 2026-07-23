# -*- coding: utf-8 -*-
"""Merge CloudPendulum system-identification npz datasets.

Example:
  python sysid_merge_data.py system_id_data_203.npz system_id_data_203_hi.npz \
      system_id_data_203_hi_long.npz -o system_id_data_203_merged_long.npz
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

import numpy as np

from system_id_data_io import merge_dataset_files, save_datasets


def describe(ds):
    dt = np.asarray(ds["dt"])
    dtv = np.full(len(ds["t"]) - 1, float(dt)) if dt.ndim == 0 else dt[: len(ds["t"]) - 1]
    duration = float(ds["t"][-1] - ds["t"][0]) if len(ds["t"]) else 0.0
    peak_dq = float(np.max(np.abs(ds["dq"]))) if len(ds["t"]) else 0.0
    return (
        f"{ds['label']}: N={len(ds['t'])}, duration={duration:.2f}s, "
        f"dt={1e3 * np.median(dtv):.2f}+/-{1e3 * np.std(dtv):.2f}ms, "
        f"peak|dq|={peak_dq:.1f}rad/s, aborted={ds['aborted']}"
    )


def main():
    ap = argparse.ArgumentParser(description="Merge CloudPendulum system-id npz datasets")
    ap.add_argument("inputs", nargs="+", help="Input system_id_data*.npz files")
    ap.add_argument("-o", "--output", required=True, help="Merged output npz file")
    ap.add_argument("--prefix-labels", action="store_true", help="Prefix labels with source file stem")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))

    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose another --output to avoid overwriting data")

    groups = merge_dataset_files(inputs, label_prefix=args.prefix_labels)
    save_datasets(groups, out)

    print(f"Saved merged dataset to {out}")
    for group in ("free", "exc"):
        print(f"{group}: {len(groups.get(group, []))} trajectories")
        for ds in groups.get(group, []):
            print("  " + describe(ds))


if __name__ == "__main__":
    main()
