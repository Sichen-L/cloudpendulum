# -*- coding: utf-8 -*-
"""Generate paper-ready SysID plots/tables from existing data and params.

This script does not run system identification. It only reads collected npz
files plus an identified parameter json, then writes:
  - one-step prediction error CSV/PNG
  - finite-horizon terminal error CSV/PNG
  - system parameter table CSV/MD/PNG

Example:
  python sysid_plot_report_artifacts.py --params identified_params_203_fit_xxx.json system_id_data_203_xxx.npz
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import csv
import json
from pathlib import Path

import casadi as ca
import numpy as np

from sysid_common import (FIT_NAMES, PARAM_NAMES, PARAM_NOMINAL, load_fit_groups,
                          param_vec, rk4_p, simulate, step_dt, usable_sets)


def angle_error(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def one_step_errors(ds, params):
    pv = ca.DM(param_vec(params))
    q_err = []
    dq_err = []
    for k in range(len(ds["t"]) - 1):
        x = ca.DM(np.r_[ds["q"][k], ds["dq"][k]])
        pred = np.asarray(rk4_p(x, float(ds["u"][k]), pv, step_dt(ds, k))).reshape(4)
        q_err.append(angle_error(pred[:2], ds["q"][k + 1]))
        dq_err.append(pred[2:] - ds["dq"][k + 1])
    q_err = np.asarray(q_err)
    dq_err = np.asarray(dq_err)
    return {
        "q_rmse_deg": np.degrees(np.sqrt(np.mean(q_err ** 2, axis=0))),
        "dq_rmse_deg_s": np.degrees(np.sqrt(np.mean(dq_err ** 2, axis=0))),
    }


def rollout_errors(ds, params):
    q_sim, dq_sim = simulate(ds, params)
    q_err = angle_error(q_sim, ds["q"])
    dq_err = dq_sim - ds["dq"]
    return {
        "duration_s": float(ds["t"][-1] - ds["t"][0]),
        "q_rmse_deg": np.degrees(np.sqrt(np.mean(q_err ** 2, axis=0))),
        "dq_rmse_deg_s": np.degrees(np.sqrt(np.mean(dq_err ** 2, axis=0))),
        "terminal_q_deg": np.degrees(q_err[-1]),
        "terminal_dq_deg_s": np.degrees(dq_err[-1]),
        "terminal_q_norm_deg": float(np.degrees(np.linalg.norm(q_err[-1]))),
        "terminal_dq_norm_deg_s": float(np.degrees(np.linalg.norm(dq_err[-1]))),
    }


def validation_rows(valid_sets, identified):
    rows = []
    for ds in valid_sets:
        for model_name, params in (("nominal", PARAM_NOMINAL), ("identified", identified)):
            one = one_step_errors(ds, params)
            roll = rollout_errors(ds, params)
            rows.append({
                "dataset": ds["label"],
                "model": model_name,
                "duration_s": roll["duration_s"],
                "one_q1_rmse_deg": one["q_rmse_deg"][0],
                "one_q2_rmse_deg": one["q_rmse_deg"][1],
                "one_dq1_rmse_deg_s": one["dq_rmse_deg_s"][0],
                "one_dq2_rmse_deg_s": one["dq_rmse_deg_s"][1],
                "rollout_q1_rmse_deg": roll["q_rmse_deg"][0],
                "rollout_q2_rmse_deg": roll["q_rmse_deg"][1],
                "rollout_dq1_rmse_deg_s": roll["dq_rmse_deg_s"][0],
                "rollout_dq2_rmse_deg_s": roll["dq_rmse_deg_s"][1],
                "terminal_q1_deg": roll["terminal_q_deg"][0],
                "terminal_q2_deg": roll["terminal_q_deg"][1],
                "terminal_dq1_deg_s": roll["terminal_dq_deg_s"][0],
                "terminal_dq2_deg_s": roll["terminal_dq_deg_s"][1],
                "terminal_q_norm_deg": roll["terminal_q_norm_deg"],
                "terminal_dq_norm_deg_s": roll["terminal_dq_norm_deg_s"],
            })
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parameter_rows(identified, fit_names):
    rows = []
    for name in PARAM_NAMES:
        nominal = float(PARAM_NOMINAL[name])
        value = float(identified.get(name, PARAM_NOMINAL[name]))
        delta = value - nominal
        pct = 100.0 * delta / nominal if nominal else float("nan")
        rows.append({
            "parameter": name,
            "nominal": nominal,
            "identified": value,
            "delta": delta,
            "delta_percent": pct,
            "fitted": "yes" if name in fit_names else "no",
        })
    return rows


def write_parameter_markdown(path, rows):
    lines = [
        "| Parameter | Nominal | Identified | Delta | Delta [%] | Fitted |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        pct = row["delta_percent"]
        pct_text = "" if not np.isfinite(pct) else f"{pct:.2f}"
        lines.append(
            f"| {row['parameter']} | {row['nominal']:.6g} | {row['identified']:.6g} | "
            f"{row['delta']:.6g} | {pct_text} | {row['fitted']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_data_files(args_data, params_path, params):
    files = list(args_data)
    if not files:
        files = list(params.get("_meta", {}).get("data_files", []))
    if not files:
        raise SystemExit("No npz data files provided, and params _meta.data_files is empty.")

    out = []
    for item in files:
        p = Path(item)
        if p.is_absolute() and p.exists():
            out.append(p)
            continue
        candidates = [Path.cwd() / p, params_path.parent / p]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError(f"Data file not found: {item}")
        out.append(found)
    return out


def select_validation_sets(groups, mode):
    free_ok, exc_ok = usable_sets(groups)
    if mode == "all":
        valid = free_ok + exc_ok
    else:
        valid = []
        if free_ok:
            valid.append(free_ok[-1])
        if exc_ok:
            valid.append(exc_ok[-1])
    if not valid:
        raise SystemExit("No usable validation datasets found.")
    return valid


def save_plots(out_dir, prefix, rows, param_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in rows})
    nominal = {r["dataset"]: r for r in rows if r["model"] == "nominal"}
    identified = {r["dataset"]: r for r in rows if r["model"] == "identified"}
    x = np.arange(len(datasets))
    width = 0.36

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].bar(x - width / 2, [np.hypot(nominal[d]["one_q1_rmse_deg"], nominal[d]["one_q2_rmse_deg"]) for d in datasets],
              width, label="nominal")
    ax[0].bar(x + width / 2, [np.hypot(identified[d]["one_q1_rmse_deg"], identified[d]["one_q2_rmse_deg"]) for d in datasets],
              width, label="identified")
    ax[0].set_ylabel("q RMSE [deg]")
    ax[0].set_title("One-step prediction error")
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend()
    ax[1].bar(x - width / 2, [np.hypot(nominal[d]["one_dq1_rmse_deg_s"], nominal[d]["one_dq2_rmse_deg_s"]) for d in datasets],
              width, label="nominal")
    ax[1].bar(x + width / 2, [np.hypot(identified[d]["one_dq1_rmse_deg_s"], identified[d]["one_dq2_rmse_deg_s"]) for d in datasets],
              width, label="identified")
    ax[1].set_ylabel("dq RMSE [deg/s]")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(datasets, rotation=15, ha="right")
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    one_png = out_dir / f"{prefix}_onestep_error.png"
    fig.savefig(one_png, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].bar(x - width / 2, [nominal[d]["terminal_q_norm_deg"] for d in datasets], width, label="nominal")
    ax[0].bar(x + width / 2, [identified[d]["terminal_q_norm_deg"] for d in datasets], width, label="identified")
    ax[0].set_ylabel("terminal |q error| [deg]")
    ax[0].set_title("Finite-horizon terminal error")
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend()
    ax[1].bar(x - width / 2, [nominal[d]["terminal_dq_norm_deg_s"] for d in datasets], width, label="nominal")
    ax[1].bar(x + width / 2, [identified[d]["terminal_dq_norm_deg_s"] for d in datasets], width, label="identified")
    ax[1].set_ylabel("terminal |dq error| [deg/s]")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(datasets, rotation=15, ha="right")
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    terminal_png = out_dir / f"{prefix}_terminal_error.png"
    fig.savefig(terminal_png, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(param_rows) + 1.0))
    ax.axis("off")
    table_data = [
        [r["parameter"], f"{r['nominal']:.3e}", f"{r['identified']:.3e}",
         "" if not np.isfinite(r["delta_percent"]) else f"{r['delta_percent']:.1f}%",
         r["fitted"]]
        for r in param_rows
    ]
    table = ax.table(cellText=table_data,
                     colLabels=["param", "nominal", "identified", "delta %", "fitted"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    param_png = out_dir / f"{prefix}_parameter_table.png"
    fig.tight_layout()
    fig.savefig(param_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [one_png, terminal_png, param_png]


def main():
    parser = argparse.ArgumentParser(description="Generate SysID report plots/tables without refitting.")
    parser.add_argument("data", nargs="*", help="Collected npz files. If omitted, use _meta.data_files from params json.")
    parser.add_argument("--params", required=True, help="identified_params_*.json")
    parser.add_argument("--out-dir", default=None, help="Output folder, default: params file folder")
    parser.add_argument("--prefix", default=None, help="Output filename prefix, default: params file stem")
    parser.add_argument("--validation", choices=["holdout", "all"], default="holdout",
                        help="holdout uses last usable free and exc trajectory, matching RUN3.")
    args = parser.parse_args()

    params_path = Path(args.params).resolve()
    params = json.loads(params_path.read_text(encoding="utf-8"))
    identified = dict(PARAM_NOMINAL)
    identified.update({k: v for k, v in params.items() if k in PARAM_NAMES})
    fit_names = params.get("_meta", {}).get("fit_names", FIT_NAMES)

    data_files = resolve_data_files(args.data, params_path, params)
    groups = load_fit_groups(data_files)
    valid_sets = select_validation_sets(groups, args.validation)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else params_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or params_path.stem

    rows = validation_rows(valid_sets, identified)
    one_csv = out_dir / f"{prefix}_onestep_error.csv"
    terminal_csv = out_dir / f"{prefix}_terminal_error.csv"
    param_csv = out_dir / f"{prefix}_parameter_table.csv"
    param_md = out_dir / f"{prefix}_parameter_table.md"

    write_csv(one_csv, rows, [
        "dataset", "model", "duration_s",
        "one_q1_rmse_deg", "one_q2_rmse_deg",
        "one_dq1_rmse_deg_s", "one_dq2_rmse_deg_s",
    ])
    write_csv(terminal_csv, rows, [
        "dataset", "model", "duration_s",
        "terminal_q1_deg", "terminal_q2_deg",
        "terminal_dq1_deg_s", "terminal_dq2_deg_s",
        "terminal_q_norm_deg", "terminal_dq_norm_deg_s",
        "rollout_q1_rmse_deg", "rollout_q2_rmse_deg",
        "rollout_dq1_rmse_deg_s", "rollout_dq2_rmse_deg_s",
    ])

    p_rows = parameter_rows(identified, fit_names)
    write_csv(param_csv, p_rows, ["parameter", "nominal", "identified", "delta", "delta_percent", "fitted"])
    write_parameter_markdown(param_md, p_rows)

    written = [one_csv, terminal_csv, param_csv, param_md]
    try:
        written.extend(save_plots(out_dir, prefix, rows, p_rows))
    except ImportError:
        print("matplotlib is unavailable; wrote CSV/Markdown only.")

    print("Wrote report artifacts:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
