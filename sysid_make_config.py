# -*- coding: utf-8 -*-
"""RUN 0: create sysid_config.json once.

The config is local to sysid_pipeline. Tokens are read in this order:
CLOUDPENDULUM_TOKEN, an existing sysid_config.json, then an optional local
system_identification_acrobot.ipynb. Existing configs are not overwritten unless
--force is supplied.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_NB_CANDIDATES = [HERE / "system_identification_acrobot.ipynb"]
CONFIG = HERE / "sysid_config.json"


def token_from_local_notebook():
    notebook = next((p for p in _NB_CANDIDATES if p.exists()), None)
    if notebook is None:
        return None
    raw = notebook.read_text(encoding="utf-8")
    matches = re.findall(r"USER_TOKEN\s*=\s*'(\d+)'", raw)
    if len(matches) != 1:
        print(f"WARNING: found {len(matches)} USER_TOKEN assignments in {notebook.name}; not importing a token.")
        return None
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description="Create local sysid_config.json")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing config file")
    args = ap.parse_args()

    if CONFIG.exists() and not args.force:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        tok = cfg.get("user_token", "")
        suffix = tok[-3:] if len(tok) >= 6 else ""
        print(f"{CONFIG.name} already exists (token={tok[:3]}...{suffix}, len={len(tok)}); not overwriting. Use --force to rebuild.")
        return 0

    old_cfg = {}
    if CONFIG.exists():
        old_cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    token = (
        os.environ.get("CLOUDPENDULUM_TOKEN", "").strip()
        or str(old_cfg.get("user_token", "")).strip()
        or token_from_local_notebook()
        or ""
    )

    cfg = {
        "user_token": token,
        "data_dir": ".",
        "experiment_type": "DoublePendulum",
        "cell_id": 203,
        "tau_max_id": 0.06,
        "dq_abort": 40.0,
        "ctrl_hz": 200.0,
        "inter_run_cooldown": 35.0,
        "start_retry_max": 4,
        "fit_data": [],
        "design_params": "",
    }
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    back = json.loads(CONFIG.read_text(encoding="utf-8"))["user_token"]
    assert back == token, "config read-back check failed: token mismatch"
    print(f"Created {CONFIG.name}")
    if token:
        print(f"  token: {token[:3]}...{token[-3:]}(len={len(token)})")
    else:
        print("  token: <empty> (offline RUN1/RUN3 can run; fill user_token or set CLOUDPENDULUM_TOKEN before RUN2)")
    print("  cell_id=203  tau_max_id=0.06  ctrl_hz=200")
    print("Tip: after token rotation, edit user_token here or set CLOUDPENDULUM_TOKEN to override it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
