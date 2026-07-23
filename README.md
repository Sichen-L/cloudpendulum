# CloudPendulum — System Identification & NMPC Swing-Up (1st Evaluation Snapshot)

Development-stage system identification and nonlinear model predictive control (NMPC)
for the **CloudPendulum** underactuated double pendulum, in both **acrobot** (elbow-driven)
and **pendubot** (shoulder-driven) configurations. This folder is the frozen snapshot that
accompanies the 1st-evaluation report
(`System_Identification_and_Development_Stage_NMPC...pdf`).

The task: swing the pendulum from hanging to upright and hold it there using a single
actuated joint, on remote hardware whose parameters are unknown and cell-specific.

---

## How the system works

The pipeline has three stages. Only stages 2–3 live in this folder; stage 1 (system
identification) runs upstream and produces the parameter JSON used here.

```
 (1) System ID            (2) Reference generation        (3) Real-time NMPC (hardware)
 hardware data      ->    ctrl_make_reference.py     ->   ctrl_run_{plant}.py
 => identified_params      => {plant}_reference_{cell}.npz  => ctrlrun_{plant}_{cell}_*.npz
    _{cell}_fit_*.json        (swing-up OCP, offline)          (acados SQP-RTI, ~400 Hz)
```

1. **System identification (upstream).** A 12-parameter base-parameter model
   (inertia `P1..P6`, gravity `P4,P5`, per-joint friction `b,cf,d`) is fitted from real
   free-swing / excitation data. Result: `identified_params_{cell}_fit_*.json`.

2. **Reference generation — `ctrl_make_reference.py`** *(offline, CasADi/IPOPT, no hardware)*.
   Solves a multiple-shooting swing-up optimal control problem from the identified model and
   writes `{plant}_reference_{cell}.npz`. The reference carries the parameter **sha256** so
   the controller can reject a reference that does not match the loaded parameters. The OCP
   is warm-started from a canonical seed (`swingup_reference_seed_{plant}.npz`, shipped with
   the pipeline) because the cold solve is unreliable for underactuated swing-up.

3. **Real-time control — `ctrl_run_acrobot.py` / `ctrl_run_pendubot.py`**
   *(on CloudPendulum JupyterHub; needs acados + cloudpendulumclient)*.
   Both are thin entry points into `ctrl_run_common.py`, which: loads params + reference,
   builds an **acados SQP-RTI** tracker, discovers and reserves the hardware cell, runs the
   real-time loop at a target 400 Hz (soft velocity + hard torque bounds, one QP iteration
   per cycle), and logs every array to `ctrlrun_{plant}_{cell}_{stamp}.npz` plus a video.

`ctrl_plants.py` is the shared core (plant specs, dynamics/RK4, swing-up OCP, reference
I/O, state estimator). It imports **no** acados/hardware modules, so reference generation
is fully offline.

### Plant configurations & platform limits

| | Acrobot | Pendubot |
|---|---|---|
| Actuated joint | elbow (`TAU_MAX = 0.13` Nm) | shoulder (`TAU1_MAX = 0.13` Nm) |
| Passive-joint assist | shoulder `≤ 5` mNm | elbow `≤ 5` mNm |
| Swing-up horizon | `T = 3.0` s, `N = 150` | `T = 3.5` s, `N = 175` (both `dt_ref = 20` ms) |

Platform limits (respected by the controller): torque ≤ 0.15 Nm, velocity ≤ 50 rad/s
(`DQ_ABORT = 50`; reference/NMPC capped at `20`), control rate ≤ 500 Hz (target 400),
attempt ≤ 60 s. The 5 mNm passive-joint assist is a development aid and a scored deduction,
not evaluation-compliant single-actuator control.

---

## Using the notebooks

Two notebooks drive the acrobot workflow end-to-end; each cell is a stage of the pipeline:

- **`acrobot_workflow1.ipynb`** — the current version (adds a diagnostic plots section).
- **`acrobot_workflow.ipynb`** — the earlier version (same flow, no plots).

They must run **on the CloudPendulum JupyterHub** (see prerequisites). Steps:

1. **Settings (cell 4)** — the only cell you normally edit:
   ```python
   PLANT          = "acrobot"   # plant selector
   MAKE_REFERENCE = True        # regenerate the reference before running
   RUN_HARDWARE   = False       # SAFETY GATE — keep False until ready to go on hardware
   T_SIM          = 20.0        # experiment length (s)
   RECORD         = True        # request server-side video
   PARAM_FILE     = None        # None = auto-pick latest identified_params_{cell}_fit_*.json
   ```
2. **Generate Reference (cell 8)** — runs `ctrl_make_reference.py` for the selected plant.
   Succeeds silently; on failure it raises and stops (check the log line `warm start <- ...`).
3. **Run Hardware Control (cell 10)** — only fires when `RUN_HARDWARE = True`; otherwise it
   prints `RUN_HARDWARE=False; hardware control not started.` **This reserves a real cell and
   moves the motors** — set `True` only after the cell id, token, and reference are confirmed.
4. **Latest Logs / Plots (cells 12–14)** — locate and plot the newest
   `ctrlrun_{plant}_{cell}_*.npz`.

**Typical run order:** set `RUN_HARDWARE=False` → run all cells to confirm the reference
generates → set `RUN_HARDWARE=True` → re-run cells 4 and 10.

### Command-line equivalent (no notebook)

```bash
# offline, anywhere with CasADi/IPOPT:
python ctrl_make_reference.py --plant pendubot --params identified_params_202_fit_20260721-231023.json --force

# on JupyterHub, needs acados + cloudpendulumclient:
python ctrl_run_pendubot.py --params identified_params_202_fit_20260721-231023.json --t-sim 20
```

There is no separate pendubot notebook in this snapshot; run the pendubot side either from
the CLI above, or by copying `acrobot_workflow1.ipynb` and setting `PLANT="pendubot"`
(cell 10 already calls `ctrl_run_pendubot.py` for that plant).

---

## Prerequisites (to actually run it)

This folder is an **evaluation snapshot, not a self-contained package.** Running the
notebooks/scripts additionally requires, from the parent `sysid_pipeline`:

- **`sysid_common.py`** (`load_config`, `resolve_data`, `data_dir`, `mask_token`) and
  **`system_id_data_io.py`** — imported by the control scripts but not included here.
- **`sysid_config.json`** — holds `cell_id` and `user_token` (the CloudPendulum token).
  **Not committed** (it is a secret); provide your own, or set the `CLOUDPENDULUM_TOKEN`
  environment variable.

Environment:

- **CloudPendulum JupyterHub** account and hardware access.
- **acados** (+ `acados_template`) for the real-time SQP-RTI solver.
- **cloudpendulumclient** for the hardware API.
- **CasADi** (with IPOPT) and **NumPy** for reference generation.

---

## Configuration (`sysid_config.json`)

Every script and notebook reads a local `sysid_config.json`. It is **not committed**
(it holds your access token). Create it one of two ways:

- Copy the template: `cp sysid_config.example.json sysid_config.json`, then fill in
  `user_token` (and `cell_id`), **or**
- Run `python sysid_make_config.py` (reads `CLOUDPENDULUM_TOKEN` from the environment, or
  an existing config; writes the file with defaults).

Fields (see `sysid_config.example.json`):

| Field | Meaning |
|---|---|
| `user_token` | **Your CloudPendulum access token** (secret). Leave `""` to work offline; required before any hardware step. `CLOUDPENDULUM_TOKEN` env var overrides it. |
| `data_dir` | Where params / references / logs live, relative to the scripts (`"."` = here). |
| `experiment_type` | `"DoublePendulum"` (both actuators). `"Acrobot"` / `"Pendubot"` restrict torque to the single evaluation-legal joint. |
| `cell_id` | **Your assigned hardware cell** (device-specific; e.g. `202`, `203`). |
| `tau_max_id` | Torque cap (Nm) used during system-ID data collection (conservative). |
| `dq_abort` | Velocity abort threshold (rad/s) during ID collection. |
| `ctrl_hz` | Control rate (Hz) during ID collection. |
| `inter_run_cooldown` | Seconds to wait between collection runs so the hardware settles. |
| `start_retry_max` | Max retries when starting an experiment. |
| `fit_data` | NPZ filenames used by RUN 3 fitting (empty = pass them on the command line). |
| `design_params` | Optional path to a nominal-parameter JSON (empty = built-in nominal). |

> The token is never printed in full — the tools mask it as `abc...xyz(len=19)`.

---

## Repository contents

| File | Role |
|---|---|
| `ctrl_plants.py` | Shared core: plant specs, dynamics/RK4, swing-up OCP, reference I/O, estimator |
| `ctrl_make_reference.py` | Offline swing-up reference generator (CasADi/IPOPT) |
| `ctrl_run_common.py` | Real-time controller runtime (acados SQP-RTI, device I/O, logging) |
| `ctrl_run_acrobot.py` / `ctrl_run_pendubot.py` | Thin per-plant entry points |
| `acrobot_workflow1.ipynb` / `acrobot_workflow.ipynb` | End-to-end acrobot workflow notebooks |
| `identified_params_202_fit_20260721-231023.json` | Identified base parameters (cell 202) |
| `acrobot_reference_202.npz` / `pendubot_reference_202.npz` | Generated swing-up references (with param provenance) |
| `acrobot_elbow_ocp.json` / `pendubot_ocp.json` | acados OCP export files |
| `ctrlrun_{plant}_202_*.npz` | Hardware run logs (all states, torques, timing, predictions) |
| `System_Identification_and_Development_Stage_NMPC...pdf` | The 1st-evaluation report |

Each `ctrlrun_*.npz` stores the full run: `q_hist`, `dq_*_hist`, `tau_hist`, `u_cmd/ref_hist`,
`solve_hist`, per-step timing, one-step prediction errors, `Xref/Uref`, and metadata
(`n_act`, `aborted`, `N_SWINGUP`, `dt_ref`, torque/velocity limits, param sha256).
