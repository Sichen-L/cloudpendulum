import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = ("t", "q", "dq", "u", "dt")


def load_datasets(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        groups = {"free": [], "exc": []}
        for entry in manifest:
            key = entry["key"]
            ds = {name: np.array(archive[f"{key}_{name}"]) for name in REQUIRED_FIELDS}
            ds.update(label=entry.get("label", key), aborted=bool(entry.get("aborted", False)))
            _validate_dataset(ds, key)
            groups.setdefault(entry["group"], []).append(ds)
    return groups


def save_datasets(groups, path):
    path = Path(path)
    payload = {}
    manifest = []
    counters = {}
    for group, datasets in groups.items():
        counters.setdefault(group, 0)
        for ds in datasets:
            key = f"{group}_{counters[group]:02d}"
            counters[group] += 1
            for name in REQUIRED_FIELDS:
                payload[f"{key}_{name}"] = np.asarray(ds[name])
            manifest.append({
                "key": key,
                "group": group,
                "label": str(ds.get("label", key)),
                "aborted": bool(ds.get("aborted", False)),
            })
    if not manifest:
        raise ValueError("No datasets to save")
    payload["manifest_json"] = np.array(json.dumps(manifest, ensure_ascii=False))
    np.savez_compressed(path, **payload)
    return path


def merge_dataset_files(paths, label_prefix=False):
    merged = {"free": [], "exc": []}
    for path in paths:
        path = Path(path)
        groups = load_datasets(path)
        for group, datasets in groups.items():
            merged.setdefault(group, [])
            for ds in datasets:
                item = {name: np.asarray(ds[name]).copy() for name in REQUIRED_FIELDS}
                label = str(ds.get("label", ""))
                if label_prefix:
                    label = f"{path.stem}:{label}"
                item.update(label=label, aborted=bool(ds.get("aborted", False)))
                merged[group].append(item)
    return merged


def _validate_dataset(ds, key):
    n = len(ds["t"])
    if n < 2 or ds["q"].shape != (n, 2) or ds["dq"].shape != (n, 2):
        raise ValueError(f"Invalid state shapes in {key}")
    if len(ds["u"]) < n - 1:
        raise ValueError(f"Input sequence is too short in {key}")
    dt = np.asarray(ds["dt"])
    if dt.ndim > 0 and dt.size not in (n - 1, n):
        raise ValueError(f"dt must be scalar or contain N-1/N values in {key}")
    for name in REQUIRED_FIELDS:
        if not np.all(np.isfinite(ds[name])):
            raise ValueError(f"Non-finite {name} values in {key}")
