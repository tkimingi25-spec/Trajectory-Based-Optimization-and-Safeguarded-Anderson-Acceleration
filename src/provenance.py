"""
Small helpers for writing machine-readable experiment provenance.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import torch


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, range):
        return list(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_experiment_result(path, *, experiment, config, results, per_seed=None, notes=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "experiment": experiment,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "config": to_jsonable(config),
        "results": to_jsonable(results),
        "per_seed": to_jsonable(per_seed or {}),
        "notes": notes or [],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
