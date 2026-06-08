"""Small shared CSV/JSON helpers (stdlib only)."""
from __future__ import annotations

import csv
import json
import os
from typing import List, Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        # still create an empty file with no header (downstream tolerates this)
        open(path, "w").close()
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def read_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
