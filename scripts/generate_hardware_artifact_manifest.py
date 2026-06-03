#!/usr/bin/env python3
"""
generate_hardware_artifact_manifest.py
Build a JSON manifest of all .fc32 hardware captures and their associated
analysis JSON / plot artifacts. References paths only — NEVER copies or embeds
raw IQ data.

For each capture foo.fc32 the script looks (in the same directory) for:
  - foo_analysis.json       (preferred matching convention)
  - any *_analysis.json whose "input_file" references the capture (fallback)
  - foo_waterfall.png, foo_maxhold.png, and any foo*.png plots

validation_status / signal_detected / note are pulled from the new-schema
analysis JSON. Legacy analysis JSONs (which lack validation_status) are handled
gracefully: validation_status/signal_detected -> null, note ->
"legacy analysis, no validation_status".

Usage:
    uv run python scripts/generate_hardware_artifact_manifest.py \
        [--capture-dir hardware/captures] \
        [--output hardware/hardware_artifact_manifest.json]
"""

import argparse
import datetime
import json
from pathlib import Path


def _repo_root() -> Path:
    # scripts/ is directly under the repo root.
    return Path(__file__).resolve().parent.parent


def _load_json(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _index_analysis_jsons(capture_dir: Path):
    """
    Build an index mapping resolved capture path -> analysis json path,
    using each analysis json's "input_file" field (fallback matching).
    """
    by_input = {}
    for jp in capture_dir.rglob("*_analysis.json"):
        data = _load_json(jp)
        if not isinstance(data, dict):
            continue
        inp = data.get("input_file")
        if not inp:
            continue
        # input_file may be relative to repo root or absolute.
        candidates = [Path(inp), _repo_root() / inp, jp.parent / Path(inp).name]
        for c in candidates:
            try:
                rc = c.resolve()
            except Exception:
                continue
            by_input[rc] = jp
    return by_input


def _find_analysis_json(capture: Path, by_input: dict):
    """Resolve the analysis json for a capture via convention, then fallback."""
    # Convention: foo.fc32 -> foo_analysis.json in same dir.
    conv = capture.with_name(capture.stem + "_analysis.json")
    if conv.exists():
        return conv
    # Fallback: any analysis json referencing this capture via input_file.
    rc = capture.resolve()
    if rc in by_input:
        return by_input[rc]
    return None


def _find_plots(capture: Path):
    """All PNGs in the same dir whose stem starts with the capture stem."""
    plots = []
    for png in sorted(capture.parent.glob(f"{capture.stem}*.png")):
        plots.append(png)
    return plots


def _rel(path: Path) -> str:
    """Path relative to repo root if possible, else absolute string."""
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except Exception:
        return str(path)


def build_entry(capture: Path, by_input: dict) -> dict:
    size_bytes = capture.stat().st_size
    sample_count = size_bytes // 8  # complex64 = 8 bytes/sample

    analysis_path = _find_analysis_json(capture, by_input)
    validation_status = None
    signal_detected = None
    note = None
    analysis_json_rel = None

    if analysis_path is not None and analysis_path.exists():
        analysis_json_rel = _rel(analysis_path)
        data = _load_json(analysis_path)
        if isinstance(data, dict):
            if "validation_status" in data:
                validation_status = data.get("validation_status")
                signal_detected = data.get("signal_detected")
                note = data.get("analysis_note")
            else:
                # Legacy schema: no validation fields.
                note = "legacy analysis, no validation_status"
        else:
            note = "analysis json present but unreadable"
    else:
        note = "no analysis json"

    return {
        "path": _rel(capture),
        "size_bytes": int(size_bytes),
        "sample_count": int(sample_count),
        "analysis_json": analysis_json_rel,
        "plots": [_rel(p) for p in _find_plots(capture)],
        "validation_status": validation_status,
        "signal_detected": signal_detected,
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate hardware artifact manifest")
    parser.add_argument(
        "--capture-dir",
        type=str,
        default=str(_repo_root() / "hardware" / "captures"),
        help="Directory to scan recursively for .fc32 captures",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_repo_root() / "hardware" / "hardware_artifact_manifest.json"),
        help="Output manifest JSON path",
    )
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    captures = []

    if capture_dir.exists() and capture_dir.is_dir():
        by_input = _index_analysis_jsons(capture_dir)
        for fc in sorted(capture_dir.rglob("*.fc32")):
            if fc.is_file():
                try:
                    captures.append(build_entry(fc, by_input))
                except Exception as e:
                    captures.append(
                        {
                            "path": _rel(fc),
                            "size_bytes": None,
                            "sample_count": None,
                            "analysis_json": None,
                            "plots": [],
                            "validation_status": None,
                            "signal_detected": None,
                            "note": f"error building entry: {e}",
                        }
                    )
    else:
        print(f"[manifest] capture dir not found: {capture_dir} (writing empty manifest)")

    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "capture_dir": _rel(capture_dir) if capture_dir.exists() else str(capture_dir),
        "n_captures": len(captures),
        "captures": captures,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[manifest] {len(captures)} capture(s) → {out_path}")
    for c in captures:
        print(f"  - {c['path']}  status={c['validation_status']}  note={c['note']}")


if __name__ == "__main__":
    main()
