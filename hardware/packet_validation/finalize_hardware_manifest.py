"""Finalize a hardware experiment run: classify artifacts for commit vs local.

Produces (in the run dir):
  manifest_final.yaml          - augmented manifest + artifact counts
  artifact_index.csv           - path, size_bytes, sha256, category
  sha256sums.txt               - checksums for committable artifacts
  recommended_commit_files.txt - small summaries/manifests safe to commit
  do_not_commit_files.txt      - raw IQ + oversized artifacts (local-only)

Raw IQ (.fc32/.cfile/.sigmf-data) and oversized files are marked local-only and
never listed for commit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation import _yaml  # noqa: E402

RAW_IQ_EXT = (".fc32", ".cfile", ".sigmf-data", ".dat")
MAX_COMMIT_BYTES = 5 * 1024 * 1024  # 5 MB


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize(run_dir: str) -> dict:
    files = []
    for root, _, names in os.walk(run_dir):
        for n in sorted(names):
            files.append(os.path.relpath(os.path.join(root, n), run_dir))

    commit, do_not = [], []
    index = []
    for rel in sorted(files):
        full = os.path.join(run_dir, rel)
        size = os.path.getsize(full)
        ext = os.path.splitext(rel)[1].lower()
        is_raw = ext in RAW_IQ_EXT
        too_big = size > MAX_COMMIT_BYTES
        if is_raw or too_big:
            category = "do_not_commit_raw_iq" if is_raw else "do_not_commit_large"
            sha = ""  # do not hash large/raw
            do_not.append(rel)
        else:
            category = "commit"
            sha = _sha256(full)
            commit.append(rel)
        index.append({"path": rel, "size_bytes": size, "sha256": sha,
                      "category": category})

    # write artifact_index.csv
    with open(os.path.join(run_dir, "artifact_index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "size_bytes", "sha256", "category"])
        w.writeheader()
        w.writerows(index)

    # sha256sums.txt (committable only)
    with open(os.path.join(run_dir, "sha256sums.txt"), "w") as fh:
        for row in index:
            if row["category"] == "commit":
                fh.write(f"{row['sha256']}  {row['path']}\n")

    with open(os.path.join(run_dir, "recommended_commit_files.txt"), "w") as fh:
        fh.write("# Small summaries/manifests. NOTE: validation_runs/ is gitignored;\n"
                 "# copy these to a committed location (e.g. docs/) if needed.\n")
        fh.write("\n".join(commit) + ("\n" if commit else ""))

    with open(os.path.join(run_dir, "do_not_commit_files.txt"), "w") as fh:
        fh.write("# Raw IQ (local-only) and oversized artifacts. Never commit.\n")
        fh.write("\n".join(do_not) + ("\n" if do_not else ""))

    # manifest_final.yaml
    base = {}
    mpath = os.path.join(run_dir, "manifest.yaml")
    if os.path.exists(mpath):
        with open(mpath) as fh:
            base = _yaml.loads(fh.read()) or {}
    base["finalized"] = {
        "n_artifacts": len(index),
        "n_commit": len(commit),
        "n_do_not_commit": len(do_not),
        "raw_iq_local_only": True,
    }
    with open(os.path.join(run_dir, "manifest_final.yaml"), "w") as fh:
        fh.write(_yaml.dumps(base) + "\n")

    return {"n_commit": len(commit), "n_do_not_commit": len(do_not),
            "commit": commit, "do_not_commit": do_not}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Finalize a hardware experiment run.")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    res = finalize(args.run_dir)
    print(f"[finalize] commit={res['n_commit']} do_not_commit={res['n_do_not_commit']}")
    return res


if __name__ == "__main__":
    main()
