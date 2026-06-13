#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT = Path("dataraw/spacetrack")
CANDIDATE_NAMES = [
    "BLACK KITE-1",
    "BLACK KITE 1",
    "BLACKKITE-1",
    "BLACKKITE 1",
    "BLACK KITE-2",
    "BLACK KITE 2",
    "BLACKKITE-2",
    "BLACKKITE 2",
]

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"

def open_url(opener, url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "LEO-Hybrid-PGRL artifact fetch"})
    with opener.open(req, timeout=90) as resp:
        return resp.read()

def fetch_to(opener, url: str, out: Path) -> None:
    data = open_url(opener, url)
    out.write_bytes(data)
    print(f"WROTE {out} bytes={len(data)}")

def main() -> int:
    load_dotenv(Path(".env.spacetrack"))
    username = os.environ.get("SPACETRACK_USERNAME") or os.environ.get("SPACETRACKUSER")
    password = os.environ.get("SPACETRACK_PASSWORD") or os.environ.get("SPACETRACKPASS")
    if not username or not password:
        print("ERROR: Space-Track credentials not set", file=sys.stderr)
        return 2

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    login_data = urllib.parse.urlencode({"identity": username, "password": password}).encode()
    login_req = urllib.request.Request(
        "https://www.space-track.org/ajaxauth/login",
        data=login_data,
        headers={"User-Agent": "LEO-Hybrid-PGRL artifact fetch"},
    )
    with opener.open(login_req, timeout=90) as resp:
        login_body = resp.read()
    print(f"LOGIN ok, response_bytes={len(login_body)}, credentials hidden")

    records = {}
    for name in CANDIDATE_NAMES:
        enc = urllib.parse.quote(name)
        url = f"https://www.space-track.org/basicspacedata/query/class/satcat/OBJECT_NAME/{enc}/format/json"
        try:
            data = open_url(opener, url)
            parsed = json.loads(data.decode())
        except Exception as e:
            print(f"SATCAT query failed for {name}: {e}")
            continue
        for rec in parsed if isinstance(parsed, list) else []:
            norad = str(rec.get("NORAD_CAT_ID", "")).strip()
            obj = str(rec.get("OBJECT_NAME", "")).strip()
            if norad:
                records[norad] = rec
                print(f"FOUND NORAD={norad} OBJECT_NAME={obj}")

    if not records:
        print("No BLACK KITE family records found by exact-name candidates.")
        return 1

    for norad, rec in sorted(records.items()):
        obj = rec.get("OBJECT_NAME", f"NORAD_{norad}")
        slug = f"{safe_slug(obj)}_{norad}"
        outdir = ROOT / slug
        outdir.mkdir(parents=True, exist_ok=True)

        (outdir / "satcat_record_from_name_query.json").write_text(json.dumps(rec, indent=2) + "\n")

        queries = {
            f"gp_latest_{norad}.tle":
                f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{norad}/orderby/EPOCH%20desc/format/tle",
            f"gp_latest_{norad}.json":
                f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{norad}/orderby/EPOCH%20desc/format/json",
            f"gp_history_{norad}.tle":
                f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{norad}/orderby/EPOCH%20asc/format/tle",
            f"gp_history_{norad}.json":
                f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{norad}/orderby/EPOCH%20asc/format/json",
            f"satcat_{norad}.json":
                f"https://www.space-track.org/basicspacedata/query/class/satcat/NORAD_CAT_ID/{norad}/format/json",
        }
        for fname, url in queries.items():
            fetch_to(opener, url, outdir / fname)

        files = sorted(p for p in outdir.iterdir() if p.is_file() and p.name not in {"fetch_manifest.json", "SHA256SUMS.txt"})
        manifest = {
            "object_name": obj,
            "norad_cat_id": norad,
            "fetched_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            "source": "Space-Track basicspacedata API",
            "credentials_not_stored": True,
            "reference_is_measured_truth": False,
            "files": [],
        }
        for p in files:
            manifest["files"].append({
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
        (outdir / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (outdir / "SHA256SUMS.txt").open("w") as f:
            for p in files + [outdir / "fetch_manifest.json"]:
                f.write(f"{sha256_file(p)}  {p.name}\n")

    print("DONE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
