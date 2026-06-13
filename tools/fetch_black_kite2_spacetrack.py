#!/usr/bin/env python3
"""
Fetch BLACK KITE-2 / NORAD 68474 raw Space-Track artifacts.

Credentials are read from environment variables only:
- SPACETRACK_USERNAME / SPACETRACK_PASSWORD
- or legacy SPACETRACKUSER / SPACETRACKPASS

This script never prints credentials and writes outputs only under dataraw/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request


NORAD_CAT_ID = 68474
OBJECT_NAME = "BLACK KITE-2"
ROOT = Path("dataraw/spacetrack/black_kite_2_68474")


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
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(opener: urllib.request.OpenerDirector, url: str, out: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "LEO-Hybrid-PGRL artifact fetch"})
    with opener.open(req, timeout=60) as resp:
        data = resp.read()
    out.write_bytes(data)
    print(f"WROTE {out} bytes={len(data)}")


def main() -> int:
    load_dotenv(Path(".env.spacetrack"))

    username = os.environ.get("SPACETRACK_USERNAME") or os.environ.get("SPACETRACKUSER")
    password = os.environ.get("SPACETRACK_PASSWORD") or os.environ.get("SPACETRACKPASS")

    if not username or not password:
        print("ERROR: Space-Track credentials not set.", file=sys.stderr)
        print("Set SPACETRACK_USERNAME/SPACETRACK_PASSWORD or SPACETRACKUSER/SPACETRACKPASS.", file=sys.stderr)
        return 2

    ROOT.mkdir(parents=True, exist_ok=True)

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    print("LOGIN Space-Track credentials hidden")
    login_data = urllib.parse.urlencode({"identity": username, "password": password}).encode()
    login_req = urllib.request.Request(
        "https://www.space-track.org/ajaxauth/login",
        data=login_data,
        headers={"User-Agent": "LEO-Hybrid-PGRL artifact fetch"},
    )
    with opener.open(login_req, timeout=60) as resp:
        login_body = resp.read()
    (ROOT / "login_response_redacted.txt").write_text(
        f"login_http_response_bytes={len(login_body)}\ncredentials_redacted=true\n"
    )

    queries = {
        "gp_latest_68474.tle":
            "https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/68474/orderby/EPOCH%20desc/format/tle",
        "gp_latest_68474.json":
            "https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/68474/orderby/EPOCH%20desc/format/json",
        "gp_history_68474.tle":
            "https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/68474/orderby/EPOCH%20asc/format/tle",
        "gp_history_68474.json":
            "https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/68474/orderby/EPOCH%20asc/format/json",
        "satcat_68474.json":
            "https://www.space-track.org/basicspacedata/query/class/satcat/NORAD_CAT_ID/68474/format/json",
    }

    for name, url in queries.items():
        fetch(opener, url, ROOT / name)

    files = sorted(p for p in ROOT.iterdir() if p.is_file() and p.name != "fetch_manifest.json")
    manifest = {
        "object_name": OBJECT_NAME,
        "norad_cat_id": NORAD_CAT_ID,
        "fetched_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
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

    (ROOT / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with (ROOT / "SHA256SUMS.txt").open("w") as f:
        for p in files + [ROOT / "fetch_manifest.json"]:
            f.write(f"{sha256_file(p)}  {p.name}\n")

    print("DONE")
    print(f"manifest={ROOT / 'fetch_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
