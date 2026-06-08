"""Generate deterministic packet payloads for a validation run.

Each payload embeds the seq number (first 2 bytes, big-endian) followed by
seed-derived random bytes, and a trailing CRC32 (4 bytes) for local integrity
checks. Fully deterministic given (run_id, seed).

Outputs:
  <out>/tx_payloads.csv
  <out>/tx_payloads.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib
import random
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hardware.packet_validation._io import ensure_dir, write_csv, write_jsonl  # noqa: E402


def build_payloads(run_id: str, n_packets: int, seed: int = 42,
                   body_len: int = 6):
    rng = random.Random(f"{run_id}:{seed}")
    rows = []
    ts = datetime.now(timezone.utc).isoformat()
    for seq in range(n_packets):
        body = bytes([rng.randrange(256) for _ in range(body_len)])
        core = seq.to_bytes(2, "big") + body
        crc = zlib.crc32(core) & 0xFFFFFFFF
        payload = core + crc.to_bytes(4, "big")
        rows.append({
            "run_id": run_id,
            "seq": seq,
            "timestamp_utc": ts,
            "seed": seed,
            "payload_hex": payload.hex(),
            "payload_len": len(payload),
            "crc32": f"{crc:08x}",
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate deterministic TX payloads.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-packets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--body-len", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ensure_dir(args.out)
    rows = build_payloads(args.run_id, args.n_packets, args.seed, args.body_len)
    write_csv(os.path.join(args.out, "tx_payloads.csv"), rows)
    write_jsonl(os.path.join(args.out, "tx_payloads.jsonl"), rows)
    print(f"[generate_payloads] wrote {len(rows)} payloads to {args.out}")
    return rows


if __name__ == "__main__":
    main()
