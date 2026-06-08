"""Multi-format receiver decoded-packet log parser.

Supports (auto-detected per line):
  Format A: RX seq=12 payload=abcd1234 crc=ok rssi=-91.5 snr=7.2 cfo=123.4 timestamp=...
  Format B: [RX] seq:12 payload:abcd1234 CRC_OK RSSI=-91.5 SNR=7.2 CFO=123.4
  Format C: packet_received seq=12 len=16 payload_hex=abcd1234 crc_ok=true
  JSONL   : {"seq":12,"payload_hex":"abcd1234","crc_ok":true,"rssi_dbm":-91.5,...}

Returns (records, parser_stats). seq/payload_hex/crc_ok are required for a line
to count as parsed; rssi_dbm/snr_db/cfo_hz/rx_timestamp_utc are optional.
Unparseable lines are ignored but counted in parser_stats. Bad-CRC packets are
included with decode_status="crc_fail" (NOT counted as delivered downstream).
Duplicates are preserved (the evaluator handles them).
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple, Dict

from hardware.packet_validation.schemas import RxPacketRecord

_SEQ = re.compile(r"\bseq[=:]\s*(\d+)", re.IGNORECASE)
_PAYLOAD = re.compile(r"\bpayload(?:_hex)?[=:]\s*([0-9a-fA-F]+)", re.IGNORECASE)
_CRC_KV = re.compile(r"\bcrc(?:_ok)?[=:]\s*(ok|fail|true|false|1|0)", re.IGNORECASE)
_CRC_TOKEN = re.compile(r"\bCRC[_ ]?(OK|FAIL)\b", re.IGNORECASE)
_RSSI = re.compile(r"\brssi(?:_dbm)?[=:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_SNR = re.compile(r"\bsnr(?:_db)?[=:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_CFO = re.compile(r"\bcfo(?:_hz)?[=:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_TS = re.compile(r"\b(?:timestamp|rx_timestamp_utc|time)[=:]\s*(\S+)", re.IGNORECASE)

_TRUE = {"ok", "true", "1"}


def _fnum(m) -> Optional[float]:
    return float(m.group(1)) if m else None


def _crc_from_text(line: str) -> Optional[bool]:
    m = _CRC_KV.search(line)
    if m:
        return m.group(1).lower() in _TRUE
    m = _CRC_TOKEN.search(line)
    if m:
        return m.group(1).lower() == "ok"
    return None


def _record_from_json(run_id: str, obj: dict, log_path: str) -> Optional[RxPacketRecord]:
    if "seq" not in obj or "payload_hex" not in obj or "crc_ok" not in obj:
        return None
    payload = str(obj["payload_hex"])
    ok = bool(obj["crc_ok"])
    return RxPacketRecord(
        run_id=run_id, seq=int(obj["seq"]), payload_hex=payload,
        payload_len=obj.get("len", len(payload) // 2),
        rx_timestamp_utc=obj.get("timestamp") or obj.get("rx_timestamp_utc"),
        rx_backend="log_parser", crc_ok=ok,
        rssi_dbm=obj.get("rssi_dbm"), snr_db=obj.get("snr_db"),
        cfo_hz=obj.get("cfo_hz"),
        decode_status="decoded" if ok else "crc_fail",
        raw_log_path=log_path)


def _record_from_text(run_id: str, line: str, log_path: str) -> Optional[RxPacketRecord]:
    s, p, c = _SEQ.search(line), _PAYLOAD.search(line), _crc_from_text(line)
    if not (s and p and c is not None):
        return None
    payload = p.group(1)
    return RxPacketRecord(
        run_id=run_id, seq=int(s.group(1)), payload_hex=payload,
        payload_len=len(payload) // 2,
        rx_timestamp_utc=(_TS.search(line).group(1) if _TS.search(line) else None),
        rx_backend="log_parser", crc_ok=c,
        rssi_dbm=_fnum(_RSSI.search(line)), snr_db=_fnum(_SNR.search(line)),
        cfo_hz=_fnum(_CFO.search(line)),
        decode_status="decoded" if c else "crc_fail",
        raw_log_path=log_path)


def parse_rx_log(run_id: str, rx_log: str) -> Tuple[List[RxPacketRecord], Dict[str, int]]:
    stats = {"n_lines": 0, "n_parsed": 0, "n_unparseable": 0, "n_crc_ok": 0,
             "n_crc_fail": 0}
    recs: List[RxPacketRecord] = []
    if not rx_log or not os.path.exists(rx_log):
        return recs, stats
    log_path = os.path.abspath(rx_log)
    with open(rx_log) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            stats["n_lines"] += 1
            rec = None
            if line.startswith("{"):
                try:
                    rec = _record_from_json(run_id, json.loads(line), log_path)
                except (json.JSONDecodeError, ValueError, TypeError):
                    rec = None
            if rec is None:
                rec = _record_from_text(run_id, line, log_path)
            if rec is None:
                stats["n_unparseable"] += 1
                continue
            stats["n_parsed"] += 1
            stats["n_crc_ok" if rec.crc_ok else "n_crc_fail"] += 1
            recs.append(rec)
    return recs, stats
