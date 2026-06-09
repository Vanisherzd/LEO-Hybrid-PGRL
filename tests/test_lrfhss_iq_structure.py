"""Test the LR-FHSS IQ-structure analyzer on synthetic hopping IQ (no hardware).

Runnable with pytest OR plain python:
    python tests/test_lrfhss_iq_structure.py
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import analyze_lrfhss_iq_structure as ana  # noqa: E402

FS = 1_000_000.0


def _make_hopping_iq(fs=FS, n_hops=8, hop_s=0.004, gap_s=0.003, seed=1):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_hops):
        # gap (noise only)
        ng = int(gap_s * fs)
        parts.append(0.05 * (rng.standard_normal(ng) + 1j * rng.standard_normal(ng)))
        # hop tone at a random offset, well above noise
        nh = int(hop_s * fs)
        f0 = rng.uniform(-3e5, 3e5)
        tt = np.arange(nh) / fs
        tone = np.exp(2j * np.pi * f0 * tt).astype(np.complex64)
        noise = 0.05 * (rng.standard_normal(nh) + 1j * rng.standard_normal(nh))
        parts.append(tone + noise)
    return np.concatenate(parts).astype(np.complex64)


def _run():
    d = tempfile.mkdtemp()
    iqpath = os.path.join(d, "synthetic.fc32")
    _make_hopping_iq().tofile(iqpath)
    out = os.path.join(d, "out")
    summary = ana.main(["--iq", iqpath, "--sample-rate", str(FS),
                        "--center-frequency", "868000000",
                        "--run-id", "synthtest", "--out", out])
    return out, summary


def test_summary_and_csv_exist():
    out, _ = _run()
    assert os.path.exists(os.path.join(out, "iq_structure_summary.json"))
    assert os.path.exists(os.path.join(out, "burst_candidates.csv"))
    assert os.path.exists(os.path.join(out, "psd_maxhold.csv"))
    assert os.path.exists(os.path.join(out, "tone_occupancy.csv"))


def test_several_bursts_detected():
    out, summary = _run()
    assert summary["n_candidate_bursts"] >= 3, summary["n_candidate_bursts"]


def test_structure_score_positive():
    out, summary = _run()
    assert summary["structure_score"] > 0


def test_no_decode_per_crc_claim():
    out, summary = _run()
    blob = json.dumps(summary).lower()
    # forbidden POSITIVE claims (negation label "no ... per/crc/decode" is fine)
    for bad in ["decoded packet", "packet delivered", '"per":', "per=",
                "crc_ok\": true", "payload_decoded"]:
        assert bad not in blob, f"unexpected claim token: {bad}"
    # the conservative label must be present
    assert "no payload decode" in summary["label"].lower()


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
