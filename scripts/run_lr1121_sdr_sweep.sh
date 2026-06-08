#!/usr/bin/env bash
# run_lr1121_sdr_sweep.sh — Repeatable USRP B210 capture sweep for LR-FHSS validation
#
# Sweeps across center frequencies AND RX antennas, capturing IQ to .fc32 with the
# native C++ capture tool, then runs the Python analyzers on each capture.
#
# IMPORTANT (honesty): This performs IQ-level RX validation ONLY. A "signal_detected"
# result means RF energy above the noise floor was captured at the configured center
# frequency. It does NOT decode LR-FHSS, and does NOT prove LR-FHSS waveform integrity.
#
# Transmission is performed by the Semtech LR1121 board (SWDM001 firmware), NOT here.
#
# Example:
#   bash scripts/run_lr1121_sdr_sweep.sh \
#     --serial 8000304 \
#     --freqs "868e6,915e6,923e6" \
#     --antennas "RX2,TX/RX" \
#     --rate 2e6 \
#     --gain 45 \
#     --duration 30 \
#     --outdir hardware/captures/sweep_YYYYMMDD_HHMMSS
#
# The capture binary CLI was discovered from rx_capture_to_file_cpp.cpp:
#   --freq <hz>  --rate <hz>  --gain <db>  --duration <s>  --out <path>
#   --antenna <RX2|TX/RX>  --channel <n>  --args <uhd-args, e.g. serial=8000304>
#   NOTE: there is NO dedicated --serial flag; the serial is passed inside --args
#         as "serial=<SERIAL>". The capture tool defaults --args to "type=b200".

set -euo pipefail

# --- Resolve repo root from script location ----------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CAPTURE_BIN="$REPO_ROOT/hardware/usrp_scripts/rx_capture_to_file_cpp"
CAPTURE_SRC="$REPO_ROOT/hardware/usrp_scripts/rx_capture_to_file_cpp.cpp"
ANALYZE_PY="$REPO_ROOT/hardware/usrp_scripts/analyze_capture.py"
MAXHOLD_PY="$REPO_ROOT/hardware/usrp_scripts/quick_maxhold.py"

# --- Defaults ----------------------------------------------------------------
SERIAL=""
FREQS="868e6,915e6,923e6"
ANTENNAS="RX2,TX/RX"
RATE="2e6"
GAIN="45"
DURATION="30"
OUTDIR=""

usage() {
  cat <<EOF
Usage: bash scripts/run_lr1121_sdr_sweep.sh [options]

Options:
  --serial   <id>     USRP serial (optional; passed to capture tool as args=serial=<id>)
  --freqs    <list>   Comma-separated center freqs in Hz (default: $FREQS)
  --antennas <list>   Comma-separated RX antennas (default: $ANTENNAS)
  --rate     <hz>     Sample rate, sci-notation ok (default: $RATE)
  --gain     <db>     RX gain (default: $GAIN)
  --duration <s>      Capture duration seconds (default: $DURATION)
  --outdir   <path>   Output dir (default: hardware/captures/sweep_<timestamp>)
  -h|--help           Show this help

IQ-level RX validation only. Does NOT decode LR-FHSS.
EOF
}

# --- Manual long-flag parsing ------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial)   SERIAL="$2";   shift 2 ;;
    --freqs)    FREQS="$2";    shift 2 ;;
    --antennas) ANTENNAS="$2"; shift 2 ;;
    --rate)     RATE="$2";     shift 2 ;;
    --gain)     GAIN="$2";     shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --outdir)   OUTDIR="$2";   shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Auto outdir if not given (or if the example placeholder was passed literally).
if [[ -z "$OUTDIR" || "$OUTDIR" == *"YYYYMMDD_HHMMSS"* ]]; then
  OUTDIR="$REPO_ROOT/hardware/captures/sweep_$(date +%Y%m%d_%H%M%S)"
fi

# --- Preconditions -----------------------------------------------------------
if [[ ! -x "$CAPTURE_BIN" ]]; then
  echo "ERROR: capture binary not found or not executable:" >&2
  echo "       $CAPTURE_BIN" >&2
  echo "" >&2
  echo "Build hint (requires UHD dev headers + a C++ toolchain):" >&2
  echo "       g++ -std=c++14 -O2 \\" >&2
  echo "         \"$CAPTURE_SRC\" \\" >&2
  echo "         -o \"$CAPTURE_BIN\" \\" >&2
  echo "         \$(pkg-config --cflags --libs uhd) -lpthread" >&2
  exit 1
fi

if [[ ! -f "$ANALYZE_PY" ]]; then
  echo "ERROR: analyzer not found: $ANALYZE_PY" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

# Sample rate as integer Hz for analyze_capture --sample-rate.
RATE_INT="$(python3 -c "print(int(float('$RATE')))")"

GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

CSV_FILE="$OUTDIR/sweep_summary.csv"
JSON_FILE="$OUTDIR/sweep_summary.json"

echo "capture_file,frequency_hz,antenna,rate,gain,duration,validation_status,signal_detected,peak_to_median_db,burst_energy_excess_db,peak_frequency_offset_hz" > "$CSV_FILE"

# JSON rows collected as files, assembled at the end with python3.
ROWS_DIR="$(mktemp -d)"
trap 'rm -rf "$ROWS_DIR"' EXIT

# --- Iterate over (freq, antenna) pairs --------------------------------------
IFS=',' read -r -a FREQ_ARR <<< "$FREQS"
IFS=',' read -r -a ANT_ARR <<< "$ANTENNAS"

TOTAL=0
DETECTED=0
ROW_IDX=0

sanitize_antenna() {
  # "TX/RX" -> "txrx", "RX2" -> "rx2"
  local a="$1"
  a="$(printf '%s' "$a" | tr 'A-Z' 'a-z')"
  a="$(printf '%s' "$a" | tr -cd 'a-z0-9')"
  printf '%s' "$a"
}

for freq in "${FREQ_ARR[@]}"; do
  freq="$(printf '%s' "$freq" | tr -d '[:space:]')"
  [[ -z "$freq" ]] && continue
  for ant in "${ANT_ARR[@]}"; do
    ant="$(printf '%s' "$ant" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$ant" ]] && continue

    TOTAL=$((TOTAL + 1))
    ant_safe="$(sanitize_antenna "$ant")"
    base="$OUTDIR/cap_${freq}_${ant_safe}"
    cap_file="${base}.fc32"

    echo ""
    echo "=================================================================="
    echo "[SWEEP] Capture $TOTAL : freq=$freq  antenna=$ant"
    echo "------------------------------------------------------------------"
    echo "ACTION: reset NUCLEO-L476RG now so SWDM001 retransmits LR-FHSS"
    echo "        pings, then capture starts (${DURATION}s)."
    echo "=================================================================="

    # Build capture invocation from the REAL flags found in the .cpp.
    cap_cmd=( "$CAPTURE_BIN"
      --freq "$freq"
      --rate "$RATE"
      --gain "$GAIN"
      --duration "$DURATION"
      --antenna "$ant"
      --out "$cap_file" )
    # serial: the .cpp has NO --serial flag; it is passed inside --args.
    if [[ -n "$SERIAL" ]]; then
      cap_cmd+=( --args "serial=$SERIAL" )
    fi

    echo "[SWEEP] capture command:"
    printf '   %q' "${cap_cmd[@]}"; echo ""
    "${cap_cmd[@]}"

    # --- Analyzer ------------------------------------------------------------
    analysis_json="${base}_analysis.json"
    analyze_cmd=( uv run python "$ANALYZE_PY" "$cap_file"
      --sample-rate "$RATE_INT"
      --output-json "$analysis_json"
      --plot "${base}_waterfall.png"
      --maxhold-plot "${base}_maxhold.png"
      --signal-threshold-db 8 )
    echo "[SWEEP] analyze command:"
    printf '   %q' "${analyze_cmd[@]}"; echo ""
    ( cd "$REPO_ROOT" && "${analyze_cmd[@]}" )

    # --- Quick max-hold ------------------------------------------------------
    if [[ -f "$MAXHOLD_PY" ]]; then
      maxhold_cmd=( uv run python "$MAXHOLD_PY" "$cap_file"
        --sample-rate "$RATE_INT"
        --out "${base}_maxhold_quick.png"
        --json "${base}_maxhold.json" )
      echo "[SWEEP] quick_maxhold command:"
      printf '   %q' "${maxhold_cmd[@]}"; echo ""
      ( cd "$REPO_ROOT" && "${maxhold_cmd[@]}" ) || \
        echo "[SWEEP] WARN: quick_maxhold failed (non-fatal)"
    else
      echo "[SWEEP] WARN: quick_maxhold.py not found, skipping."
    fi

    # --- Parse analysis JSON (robust, via python3 json.load) -----------------
    if [[ -f "$analysis_json" ]]; then
      read -r v_status v_detected v_ptm v_burst v_off < <(
        python3 - "$analysis_json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
def g(k, default="NA"):
    v = d.get(k, default)
    return default if v is None else v
print(
    g("validation_status"),
    str(bool(d.get("signal_detected", False))).lower(),
    g("peak_to_median_db"),
    g("burst_energy_excess_db"),
    g("peak_frequency_offset_hz"),
)
PY
      )
    else
      echo "[SWEEP] WARN: analysis JSON missing: $analysis_json"
      v_status="missing"; v_detected="false"; v_ptm="NA"; v_burst="NA"; v_off="NA"
    fi

    case "$v_status" in
      signal_detected)        echo "[SWEEP] $freq $ant -> SIGNAL DETECTED" ;;
      weak_signal_candidate)  echo "[SWEEP] $freq $ant -> weak_signal_candidate" ;;
      noise_floor_only)       echo "[SWEEP] $freq $ant -> noise_floor_only" ;;
      *)                      echo "[SWEEP] $freq $ant -> $v_status" ;;
    esac

    if [[ "$v_detected" == "true" ]]; then
      DETECTED=$((DETECTED + 1))
    fi

    # --- Append CSV row ------------------------------------------------------
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$cap_file" "$freq" "$ant" "$RATE" "$GAIN" "$DURATION" \
      "$v_status" "$v_detected" "$v_ptm" "$v_burst" "$v_off" >> "$CSV_FILE"

    # --- Stash JSON row ------------------------------------------------------
    python3 - "$ROWS_DIR/$ROW_IDX.json" "$cap_file" "$freq" "$ant" "$RATE" \
      "$GAIN" "$DURATION" "$v_status" "$v_detected" "$v_ptm" "$v_burst" "$v_off" <<'PY'
import json, sys
(out, cap, freq, ant, rate, gain, dur, status, detected,
 ptm, burst, off) = sys.argv[1:13]
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
row = {
    "capture_file": cap,
    "frequency_hz": freq,
    "antenna": ant,
    "rate": rate,
    "gain": gain,
    "duration": dur,
    "validation_status": status,
    "signal_detected": detected == "true",
    "peak_to_median_db": num(ptm),
    "burst_energy_excess_db": num(burst),
    "peak_frequency_offset_hz": num(off),
}
with open(out, "w") as f:
    json.dump(row, f)
PY
    ROW_IDX=$((ROW_IDX + 1))
  done
done

# --- Assemble sweep_summary.json ---------------------------------------------
python3 - "$JSON_FILE" "$ROWS_DIR" "${SERIAL:-}" "$GENERATED_AT" "$OUTDIR" <<'PY'
import json, os, sys
out, rows_dir, serial, generated_at, outdir = sys.argv[1:6]
rows = []
files = sorted(os.listdir(rows_dir),
               key=lambda n: int(n.split(".")[0]) if n.split(".")[0].isdigit() else 0)
for name in files:
    if name.endswith(".json"):
        with open(os.path.join(rows_dir, name)) as f:
            rows.append(json.load(f))
doc = {
    "metadata": {
        "serial": serial or None,
        "generated_at": generated_at,
        "outdir": outdir,
        "validation_scope": "IQ-level RX only; not an LR-FHSS decode",
        "count": len(rows),
        "signal_detected_count": sum(1 for r in rows if r.get("signal_detected")),
    },
    "captures": rows,
}
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
PY

# --- Final summary -----------------------------------------------------------
echo ""
echo "=================================================================="
echo "[SWEEP] SUMMARY"
echo "  captures:        $TOTAL"
echo "  signal_detected: $DETECTED"
echo "  outdir:          $OUTDIR"
echo "  csv:             $CSV_FILE"
echo "  json:            $JSON_FILE"
if [[ "$DETECTED" -gt 0 ]]; then
  echo "HARDWARE STATUS: signal-detected"
else
  echo "HARDWARE STATUS: pending (no signal above noise floor)"
fi
echo "=================================================================="
