# LR1121 LR-FHSS signal detection (IQ-level)

How to recognise an LR-FHSS transmission from a Semtech LR1121 (SWDM001) in a
USRP B210 `.fc32` IQ capture, why EVM is **not** the metric to trust here, and
the conservative detection ladder the analyzer uses to decide what may and may
not be claimed.

> **Scope discipline (read first).** Everything in this document is IQ-level
> measurement only. The SDR path is **not** a standard-compliant LR-FHSS
> gateway. Do **not** report LR-FHSS PER, standard decoding, or "hardware
> validated" from any metric here. The current real status of the LR1121 capture
> is **`weak_signal_candidate`**, which is **not** hardware validation. The
> honest claim today is "firmware running" + "IQ capture done" + "weak LR-FHSS
> candidate observed". Only `validation_status == "signal_detected"` (with the
> corroboration in section 3) advances RF claims, and even that is RF energy
> detection, not PER.

Cross-references:
[docs/lr1121_sdr_validation_workflow.md](lr1121_sdr_validation_workflow.md) (end-to-end
capture/analyze/sweep procedure and the validation-label ladder) and
[docs/swdm001_auto_tx_mode.md](swdm001_auto_tx_mode.md) (how to get a burst to
land inside the capture window: reset-triggered Mode A vs continuous-TX Mode B).

---

## 1. What LR-FHSS actually looks like in IQ / spectrogram

LR-FHSS is **not** a continuous burst and **not** a clean QPSK constellation. It
is a **sparse, fast frequency-hopping, GMSK-like** waveform. In the captured
data the signature is:

- **Waterfall (spectrogram):** a scattering of **short horizontal dashes** at
  different frequency offsets and different times. Each dash is one narrow
  frequency hop that is on-air only briefly, then the signal jumps elsewhere.
  Across the whole capture, **many** frequency bins are touched, but **each only
  for a short slice of time**. This exactly matches the current LR1121 capture:
  sparse short horizontal frequency-hop dashes spread across the band.
- **Max-hold spectrum:** because max-hold accumulates the peak power per
  frequency bin over the whole capture, all the brief hops "stack up" into
  **multiple narrow peaks spread across the band** — far more revealing than a
  single FFT frame, which would usually catch zero or one hop. The current
  capture shows exactly this: multiple narrow max-hold peaks across the band.
- **What it is NOT:** not a single wide continuous burst sitting at one
  frequency for the whole capture; not a dense, fully-populated band; not a
  recognisable QPSK / PSK constellation cluster.

### Contrast with the original burst / EVM detector

The original `analyze_capture.py` pipeline was built around a **single dominant
burst** model: find the strongest in-band bin, score `peak_to_median_db`, check
a `burst_energy_excess_db` for one concentrated burst, then (only if detected)
compute a CFO and a **QPSK EVM proxy**. That model fits a continuous, single-tone
/ single-burst signal. It does **not** fit LR-FHSS, whose energy is deliberately
spread thin across many short hops: at any one instant the band looks almost
empty, and there is no persistent carrier to lock a constellation onto. A
sparse-hopping signal can therefore be **real RF** and still produce a modest
`peak_to_median_db` and a meaningless EVM. The LR-FHSS-aware metrics below exist
to measure the *hopping structure* directly instead of assuming one burst.

---

## 2. Why EVM is NOT the main metric here

The QPSK EVM proxy (`rx_evm_percent`, `evm_method = qpsk_constellation_mse`)
maps received samples onto a QPSK constellation and reports the error. That only
means something for a QPSK-like signal. LR-FHSS is **GMSK-like sparse hopping**,
not QPSK, so:

- Projecting LR-FHSS (or, worse, noise-floor samples between hops) onto a QPSK
  grid yields a number with **no physical meaning** for this waveform.
- It is **not** a quality metric and is **absolutely not** a PER. It must never
  be reported as LR-FHSS quality, link quality, or PER.

The analyzer already withholds CFO/EVM unless a signal is detected, and even when
emitted they are labelled "RF-quality proxy only; not a standard LR-FHSS PER
measurement". For LR-FHSS the **relevant** evidence is the hopping-structure
metrics, all computed with the **DC / LO spike always excluded** (the B210 front
end produces a strong centre-bin spike that must never be mistaken for signal):

| Metric | Definition |
|--------|-----------|
| `time_frequency_hot_bin_count` | Number of time-frequency cells in the spectrogram whose power exceeds the noise-floor threshold (DC excluded). LR-FHSS lights up many scattered cells rather than one solid block. |
| `occupied_frequency_bins` | How many distinct **frequency** bins are touched at any time during the capture (DC excluded). Sparse hopping spreads energy over **many** frequency bins. |
| `occupied_time_bins` | How many distinct **time** slices contain above-threshold in-band energy. Hops are short, so each individual hop occupies only a small fraction of time. |
| `hop_like_segment_count` | Count of short, narrowband, time-localised energy segments — i.e. dash-like events consistent with individual frequency hops (as opposed to one long continuous burst). |
| `maxhold_peak_count_excluding_dc` | Number of distinct narrow peaks in the max-hold spectrum after removing the DC/LO bin. Multiple separated peaks = multiple hop frequencies stacked over time. |
| `lr_fhss_candidate_score` | Aggregate score combining the above (many occupied frequency bins, each briefly occupied in time, multiple hop-like segments and max-hold peaks) into a single 0..1-style confidence that the capture contains LR-FHSS-like hopping. |
| `lr_fhss_candidate` | Boolean: `lr_fhss_candidate_score` cleared the candidate threshold. **Candidate is not detection** — see the ladder below. |

The defining LR-FHSS fingerprint these encode: **many frequency bins occupied
across the capture, but each occupied only briefly in time** — sparse hopping,
not a dense or continuous occupancy, and never the DC spike.

---

## 3. Detection ladder / decision rules

The analyzer classifies every capture into exactly one `validation_status`. The
rungs (lowest to highest):

### `noise_floor_only`
No usable evidence. `peak_to_median_db` below threshold, no hop-like structure,
`lr_fhss_candidate_score` below the candidate threshold. The capture is
consistent with the noise floor (DC/LO spike excluded). CFO/EVM are withheld.

### `weak_signal_candidate`  ← CURRENT STATUS
Some LR-FHSS-like evidence is present (e.g. a non-trivial
`lr_fhss_candidate_score`, a few hop-like segments / max-hold peaks, sparse
dashes in the waterfall) **but it is not strong enough and/or not corroborated**
to call detection. **This is not hardware validation.** CFO/EVM stay withheld
because they are not meaningful for a weak, sparse, GMSK-like signal. This is the
present state of the LR1121 capture (RF=868 MHz, PWR=10 dBm, shortened packet
interval).

### `signal_detected`  (strong evidence — all that apply must hold)
Promotion to `signal_detected` requires **all** of:

1. **(a)** `lr_fhss_candidate_score >= threshold` (strong hopping structure:
   enough occupied frequency bins, hop-like segments, and max-hold peaks, DC
   excluded);
2. **(b)** if a UART log is provided, `packet_sent_count > 0` (firmware actually
   issued TX commands during the capture window);
3. **(c)** if a TX ON/OFF comparison is provided, the **TX-ON** capture shows
   occupancy / hop evidence **significantly stronger** than the **TX-OFF**
   reference (differential proof that the energy is the LR1121, not ambient RF or
   receiver artefacts).

### What does NOT count as detection
- **`Packet sent!` on UART proves only firmware TX-command completion** — that
  the firmware pipeline ran. It does **not** prove RF was emitted on the correct
  sub-GHz port, nor that the SDR captured it.
- A non-zero `lr_fhss_candidate_score` alone is a *candidate*, not detection.
- The **signal detector plus the TX ON/OFF differential are the authority** for
  any RF claim. Until `signal_detected == true` with the ON/OFF corroboration in
  hand, the validation label stays at hardware-bringup / `weak_signal_candidate`.

---

## 4. How to run it

### 4.1 LR-FHSS-mode single analysis

```bash
uv run python hardware/usrp_scripts/analyze_capture.py CAP.fc32 \
    --sample-rate 1000000 \
    --lr-fhss-mode \
    --output-json out.json \
    --plot out.png \
    --maxhold-plot out_maxhold.png
```

`--lr-fhss-mode` switches scoring to the hopping-structure metrics in section 2.
The waterfall (`out.png`) shows the time-frequency dashes; the max-hold
(`out_maxhold.png`) stacks the hops into narrow peaks across the band.

### 4.2 TX ON/OFF comparison (differential proof)

```bash
uv run python hardware/usrp_scripts/compare_tx_on_off.py \
    --tx-on on.fc32 \
    --tx-off off.fc32 \
    --sample-rate 1000000 \
    --out-json comparison.json \
    --out-plot comparison.png
```

This is the corroboration required by rule 3(c): it confirms the TX-ON occupancy
is significantly above the TX-OFF reference, so the observed energy is
attributable to the LR1121 rather than ambient RF or the receiver's own
artefacts.

### 4.3 Automated on/off sweep

```bash
uv run python scripts/run_lr1121_sdr_sweep_auto.py \
    --serial 8000304 \
    --freqs "868e6" \
    --antennas "TX/RX" \
    --rate 1e6 \
    --gain 45 \
    --duration 10 \
    --mode on-off \
    --reset-method none \
    --uart /dev/tty.usbmodem1303
```

`--mode on-off` records a TX-ON and a TX-OFF capture per cell and runs the
differential comparison automatically. `--reset-method none` assumes
continuous-TX firmware (Mode B in
[docs/swdm001_auto_tx_mode.md](swdm001_auto_tx_mode.md)). Safe defaults:
`--rate 1e6 --gain 45 --duration 10` (`2e6 / 30s` caused a UHD overflow on this
Mac).

---

## 5. Interpretation — moving from candidate toward detection

`weak_signal_candidate` means the hops are there but thin / unconfirmed. To push
toward `signal_detected`:

- **Raise TX power.** The stock demo macro is low (`POWER_IN_DBM (-9)`); the
  current capture used 10 dBm. Higher (legal, attenuated) power makes each hop
  easier to clear the noise floor and lifts `lr_fhss_candidate_score`. Never
  cable a high-power TX into the USRP without attenuation (~30 dB).
- **Shorten the packet interval.** Lowering `INTER_PKT_DELAY_IN_MS` (e.g. 20000
  -> 1000, Mode B) puts **more hops inside the same capture window**, raising
  `occupied_frequency_bins`, `hop_like_segment_count`, and
  `maxhold_peak_count_excluding_dc`.
- **Longer / lower-rate capture.** A longer (or slower) capture window catches
  more hops, improving the hopping-structure statistics.
- **Always take a TX-OFF reference.** Run `--mode on-off` (or
  `compare_tx_on_off.py`) every time. The ON/OFF differential is the strongest,
  most defensible evidence and is **required** for promotion (rule 3(c)).

If, after these, the analyzer reports `lr_fhss_candidate_score >= threshold`,
`packet_sent_count > 0`, **and** TX-ON occupancy significantly above TX-OFF, the
capture promotes to `signal_detected` — RF energy detected above noise,
attributable to the LR1121. That is still RF detection, not PER.

---

## 6. Effect on paper hardware claims

**No paper hardware claim changes because of this document.** The current,
honest status is **hardware bring-up / `weak_signal_candidate`**:
firmware runs (UART `Packet sent!`), IQ captures exist, and a weak LR-FHSS-like
hopping candidate has been observed at the IQ level. The status remains
hardware-bringup / weak candidate **until** `signal_detected == true` with TX
ON/OFF corroboration. Even then, claims stay at RF-energy-detection level: no
LR-FHSS PER and no "hardware validated" language is permitted from this IQ-level
path. See the validation-label ladder in
[docs/lr1121_sdr_validation_workflow.md](lr1121_sdr_validation_workflow.md) and
report only the highest rung whose evidence is actually in hand.
