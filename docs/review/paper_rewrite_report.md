# Paper Rewrite Report — Simulation / Trace-Driven Evidence-Gated Controller

*Records the pivot of `paper/icc_main.tex` away from hardware validation to a
software-only, model-derived, evidence-gated controller paper. No hardware/RF
command was run for this rewrite; `dataraw/` was not touched.
`reference_is_measured_truth=false`.*

*Generated: 2026-06-14 UTC. Branch `experiment-bk2-tle-residual`.*

---

## 1. New title

`Physics-First Evidence-Gated Uplink Control for LR-FHSS Direct-to-Satellite IoT`

## 2. Sections changed (full-body rewrite of `paper/icc_main.tex`)

| Section | Change |
|---|---|
| Title | Replaced uncertainty-head title with the physics-first evidence-gated title. |
| Abstract | Fully rewritten: stale-TLE / LR-FHSS terminal control problem, SGP4 baseline, evidence-gated learned residual, real BK1/BK2 negative result, controlled synthetic stress, guard/outage/energy proxy, explicit no-hardware/no-measured/no-live non-claims. |
| Keywords | Replaced "calibrated uncertainty, risk-aware control" with "transmitter-side control, stale TLE, Doppler residual, evidence gate, safe-by-default learning". |
| Introduction | Rewritten around transmitter-side control before transmission, when-to-learn vs not, real-TLE evidence that learning may hurt, the gate as a safety mechanism, synthetic stress as conditional benefit. |
| Contributions | Replaced with the 5 new contributions (gated controller; real negative result; controlled stress characterisation; guard/outage/energy proxy formulation; claim-evidence discipline). |
| Fig.~1 (architecture) | TikZ figure rewritten: Prediction (stale TLE→SGP4→$f_{phys}$/$f_{ml}$) → Evidence Gate → terminal control + proxies. Removed the "Conducted IQ Evidence" group and the 9.82 dB box. |
| Related Work | Trimmed calibration framing; re-pointed the ML-orbit subsection to the "is learning justified at all" question; updated positioning table last rows to evidence-gated control. |
| System Model | Kept Doppler / residual-CFO equations; replaced the PGRL uncertainty-head architecture with the stale-TLE open-loop baseline + inter-TLE residual definition $r=D_{ref}-f_{phys}$ (model-derived). |
| Method (new) | Added the formal Evidence Gate: $\mathrm{MAE}_{phys}(V)$, $\mathrm{MAE}_{ml}(V)$, $G=\mathbf{1}[\mathrm{MAE}_{ml}(V)<\gamma\,\mathrm{MAE}_{phys}(V)]$, $\hat f=Gf_{ml}+(1-G)f_{phys}$; honest validation-window-only scope; guard/outage/energy proxies; datasets + chronological splits; control algorithm. |
| Evaluation | Rebuilt: real BK negative table (Tab.~\ref{tab:bk}-equiv), synthetic stress table, $\gamma$/validation-window sensitivity table; software-only scope note. |
| Old eval removed | Removed Stage 3E/3F/4 uncertainty-head/temperature/$\alpha$-sweep tables and figures (tempcal/risk/guardres) and the 5.35 m/Cov68-Cov95 centerpiece. |
| Hardware section | Removed as a results section; converted to a short Limitations/Future-Work paragraph documenting the inconclusive conducted-HIL attempt (missing coax + attenuator). |
| Limitations (new) | Explicit non-claims: no measured Doppler truth, no live satellite, no PER/BER/CRC/PDR/gateway ACK, no standards-compliant LR-FHSS decoding, no valid conducted-HIL evidence yet, synthetic stress is not real evidence, gate is validation-window evidence not a worst-case bound. |
| Conclusion | Rewritten to the safe-by-default, evidence-gated, software-only narrative. |
| Figures | All external `\includegraphics` removed (0 remain); only the self-contained TikZ architecture figure is kept. Orphaned `figures/fig2..5*.pdf` left on disk, unreferenced. |

## 3. Old claims removed

- 5.35 m position RMSE / Cov68=0.713 / Cov95=0.947 / $T=1.0$ as the main result.
- Risk-aware guard $g=g_{base}+\alpha\sigma_r$ and the outage 5.0%→1.7% @ 13.8% overhead $\alpha$-sweep as the centerpiece.
- "Conducted LR1121-to-USRP B210 capture … 9.82 dB TX-ON/OFF margin" as validation evidence.
- LR-FHSS-candidate score 0.76 / 101 candidate bursts / 19 bins IQ-structure evidence.
- Any phrasing implying hardware validation or signal presence.

## 4. New evidence used (all software-only / model-derived, from `docs/review/`)

- `bk_negative_result_compact.md` / `.csv` — real BK1 (8–168 h) + BK1→BK2 negative table.
- `gate_stress_compact.md` / `.csv` — synthetic stress regimes + guard/outage proxies.
- `evidence_gate_stress_experiment.md` — $\gamma$-sweep and validation-window sweep.
- `gate_threshold_interpretation.md` — $\gamma=0.95$ default rationale.
- `validation_window_sensitivity.md` — window-stability interpretation.
- `black_kite_residual_evidence_gate.md` — gate-closes-on-real-data record.
- `black_kite_1_target_specific_residual_experiment.md`, `black_kite_tle_history_residual_experiment.md` — source experiments.
- `claim_evidence_matrix.md`, `paper_reframing_blueprint.md` — wording discipline.
- `mac_conducted_hil_result_summary.md` — inconclusive HIL attempt (limitations paragraph).

## 5. Remaining limitations (as stated in the paper)

- Negative result is for two BLACK KITE satellites, not universal.
- Positive learning benefit is synthetic-only; gate gives a validation-window property, not a held-out/worst-case bound.
- Guard/outage/energy are control proxies, not link-layer measurements.
- No measured Doppler, no live satellite, no decoding, no PER/BER/CRC/PDR/gateway ACK.
- No valid conducted-HIL evidence yet (attempt halted: missing coax + fixed attenuator).

## 6. Compile result

`tectonic paper/icc_main.tex` → **success**, `paper/icc_main.pdf` written (~108.9 KiB).
Remaining warnings are cosmetic only: two small overfull `\hbox` (1.6 pt related-work
table, 4.8 pt BK table) and harmless "Object already defined" notes from the
pre-existing table-note `\patchcmd` hack. No errors.

## 7. Grep audit result

Command (as required):
```
grep -Rni "guarantee\|worst-case\|measured Doppler\|live satellite\|PER\|BER\|CRC\|PDR\|gateway ACK\|can only help\|hardware validates\|hardware-validated\|conducted LR1121-to-USRP capture confirms\|learned residual.*real BLACK KITE" paper/icc_main.tex
```
- Genuine dangerous tokens (`guarantee`, `worst-case`, `can only help`,
  `hardware-validated`, `hardware validates`, `conducted LR1121-to-USRP capture
  confirms`): **0 matches.**
- Word-bounded `PER|BER|CRC|PDR`: only the single Limitations sentence listing
  them as **not measured** (explicit non-claim).
- `measured Doppler` / `live satellite`: only in **"no measured Doppler" / "no
  live-satellite"** non-claim contexts.
- Remaining case-insensitive hits are the substring "per" inside *paper,
  property, percentile, performance, unexplored, Experimental* — false positives.

**Conclusion: no overclaim remains; every flagged token is an honest disclaimer.**

## 8. `git diff --stat`

```
paper/icc_main.tex | 627 +++++++++++++++++++++++++-----------
 1 file changed, 427 insertions(+), 200 deletions(-)
```
(`paper/icc_main.pdf` is git-ignored; not shown.)

## 9. Confirmations

- **No hardware/RF/UART/TX/capture command was run** for this rewrite (edit +
  `grep` + `tectonic` only).
- **`dataraw/` not touched** (`git status` shows no `dataraw` path).
- **LoRa antenna not used as evidence**; the inconclusive HIL attempt is in
  limitations only.
- Paper claim is now **simulation / trace-driven / model-derived only.**
- Not committed.

---

## 10. Final polishing edits (low-risk, narrative unchanged)

Three targeted insertions; no main-narrative change, no hardware/measured/live
claim reintroduced.

1. **§V-B (real BK negative result):** added a cautious physical interpretation of
   why the inter-TLE residual is unpredictable — *consistent with*
   orbit-determination updates, drag-model mismatch, tracking/fit noise, and
   occasional maneuver or bad-fit events, *not exposed as predictable
   terminal-side features*. Explicitly not a propagator-optimality claim; no
   "station-keeping caused", no "guarantees", no "worst-case".
2. **§V-C (extreme synthetic stress):** added a proxy interpretation — guard proxy
   $14.1$~kHz$\to$$950$~Hz means a much smaller reserved frequency margin that can
   lower conservative margin/overhead, stated as an energy/overhead **proxy**, not
   a measured power/battery saving, gateway search-window measurement, or
   link-layer result; benefit is synthetic-only.
3. **§V-A (experimental setup):** added one sentence that 868~MHz is a
   representative carrier for Doppler scaling/reproducibility and the controller is
   frequency-parametric (rescale $f_c$); no NCC/AS923 or regulatory claim.

Hardware attempt **kept** as a limitation/future-work paragraph (not a
contribution).

**Compile:** `tectonic paper/icc_main.tex` → success, `paper/icc_main.pdf`
(~110.5 KiB); only two cosmetic overfull hboxes (1.6 pt, 4.8 pt). No errors.

**Grep (polish pass):**
```
grep -Rni "guarantee\|worst-case\|measured Doppler\|live satellite\|can only help\|hardware validates\|hardware-validated\|conducted LR1121-to-USRP capture confirms\|learned residual.*real BLACK KITE\|battery saving\|power saving\|station-keeping caused" paper/icc_main.tex
```
3 matches, **all explicit non-claims**: "no measured Doppler, no live-satellite…"
(Fig.~1 note), "not a measured power saving, a gateway receiver…" (§V-C
disclaimer), "no measured Doppler truth, no live-satellite…" (limitations). No
"station-keeping caused", no genuine overclaim.

**Confirmations (polish pass):** no hardware/RF command run (edit + grep +
tectonic only); `dataraw/` not touched; paper remains software-only /
model-derived. Not committed.

---

## 11. Finalization pass (parallel tracks A–G, coordinator-applied)

Maximum-depth review across seven tracks; only minimal high-value edits applied.
No narrative pivot; no hardware/measured/live content reintroduced.

- **Track A (overclaim auditor):** whole-paper sweep found no dangerous claim. All
  PER/BER/CRC/PDR/measured-Doppler/live-satellite/power-saving occurrences are
  explicit non-claims/limitations. No edit needed.
- **Track B (narrative):** story chain verified coherent
  (problem $\to$ stale-TLE baseline $\to$ naive-learning failure $\to$ Evidence
  Gate $\to$ real BK negative $\to$ synthetic stress open/close $\to$
  guard/outage/energy proxy $\to$ limitations). One reviewer-facing sentence added
  in §V-B framing the negative result constructively ("a rigorous real-data case
  showing why always-on residual learning is unsafe … why an evidence gate … is
  the appropriate design").
- **Track C (math/method):** notation verified consistent and defined before use:
  $f_{\mathrm{phys}}, D_{\mathrm{ref}}, r, f_{\mathrm{ml}}, G, \gamma,
  \mathrm{MAE}_{\mathrm{phys}}(V), \mathrm{MAE}_{\mathrm{ml}}(V), \hat f, e,
  g=2p_{99}(|e|), \rho=\Pr[|e|>F_{\mathrm{tol}}]$. No edit needed.
- **Track D (evaluation):** real BK table framed as negative evidence; synthetic
  stress labelled controlled simulation / not real BK evidence; $\gamma$ and
  validation-window tables explained as gate behavior, not guarantee; energy
  language kept proxy-only. No further edit needed.
- **Track E (figure):** Fig.~1 already matches the preferred structure (stale TLE
  $\to$ SGP4 baseline + optional ML path $\to$ Evidence Gate central block $\to$
  default-to-physics mux $\hat f=Gf_{\mathrm{ml}}+(1-G)f_{\mathrm{phys}}$ $\to$
  guard/outage/energy proxy; bottom software-only note; no hardware path). Left
  unchanged to avoid risk to a clean-compiling figure.
- **Track F (compile/layout):** `tectonic paper/icc_main.tex` $\to$ success,
  `paper/icc_main.pdf` (~110.8 KiB), **6 pages** (within workshop limit). Only two
  cosmetic overfull hboxes (1.6 pt, 4.8 pt); left per instructions. No undefined
  citations.
- **Track G (references):** citations consistent; no new references added; none
  invented; `refs.bib` untouched.

**Finalization grep (full, incl. PER/BER/CRC/PDR/gateway ACK):** every match is an
explicit non-claim/limitation (e.g.\ "not a measured power saving, a gateway
receiver…", "not link-budget, packet-error rate (PER), …, (PDR) measurements",
"no measured Doppler truth, no live-satellite…"). No "guarantee", "worst-case",
"hardware-validated", "station-keeping caused", or "battery/power saving" claim.
**Zero dangerous overclaims.**

**Confirmations (finalization):** no hardware/RF/UART/TX/capture command run
(edit + grep + tectonic only); `dataraw/` untouched; paper remains
simulation/trace-driven/model-derived only. Not committed.

---

## Body densification pass

Expanded the main body into a full 6-page contribution while keeping the PDF at
**6 pages** (references end on page 6). No narrative pivot; no hardware/PGRL/
measured/live content reintroduced.

**Expansions made:**
- New §IV-B *Statistical Role of the Gate*: false open (Type-I) / missed open
  (Type-II) error modes, $\gamma$ as a conservative knob, $\gamma=0.95$ default.
  Avoids "guarantee/worst-case/bound/can only help/optimal".
- Expanded §IV-C proxies: cross-layer rationale (frequency error $\to$ LR-FHSS
  margin), guard proxy $g=2p_{99}(|e|)$, outage proxy $\rho=\Pr[|e|>F_{\mathrm{tol}}]$,
  and an illustrative overhead proxy $E_{\mathrm{proxy}}\propto(1+\alpha_g g/B)(1+\rho)$
  explicitly labelled not a measured energy/packet model.
- Expanded §V-B physical interpretation: "weakly structured inter-TLE residual";
  cautious wording; not-white-for-all-satellites caveat; no station-keeping-as-fact.
- New §V-E *Implications for Terminal Control* (ties Tables BK/stress/gamma) and
  §V-F *Design Implications* (fresh-TLE hard to beat; stale/biased only if
  validated; deploy with evidence logging, not always-on; practical rule, not a
  guarantee).
- Limitations restructured into a compact itemized scope (model-derived reference;
  no live/link-layer PER/BER/CRC/PDR/gateway ACK; negative result not universal;
  synthetic not real; gate validation-window only; proxies only); hardware kept as
  a limitation/future-work item.

**Layout actions to hold 6 pages:** added `enumitem` tight lists + compact
float/display spacing; shrank Fig.~1 vertical geometry; removed the qualitative
positioning table (content kept in §II-D prose) and the redundant algorithm float
(method fully specified by Eqs.\ and a one-paragraph deployment recipe); trimmed
several verbose sentences. Figure~1 unchanged in structure (no hardware path).

**Compile:** `tectonic paper/icc_main.tex` $\to$ success; `pdfinfo` reports
**Pages: 6**; references end on page 6 (no page-7 overflow). One cosmetic 4.8 pt
overfull hbox (BK table); no errors.

**Grep audit (full, incl. PER/BER/CRC/PDR/gateway ACK):** 5 matches, all explicit
non-claims (Fig.~1 note "no measured Doppler, no live-satellite"; §IV-C "not a
measured power saving, a gateway receiver…"; Limitations "no … PER, BER, CRC,
PDR, or gateway acknowledgement"). No "guarantee", "worst-case",
"hardware-validated", "station-keeping caused", "battery/power saving" claim; no
"learned residual improves real BLACK KITE". **Zero overclaims.**

**Confirmations (densification):** no hardware/RF/UART/TX/capture command run
(edit + grep + tectonic + pdfinfo only); `dataraw/` untouched; paper remains
software-only / model-derived. Not committed.
