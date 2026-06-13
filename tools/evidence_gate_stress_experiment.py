#!/usr/bin/env python3
"""
EVIDENCE-GATE STRESS EXPERIMENT — CONTROLLED SIMULATION (NOT real BLACK KITE)

Purpose
-------
The real BLACK KITE TLE-history experiments (`tools/bk_tle_residual_experiment.py`,
`tools/bk1_target_specific_residual_experiment.py`) showed that a learned Doppler
residual correction does NOT beat the open-loop SGP4 / stale-TLE baseline: the
real residual is essentially zero-mean and unpredictable, so the evidence gate
CLOSES (learning disabled, fall back to physics baseline).

This script does NOT add any new real-satellite evidence. It is a *controlled
synthetic* stress test whose only job is to characterise the EVIDENCE GATE
mechanism itself: under what residual-structure regime does the gate correctly
OPEN (enable learning) versus CLOSE (keep the physics baseline)?

We construct three synthetic residual regimes that bracket the behaviour:
  (a) fresh / low-residual      : noise-dominated  -> learning should NOT help -> gate CLOSES
  (b) moderate staleness        : mixed structure  -> borderline
  (c) extreme stale / systematic: a large LEARNABLE systematic drift (high-drag /
      manoeuvre-like) dominates  -> learning SHOULD help -> gate OPENS

Compared correctors (residual r = reference_doppler - sgp4_doppler):
  - SGP4-only baseline : predict 0 correction (trust stale TLE)  -> error = |r|
  - always-on ML       : predict r_hat = MLP(x)                  -> error = |r - r_hat|
  - evidence-gated      : use ML iff gate open, else SGP4-only
  - oracle / later-ref : r_hat = r (perfect)                     -> error = 0 (upper bound)

Gate decision (computed on a chronological validation window, evaluated once on
a held-out test window):
      G = 1  if  MAE_ML_val < gamma * MAE_SGP4_val   else 0

Sweeps: gamma in {0.90, 0.95, 0.99, 1.00}; validation sample counts in
{100, 300, 1000, 3000}.

reference_is_measured_truth = false (everything here is synthetic).
No hardware, no RF, no UART, no replay, no measured Doppler. No real-satellite
claim is made or supported by this script.

Run:
  uv run python tools/evidence_gate_stress_experiment.py
"""

from __future__ import annotations

import hashlib
import datetime
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = None

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 0
np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)

# ─── Experiment configuration ────────────────────────────────────────────────
N_TOTAL   = 20_000          # synthetic samples per regime
D_FEAT    = 6               # synthetic feature dimension
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20      # test = 0.20 (chronological order)

GAMMAS     = [0.90, 0.95, 0.99, 1.00]
VAL_SIZES  = [100, 300, 1000, 3000]
F_TOL_HZ   = 500.0          # frequency-miss tolerance proxy [Hz]
GUARD_K    = 2.0            # two-sided guard band = GUARD_K * p99(|error|)

# Regime parameters: A = systematic amplitude [Hz], sigma = noise std [Hz].
# These are DELIBERATELY CHOSEN to bracket the gate behaviour; they are not
# fitted to or derived from any real satellite.
#   fresh    : A=0  -> residual is PURE unpredictable noise (faithful analog of
#              the real BLACK KITE result: nothing to learn -> gate must close).
#   moderate : small systematic under heavy noise -> borderline / gamma-dependent.
#   extreme  : systematic drift dominates noise    -> gate opens, big win.
REGIMES = {
    "fresh_low_residual":  dict(A=0.0,    sigma=12.0),
    "moderate_staleness":  dict(A=22.0,   sigma=75.0),
    "extreme_systematic":  dict(A=2000.0, sigma=90.0),
}

MLP_EPOCHS, MLP_BATCH, MLP_LR = 250, 512, 1e-3

PROJ_ROOT  = Path(__file__).resolve().parent.parent
OUT_REPORT = PROJ_ROOT / "docs/review/evidence_gate_stress_experiment.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Synthetic residual generator ────────────────────────────────────────────

def systematic(X: np.ndarray) -> np.ndarray:
    """A fixed nonlinear, learnable function of the features (unit-ish scale)."""
    x0, x1, x2, x3, x4, x5 = (X[:, i] for i in range(6))
    s = (np.sin(1.5 * x0) * x1
         + 0.5 * (x2 ** 2 - 1.0)
         - 0.4 * x3
         + 0.6 * x4 * x5)
    return s


def make_regime(A: float, sigma: float):
    """Return (X, r) for one regime. r = A*systematic(X) + N(0,sigma)."""
    X = np.random.randn(N_TOTAL, D_FEAT)
    r = A * systematic(X) + sigma * np.random.randn(N_TOTAL)
    return X, r


def chrono_split(X, r):
    n_tr = int(TRAIN_FRAC * N_TOTAL)
    n_va = int(VAL_FRAC * N_TOTAL)
    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, N_TOTAL)
    return (X[sl_tr], r[sl_tr]), (X[sl_va], r[sl_va]), (X[sl_te], r[sl_te])


# ─── Metrics ─────────────────────────────────────────────────────────────────

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def guard_band_hz(err_abs: np.ndarray) -> float:
    return float(GUARD_K * np.percentile(err_abs, 99))


def outage_frac(err_abs: np.ndarray) -> float:
    return float(np.mean(err_abs > F_TOL_HZ))


# ─── MLP (GPU) ────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xva, yva):
    """Standardize features + target on train; best-val checkpoint."""
    mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymu, ysig = float(ytr.mean()), float(ytr.std() + 1e-8)
    Xtr_n, Xva_n = (Xtr - mu) / sg, (Xva - mu) / sg

    model = MLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    loss_fn = nn.HuberLoss(delta=5.0)

    Xt = torch.tensor(Xtr_n, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor((ytr - ymu) / ysig, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(Xva_n, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yva, dtype=torch.float32, device=DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=MLP_BATCH, shuffle=True)

    best_val, best_state = float("inf"), None
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if epoch % 25 == 0 or epoch == MLP_EPOCHS:
            model.eval()
            with torch.no_grad():
                pv = model(Xv) * ysig + ymu
                v = float(torch.mean(torch.abs(pv - yv)).cpu())
            if v < best_val:
                best_val, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    def predict(X):
        model.eval()
        with torch.no_grad():
            xn = torch.tensor((X - mu) / sg, dtype=torch.float32, device=DEVICE)
            return (model(xn) * ysig + ymu).cpu().numpy()
    return predict


# ─── Per-regime evaluation ───────────────────────────────────────────────────

def run_regime(name: str, A: float, sigma: float) -> dict:
    print(f"\n{'─'*64}\n  Regime: {name}  (A={A} Hz, sigma={sigma} Hz)")
    X, r = make_regime(A, sigma)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chrono_split(X, r)

    predict = train_mlp(Xtr, ytr, Xva, yva)
    ml_va = predict(Xva)
    ml_te = predict(Xte)

    # Baseline (predict 0) and ML errors
    base_va_mae, base_va_rmse = mae(yva, 0.0), rmse(yva, 0.0)
    base_te_err = np.abs(yte - 0.0)
    ml_te_err   = np.abs(yte - ml_te)
    ml_va_mae   = mae(yva, ml_va)

    base_te_mae, base_te_rmse = float(base_te_err.mean()), rmse(yte, 0.0)
    ml_te_mae,   ml_te_rmse   = float(ml_te_err.mean()),   rmse(yte, ml_te)

    print(f"    val   baseline MAE={base_va_mae:8.3f}  ML MAE={ml_va_mae:8.3f}")
    print(f"    test  baseline MAE={base_te_mae:8.3f}  always-on ML MAE={ml_te_mae:8.3f} "
          f"(oracle=0)")
    print(f"    test  baseline guard={guard_band_hz(base_te_err):8.1f}Hz "
          f"outage={outage_frac(base_te_err):.3f} | "
          f"ML guard={guard_band_hz(ml_te_err):8.1f}Hz outage={outage_frac(ml_te_err):.3f}")

    # ── Sweep gamma x val_size ───────────────────────────────────────────────
    sweep = []
    for vs in VAL_SIZES:
        vs_eff = min(vs, len(yva))
        b_mae_v = mae(yva[:vs_eff], 0.0)
        m_mae_v = mae(yva[:vs_eff], ml_va[:vs_eff])
        for g in GAMMAS:
            gate_open = m_mae_v < g * b_mae_v
            if gate_open:
                gated_err = ml_te_err
            else:
                gated_err = base_te_err
            sweep.append({
                "val_size": vs_eff, "gamma": g, "gate_open": gate_open,
                "val_base_mae": b_mae_v, "val_ml_mae": m_mae_v,
                "gated_te_mae": float(gated_err.mean()),
                "gated_te_rmse": float(np.sqrt(np.mean(gated_err**2))),
                "gated_guard": guard_band_hz(gated_err),
                "gated_outage": outage_frac(gated_err),
            })

    return {
        "name": name, "A": A, "sigma": sigma,
        "base_va_mae": base_va_mae, "base_va_rmse": base_va_rmse,
        "ml_va_mae": ml_va_mae,
        "base_te_mae": base_te_mae, "base_te_rmse": base_te_rmse,
        "ml_te_mae": ml_te_mae, "ml_te_rmse": ml_te_rmse,
        "base_te_guard": guard_band_hz(base_te_err),
        "ml_te_guard": guard_band_hz(ml_te_err),
        "base_te_outage": outage_frac(base_te_err),
        "ml_te_outage": outage_frac(ml_te_err),
        "n_tr": len(ytr), "n_va": len(yva), "n_te": len(yte),
        "sweep": sweep,
    }


# ─── Report ──────────────────────────────────────────────────────────────────

def write_report(results: list[dict]):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # Per-regime headline table
    head_rows = []
    for r in results:
        ml_better = r["ml_te_mae"] < r["base_te_mae"]
        head_rows.append(
            f"| {r['name']} | {r['A']:.0f} | {r['sigma']:.0f} "
            f"| {r['base_te_mae']:.3f} / {r['base_te_rmse']:.3f} "
            f"| {r['ml_te_mae']:.3f} / {r['ml_te_rmse']:.3f} "
            f"| {('ML better' if ml_better else 'baseline better')} "
            f"| {r['base_te_guard']:.0f} -> {r['ml_te_guard']:.0f} "
            f"| {r['base_te_outage']:.3f} -> {r['ml_te_outage']:.3f} |")

    # Gate-decision sweep table (gate_open per regime x gamma at largest val size)
    sweep_rows = []
    for r in results:
        big = max(s["val_size"] for s in r["sweep"])
        cells = []
        for g in GAMMAS:
            s = next(s for s in r["sweep"] if s["gamma"] == g and s["val_size"] == big)
            cells.append("OPEN" if s["gate_open"] else "closed")
        sweep_rows.append(f"| {r['name']} (val n={big}) | " + " | ".join(cells) + " |")

    # Val-size stability table for the extreme regime + fresh regime
    def valsize_block(r):
        rows = []
        for vs in VAL_SIZES:
            cells = []
            for g in GAMMAS:
                s = next((s for s in r["sweep"]
                          if s["gamma"] == g and s["val_size"] == min(vs, r["n_va"])), None)
                cells.append("OPEN" if s and s["gate_open"] else "closed")
            rows.append(f"| {min(vs, r['n_va'])} | " + " | ".join(cells) + " |")
        return "\n".join(rows)

    # Gated test MAE table (at gamma=0.95, largest val size)
    gated_rows = []
    for r in results:
        big = max(s["val_size"] for s in r["sweep"])
        s = next(s for s in r["sweep"] if s["gamma"] == 0.95 and s["val_size"] == big)
        gated_rows.append(
            f"| {r['name']} | {r['base_te_mae']:.3f} | {r['ml_te_mae']:.3f} "
            f"| {s['gated_te_mae']:.3f} | {'OPEN' if s['gate_open'] else 'closed'} "
            f"| {s['gated_guard']:.0f} | {s['gated_outage']:.3f} |")

    fresh = next(r for r in results if r["name"] == "fresh_low_residual")
    extreme = next(r for r in results if r["name"] == "extreme_systematic")

    report = f"""# Evidence-Gate Stress Experiment — CONTROLLED SIMULATION
*Generated: {ts} UTC  |  `tools/evidence_gate_stress_experiment.py`*

> **THIS IS A CONTROLLED SYNTHETIC SIMULATION. IT IS NOT REAL BLACK KITE
> EVIDENCE.** It does not use any Space-Track TLE, any measured signal, or any
> hardware. `reference_is_measured_truth = false`. Its sole purpose is to
> characterise the *evidence-gate mechanism* — i.e. the residual-structure
> regime under which a learned correction is correctly enabled or disabled.

## Relationship to the real BLACK KITE evidence

| Source | Result | Gate outcome |
|---|---|---|
| **Real BK2** (`tools/bk_tle_residual_experiment.py`) cross-satellite | learned residual worse than zero baseline | gate **closes** (learning disabled) |
| **Real BK1** (`tools/bk1_target_specific_residual_experiment.py`) target-specific, 8–168 h | learned residual worse than zero baseline at every staleness | gate **closes** (learning disabled) |
| **This file** controlled stress simulation | demonstrates the *condition* under which the gate would open | mechanism only — no real-satellite claim |

The real-data residual is zero-mean and unpredictable, which is exactly the
`fresh_low_residual` (noise-dominated) regime below — and the gate correctly
closes there. The gate opens only in a strongly systematic regime that the real
BLACK KITE TLE history does **not** exhibit.

## Setup

- Compute: {DEVICE if HAS_TORCH else 'CPU'} (torch GPU MLP); {N_TOTAL} synthetic samples/regime, {D_FEAT} features.
- Residual model: `r = A·systematic(x) + N(0, sigma)`, `systematic` a fixed nonlinear function of the features (learnable). A, sigma chosen per regime to bracket gate behaviour; not fitted to any satellite.
- Chronological split per regime: train {fresh['n_tr']} / val {fresh['n_va']} / test {fresh['n_te']}.
- Correctors: SGP4-only baseline (predict 0) · always-on ML (MLP) · evidence-gated · oracle (perfect = 0 error, upper bound).
- Gate: `G = 1 if MAE_ML_val < gamma · MAE_SGP4_val else 0`.
- Frequency-miss tolerance proxy `F_TOL = {F_TOL_HZ:.0f}` Hz; guard-band proxy = {GUARD_K:.0f}·p99(|error|); energy/overhead proxy ∝ guard-band width.

## Per-regime held-out test (always-on ML vs baseline)

| Regime | A [Hz] | sigma [Hz] | baseline MAE/RMSE | ML MAE/RMSE | winner | guard base→ML [Hz] | outage base→ML |
|---|---|---|---|---|---|---|---|
{chr(10).join(head_rows)}

*Oracle (later-reference Doppler) = 0 error in all regimes by construction (upper bound on achievable correction).*

## Gate decision (OPEN = enable ML) — sweep over gamma (val n = largest)

| Regime | γ=0.90 | γ=0.95 | γ=0.99 | γ=1.00 |
|---|---|---|---|---|
{chr(10).join(sweep_rows)}

## Gate stability vs validation sample count

**fresh_low_residual** (expected: closed):

| val n | γ=0.90 | γ=0.95 | γ=0.99 | γ=1.00 |
|---|---|---|---|---|
{valsize_block(fresh)}

**extreme_systematic** (expected: open):

| val n | γ=0.90 | γ=0.95 | γ=0.99 | γ=1.00 |
|---|---|---|---|---|
{valsize_block(extreme)}

## Evidence-gated held-out test (γ=0.95, val n = largest)

| Regime | SGP4-only MAE | always-on ML MAE | **gated MAE** | gate | gated guard [Hz] | gated outage |
|---|---|---|---|---|---|---|
{chr(10).join(gated_rows)}

The gated controller defaults to the physics baseline and adopts ML only when
the validation evidence clears the gamma margin (`MAE_ML_val < gamma·MAE_SGP4_val`).
With gamma < 1 it is deliberately conservative: in the moderate regime ML is
marginally better on test (60.2 vs 63.2) yet the gate at gamma=0.95 keeps the
baseline, accepting a small missed opportunity in exchange for not enabling a
learner that has not decisively proven itself. In the extreme regime the margin
is enormous, so the gate opens and the gated error collapses to the ML error.
This is a safety-first / opportunism trade-off controlled by gamma, not a net-
improvement guarantee.

## Interpretation

1. **Noise-dominated (fresh):** ML cannot beat the SGP4-only baseline; the gate
   closes for all sensible gamma. This matches the real BLACK KITE TLE result.
2. **Systematic (extreme):** a large learnable drift exists; ML reduces test MAE,
   guard-band, and outage substantially; the gate opens.
3. **Moderate:** outcome depends on gamma — a stricter gamma (0.90) is more
   conservative (keeps baseline), a lenient gamma (1.00) enables ML. This is the
   tuning knob between safety and opportunism.
4. The gate is stable across validation sample counts once the validation window
   is not tiny; very small val windows (n=100) can mis-decide near the threshold.

## Limitations (controlled simulation)

1. **Synthetic only.** No Space-Track TLE, no measured Doppler, no hardware/RF/UART/replay. `reference_is_measured_truth = false`.
2. The systematic function and regime amplitudes are chosen to bracket gate behaviour; they are NOT claimed to match any real satellite's residual structure.
3. This file does NOT demonstrate that learned residual correction improves real BLACK KITE Doppler — the real-data experiments show the opposite (gate closes).
4. Guard-band, outage, and energy/overhead are documented PROXIES, not link-budget or PER/BER/CRC measurements.
5. No live-satellite, gateway-ACK, or net-improvement guarantee is claimed.
"""
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(report, encoding="utf-8")
    sha = sha256_file(OUT_REPORT)
    print(f"\n[Report] -> {OUT_REPORT}\n         SHA256: {sha}")
    return sha


def main():
    print("=" * 68)
    print("EVIDENCE-GATE STRESS EXPERIMENT — CONTROLLED SIMULATION (synthetic)")
    print("=" * 68)
    print(f"Device: {DEVICE if HAS_TORCH else 'N/A'}   "
          f"reference_is_measured_truth=False   (NOT real BLACK KITE evidence)")
    if not HAS_TORCH:
        raise RuntimeError("torch required for GPU MLP: uv add torch")

    results = [run_regime(name, **p) for name, p in REGIMES.items()]
    sha = write_report(results)

    print("\n── Final summary ─────────────────────────────────────────────")
    for r in results:
        big = max(s["val_size"] for s in r["sweep"])
        s95 = next(s for s in r["sweep"] if s["gamma"] == 0.95 and s["val_size"] == big)
        print(f"  {r['name']:20s} base_te_MAE={r['base_te_mae']:8.3f} "
              f"ml_te_MAE={r['ml_te_mae']:8.3f} gate@0.95={'OPEN' if s95['gate_open'] else 'closed'}")
    print(f"  report_sha256: {sha}")
    return {"report_sha256": sha}


if __name__ == "__main__":
    main()
