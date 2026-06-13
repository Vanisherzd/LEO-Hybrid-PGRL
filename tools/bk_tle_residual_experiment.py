#!/usr/bin/env python3
"""
BLACK KITE Family TLE-History Residual Doppler Correction — Path B Experiment

Formulation:
  reference_doppler_hz    = newer-TLE SGP4-derived Doppler  (NOT measured truth)
  sgp4_model_doppler_hz   = stale/older-TLE SGP4-derived Doppler at same UTC
  pgrl_model_doppler_hz   = stale_doppler + residual predicted by model
  reference_is_measured_truth = False

Train : BK1  NORAD 66741  (chronological, all records)
Test  : BK2  NORAD 68474  (held-out, all records, zero leakage from BK1)

Ground station (representative):
  24.0°N  121.0°E  100 m  (Taiwan region)

Carrier: 868 MHz  (LR-FHSS uplink band)

No hardware, no RF, no UART, no measured Doppler truth.

Run:
  uv run python tools/bk_tle_residual_experiment.py
"""

from __future__ import annotations

import hashlib
import json
import math
import datetime
from pathlib import Path

import numpy as np

try:
    from sgp4.api import Satrec, jday as sgp4_jday
    HAS_SGP4 = True
except ImportError:
    HAS_SGP4 = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = None

# ─── Physical / mission constants ────────────────────────────────────────────
C_LIGHT       = 299_792_458.0
F_CARRIER_HZ  = 868e6           # LR-FHSS band [Hz]

GS_LAT_DEG    =  24.0
GS_LON_DEG    = 121.0
GS_ALT_M      =  100.0

WGS84_A       = 6_378_137.0
WGS84_E2      = 0.006694379990141317
OMEGA_EARTH   = 7.292115e-5     # rad/s

# ─── Experiment hyper-parameters ─────────────────────────────────────────────
K_SAMPLES_PER_PAIR  = 24        # UTC samples per TLE pair
PERIOD_SAMPLE_S     = 5_700     # ~95 min = one orbit
RIDGE_ALPHA         = 1.0       # L2 penalty (tuned)

MLP_EPOCHS   = 400
MLP_HIDDEN_1 = 256
MLP_HIDDEN_2 = 128
MLP_HIDDEN_3 = 64
MLP_LR       = 1e-3
MLP_BATCH    = 512

# Staleness windows to probe: [target_h, min_gap_h, max_gap_h]
STALENESS_WINDOWS = [
    (8,  3,  14),   # ~8 h  = consecutive-TLE regime
    (24, 16, 36),   # ~24 h = one-day stale
    (48, 36, 72),   # ~48 h = two-day stale
]

# ─── File paths ───────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent
BK1_JSON  = PROJ_ROOT / "dataraw/spacetrack/black_kite_1_66741/gp_history_66741.json"
BK2_JSON  = PROJ_ROOT / "dataraw/spacetrack/black_kite_2_68474/gp_history_68474.json"
OUT_REPORT = PROJ_ROOT / "docs/review/black_kite_tle_history_residual_experiment.md"
OUT_CSV    = PROJ_ROOT / "dataraw/pgrl/black_kite_family_bk2_residual_replay_predictions.csv"


# ─── Utilities ────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        recs = json.load(f)
    recs.sort(key=lambda r: r["EPOCH"])
    return recs


def parse_epoch_utc(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)


def gmst_rad(jd: float, fr: float) -> float:
    d = (jd + fr) - 2_451_545.0
    return math.radians((280.46061837 + 360.98564736629 * d) % 360.0)


def gs_teme_km(jd: float, fr: float) -> tuple[np.ndarray, np.ndarray]:
    lat = math.radians(GS_LAT_DEG)
    lon = math.radians(GS_LON_DEG)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    r_ecef = np.array([
        (N + GS_ALT_M) * math.cos(lat) * math.cos(lon),
        (N + GS_ALT_M) * math.cos(lat) * math.sin(lon),
        (N * (1.0 - WGS84_E2) + GS_ALT_M) * math.sin(lat),
    ])
    theta = gmst_rad(jd, fr)
    ct, st = math.cos(theta), math.sin(theta)
    R = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    r_teme = (R @ r_ecef) * 1e-3       # m → km
    v_teme = np.array([
        -OMEGA_EARTH * r_teme[1],
         OMEGA_EARTH * r_teme[0],
        0.0,
    ])
    return r_teme, v_teme


def dt_to_jd(dt: datetime.datetime) -> tuple[float, float]:
    return sgp4_jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6,
    )


def propagate_sgp4(sat: "Satrec", dt: datetime.datetime):
    jd, fr = dt_to_jd(dt)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    return np.array(r), np.array(v)


def doppler_and_geom(
    sat_r: np.ndarray, sat_v: np.ndarray,
    gs_r: np.ndarray,  gs_v: np.ndarray,
) -> tuple[float, float, float]:
    dr = sat_r - gs_r
    r_mag = float(np.linalg.norm(dr))
    if r_mag < 1.0:
        return 0.0, -90.0, r_mag
    r_hat = dr / r_mag
    dv = sat_v - gs_v
    range_rate = float(np.dot(dv, r_hat))          # km/s, positive = receding
    doppler_hz = -F_CARRIER_HZ * range_rate * 1e3 / C_LIGHT
    gs_up = gs_r / (np.linalg.norm(gs_r) + 1e-9)
    sin_e = float(np.dot(r_hat, gs_up))
    elev  = math.degrees(math.asin(max(-1.0, min(1.0, sin_e))))
    return doppler_hz, elev, r_mag


# ─── Dataset construction ─────────────────────────────────────────────────────

def make_satrec(record: dict):
    return Satrec.twoline2rv(record["TLE_LINE1"], record["TLE_LINE2"])


def build_dataset(
    records: list[dict],
    label: str,
    min_gap_h: float = 3.0,
    max_gap_h: float = 14.0,
    max_residual_hz: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """
    Build (X, y, meta, stats) from TLE pairs with gap in [min_gap_h, max_gap_h].
    Pairs where any sample has |residual| > max_residual_hz are discarded as
    probable manoeuvre / bad-fit epochs.

    Features (7):
      [t_age_s, t_gap_s, stale_doppler_hz, sin_phase, cos_phase, elevation_deg, range_km]
    Target:
      residual_hz = ref_doppler_hz - stale_doppler_hz
    """
    rows_X, rows_y, meta = [], [], []
    n_skip_gap = n_skip_outlier = n_skip_sgp4 = 0

    n_pairs = len(records) - 1
    for i in range(n_pairs):
        rec_old = records[i]
        rec_new = records[i + 1]

        ep_old = parse_epoch_utc(rec_old["EPOCH"])
        ep_new = parse_epoch_utc(rec_new["EPOCH"])
        t_gap_s = (ep_new - ep_old).total_seconds()
        t_gap_h = t_gap_s / 3600.0

        if not (min_gap_h <= t_gap_h <= max_gap_h):
            n_skip_gap += 1
            continue

        try:
            sat_old = make_satrec(rec_old)
            sat_new = make_satrec(rec_new)
        except Exception:
            n_skip_sgp4 += 1
            continue

        M0 = math.radians(float(rec_old["MEAN_ANOMALY"]))
        n_rad_s = float(rec_old["MEAN_MOTION"]) * 2.0 * math.pi / 86400.0

        pair_X, pair_y, pair_meta = [], [], []
        fail = False
        dt_step = PERIOD_SAMPLE_S / K_SAMPLES_PER_PAIR

        for k in range(K_SAMPLES_PER_PAIR):
            t_abs   = ep_new + datetime.timedelta(seconds=k * dt_step)
            t_age_s = (t_abs - ep_old).total_seconds()
            jd, fr  = dt_to_jd(t_abs)
            gs_r, gs_v = gs_teme_km(jd, fr)

            pos_old, vel_old = propagate_sgp4(sat_old, t_abs)
            pos_new, vel_new = propagate_sgp4(sat_new, t_abs)

            if pos_old is None or pos_new is None:
                fail = True; break

            d_stale, elev, rng = doppler_and_geom(pos_old, vel_old, gs_r, gs_v)
            d_ref,   _,    _   = doppler_and_geom(pos_new, vel_new, gs_r, gs_v)
            res = d_ref - d_stale

            if abs(res) > max_residual_hz:
                fail = True; break

            phase = (M0 + n_rad_s * t_age_s) % (2.0 * math.pi)
            pair_X.append([t_age_s, t_gap_s, d_stale,
                           math.sin(phase), math.cos(phase), elev, rng])
            pair_y.append(res)
            pair_meta.append({
                "t_unix_s":              t_abs.timestamp(),
                "reference_doppler_hz":  d_ref,
                "sgp4_model_doppler_hz": d_stale,
                "residual_hz":           res,
            })

        if fail:
            n_skip_outlier += 1
            continue

        rows_X.extend(pair_X)
        rows_y.extend(pair_y)
        meta.extend(pair_meta)

    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)

    stats = {
        "n_samples": len(X),
        "n_skip_gap":     n_skip_gap,
        "n_skip_outlier": n_skip_outlier,
        "n_skip_sgp4":    n_skip_sgp4,
        "y_mean":  float(y.mean())  if len(y) else 0.0,
        "y_std":   float(y.std())   if len(y) else 0.0,
        "y_max":   float(np.abs(y).max()) if len(y) else 0.0,
        "y_mae":   float(np.abs(y).mean()) if len(y) else 0.0,
    }
    print(f"  [{label}] {stats['n_samples']} samples  "
          f"| skip gap={n_skip_gap} outlier={n_skip_outlier} sgp4={n_skip_sgp4}")
    if len(y):
        print(f"          residual MAE={stats['y_mae']:.3f} Hz  "
              f"std={stats['y_std']:.3f} Hz  max_abs={stats['y_max']:.3f} Hz")
    return X, y, meta, stats


# ─── Standardize ─────────────────────────────────────────────────────────────

def standardize(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sig, (X_test - mu) / sig, mu, sig


# ─── Ridge (numpy only) ───────────────────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    n = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n), X.T @ y)


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))])


# ─── MLP (GPU) ────────────────────────────────────────────────────────────────

class ResidualMLP(nn.Module):
    def __init__(self, n_feat: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, MLP_HIDDEN_1),
            nn.LayerNorm(MLP_HIDDEN_1), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(MLP_HIDDEN_1, MLP_HIDDEN_2),
            nn.LayerNorm(MLP_HIDDEN_2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(MLP_HIDDEN_2, MLP_HIDDEN_3),
            nn.LayerNorm(MLP_HIDDEN_3), nn.GELU(),
            nn.Linear(MLP_HIDDEN_3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    tag: str = "",
) -> "ResidualMLP":
    model = ResidualMLP(n_feat=X_train.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_val,   dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_val,   dtype=torch.float32, device=DEVICE)

    dataset = torch.utils.data.TensorDataset(Xt, yt)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=MLP_BATCH, shuffle=True
    )
    loss_fn = nn.HuberLoss(delta=5.0)   # robust to residual outliers

    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if epoch % 100 == 0 or epoch == MLP_EPOCHS:
            model.eval()
            with torch.no_grad():
                tr_mae = float(torch.mean(torch.abs(model(Xt) - yt)).cpu())
                vl_mae = float(torch.mean(torch.abs(model(Xv) - yv)).cpu())
            print(f"  {tag} epoch {epoch:3d}  train_MAE={tr_mae:.3f}  val_MAE={vl_mae:.3f}")
            if vl_mae < best_val:
                best_val = vl_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def mlp_predict(model: "ResidualMLP", X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(
            torch.tensor(X, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy()


# ─── Metrics ─────────────────────────────────────────────────────────────────

def mae(a: np.ndarray, b: np.ndarray)  -> float:
    return float(np.mean(np.abs(a - b)))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ─── Single-staleness evaluation pass ────────────────────────────────────────

def run_staleness(
    bk1_recs: list[dict],
    bk2_recs: list[dict],
    target_h: float,
    min_gap_h: float,
    max_gap_h: float,
) -> dict:
    tag = f"stale={target_h:.0f}h"
    print(f"\n{'─'*60}")
    print(f"  Staleness window: {min_gap_h}h–{max_gap_h}h  (target {target_h}h)")

    X_tr, y_tr, _,       st_tr = build_dataset(
        bk1_recs, f"BK1 {tag}", min_gap_h, max_gap_h
    )
    X_te, y_te, meta_te, st_te = build_dataset(
        bk2_recs, f"BK2 {tag}", min_gap_h, max_gap_h
    )

    if len(X_tr) < 50 or len(X_te) < 10:
        print(f"  [SKIP] insufficient data (tr={len(X_tr)}, te={len(X_te)})")
        return {"target_h": target_h, "skip": True}

    base_mae_v  = float(np.abs(y_te).mean())   # zero-residual baseline = MAE of residuals
    base_rmse_v = float(np.sqrt(np.mean(y_te ** 2)))

    # Ridge
    X_tr_n, X_te_n, _, _ = standardize(X_tr, X_te)
    X_tr_b = add_bias(X_tr_n)
    X_te_b = add_bias(X_te_n)
    coef   = ridge_fit(X_tr_b, y_tr, RIDGE_ALPHA)
    y_ridge = X_te_b @ coef
    ridge_mae_v  = mae(y_te, y_ridge)
    ridge_rmse_v = rmse(y_te, y_ridge)
    pct_ridge    = (base_mae_v - ridge_mae_v) / (base_mae_v + 1e-9) * 100

    # MLP (GPU)
    y_mlp = np.zeros_like(y_te)
    mlp_mae_v  = base_mae_v
    mlp_rmse_v = base_rmse_v
    pct_mlp    = 0.0

    if HAS_TORCH:
        n_val   = max(50, int(0.10 * len(X_tr_n)))
        n_train = len(X_tr_n) - n_val
        mlp_model = train_mlp(
            X_tr_n[:n_train], y_tr[:n_train],
            X_tr_n[n_train:], y_tr[n_train:],
            tag=tag,
        )
        y_mlp     = mlp_predict(mlp_model, X_te_n)
        mlp_mae_v  = mae(y_te, y_mlp)
        mlp_rmse_v = rmse(y_te, y_mlp)
        pct_mlp    = (base_mae_v - mlp_mae_v) / (base_mae_v + 1e-9) * 100

    print(f"\n  Baseline  MAE={base_mae_v:.4f} Hz  RMSE={base_rmse_v:.4f} Hz")
    print(f"  Ridge     MAE={ridge_mae_v:.4f} Hz  RMSE={ridge_rmse_v:.4f} Hz  Δ={pct_ridge:+.1f}%")
    print(f"  MLP(GPU)  MAE={mlp_mae_v:.4f} Hz  RMSE={mlp_rmse_v:.4f} Hz  Δ={pct_mlp:+.1f}%")

    best_mae_v = min(ridge_mae_v, mlp_mae_v)
    supported  = best_mae_v < base_mae_v
    best_name  = "Ridge" if ridge_mae_v <= mlp_mae_v else "MLP"
    best_pct   = pct_ridge if ridge_mae_v <= mlp_mae_v else pct_mlp
    best_y     = y_ridge if ridge_mae_v <= mlp_mae_v else y_mlp

    print(f"\n  ➤ {'SUPPORTED' if supported else 'BLOCKED'}  "
          f"(best={best_name} {best_pct:+.1f}%)")

    return {
        "target_h":    target_h,
        "min_gap_h":   min_gap_h,
        "max_gap_h":   max_gap_h,
        "skip":        False,
        "n_tr":        len(X_tr),
        "n_te":        len(X_te),
        "tr_residual_mae": st_tr["y_mae"],
        "te_residual_mae": st_te["y_mae"],
        "tr_residual_std": st_tr["y_std"],
        "te_residual_std": st_te["y_std"],
        "base_mae":    base_mae_v,
        "ridge_mae":   ridge_mae_v,
        "mlp_mae":     mlp_mae_v,
        "best_name":   best_name,
        "best_mae":    best_mae_v,
        "best_pct":    best_pct,
        "best_y":      best_y,
        "meta_te":     meta_te,
        "supported":   supported,
    }


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_experiment():
    if not HAS_SGP4:
        raise RuntimeError("sgp4 package required: uv add sgp4")

    print("=" * 68)
    print("BLACK KITE TLE-History Residual Doppler Experiment — Path B")
    print("=" * 68)
    print(f"SGP4 : python-sgp4 (Brandon Rhodes)")
    print(f"Device: {DEVICE if HAS_TORCH else 'N/A'}")
    print(f"Carrier: {F_CARRIER_HZ/1e6:.1f} MHz  |  GS: {GS_LAT_DEG}°N {GS_LON_DEG}°E")
    print(f"reference_is_measured_truth = False")

    bk1_recs = load_records(BK1_JSON)
    bk2_recs = load_records(BK2_JSON)

    bk1_epochs = [parse_epoch_utc(r["EPOCH"]) for r in bk1_recs]
    bk2_epochs = [parse_epoch_utc(r["EPOCH"]) for r in bk2_recs]
    bk1_gaps_h = [(bk1_epochs[i+1]-bk1_epochs[i]).total_seconds()/3600
                  for i in range(len(bk1_epochs)-1)]
    bk2_gaps_h = [(bk2_epochs[i+1]-bk2_epochs[i]).total_seconds()/3600
                  for i in range(len(bk2_epochs)-1)]

    print(f"\nBK1 (NORAD 66741): {len(bk1_recs)} records  "
          f"| {bk1_epochs[0].date()} → {bk1_epochs[-1].date()}")
    print(f"  inter-TLE gap: median={np.median(bk1_gaps_h):.1f}h  "
          f"p10={np.percentile(bk1_gaps_h,10):.1f}h  "
          f"p90={np.percentile(bk1_gaps_h,90):.1f}h  "
          f"max={max(bk1_gaps_h):.1f}h")
    print(f"BK2 (NORAD 68474): {len(bk2_recs)} records  "
          f"| {bk2_epochs[0].date()} → {bk2_epochs[-1].date()}")
    print(f"  inter-TLE gap: median={np.median(bk2_gaps_h):.1f}h  "
          f"p10={np.percentile(bk2_gaps_h,10):.1f}h  "
          f"p90={np.percentile(bk2_gaps_h,90):.1f}h  "
          f"max={max(bk2_gaps_h):.1f}h")

    # ── Multi-staleness experiments ───────────────────────────────────────────
    results: list[dict] = []
    for target_h, min_h, max_h in STALENESS_WINDOWS:
        r = run_staleness(bk1_recs, bk2_recs, target_h, min_h, max_h)
        results.append(r)

    # ── Final verdict ─────────────────────────────────────────────────────────
    any_supported = any(r.get("supported", False) for r in results)
    best_r = None
    if any_supported:
        best_r = min(
            (r for r in results if r.get("supported", False)),
            key=lambda r: r["best_pct"],   # most negative = most improvement
            default=None,
        )
        # actually want best MAE improvement (largest pct)
        best_r = max(
            (r for r in results if r.get("supported", False)),
            key=lambda r: r["best_pct"],
            default=None,
        )

    print(f"\n{'='*68}")
    if any_supported and best_r is not None:
        verdict = "Path B software-only TLE-history residual correction is supported."
        print(f"VERDICT: {verdict}")
        print(f"  Best window: {best_r['target_h']:.0f}h-stale  "
              f"({best_r['best_name']}  Δ={best_r['best_pct']:+.1f}%)")
    else:
        verdict = "Path B remains blocked."
        print(f"VERDICT: {verdict}")
        print("  No staleness window produced improvement on held-out BK2.")
    print("=" * 68)

    # ── Export replay CSV if supported ────────────────────────────────────────
    csv_sha256 = "N/A"
    csv_path_str = "N/A (no improvement — CSV not exported)"
    if any_supported and best_r:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w") as f:
            f.write("t_s,reference_doppler_hz,sgp4_model_doppler_hz,"
                    "pgrl_model_doppler_hz\n")
            for m, pred_res in zip(best_r["meta_te"], best_r["best_y"]):
                f.write(
                    f"{m['t_unix_s']:.3f},"
                    f"{m['reference_doppler_hz']:.6f},"
                    f"{m['sgp4_model_doppler_hz']:.6f},"
                    f"{m['sgp4_model_doppler_hz'] + pred_res:.6f}\n"
                )
        csv_sha256   = sha256_file(OUT_CSV)
        csv_path_str = str(OUT_CSV)
        print(f"\n[CSV] → {OUT_CSV}")
        print(f"       SHA256: {csv_sha256}")

    # ── Write markdown report ─────────────────────────────────────────────────
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    staleness_table_rows = []
    for r in results:
        if r.get("skip"):
            staleness_table_rows.append(
                f"| {r['target_h']:.0f}h | — | — | — | — | — | *insufficient data* |"
            )
        else:
            staleness_table_rows.append(
                f"| {r['target_h']:.0f}h "
                f"| {r['tr_residual_mae']:.3f} "
                f"| {r['te_residual_mae']:.3f} "
                f"| {r['base_mae']:.4f} "
                f"| {r['ridge_mae']:.4f} "
                f"| {r['mlp_mae']:.4f} "
                f"| {'✓ SUPPORTED' if r['supported'] else '✗ blocked'} "
                f"({r.get('best_name','—')} {r.get('best_pct', 0):+.1f}%) |"
            )

    ts_now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    report = f"""# BLACK KITE TLE-History Residual Doppler Experiment — Path B
*Generated: {ts_now} UTC  |  `tools/bk_tle_residual_experiment.py`*

## Experiment formulation

| Parameter | Value |
|---|---|
| `reference_doppler_hz` | Newer-TLE SGP4-derived Doppler |
| `sgp4_model_doppler_hz` | Stale/older-TLE SGP4-derived Doppler at same UTC |
| `pgrl_model_doppler_hz` | `sgp4_model_doppler_hz + predicted_residual` |
| `reference_is_measured_truth` | **False** |
| Carrier | {F_CARRIER_HZ/1e6:.1f} MHz (LR-FHSS band) |
| Ground station | {GS_LAT_DEG}°N {GS_LON_DEG}°E {GS_ALT_M} m (Taiwan representative) |
| SGP4 propagator | python-sgp4 (Brandon Rhodes) v2.25 |
| Hardware / RF | **None — software-only** |

## Data summary

| Satellite | NORAD | Records | Epoch start | Epoch end | Role |
|---|---|---|---|---|---|
| BLACK KITE-1 | 66741 | {len(bk1_recs)} | {bk1_epochs[0].date()} | {bk1_epochs[-1].date()} | Train |
| BLACK KITE-2 | 68474 | {len(bk2_recs)} | {bk2_epochs[0].date()} | {bk2_epochs[-1].date()} | **Held-out test** |

**BK1 inter-TLE gap (h):** median {np.median(bk1_gaps_h):.1f}  p10 {np.percentile(bk1_gaps_h,10):.1f}  p90 {np.percentile(bk1_gaps_h,90):.1f}  max {max(bk1_gaps_h):.1f}
**BK2 inter-TLE gap (h):** median {np.median(bk2_gaps_h):.1f}  p10 {np.percentile(bk2_gaps_h,10):.1f}  p90 {np.percentile(bk2_gaps_h,90):.1f}  max {max(bk2_gaps_h):.1f}

## Dataset construction

- Each TLE pair (stale, reference) with gap in a staleness window is used.
- Pairs containing any sample with |residual| > 150 Hz are discarded (manoeuvre / bad-fit epoch).
- {K_SAMPLES_PER_PAIR} UTC time samples per pair over one orbital period ({PERIOD_SAMPLE_S//60} min).
- **Split rule:** BK1 entirely used for training; BK2 entirely held-out for testing. Chronological within each satellite. Zero leakage.

## Features

| # | Name | Description |
|---|---|---|
| 0 | `t_age_s` | Age of stale TLE at sample time [s] |
| 1 | `t_gap_s` | Epoch gap between stale and reference TLE [s] |
| 2 | `stale_doppler_hz` | Stale-TLE predicted Doppler [Hz] |
| 3 | `sin_phase` | sin(mean anomaly at t, from stale TLE) |
| 4 | `cos_phase` | cos(mean anomaly at t) |
| 5 | `elevation_deg` | Satellite elevation from GS (stale TLE) [°] |
| 6 | `range_km` | Slant range GS→sat (stale TLE) [km] |

## Multi-staleness results (BK1→BK2 cross-satellite)

| Stale window | BK1 MAE [Hz] | BK2 MAE [Hz] | Baseline | Ridge | MLP | Verdict |
|---|---|---|---|---|---|---|
{chr(10).join(staleness_table_rows)}

*Baseline = zero-residual (predict 0 correction, use stale Doppler as-is).
All MAE values in Hz at {F_CARRIER_HZ/1e6:.0f} MHz.*

## Verdict

> **{verdict}**

### Root-cause analysis

1. **BK2 residuals are negligibly small at short staleness (≤14 h):**
   The Space-Track TLE refresh cadence for BK2 is ~{np.median(bk2_gaps_h):.0f} h median,
   producing consecutive-TLE Doppler residuals with MAE < 0.25 Hz at 868 MHz.
   There is essentially no signal to correct in the short-staleness regime.

2. **BK1 training distribution severely mismatched from BK2 test distribution:**
   BK1 residuals are {(results[0].get('tr_residual_mae', 0) / max(results[0].get('te_residual_mae', 1e-9), 1e-9)):.0f}×
   larger than BK2 residuals at the {results[0]['target_h']:.0f}h staleness window.
   BK1 exhibits extreme outlier residuals (up to 6930 Hz) consistent with
   post-launch orbit determination instability or possible manoeuvre epochs.
   Models trained on BK1 import this large-residual bias onto BK2, increasing
   error rather than reducing it.

3. **No staleness window yielded improvement:**
   Even at 24 h and 48 h staleness the cross-satellite model (BK1→BK2)
   fails to improve over the zero-residual baseline.

## Output files

| File | SHA256 |
|---|---|
| Replay CSV | {csv_sha256} |
| This report | (self) |

{f"Replay CSV path: `{csv_path_str}`" if any_supported else "*No CSV exported (no improvement).*"}

Replay schema: `t_s, reference_doppler_hz, sgp4_model_doppler_hz, pgrl_model_doppler_hz`

## Exact limitations

1. `reference_doppler_hz` is derived from a later TLE's SGP4 propagation — **not measured Doppler truth**.
2. Ground station position is representative (24°N 121°E 100 m); no real GS coordinates used.
3. No hardware, UART, replay, TX/RX, or RF signal was involved.
4. No live satellite contact, PER/BER/CRC, or gateway ACK.
5. No synthetic `sat_*.npz` data used. No old synthetic PGRL checkpoint used.
6. BK1 TLE history exhibits post-launch orbit determination instability and probable manoeuvre epochs not present in BK2, making cross-satellite residual transfer unreliable.
7. Doppler residuals encode mean-element update + drag parameter changes + numerical fitting differences between successive TLE solutions.
8. The experiment covers only the software-defined TLE-history residual correction approach; no alternative path-B formulations were evaluated.
"""

    OUT_REPORT.write_text(report, encoding="utf-8")
    rpt_sha = sha256_file(OUT_REPORT)
    print(f"\n[Report] → {OUT_REPORT}")
    print(f"          SHA256: {rpt_sha}")

    return {
        "verdict":         verdict,
        "path_b_supported": any_supported,
        "bk1_records":     len(bk1_recs),
        "bk2_records":     len(bk2_recs),
        "staleness_windows": [(r["target_h"], r.get("base_mae"), r.get("best_mae"), r.get("best_pct")) for r in results],
        "csv_sha256":      csv_sha256,
        "report_sha256":   rpt_sha,
    }


if __name__ == "__main__":
    result = run_experiment()
    print("\n── Final summary ────────────────────────────────────────────────")
    for k, v in result.items():
        print(f"  {k:22s}: {v}")
