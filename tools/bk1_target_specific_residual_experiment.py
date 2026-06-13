#!/usr/bin/env python3
"""
BLACK KITE-1 TARGET-SPECIFIC TLE-History Residual Doppler Correction — Path B

Target  : BLACK KITE-1  NORAD 66741  (single-satellite, target-specific)
Control : BLACK KITE-2  NORAD 68474  (negative-transfer / evidence-gate case,
          evaluated by the separate cross-satellite blocker experiment)

Formulation (target-specific):
  reference_doppler_hz    = later/newer-TLE SGP4-derived Doppler (NOT measured truth)
  sgp4_model_doppler_hz   = stale/older-TLE SGP4-derived open-loop Doppler at same UTC
  pgrl_model_doppler_hz   = stale_doppler + residual predicted by a model trained
                            ONLY on BK1 historical TLE samples available before the
                            test period.
  reference_is_measured_truth = False

Split (chronological by REFERENCE epoch — no future-TLE leakage):
  early  60%  -> train
  middle 20%  -> validation
  late   20%  -> held-out test
Split boundaries are two fixed UTC datetimes derived from the full BK1 record
epoch span (60th / 80th time-quantile). The SAME two dates gate every staleness
window. A TLE pair is assigned by its reference epoch; its stale TLE is always
older, so training never sees a future TLE.

Staleness windows: 8 / 24 / 48 / 72 / 96 / 168 h (each with a gap band).

Models:
  - zero-residual baseline   (predict 0 correction = use stale Doppler as-is)
  - median-residual baseline (constant = median train residual)
  - ridge regression         (alpha tuned on validation)
  - random forest            (config tuned on validation, sklearn)
  - gradient boosting        (HistGradientBoosting, sklearn)
  - MLP                      (torch, CUDA GPU, best-validation checkpoint)

Model selection: the model with the best VALIDATION MAE is chosen; its
held-out TEST MAE/RMSE is the decision metric (no test peeking for selection).

Decision rule:
  Path B for BK1 is SUPPORTED for a window iff the selected model's held-out
  TEST MAE < zero-residual stale-TLE baseline TEST MAE.
  CSV is exported iff at least one window is supported (best improving window).

No hardware, no RF, no UART, no replay, no measured Doppler truth.

Run:
  uv run python tools/bk1_target_specific_residual_experiment.py
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

try:
    from sklearn.ensemble import (
        RandomForestRegressor,
        HistGradientBoostingRegressor,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 0
np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)

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
MANEUVER_CAP_HZ     = 1500.0    # per-pair reject if max|residual| exceeds this
                                # (removes manoeuvre / bad-OD epochs, keeps drift)

RIDGE_ALPHAS = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]

MLP_EPOCHS   = 300
MLP_HIDDEN_1 = 256
MLP_HIDDEN_2 = 128
MLP_HIDDEN_3 = 64
MLP_LR       = 1e-3
MLP_BATCH    = 512

# Chronological time-quantiles for the global split boundaries
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20   # -> test = remaining 0.20

# Staleness windows to probe: (target_h, min_gap_h, max_gap_h)
STALENESS_WINDOWS = [
    (8,   4,   14),
    (24,  16,  36),
    (48,  36,  60),
    (72,  60,  84),
    (96,  84,  120),
    (168, 144, 192),
]

# Minimum samples required in each segment to evaluate a window
MIN_TRAIN = 60
MIN_VAL   = 20
MIN_TEST  = 20

# A window is "supported" only if a LEARNED residual model (ridge / RF / GBR /
# MLP) beats the zero baseline by at least this relative MAE margin AND also
# improves RMSE. This guards against declaring success on a noise-level
# difference (e.g. a sub-0.1% constant-offset shift, which is NOT a learned
# residual correction). Baselines (zero, median) can never trigger "supported".
MIN_SUPPORT_FRAC = 0.01     # 1% relative MAE improvement required
LEARNED_MODELS = ("ridge", "random_forest", "grad_boost", "mlp_gpu")

# ─── File paths ───────────────────────────────────────────────────────────────
PROJ_ROOT  = Path(__file__).resolve().parent.parent
BK1_JSON   = PROJ_ROOT / "dataraw/spacetrack/black_kite_1_66741/gp_history_66741.json"
OUT_REPORT = PROJ_ROOT / "docs/review/black_kite_1_target_specific_residual_experiment.md"
OUT_GATE   = PROJ_ROOT / "docs/review/black_kite_residual_evidence_gate.md"
OUT_CSV    = PROJ_ROOT / "dataraw/pgrl/black_kite_1_target_residual_replay_predictions.csv"


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


def make_satrec(record: dict):
    return Satrec.twoline2rv(record["TLE_LINE1"], record["TLE_LINE2"])


# ─── Dataset construction (target-specific, single satellite) ────────────────

N_FEAT = 10
FEATURE_NAMES = [
    "t_age_s", "t_gap_s", "stale_doppler_hz", "sin_phase", "cos_phase",
    "elevation_deg", "range_km", "stale_mean_motion", "stale_bstar", "stale_ecc",
]


def build_pairs(
    records: list[dict],
    target_h: float,
    min_gap_h: float,
    max_gap_h: float,
    cap_hz: float = MANEUVER_CAP_HZ,
):
    """
    Build per-pair sample blocks for one staleness window.

    Staleness is the OPERATIONAL scenario: a receiver holds a TLE that is ~N h
    old and propagates it open-loop; the reference is a later (newer) TLE. So
    for each reference (newer) record j we select the OLDER record i whose epoch
    gap to j is inside [min_gap_h, max_gap_h] and closest to target_h. This is
    NOT restricted to consecutive TLEs — it is the genuine stale->reference
    pairing and is what fills the long-staleness windows (BK1 refreshes ~6 h, so
    a 48 h-stale pair is an older TLE 48 h back, not a 48 h consecutive gap).

    Returns a list of dicts, one per accepted TLE pair:
      { "ref_epoch": datetime, "X": (K,N_FEAT), "y": (K,), "meta": [..] }
    plus stats including outlier (manoeuvre) pair count and their magnitudes.
    """
    epochs = [parse_epoch_utc(r["EPOCH"]) for r in records]
    pairs = []
    n_no_partner = n_skip_sgp4 = 0
    outlier_mags: list[float] = []

    for j in range(len(records)):
        ep_new = epochs[j]
        # Find the older record closest to target staleness within the gap band.
        # Gaps increase monotonically as i decreases (epochs sorted ascending).
        best_i, best_d = None, None
        for i in range(j - 1, -1, -1):
            gap_h = (ep_new - epochs[i]).total_seconds() / 3600.0
            if gap_h < min_gap_h:
                continue
            if gap_h > max_gap_h:
                break
            d = abs(gap_h - target_h)
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_i is None:
            n_no_partner += 1
            continue

        rec_old = records[best_i]
        rec_new = records[j]
        ep_old  = epochs[best_i]
        t_gap_s = (ep_new - ep_old).total_seconds()

        try:
            sat_old = make_satrec(rec_old)
            sat_new = make_satrec(rec_new)
        except Exception:
            n_skip_sgp4 += 1
            continue

        M0      = math.radians(float(rec_old["MEAN_ANOMALY"]))
        n_rad_s = float(rec_old["MEAN_MOTION"]) * 2.0 * math.pi / 86400.0
        stale_mm   = float(rec_old["MEAN_MOTION"])
        stale_bstar = float(rec_old["BSTAR"])
        stale_ecc  = float(rec_old["ECCENTRICITY"])

        pair_X, pair_y, pair_meta = [], [], []
        fail_sgp4 = False
        max_abs_res = 0.0
        dt_step = PERIOD_SAMPLE_S / K_SAMPLES_PER_PAIR

        for k in range(K_SAMPLES_PER_PAIR):
            t_abs   = ep_new + datetime.timedelta(seconds=k * dt_step)
            t_age_s = (t_abs - ep_old).total_seconds()
            jd, fr  = dt_to_jd(t_abs)
            gs_r, gs_v = gs_teme_km(jd, fr)

            pos_old, vel_old = propagate_sgp4(sat_old, t_abs)
            pos_new, vel_new = propagate_sgp4(sat_new, t_abs)
            if pos_old is None or pos_new is None:
                fail_sgp4 = True
                break

            d_stale, elev, rng = doppler_and_geom(pos_old, vel_old, gs_r, gs_v)
            d_ref,   _,    _   = doppler_and_geom(pos_new, vel_new, gs_r, gs_v)
            res = d_ref - d_stale
            max_abs_res = max(max_abs_res, abs(res))

            phase = (M0 + n_rad_s * t_age_s) % (2.0 * math.pi)
            pair_X.append([
                t_age_s, t_gap_s, d_stale,
                math.sin(phase), math.cos(phase), elev, rng,
                stale_mm, stale_bstar, stale_ecc,
            ])
            pair_y.append(res)
            pair_meta.append({
                "t_unix_s":              t_abs.timestamp(),
                "reference_doppler_hz":  d_ref,
                "sgp4_model_doppler_hz": d_stale,
                "residual_hz":           res,
            })

        if fail_sgp4:
            n_skip_sgp4 += 1
            continue
        if max_abs_res > cap_hz:
            outlier_mags.append(max_abs_res)
            continue

        pairs.append({
            "ref_epoch": ep_new,
            "X":  np.asarray(pair_X, dtype=np.float64),
            "y":  np.asarray(pair_y, dtype=np.float64),
            "meta": pair_meta,
        })

    stats = {
        "n_pairs_kept":  len(pairs),
        "n_no_partner":  n_no_partner,
        "n_skip_sgp4":   n_skip_sgp4,
        "n_outlier":     len(outlier_mags),
        "outlier_mags":  outlier_mags,
    }
    return pairs, stats


def split_pairs(pairs, t_train_end, t_val_end):
    """Chronological split by reference epoch into (train, val, test) blocks."""
    tr, va, te = [], [], []
    for p in pairs:
        if p["ref_epoch"] <= t_train_end:
            tr.append(p)
        elif p["ref_epoch"] <= t_val_end:
            va.append(p)
        else:
            te.append(p)
    return tr, va, te


def stack(pairs):
    if not pairs:
        return (np.empty((0, N_FEAT)), np.empty((0,)), [])
    X = np.vstack([p["X"] for p in pairs])
    y = np.concatenate([p["y"] for p in pairs])
    meta = [m for p in pairs for m in p["meta"]]
    return X, y, meta


# ─── Standardize ─────────────────────────────────────────────────────────────

def fit_scaler(X: np.ndarray):
    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-8
    return mu, sig


def apply_scaler(X, mu, sig):
    return (X - mu) / sig


# ─── Ridge (numpy normal equation) ───────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n = X.shape[1]
    A = X.T @ X + alpha * np.eye(n)
    return np.linalg.solve(A, X.T @ y)


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))])


# ─── MLP (GPU) ────────────────────────────────────────────────────────────────

class ResidualMLP(nn.Module):
    def __init__(self, n_feat: int = N_FEAT):
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


def train_mlp(X_tr, y_tr, X_va, y_va, tag: str = ""):
    """Train MLP on GPU with standardized target; return (model, y_mu, y_sig)."""
    y_mu  = float(y_tr.mean())
    y_sig = float(y_tr.std() + 1e-8)

    model = ResidualMLP(n_feat=X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor((y_tr - y_mu) / y_sig, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=MLP_BATCH, shuffle=True,
    )
    loss_fn = nn.HuberLoss(delta=5.0)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if epoch % 25 == 0 or epoch == MLP_EPOCHS:
            model.eval()
            with torch.no_grad():
                pv = model(Xv) * y_sig + y_mu
                vl_mae = float(torch.mean(torch.abs(pv - yv)).cpu())
            if vl_mae < best_val:
                best_val = vl_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"    [MLP {tag}] best val_MAE={best_val:.4f} Hz")
    return model, y_mu, y_sig


def mlp_predict(model, X, y_mu, y_sig):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        return (out * y_sig + y_mu).cpu().numpy()


# ─── Metrics ─────────────────────────────────────────────────────────────────

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def residual_dist(y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"mean": 0, "std": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    a = np.abs(y)
    return {
        "mean": float(y.mean()),
        "std":  float(y.std()),
        "p50":  float(np.percentile(a, 50)),
        "p90":  float(np.percentile(a, 90)),
        "p99":  float(np.percentile(a, 99)),
        "max":  float(a.max()),
    }


# ─── Single staleness-window evaluation ──────────────────────────────────────

def run_window(records, target_h, min_h, max_h, t_train_end, t_val_end):
    print(f"\n{'─'*64}")
    print(f"  Staleness window: target {target_h}h  (gap band {min_h}-{max_h}h)")

    pairs, st = build_pairs(records, target_h, min_h, max_h)
    tr_p, va_p, te_p = split_pairs(pairs, t_train_end, t_val_end)
    X_tr, y_tr, _       = stack(tr_p)
    X_va, y_va, _       = stack(va_p)
    X_te, y_te, meta_te = stack(te_p)

    print(f"    pairs kept={st['n_pairs_kept']}  "
          f"(no_partner={st['n_no_partner']} sgp4={st['n_skip_sgp4']} "
          f"outlier={st['n_outlier']})")
    print(f"    samples  train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    base = {
        "target_h": target_h, "min_h": min_h, "max_h": max_h,
        "n_pairs": st["n_pairs_kept"], "n_outlier": st["n_outlier"],
        "outlier_mags": st["outlier_mags"],
        "n_tr": len(X_tr), "n_va": len(X_va), "n_te": len(X_te),
        "dist_all": residual_dist(np.concatenate([y_tr, y_va, y_te])
                                  if len(X_tr)+len(X_va)+len(X_te) else np.empty(0)),
        "dist_te": residual_dist(y_te),
    }

    if len(X_tr) < MIN_TRAIN or len(X_va) < MIN_VAL or len(X_te) < MIN_TEST:
        print("    [SKIP] insufficient samples in one or more segments")
        base["skip"] = True
        return base
    base["skip"] = False

    # ── Baselines ────────────────────────────────────────────────────────────
    zero_mae_te,  zero_rmse_te  = mae(y_te, 0.0), rmse(y_te, 0.0)
    zero_mae_va,  zero_rmse_va  = mae(y_va, 0.0), rmse(y_va, 0.0)
    med = float(np.median(y_tr))
    med_mae_va, med_mae_te = mae(y_va, med), mae(y_te, med)
    med_rmse_te = rmse(y_te, med)

    # ── Standardize (for ridge + MLP) ────────────────────────────────────────
    mu, sig = fit_scaler(X_tr)
    Xtr_n, Xva_n, Xte_n = (apply_scaler(X, mu, sig) for X in (X_tr, X_va, X_te))
    Xtr_b, Xva_b, Xte_b = add_bias(Xtr_n), add_bias(Xva_n), add_bias(Xte_n)

    models: dict[str, dict] = {}
    models["zero_baseline"]   = {"va_mae": zero_mae_va, "te_mae": zero_mae_te,
                                 "te_rmse": zero_rmse_te,
                                 "pred_te": np.zeros_like(y_te), "selectable": False}
    models["median_baseline"] = {"va_mae": med_mae_va, "te_mae": med_mae_te,
                                 "te_rmse": med_rmse_te,
                                 "pred_te": np.full_like(y_te, med), "selectable": True}

    # ── Ridge (alpha tuned on val) ───────────────────────────────────────────
    best_alpha, best_va = None, float("inf")
    for a in RIDGE_ALPHAS:
        coef = ridge_fit(Xtr_b, y_tr, a)
        v = mae(y_va, Xva_b @ coef)
        if v < best_va:
            best_va, best_alpha = v, a
    coef = ridge_fit(Xtr_b, y_tr, best_alpha)
    ridge_pred_te = Xte_b @ coef
    models["ridge"] = {"va_mae": best_va, "te_mae": mae(y_te, ridge_pred_te),
                       "te_rmse": rmse(y_te, ridge_pred_te),
                       "pred_te": ridge_pred_te, "selectable": True,
                       "note": f"alpha={best_alpha:g}"}
    print(f"    ridge   best alpha={best_alpha:g}  val_MAE={best_va:.4f}")

    # ── Random forest (config tuned on val) ──────────────────────────────────
    if HAS_SKLEARN:
        rf_best, rf_va, rf_cfg = None, float("inf"), None
        for leaf in (1, 5, 20):
            rf = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=leaf,
                n_jobs=-1, random_state=SEED,
            ).fit(X_tr, y_tr)
            v = mae(y_va, rf.predict(X_va))
            if v < rf_va:
                rf_va, rf_best, rf_cfg = v, rf, leaf
        rf_pred_te = rf_best.predict(X_te)
        models["random_forest"] = {"va_mae": rf_va, "te_mae": mae(y_te, rf_pred_te),
                                   "te_rmse": rmse(y_te, rf_pred_te),
                                   "pred_te": rf_pred_te, "selectable": True,
                                   "note": f"min_leaf={rf_cfg}"}
        print(f"    rf      best min_leaf={rf_cfg}  val_MAE={rf_va:.4f}")

        # ── Gradient boosting ────────────────────────────────────────────────
        gb_best, gb_va, gb_cfg = None, float("inf"), None
        for lr in (0.03, 0.1):
            gb = HistGradientBoostingRegressor(
                learning_rate=lr, max_iter=400, l2_regularization=1.0,
                random_state=SEED,
            ).fit(X_tr, y_tr)
            v = mae(y_va, gb.predict(X_va))
            if v < gb_va:
                gb_va, gb_best, gb_cfg = v, gb, lr
        gb_pred_te = gb_best.predict(X_te)
        models["grad_boost"] = {"va_mae": gb_va, "te_mae": mae(y_te, gb_pred_te),
                                "te_rmse": rmse(y_te, gb_pred_te),
                                "pred_te": gb_pred_te, "selectable": True,
                                "note": f"lr={gb_cfg}"}
        print(f"    gbr     best lr={gb_cfg}  val_MAE={gb_va:.4f}")

    # ── MLP (GPU) ────────────────────────────────────────────────────────────
    if HAS_TORCH:
        mlp, ymu, ysig = train_mlp(Xtr_n, y_tr, Xva_n, y_va, tag=f"{target_h}h")
        mlp_pred_va = mlp_predict(mlp, Xva_n, ymu, ysig)
        mlp_pred_te = mlp_predict(mlp, Xte_n, ymu, ysig)
        models["mlp_gpu"] = {"va_mae": mae(y_va, mlp_pred_va),
                             "te_mae": mae(y_te, mlp_pred_te),
                             "te_rmse": rmse(y_te, mlp_pred_te),
                             "pred_te": mlp_pred_te, "selectable": True,
                             "note": str(DEVICE)}

    # ── Model selection by VALIDATION MAE among LEARNED models only ───────────
    # (no test peeking; baselines zero/median cannot win "supported")
    learned = {k: v for k, v in models.items() if k in LEARNED_MODELS}
    sel_name = min(learned, key=lambda k: learned[k]["va_mae"])
    sel = models[sel_name]
    pct = (zero_mae_te - sel["te_mae"]) / (zero_mae_te + 1e-12) * 100.0

    # Supported iff learned model beats zero baseline by >= margin on MAE AND
    # also improves RMSE — noise-level wins do not count.
    supported = (
        sel["te_mae"] < zero_mae_te * (1.0 - MIN_SUPPORT_FRAC)
        and sel["te_rmse"] < zero_rmse_te
    )

    print(f"    ─ baseline(zero)   test MAE={zero_mae_te:.4f}  RMSE={zero_rmse_te:.4f}")
    print(f"    ─ baseline(median) test MAE={med_mae_te:.4f}")
    print(f"    ─ selected(learned)='{sel_name}' ({sel.get('note','')})  "
          f"test MAE={sel['te_mae']:.4f}  RMSE={sel['te_rmse']:.4f}  Δ={pct:+.1f}%")
    print(f"    ➤ {'SUPPORTED' if supported else 'BLOCKED'}"
          f"  (margin>={MIN_SUPPORT_FRAC:.0%} & RMSE-improve required)")

    base.update({
        "zero_mae_te": zero_mae_te, "zero_rmse_te": zero_rmse_te,
        "zero_mae_va": zero_mae_va,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "pred_te"}
                   for k, v in models.items()},
        "sel_name": sel_name,
        "sel_te_mae": sel["te_mae"], "sel_te_rmse": sel["te_rmse"],
        "sel_va_mae": sel["va_mae"], "sel_pct": pct,
        "sel_note": sel.get("note", ""),
        "supported": supported,
        "sel_pred_te": sel["pred_te"],
        "meta_te": meta_te,
    })
    return base


# ─── Report generation ───────────────────────────────────────────────────────

def fmt_models_row(m: dict) -> str:
    order = ["zero_baseline", "median_baseline", "ridge",
             "random_forest", "grad_boost", "mlp_gpu"]
    cells = []
    for name in order:
        if name in m["models"]:
            cells.append(f"{m['models'][name]['te_mae']:.4f}")
        else:
            cells.append("—")
    return " | ".join(cells)


def main():
    if not HAS_SGP4:
        raise RuntimeError("sgp4 package required: uv add sgp4")

    print("=" * 68)
    print("BLACK KITE-1 TARGET-SPECIFIC TLE-History Residual Doppler — Path B")
    print("=" * 68)
    print(f"SGP4   : python-sgp4 (Brandon Rhodes)")
    print(f"Device : {DEVICE if HAS_TORCH else 'N/A'}   sklearn={HAS_SKLEARN}")
    print(f"Carrier: {F_CARRIER_HZ/1e6:.1f} MHz  GS {GS_LAT_DEG}N {GS_LON_DEG}E")
    print(f"reference_is_measured_truth = False")

    recs = load_records(BK1_JSON)
    epochs = [parse_epoch_utc(r["EPOCH"]) for r in recs]
    gaps_h = [(epochs[i+1]-epochs[i]).total_seconds()/3600
              for i in range(len(epochs)-1)]

    # ── Global chronological split boundaries (time-quantile of records) ──────
    t0, tN = epochs[0], epochs[-1]
    span_s = (tN - t0).total_seconds()
    t_train_end = t0 + datetime.timedelta(seconds=TRAIN_FRAC * span_s)
    t_val_end   = t0 + datetime.timedelta(seconds=(TRAIN_FRAC + VAL_FRAC) * span_s)
    n_tr = sum(1 for e in epochs if e <= t_train_end)
    n_va = sum(1 for e in epochs if t_train_end < e <= t_val_end)
    n_te = sum(1 for e in epochs if e > t_val_end)

    print(f"\nBK1 (NORAD 66741): {len(recs)} records  "
          f"| {t0.date()} -> {tN.date()}")
    print(f"  inter-TLE gap (h): median={np.median(gaps_h):.1f} "
          f"p10={np.percentile(gaps_h,10):.1f} p90={np.percentile(gaps_h,90):.1f} "
          f"max={max(gaps_h):.1f}")
    print(f"  split boundaries (by reference epoch):")
    print(f"    train : {t0.isoformat()}  ..  {t_train_end.isoformat()}  ({n_tr} recs)")
    print(f"    val   : {t_train_end.isoformat()}  ..  {t_val_end.isoformat()}  ({n_va} recs)")
    print(f"    test  : {t_val_end.isoformat()}  ..  {tN.isoformat()}  ({n_te} recs)")

    results = []
    for target_h, min_h, max_h in STALENESS_WINDOWS:
        results.append(run_window(recs, target_h, min_h, max_h,
                                  t_train_end, t_val_end))

    # ── Overall verdict + best improving window ──────────────────────────────
    supported_windows = [r for r in results
                         if not r.get("skip") and r.get("supported")]
    any_supported = len(supported_windows) > 0
    best_r = (max(supported_windows, key=lambda r: r["sel_pct"])
              if any_supported else None)

    print(f"\n{'='*68}")
    if any_supported:
        verdict = "Path B target-specific TLE-history residual correction is supported for BLACK KITE-1."
        print(f"VERDICT: {verdict}")
        print(f"  Best window: {best_r['target_h']}h-stale  "
              f"model={best_r['sel_name']}  Δ={best_r['sel_pct']:+.1f}%")
    else:
        verdict = "Path B remains blocked even for BLACK KITE-1."
        print(f"VERDICT: {verdict}")
        print("  No staleness window improved held-out test MAE over zero baseline.")
    print("=" * 68)

    # ── CSV export iff supported ─────────────────────────────────────────────
    csv_sha, csv_path_str = "N/A", "N/A (no improvement — CSV not exported)"
    if any_supported and best_r:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w", newline="") as f:
            f.write("# reference_is_measured_truth=false\n")
            f.write(f"# target=BLACK_KITE_1 norad=66741 carrier_hz={F_CARRIER_HZ:.0f} "
                    f"staleness_h={best_r['target_h']} model={best_r['sel_name']} "
                    f"gs={GS_LAT_DEG}N_{GS_LON_DEG}E_{GS_ALT_M}m\n")
            f.write("t_s,reference_doppler_hz,sgp4_model_doppler_hz,pgrl_model_doppler_hz\n")
            for m, pr in zip(best_r["meta_te"], best_r["sel_pred_te"]):
                f.write(f"{m['t_unix_s']:.3f},"
                        f"{m['reference_doppler_hz']:.6f},"
                        f"{m['sgp4_model_doppler_hz']:.6f},"
                        f"{m['sgp4_model_doppler_hz'] + pr:.6f}\n")
        csv_sha = sha256_file(OUT_CSV)
        csv_path_str = str(OUT_CSV.relative_to(PROJ_ROOT))
        print(f"\n[CSV] -> {OUT_CSV}\n      SHA256: {csv_sha}")

    write_reports(recs, epochs, gaps_h, t0, tN,
                  t_train_end, t_val_end, n_tr, n_va, n_te,
                  results, verdict, any_supported, best_r,
                  csv_sha, csv_path_str)

    return {"verdict": verdict, "supported": any_supported,
            "csv_sha256": csv_sha}


def write_reports(recs, epochs, gaps_h, t0, tN,
                  t_train_end, t_val_end, n_tr, n_va, n_te,
                  results, verdict, any_supported, best_r,
                  csv_sha, csv_path_str):
    ts_now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    mdl_header = ("| Window | n(tr/va/te) | outliers | zero | median | ridge | "
                  "RF | GBR | MLP | selected | test MAE | Δ% | verdict |")
    mdl_sep    = "|" + "---|" * 13
    rows = []
    for r in results:
        if r.get("skip"):
            rows.append(f"| {r['target_h']}h | "
                        f"{r['n_tr']}/{r['n_va']}/{r['n_te']} | {r['n_outlier']} | "
                        f"— | — | — | — | — | — | — | — | — | *insufficient* |")
            continue
        rows.append(
            f"| {r['target_h']}h "
            f"| {r['n_tr']}/{r['n_va']}/{r['n_te']} "
            f"| {r['n_outlier']} "
            f"| {fmt_models_row(r).replace(' | ', ' | ')} "
            f"| {r['sel_name']} "
            f"| {r['sel_te_mae']:.4f} "
            f"| {r['sel_pct']:+.1f} "
            f"| {'✓ SUPPORTED' if r['supported'] else '✗ blocked'} |"
        )

    # residual-distribution table
    dist_rows = []
    for r in results:
        d = r["dist_te"]
        dist_rows.append(
            f"| {r['target_h']}h | {r['n_te']} | {d['mean']:+.3f} | {d['std']:.3f} "
            f"| {d['p50']:.3f} | {d['p90']:.3f} | {d['p99']:.3f} | {d['max']:.3f} |"
        )

    outlier_note_rows = []
    for r in results:
        mags = r.get("outlier_mags", [])
        if mags:
            top = ", ".join(f"{m:.0f}" for m in sorted(mags, reverse=True)[:5])
            outlier_note_rows.append(
                f"| {r['target_h']}h | {len(mags)} | {top} |")
        else:
            outlier_note_rows.append(f"| {r['target_h']}h | 0 | — |")

    report = f"""# BLACK KITE-1 Target-Specific TLE-History Residual Doppler Experiment — Path B
*Generated: {ts_now} UTC  |  `tools/bk1_target_specific_residual_experiment.py`*

## Formulation (target-specific)

| Parameter | Value |
|---|---|
| Target | BLACK KITE-1  NORAD 66741 (single satellite) |
| `reference_doppler_hz` | Later/newer-TLE SGP4-derived Doppler |
| `sgp4_model_doppler_hz` | Stale/older-TLE SGP4 open-loop Doppler at same UTC |
| `pgrl_model_doppler_hz` | `sgp4_model_doppler_hz + predicted_residual` (trained only on BK1 pre-test history) |
| `reference_is_measured_truth` | **False** |
| Carrier | {F_CARRIER_HZ/1e6:.1f} MHz (LR-FHSS band) |
| Ground station | {GS_LAT_DEG}°N {GS_LON_DEG}°E {GS_ALT_M} m (Taiwan representative) |
| SGP4 propagator | python-sgp4 (Brandon Rhodes) |
| Compute | {DEVICE if HAS_TORCH else 'CPU'} (torch GPU) + scikit-learn |
| Hardware / RF | **None — software-only** |

## Data summary

| Satellite | NORAD | Records | Epoch start | Epoch end |
|---|---|---|---|---|
| BLACK KITE-1 | 66741 | {len(recs)} | {t0.date()} | {tN.date()} |

**Inter-TLE gap (h):** median {np.median(gaps_h):.1f}  p10 {np.percentile(gaps_h,10):.1f}  p90 {np.percentile(gaps_h,90):.1f}  max {max(gaps_h):.1f}

## Chronological split (by reference epoch — zero future-TLE leakage)

Two global UTC boundaries from the {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%} time-quantile of the BK1 record-epoch span gate **every** staleness window. A TLE pair is assigned by its reference (newer) epoch; its stale (older) TLE is always earlier, so training never observes a future TLE.

| Segment | From (UTC) | To (UTC) | Records |
|---|---|---|---|
| Train (early {TRAIN_FRAC:.0%}) | {t0.isoformat()} | {t_train_end.isoformat()} | {n_tr} |
| Validation (mid {VAL_FRAC:.0%}) | {t_train_end.isoformat()} | {t_val_end.isoformat()} | {n_va} |
| Held-out test (late {1-TRAIN_FRAC-VAL_FRAC:.0%}) | {t_val_end.isoformat()} | {tN.isoformat()} | {n_te} |

## Features (10)

`{', '.join(FEATURE_NAMES)}`

Target: `residual_hz = reference_doppler_hz - sgp4_model_doppler_hz`.

## Dataset construction

- **Staleness pairing (operational):** for each reference (newer) TLE, the *older* TLE whose epoch gap is inside the window band and closest to the target staleness is selected — **not** restricted to consecutive TLEs. This is the genuine "hold a TLE that is N h old, propagate open-loop, compare to a later TLE" scenario, and is what populates the long-staleness windows (BK1 refreshes ~6 h, so a 48 h-stale pair is an older TLE 48 h back, not a 48 h consecutive gap).
- {K_SAMPLES_PER_PAIR} UTC samples per pair over one orbital period (~{PERIOD_SAMPLE_S//60} min).
- **Outlier rejection:** a pair is discarded if any sample's |residual| > {MANEUVER_CAP_HZ:.0f} Hz (manoeuvre / bad-OD epoch). Counts reported below.

## Models

zero-residual baseline · median (constant-bias) baseline · ridge (alpha tuned on val) · random forest (sklearn) · gradient boosting (sklearn HistGBR) · MLP (torch, CUDA GPU).
The two baselines are references, **not** residual correctors. Selection is among the **learned** models (ridge/RF/GBR/MLP) by **validation** MAE; the selected learned model's **held-out test** MAE/RMSE is the decision metric. A window counts as supported only if that learned model beats the zero baseline by ≥{MIN_SUPPORT_FRAC:.0%} relative MAE **and** improves RMSE — sub-noise constant-offset wins do not qualify.

## Per-window held-out TEST MAE [Hz @ {F_CARRIER_HZ/1e6:.0f} MHz]

{mdl_header}
{mdl_sep}
{chr(10).join(rows)}

*Columns zero…MLP are each model's held-out test MAE. "selected" = best **learned** model by validation MAE; Δ% = its improvement over the zero baseline.*

## Held-out test residual distribution [Hz]

| Window | n | mean | std | p50(|r|) | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
{chr(10).join(dist_rows)}

## Outlier (manoeuvre / bad-OD) pairs removed

| Window | removed pairs | top |max residual| [Hz] |
|---|---|---|
{chr(10).join(outlier_note_rows)}

## Verdict

> **{verdict}**

Decision rule: Path B is supported for a window iff the validation-selected **learned** model's held-out test MAE is below the zero-residual stale-TLE baseline test MAE by ≥{MIN_SUPPORT_FRAC:.0%} **and** its test RMSE also improves. The median (constant-bias) baseline is reported but never triggers support.

## Output files

| File | Path | SHA256 |
|---|---|---|
| Replay CSV | `{csv_path_str}` | {csv_sha} |
| This report | `{OUT_REPORT.relative_to(PROJ_ROOT)}` | (self) |
| Evidence gate | `{OUT_GATE.relative_to(PROJ_ROOT)}` | (see file) |

Replay schema: `t_s, reference_doppler_hz, sgp4_model_doppler_hz, pgrl_model_doppler_hz` (+ `# reference_is_measured_truth=false` metadata header).

## Exact limitations

1. `reference_doppler_hz` is a later TLE's SGP4 propagation — **not measured Doppler truth**.
2. Ground station is representative (24°N 121°E 100 m); no real GS coordinates used.
3. No hardware, UART, replay, TX/RX, or RF signal was involved.
4. No live satellite contact, PER/BER/CRC, or gateway ACK.
5. No synthetic `sat_*.npz` data; no old synthetic PGRL checkpoint used.
6. Doppler residuals encode mean-element + drag-term + numerical-fit differences between successive TLE solutions; they are a proxy for stale-TLE open-loop error, not an absolute frequency error.
7. Split is chronological by reference epoch; the stale TLE in any test pair is always available before the test reference epoch.
8. Only the software-defined TLE-history residual correction approach is evaluated.
"""
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(report, encoding="utf-8")
    rpt_sha = sha256_file(OUT_REPORT)
    print(f"\n[Report] -> {OUT_REPORT}\n         SHA256: {rpt_sha}")

    # ── Evidence-gate note ───────────────────────────────────────────────────
    bk1_line = (
        f"BK1 (NORAD 66741) target-specific: **{'SUPPORTED' if any_supported else 'BLOCKED'}**"
    )
    if any_supported and best_r:
        bk1_detail = (
            f"Best window {best_r['target_h']}h-stale, model `{best_r['sel_name']}`, "
            f"held-out test MAE {best_r['sel_te_mae']:.4f} Hz vs zero baseline "
            f"{best_r['zero_mae_te']:.4f} Hz (Δ {best_r['sel_pct']:+.1f}%). "
            f"Residual correction ENABLED for this target/staleness regime."
        )
    else:
        bk1_detail = (
            "No staleness window improved held-out late-BK1 test MAE over the "
            "zero-residual stale-TLE baseline. Residual correction DISABLED; "
            "fall back to open-loop SGP4 / stale-TLE compensation."
        )

    gate = f"""# BLACK KITE Residual-Correction Evidence Gate
*Generated: {ts_now} UTC*

This note records the data-driven gate that decides, **per target satellite and
staleness regime**, whether a learned TLE-history Doppler residual correction is
enabled or whether the system falls back to open-loop SGP4 / stale-TLE
compensation. `reference_is_measured_truth = false` in all cases (the reference
Doppler is a later-TLE SGP4 propagation, not a measured signal).

## Gate logic

```text
if heldout_residual_model_MAE < stale_baseline_MAE:
    enable residual correction          # learned model beats open-loop
else:
    disable residual correction         # open-loop SGP4 / stale-TLE is better/equal
```

The model is selected on a chronological **validation** segment and the gate is
evaluated **once** on a chronological **held-out test** segment. Splits are by
reference epoch with no future-TLE leakage.

## Case 1 — BLACK KITE-1 (NORAD 66741): high-residual target

{bk1_line}

{bk1_detail}

- Experiment: `tools/bk1_target_specific_residual_experiment.py`
- Report: `{OUT_REPORT.relative_to(PROJ_ROOT)}` (SHA256 `{rpt_sha}`)
{f"- Replay CSV: `{csv_path_str}` (SHA256 `{csv_sha}`)" if any_supported else "- Replay CSV: not exported (correction not enabled)."}

## Case 2 — BLACK KITE-2 (NORAD 68474): negative control

BK2 is a **negative-control / negative-transfer** case, not the main target.
Its Space-Track TLE refresh cadence (~6 h median) makes consecutive-TLE Doppler
residuals negligible (held-out MAE < 0.25 Hz at 868 MHz in the short-staleness
regime). The zero-residual baseline is already excellent, so the gate
**disables** residual correction for BK2. A cross-satellite BK1→BK2 model
*increases* error (distribution mismatch + BK1 manoeuvre outliers), confirming
that residual correction must be evidence-gated rather than always-on.

- Experiment: `tools/bk_tle_residual_experiment.py`
- Report: `docs/review/black_kite_tle_history_residual_experiment.md`
  (SHA256 `fdd8b19582d1623541ee5cb6baeb73824dbeb545da94ab1d70bb4f399d2d2a93`)

## System rule

The deployed compensator MUST consult this gate:

1. If a target/staleness regime has a validated residual model with
   `heldout_MAE < baseline_MAE`, apply `pgrl_model_doppler_hz` (stale + residual).
2. Otherwise, apply open-loop `sgp4_model_doppler_hz` (stale-TLE Doppler) and do
   **not** claim a residual-correction benefit.
3. Never treat `reference_doppler_hz` as measured truth; it is model-derived.

## Limitations

No hardware, RF, UART, replay, TX/RX, PER/BER/CRC, or gateway ACK was involved.
No synthetic `sat_*.npz` or old PGRL checkpoint was used as BLACK KITE evidence.
Raw `dataraw/` Space-Track files are inputs only and are not committed.
"""
    OUT_GATE.parent.mkdir(parents=True, exist_ok=True)
    OUT_GATE.write_text(gate, encoding="utf-8")
    gate_sha = sha256_file(OUT_GATE)
    print(f"[Gate]   -> {OUT_GATE}\n         SHA256: {gate_sha}")


if __name__ == "__main__":
    out = main()
    print("\n── Final summary ─────────────────────────────────────────────")
    for k, v in out.items():
        print(f"  {k:16s}: {v}")
