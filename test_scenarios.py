"""
test_scenarios.py — Pre-production model evaluation
====================================================

Evaluates four configurations on the international test set (2023-2024)
and compares them to decide on production readiness.

  Phase 1  (loaded from saved models — no retrain)
      Train: train_intl_v2 (international only)
      Test:  test_intl_v2
      Purpose: recover prior Phase 1 results from saved models

  Scenario 1  (fresh retrain — confirms Phase 1 replication)
      Train: train_intl_v2 (international only)
      Test:  test_intl_v2
      Purpose: fresh train on intl data; should match Phase 1 closely

  Scenario 2  (fresh retrain — NEW)
      Train: train_intl_v2 + train_club_v2  (joint international + club)
      Test:  test_intl_v2
      Purpose: test whether club data transfers to international prediction

  Phase 2  (loaded from saved models — no retrain)
      Train: sw_intl + train_intl_v2  (soccerway enrichment, deduped)
      Test:  test_intl_v2
      Purpose: recover prior Phase 2 soccerway-enriched results

All four are evaluated on the same consistent base_test set.
Results saved to test_scenario_results.csv.
"""

import os, time, pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.special import gammaln
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from scipy.stats import poisson

from dc_cat_v3 import (
    add_form_features, run_pipeline, save_pipeline, SAVE_DIR,
    get_dc_log_lambdas, build_features, evaluate,
)
from preprocessor_v2 import dedup_datasets


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data …")
train_intl = pd.read_csv("train_intl_v2.csv")
test_intl  = pd.read_csv("test_intl_v2.csv")
train_club = pd.read_csv("train_club_v2.csv")
sw_intl    = pd.read_csv("sw_intl.csv")

# Consistent base test: only rows where both teams appear in intl training set
train_teams_i = set(train_intl["home_team"]).union(set(train_intl["away_team"]))
base_test = test_intl[
    test_intl["home_team"].isin(train_teams_i) &
    test_intl["away_team"].isin(train_teams_i)
].copy()

print(f"  train_intl : {len(train_intl):,} rows")
print(f"  train_club : {len(train_club):,} rows")
print(f"  sw_intl    : {len(sw_intl):,} rows")
print(f"  base_test  : {len(base_test):,} rows  (both teams seen in train_intl)")


# ─────────────────────────────────────────────────────────────────────────────
# FORM FEATURE BUILDERS
# Form is computed strictly on past data (no leakage).
# Test form is seeded from the relevant training history.
# ─────────────────────────────────────────────────────────────────────────────

print("\nPre-computing form features …")

# ── Intl-only form  (used by P1 loaded, S1 fresh) ────────────────────────────
train_s1, hist_s1 = add_form_features(
    train_intl.sort_values("date").reset_index(drop=True)
)
# test_s1 is seeded from intl-only history — reused for both P1 and S1
test_s1, _ = add_form_features(
    base_test.sort_values("date").reset_index(drop=True), seed=hist_s1
)

# ── Intl+club form  (used by S2 fresh) ───────────────────────────────────────
# Form computed separately per domain so intl history stays pure for seeding
# test; DataFrames are concatenated for joint DC + CatBoost training.
train_club_s2, _ = add_form_features(
    train_club.sort_values("date").reset_index(drop=True)
)
train_s2 = (
    pd.concat([train_s1, train_club_s2], ignore_index=True)
    .sort_values("date")
    .reset_index(drop=True)
)
# Test seeded from intl-only history (correct: test set is national teams)
test_s2 = test_s1.copy()

# ── Phase 2 form  (intl + sw_intl, deduped) ──────────────────────────────────
# sw_intl is priority source (has xG); train_intl fills historical gaps.
combined_p2 = dedup_datasets(sw_intl, train_intl)
combined_p2 = combined_p2.sort_values("date").reset_index(drop=True)
train_p2, hist_p2 = add_form_features(combined_p2)
test_p2, _ = add_form_features(
    base_test.sort_values("date").reset_index(drop=True), seed=hist_p2
)

print(f"  Intl-only  train rows  : {len(train_s1):,}")
print(f"  Intl+club  train rows  : {len(train_s2):,}")
print(f"  Phase 2 SW train rows  : {len(train_p2):,}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & EVALUATE SAVED MODELS  (no retrain)
# ─────────────────────────────────────────────────────────────────────────────

def _match_key(df):
    """Stable match identifier for cross-model alignment."""
    return (df['date'].astype(str) + '|' + df['home_team'] + '|' + df['away_team']).values

def align_lambdas(df1, lh1, la1, df2, lh2, la2):
    """Average lambda predictions from two models aligned by match key.
    Returns (common_df, ens_lh, ens_la) where common_df rows come from df1."""
    k1 = _match_key(df1)
    k2 = _match_key(df2)
    map1 = {k: i for i, k in enumerate(k1)}
    map2 = {k: i for i, k in enumerate(k2)}
    common = [k for k in map1 if k in map2]
    idx1 = np.array([map1[k] for k in common])
    idx2 = np.array([map2[k] for k in common])
    ens_lh = (lh1[idx1] + lh2[idx2]) / 2
    ens_la = (la1[idx1] + la2[idx2]) / 2
    ens_df = df1.iloc[idx1].reset_index(drop=True)
    return ens_df, ens_lh, ens_la


def load_and_evaluate(tag, test_df, save_dir=SAVE_DIR, return_lambdas=False):
    """Load saved DC params + CatBoost models; evaluate on test_df."""
    pkl_path  = f"{save_dir}/{tag}_dc_params.pkl"
    home_path = f"{save_dir}/{tag}_home.cbm"
    away_path = f"{save_dir}/{tag}_away.cbm"

    missing = [p for p in [pkl_path, home_path, away_path] if not os.path.exists(p)]
    if missing:
        print(f"  [SKIP] {tag}: file(s) not found — {missing}")
        return None

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    dc_params   = saved["dc_params"]
    team_to_idx = saved["team_to_idx"]

    m_home = CatBoostRegressor()
    m_away = CatBoostRegressor()
    m_home.load_model(home_path)
    m_away.load_model(away_path)

    # Filter test to teams known at training time
    known     = set(team_to_idx.keys())
    test_filt = test_df[
        test_df["home_team"].isin(known) & test_df["away_team"].isin(known)
    ].reset_index(drop=True)
    dropped = len(test_df) - len(test_filt)
    if dropped:
        print(f"  [info] {dropped} test rows dropped — teams unseen in saved '{tag}'")

    log_lh, log_la = get_dc_log_lambdas(test_filt, dc_params, team_to_idx)
    X_test = build_features(test_filt, log_lh, log_la)

    lh = np.clip(m_home.predict(X_test), 0.05, 10)
    la = np.clip(m_away.predict(X_test), 0.05, 10)

    metrics = evaluate(test_filt, lh, la)
    if return_lambdas:
        return metrics, len(test_filt), lh, la, test_filt
    return metrics, len(test_filt)


# ─────────────────────────────────────────────────────────────────────────────
# FRESH-RETRAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(name, train_df, test_df, dc_maxiter=5000, tune=True,
                   cb_depth=5, cb_lr=0.05, save_tag=None, cb_decay_lambda=0.0):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Train: {len(train_df):,}  |  Test (intl): {len(test_df):,}")
    print(f"{'='*70}")
    t0 = time.time()

    known     = set(train_df["home_team"]).union(set(train_df["away_team"]))
    test_filt = test_df[
        test_df["home_team"].isin(known) & test_df["away_team"].isin(known)
    ].copy()
    dropped = len(test_df) - len(test_filt)
    if dropped:
        print(f"  [info] {dropped} test rows dropped — teams unseen in this train")

    metrics, m_home, m_away, dc_params, team_to_idx, lh_test, la_test = run_pipeline(
        train_df, test_filt,
        dc_maxiter=dc_maxiter, tune=tune, cb_depth=cb_depth, cb_lr=cb_lr,
        cb_decay_lambda=cb_decay_lambda,
    )
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed/60:.1f} min")

    if save_tag:
        save_pipeline(save_tag, m_home, m_away, dc_params, team_to_idx)

    return metrics, len(test_filt), lh_test, la_test, test_filt


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL EVALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

results    = {}
test_sizes = {}

# ── Phase 1 & Phase 2 — load saved models (fast) ─────────────────────────────
print(f"\n{'='*70}")
print("  Evaluating Phase 1 & Phase 2 from saved models (no retrain)")
print(f"{'='*70}")

p1_lh = p1_la = p1_test_filt = None
result = load_and_evaluate("phase1_intl", test_s1, return_lambdas=True)
if result:
    metrics, n, p1_lh, p1_la, p1_test_filt = result
    results["P1_intl_loaded"]    = metrics
    test_sizes["P1_intl_loaded"] = n
    print(f"  Phase 1 (loaded) : {metrics}")

p2_lh = p2_la = p2_test_filt = None
result = load_and_evaluate("phase2_intl", test_p2, return_lambdas=True)
if result:
    metrics, n, p2_lh, p2_la, p2_test_filt = result
    results["P2_intl_loaded"]    = metrics
    test_sizes["P2_intl_loaded"] = n
    print(f"  Phase 2 (loaded) : {metrics}")

# ── Scenario 1 — International only  (fresh retrain, no CB decay — replication check) ─
metrics, n, s1_lh, s1_la, s1_test_filt = run_experiment(
    "Scenario 1 — International Only  [fresh retrain, no CB decay]",
    train_s1, test_s1,
    dc_maxiter=5000, tune=True,
    save_tag="scenario1_intl_only",
)
results["S1_intl_only"]    = metrics
test_sizes["S1_intl_only"] = n

# ── Scenario 5 — International only, mild CB decay (λ=0.00005) ───────────────
metrics, n, s5_lh, s5_la, s5_test_filt = run_experiment(
    "Scenario 5 — International Only  [mild CB decay λ=0.00005]",
    train_s1, test_s1,
    dc_maxiter=5000, tune=True,
    save_tag="scenario5_mild_decay",
    cb_decay_lambda=0.00005,
)
results["S5_mild_decay"]    = metrics
test_sizes["S5_mild_decay"] = n

# ── Scenario 2 — SKIPPED (club data consistently hurts; saves ~7 min) ────────
# metrics, n, _, _, _ = run_experiment(
#     "Scenario 2 — International + Club  [fresh retrain — NEW]",
#     train_s2, test_s2,
#     dc_maxiter=5000, tune=True,
#     save_tag="scenario2_intl_club",
# )
# results["S2_intl_club"]    = metrics
# test_sizes["S2_intl_club"] = n

# ── Scenario 4 — Ensemble: P2 (loaded) + P1 (loaded) — zero retrain cost ─────
if p2_lh is not None and p1_lh is not None:
    print(f"\n{'='*70}")
    print("  Scenario 4 — Ensemble: P2 (Intl+SW) ⊕ P1 (Intl) — both loaded")
    print(f"{'='*70}")
    ens_df, ens_lh, ens_la = align_lambdas(
        p1_test_filt, p1_lh, p1_la,
        p2_test_filt, p2_lh, p2_la,
    )
    ens_metrics = evaluate(ens_df, ens_lh, ens_la)
    results["S4_ensemble_p2p1"]    = ens_metrics
    test_sizes["S4_ensemble_p2p1"] = len(ens_df)
    print(f"  Aligned rows : {len(ens_df):,}  (P2 ∩ P1 by match key)")
    print(f"  Metrics      : {ens_metrics}")

# ── Scenario 6 — Ensemble: P2 (loaded) + S5 (mild decay retrain) ────────────
if p2_lh is not None and s5_lh is not None:
    print(f"\n{'='*70}")
    print("  Scenario 6 — Ensemble: P2 (Intl+SW loaded) ⊕ S5 (mild CB decay)")
    print(f"{'='*70}")
    ens_df, ens_lh, ens_la = align_lambdas(
        s5_test_filt, s5_lh, s5_la,
        p2_test_filt, p2_lh, p2_la,
    )
    ens_metrics = evaluate(ens_df, ens_lh, ens_la)
    results["S6_ens_p2_mild"]    = ens_metrics
    test_sizes["S6_ens_p2_mild"] = len(ens_df)
    print(f"  Aligned rows : {len(ens_df):,}  (P2 ∩ S5 by match key)")
    print(f"  Metrics      : {ens_metrics}")

# ── Scenario 3 — Ensemble: P2 (loaded) + S1 (time-decay retrain) ─────────────
if p2_lh is not None and s1_lh is not None:
    print(f"\n{'='*70}")
    print("  Scenario 3 — Ensemble: P2 (Intl+SW loaded) ⊕ S1 (Intl time-decay)")
    print(f"{'='*70}")
    ens_df, ens_lh, ens_la = align_lambdas(
        s1_test_filt, s1_lh, s1_la,
        p2_test_filt, p2_lh, p2_la,
    )
    ens_metrics = evaluate(ens_df, ens_lh, ens_la)
    results["S3_ensemble"]    = ens_metrics
    test_sizes["S3_ensemble"] = len(ens_df)
    print(f"  Aligned rows : {len(ens_df):,}  (P2 ∩ S1 by match key)")
    print(f"  Metrics      : {ens_metrics}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

# Ordered columns: Phase1 → S1 → S2 → Phase2
COL_ORDER = [
    "P1_intl_loaded", "S1_intl_only", "S2_intl_club",
    "P2_intl_loaded", "S4_ensemble_p2p1",
    "S5_mild_decay", "S6_ens_p2_mild",
    "S3_ensemble",
]
LABELS = {
    "P1_intl_loaded":   "P1: Intl (saved)",
    "S1_intl_only":     "S1: Intl (no decay)",
    "S2_intl_club":     "S2: Intl+Club",
    "P2_intl_loaded":   "P2: Intl+SW (saved)",
    "S4_ensemble_p2p1": "S4: Ens P2+P1",
    "S5_mild_decay":    "S5: Mild decay",
    "S6_ens_p2_mild":   "S6: Ens P2+S5",
    "S3_ensemble":      "S3: Ens P2+S1",
}
TRAIN_SIZES = {
    "P1_intl_loaded":   len(train_s1),
    "S1_intl_only":     len(train_s1),
    "S2_intl_club":     len(train_s2),
    "P2_intl_loaded":   len(train_p2),
    "S4_ensemble_p2p1": len(train_s1),
    "S5_mild_decay":    len(train_s1),
    "S6_ens_p2_mild":   len(train_s1),
    "S3_ensemble":      len(train_s1),
}

keys  = [k for k in COL_ORDER if k in results]
COL_W = 22

print("\n\n" + "=" * 90)
print(f"{'PRE-PRODUCTION EVALUATION  —  test_intl 2023-2024':^90}")
print("=" * 90)

print(f"{'Metric':<{COL_W}}", end="")
for k in keys:
    print(f"{LABELS[k]:>{COL_W}}", end="")
print()
print("-" * 90)

for metric in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
    vals = [results[k][metric] for k in keys]
    best = min(vals) if metric in ("log_loss", "rps") else max(vals)
    print(f"  {metric:<{COL_W - 2}}", end="")
    for v in vals:
        flag = " ◄" if v == best else "  "
        print(f"{v:>{COL_W - 2}.6f}{flag}", end="")
    print()

print("-" * 90)
print(f"  {'Train rows':<{COL_W - 2}}", end="")
for k in keys:
    print(f"{TRAIN_SIZES[k]:>{COL_W},}", end="")
print()
print(f"  {'Test rows':<{COL_W - 2}}", end="")
for k in keys:
    print(f"{test_sizes[k]:>{COL_W},}", end="")
print()
print("=" * 90)
print("\n  ◄ = best value in row")
print("  Interpretation: log_loss / rps ↓ better  |  accuracy / exact_score ↑ better")
print("  P1/P2 = loaded from saved models (no retrain)")
print("  S1/S2 = fresh retrain")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def delta(a, b, metric):
    """Signed improvement of b over a (positive = b is better)."""
    v_a, v_b = a[metric], b[metric]
    if metric in ("log_loss", "rps"):
        return v_a - v_b   # lower is better
    else:
        return v_b - v_a   # higher is better

p1 = results.get("P1_intl_loaded",   {})
s1 = results.get("S1_intl_only",    {})
s2 = results.get("S2_intl_club",    {})
p2 = results.get("P2_intl_loaded",  {})
s3 = results.get("S3_ensemble",     {})
s4 = results.get("S4_ensemble_p2p1",{})
s5 = results.get("S5_mild_decay",   {})
s6 = results.get("S6_ens_p2_mild",  {})

print("\n" + "─" * 90)
print("ANALYSIS")
print("─" * 90)

if p1 and s1:
    print("\n  S1 vs P1 — replication check (fresh retrain vs saved Phase 1):")
    print("  (close = pipeline is stable; large gap = randomness / data version drift)")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(p1, s1, m)
        sign = "+" if d > 0 else ""
        tag  = "S1 better" if d > 0 else "P1 better" if d < 0 else "IDENTICAL"
        print(f"    {m:<22}  P1={p1[m]:.6f}  S1={s1[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if s1 and s2:
    print("\n  S2 vs S1 — does club data transfer to international prediction?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(s1, s2, m)
        sign = "+" if d > 0 else ""
        tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
        print(f"    {m:<22}  S1={s1[m]:.6f}  S2={s2[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if s1 and p2:
    print("\n  P2 vs S1 — does soccerway enrichment beat intl-only?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(s1, p2, m)
        sign = "+" if d > 0 else ""
        tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
        print(f"    {m:<22}  S1={s1[m]:.6f}  P2={p2[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if s2 and p2:
    print("\n  S2 vs P2 — club augmentation vs soccerway enrichment?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(p2, s2, m)
        sign = "+" if d > 0 else ""
        tag  = "S2 better" if d > 0 else "P2 better" if d < 0 else "TIED     "
        print(f"    {m:<22}  P2={p2[m]:.6f}  S2={s2[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if p1 and s4:
    print("\n  S4 vs P1 — does P2+P1 ensemble beat P1 alone?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(p1, s4, m)
        sign = "+" if d > 0 else ""
        tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
        print(f"    {m:<22}  P1={p1[m]:.6f}  S4={s4[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if p2 and s4:
    print("\n  S4 vs P2 — does P2+P1 ensemble beat P2 alone?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(p2, s4, m)
        sign = "+" if d > 0 else ""
        tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
        print(f"    {m:<22}  P2={p2[m]:.6f}  S4={s4[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if s1 and s3:
    print("\n  S3 vs S1 — does ensembling P2 help over S1 (time-decay) alone?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(s1, s3, m)
        sign = "+" if d > 0 else ""
        tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
        print(f"    {m:<22}  S1={s1[m]:.6f}  S3={s3[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if p2 and s3:
    print("\n  S3 vs P2 — does ensembling beat P2 alone?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(p2, s3, m)
        sign = "+" if d > 0 else ""
        tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
        print(f"    {m:<22}  P2={p2[m]:.6f}  S3={s3[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if p1 and s5:
    print("\n  S5 vs P1 — does mild CB decay (λ=0.00005) beat no-decay baseline?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(p1, s5, m)
        sign = "+" if d > 0 else ""
        tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
        print(f"    {m:<22}  P1={p1[m]:.6f}  S5={s5[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

if p2 and s6:
    print("\n  S6 vs P2 — does P2+mild-decay ensemble beat P2 alone?")
    for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
        d = delta(p2, s6, m)
        sign = "+" if d > 0 else ""
        tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
        print(f"    {m:<22}  P2={p2[m]:.6f}  S6={s6[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION READINESS VERDICT
# ─────────────────────────────────────────────────────────────────────────────

# Primary metric: RPS (lower = better probability calibration)
# Thresholds derived from v2 DC-only baseline and football prediction literature
THRESH = {
    "rps":      0.195,   # v2 DC-only = 0.1812; set generous upper bound
    "log_loss": 0.98,    # upper bound for calibrated intl football model
    "accuracy": 0.48,    # well above coin-flip (~33% for 3-way outcome)
}

best_key   = min(results, key=lambda k: results[k]["rps"])
best_label = LABELS.get(best_key, best_key)
best       = results[best_key]

checks = {
    "rps":      (best["rps"]      <= THRESH["rps"],      best["rps"],      THRESH["rps"],      "≤"),
    "log_loss": (best["log_loss"] <= THRESH["log_loss"],  best["log_loss"], THRESH["log_loss"], "≤"),
    "accuracy": (best["accuracy"] >= THRESH["accuracy"],  best["accuracy"], THRESH["accuracy"], "≥"),
}
all_pass = all(v[0] for v in checks.values())

print("\n" + "=" * 90)
print("PRODUCTION READINESS VERDICT")
print("=" * 90)
print(f"\n  Best model : {best_label}")
print(f"  Metrics    : rps={best['rps']:.6f}  log_loss={best['log_loss']:.6f}  "
      f"accuracy={best['accuracy']:.6f}  exact_score={best['exact_score_pct']:.6f}")
print()
for metric, (passed, val, thresh, op) in checks.items():
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}]  {metric:<18}  {val:.6f} {op} {thresh}")

print()
if all_pass:
    print("  ✓  PRODUCTION READY")
    print(f"     Recommended model : {best_label}")
    print(f"     Model artifacts   : {SAVE_DIR}/{best_key.lower()}_{{home,away}}.cbm")
else:
    print("  ✗  NOT READY — one or more threshold checks failed")
    print("     Review the FAIL rows above before promoting to production.")

print()
print("  Decision guide:")
print("   • RPS is the primary metric (lower = better-calibrated probabilities)")
print("   • S1 ≈ P1       → pipeline is stable, replication confirmed")
print("   • S2 RPS < S1   → use Scenario 2 (intl+club) as production config")
print("   • P2 RPS < S1   → use Phase 2 soccerway pipeline for production")
print("   • All passing   → promote best model artifacts to production")
print("   • Any FAIL      → investigate before promoting")
print("=" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

rows = []
for k in keys:
    m   = results[k]
    row = {
        "experiment":  LABELS.get(k, k),
        "mode":        "loaded" if "loaded" in k else "fresh_retrain",
        "train_rows":  TRAIN_SIZES.get(k),
        "test_rows":   test_sizes[k],
        "is_best":     (k == best_key),
    }
    row.update(m)
    rows.append(row)

summary_df = pd.DataFrame(rows)
out_path   = "test_scenario_results.csv"
summary_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
