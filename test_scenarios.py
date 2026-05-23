"""
test_scenarios.py — Pre-production model evaluation
====================================================

Evaluates three training configurations on the international test set (2023-2024)
and compares them to decide on production readiness.

  Scenario 1 (= Phase 1 replicate)
      Train: train_intl_v2 (international only)
      Test:  test_intl_v2
      Purpose: establish the intl-only baseline; confirms Phase 1 results

  Scenario 2  (NEW)
      Train: train_intl_v2 + train_club_v2  (joint international + club)
      Test:  test_intl_v2
      Purpose: test whether club data transfers knowledge to international prediction

  Phase 2 replicate
      Train: sw_intl + train_intl_v2  (soccerway enrichment, deduped)
      Test:  test_intl_v2
      Purpose: confirms Phase 2 results; more recent intl data via soccerway

All scenarios are evaluated on the same base test set.
Results saved to test_scenario_results.csv.
"""

import os, time
import numpy as np
import pandas as pd

from dc_cat_v3 import add_form_features, run_pipeline, save_pipeline, SAVE_DIR
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

# ── Scenario 1 / Phase 1 (intl only) ────────────────────────────────────────
train_s1, hist_s1 = add_form_features(
    train_intl.sort_values("date").reset_index(drop=True)
)
test_s1, _ = add_form_features(
    base_test.sort_values("date").reset_index(drop=True), seed=hist_s1
)

# ── Scenario 2 (intl + club) ─────────────────────────────────────────────────
# Form computed separately per domain so intl history stays pure, then
# DataFrames are concatenated for joint DC + CatBoost training.
# Test is seeded from intl-only history (correct: test set is national teams).
train_intl_s2, intl_hist_s2 = add_form_features(
    train_intl.sort_values("date").reset_index(drop=True)
)
train_club_s2, _ = add_form_features(
    train_club.sort_values("date").reset_index(drop=True)
)
train_s2 = (
    pd.concat([train_intl_s2, train_club_s2], ignore_index=True)
    .sort_values("date")
    .reset_index(drop=True)
)
test_s2, _ = add_form_features(
    base_test.sort_values("date").reset_index(drop=True), seed=intl_hist_s2
)

# ── Phase 2 (intl + sw_intl, deduped) ───────────────────────────────────────
# sw_intl is priority source (has xG); train_intl fills historical gaps.
combined_p2 = dedup_datasets(sw_intl, train_intl)
combined_p2 = combined_p2.sort_values("date").reset_index(drop=True)
train_p2, hist_p2 = add_form_features(combined_p2)
test_p2, _ = add_form_features(
    base_test.sort_values("date").reset_index(drop=True), seed=hist_p2
)

print(f"  S1 / Phase 1 train rows : {len(train_s1):,}")
print(f"  S2 (intl+club) train    : {len(train_s2):,}")
print(f"  Phase 2 (intl+SW) train : {len(train_p2):,}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(name, train_df, test_df, dc_maxiter=5000, tune=True,
                   cb_depth=5, cb_lr=0.05, save_tag=None):
    print(f"\n{'='*68}")
    print(f"  {name}")
    print(f"  Train: {len(train_df):,}  |  Test (intl): {len(test_df):,}")
    print(f"{'='*68}")
    t0 = time.time()

    # Drop test rows whose teams are unseen in this training set
    known = set(train_df["home_team"]).union(set(train_df["away_team"]))
    test_filt = test_df[
        test_df["home_team"].isin(known) & test_df["away_team"].isin(known)
    ].copy()
    dropped = len(test_df) - len(test_filt)
    if dropped:
        print(f"  [info] {dropped} test rows dropped (teams unseen in this train)")

    metrics, m_home, m_away, dc_params, team_to_idx = run_pipeline(
        train_df, test_filt,
        dc_maxiter=dc_maxiter, tune=tune, cb_depth=cb_depth, cb_lr=cb_lr
    )
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed/60:.1f} min")

    if save_tag:
        save_pipeline(save_tag, m_home, m_away, dc_params, team_to_idx)

    return metrics, len(test_filt)


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

results    = {}
test_sizes = {}

# Scenario 1 — International only (replicates Phase 1)
metrics, n = run_experiment(
    "Scenario 1 — International Only  [Phase 1 replicate]",
    train_s1, test_s1,
    dc_maxiter=5000, tune=True,
    save_tag="scenario1_intl_only",
)
results["S1_intl_only"] = metrics
test_sizes["S1_intl_only"] = n

# Scenario 2 — International + Club  (NEW)
metrics, n = run_experiment(
    "Scenario 2 — International + Club  [joint training, NEW]",
    train_s2, test_s2,
    dc_maxiter=5000, tune=True,
    save_tag="scenario2_intl_club",
)
results["S2_intl_club"] = metrics
test_sizes["S2_intl_club"] = n

# Phase 2 — International + Soccerway Intl  (replicates Phase 2)
metrics, n = run_experiment(
    "Phase 2 — International + SW Intl  [Phase 2 replicate]",
    train_p2, test_p2,
    dc_maxiter=5000, tune=True,
    save_tag="phase2_intl_sw_replicate",
)
results["P2_intl_sw"] = metrics
test_sizes["P2_intl_sw"] = n


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

LABELS = {
    "S1_intl_only": "S1: Intl Only (P1)",
    "S2_intl_club": "S2: Intl+Club (NEW)",
    "P2_intl_sw":   "P2: Intl+SW",
}
TRAIN_SIZES = {
    "S1_intl_only": len(train_s1),
    "S2_intl_club": len(train_s2),
    "P2_intl_sw":   len(train_p2),
}

# Historical baselines from code comments (v2, intl, 2022 test window)
V2_BASELINES = {
    "DC Only (v2 ref)":  {"log_loss": 0.9096, "rps": 0.1812, "accuracy": 0.5996, "exact_score_pct": "—"},
    "DC+Elo (v2 ref)":   {"log_loss": 0.9166, "rps": 0.1820, "accuracy": 0.5975, "exact_score_pct": "—"},
}

COL_W = 22

print("\n\n" + "=" * 80)
print(f"{'PRE-PRODUCTION EVALUATION  —  test_intl 2023-2024':^80}")
print("=" * 80)

keys = list(results.keys())
print(f"{'Metric':<{COL_W}}", end="")
for k in keys:
    print(f"{LABELS[k]:>{COL_W}}", end="")
print()
print("-" * 80)

for metric in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
    vals = [results[k][metric] for k in keys]
    best = min(vals) if metric in ("log_loss", "rps") else max(vals)
    print(f"  {metric:<{COL_W - 2}}", end="")
    for v in vals:
        flag = " ◄" if v == best else "  "
        print(f"{v:>{COL_W - 2}.6f}{flag}", end="")
    print()

print("-" * 80)
print(f"  {'Train rows':<{COL_W - 2}}", end="")
for k in keys:
    print(f"{TRAIN_SIZES[k]:>{COL_W},}", end="")
print()
print(f"  {'Test rows':<{COL_W - 2}}", end="")
for k in keys:
    print(f"{test_sizes[k]:>{COL_W},}", end="")
print()
print("=" * 80)

print("\nv2 Baselines (intl, 2022 test window — earlier experiment):")
for name, b in V2_BASELINES.items():
    print(f"  {name:<22}  log_loss={b['log_loss']}  rps={b['rps']}  "
          f"accuracy={b['accuracy']}  exact_score={b['exact_score_pct']}")

print("\n  ◄ = best value in row")
print("  Interpretation: log_loss / rps ↓ better  |  accuracy / exact_score ↑ better")


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER LEARNING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

s1 = results["S1_intl_only"]
s2 = results["S2_intl_club"]
p2 = results["P2_intl_sw"]

print("\n" + "─" * 80)
print("TRANSFER LEARNING ANALYSIS")
print("─" * 80)

def delta(a, b, metric):
    """Signed improvement of b over a (positive = b is better)."""
    v_a, v_b = a[metric], b[metric]
    if metric in ("log_loss", "rps"):
        return v_a - v_b   # lower is better → positive delta means improvement
    else:
        return v_b - v_a   # higher is better

print(f"\n  S2 vs S1 — does club data help international prediction?")
for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
    d = delta(s1, s2, m)
    sign = "+" if d > 0 else ""
    tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
    print(f"    {m:<22}  S1={s1[m]:.6f}  S2={s2[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

print(f"\n  P2 vs S1 — does recent soccerway intl data help?")
for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
    d = delta(s1, p2, m)
    sign = "+" if d > 0 else ""
    tag  = "IMPROVES" if d > 0 else "HURTS   " if d < 0 else "NO CHANGE"
    print(f"    {m:<22}  S1={s1[m]:.6f}  P2={p2[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")

print(f"\n  P2 vs S2 — soccerway intl vs club data augmentation?")
for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
    d = delta(s2, p2, m)
    sign = "+" if d > 0 else ""
    tag  = "P2 better " if d > 0 else "S2 better " if d < 0 else "TIED     "
    print(f"    {m:<22}  S2={s2[m]:.6f}  P2={p2[m]:.6f}  Δ={sign}{d:.6f}  [{tag}]")


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION READINESS VERDICT
# ─────────────────────────────────────────────────────────────────────────────

# Pick the best model by RPS (primary metric — measures calibration)
best_key    = min(results, key=lambda k: results[k]["rps"])
best_label  = LABELS[best_key]
best        = results[best_key]

# Thresholds based on v2 DC-only baseline and football prediction literature
THRESH = {
    "rps":             0.195,   # v2 DC-only = 0.1812; v3 should be ≤ this
    "log_loss":        0.98,    # upper bound for calibrated intl football model
    "accuracy":        0.48,    # above coin-flip (~33% uniform random)
}

checks = {
    "rps":      (best["rps"]      <= THRESH["rps"],      best["rps"],      THRESH["rps"],      "≤"),
    "log_loss": (best["log_loss"] <= THRESH["log_loss"],  best["log_loss"], THRESH["log_loss"], "≤"),
    "accuracy": (best["accuracy"] >= THRESH["accuracy"],  best["accuracy"], THRESH["accuracy"], "≥"),
}

all_pass = all(v[0] for v in checks.values())

print("\n" + "=" * 80)
print("PRODUCTION READINESS VERDICT")
print("=" * 80)
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
    print(f"     Recommended model: {best_label}")
    print(f"     Model artifacts  : {SAVE_DIR}/{best_key.lower()}_{{home,away}}.cbm")
else:
    print("  ✗  NOT READY — one or more threshold checks failed")
    print("     Review the FAIL rows above before promoting to production.")

print()
print("  Guidance:")
print("   • RPS is the primary calibration metric (lower = better probability estimates)")
print("   • If S2 RPS < S1 RPS → club data improves intl prediction → use Scenario 2")
print("   • If P2 RPS < S1 RPS → soccerway enrichment helps → use Phase 2 pipeline")
print("   • Run model_comparison.py for a fuller 4-way experiment (Exp A/B/C/D)")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

rows = []
for k, m in results.items():
    row = {
        "experiment":  LABELS[k],
        "train_rows":  TRAIN_SIZES[k],
        "test_rows":   test_sizes[k],
        "is_best":     (k == best_key),
    }
    row.update(m)
    rows.append(row)

summary_df = pd.DataFrame(rows)
out_path   = "test_scenario_results.csv"
summary_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
