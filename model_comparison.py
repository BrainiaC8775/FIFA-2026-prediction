"""
model_comparison.py — Production model selection experiment

Four scenarios evaluated on the SAME test set (test_intl_v2, 2023-2024):

  Exp A  │ Intl only (no soccerway)    │ train_intl_v2
  Exp B  │ Intl + Club combined        │ train_intl_v2 + train_club_v2
  Exp C  │ Intl + SW (Phase 2)         │ train_intl_v2 + sw_intl
  Exp D  │ Intl + Club + SW (full)     │ train_intl_v2 + train_club_v2 + sw_intl + sw_club

For Exp B/D: form features computed SEPARATELY per domain (intl/club),
then DataFrames concatenated for joint DC + CatBoost training.
Keeps intl form = intl match history; club form = club match history.

All experiments evaluated on test_intl_v2 with form seeded from intl training history.
"""

import numpy as np
import pandas as pd
import os, time

from dc_cat_v3 import (
    add_form_features, run_pipeline, save_pipeline,
)
from preprocessor_v2 import dedup_datasets

SAVE_DIR = "models_dc_cat_v3"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
print("Loading data …")
train_intl = pd.read_csv("train_intl_v2.csv")
test_intl  = pd.read_csv("test_intl_v2.csv")
train_club = pd.read_csv("train_club_v2.csv")
sw_intl    = pd.read_csv("sw_intl.csv")
sw_club    = pd.read_csv("sw_club.csv")

# Base test set: filter to teams seen in train_intl (consistent across all exps)
train_teams_i = set(train_intl["home_team"]).union(set(train_intl["away_team"]))
base_test = test_intl[
    test_intl["home_team"].isin(train_teams_i) &
    test_intl["away_team"].isin(train_teams_i)
].copy()
print(f"  base_test: {len(base_test):,} rows")


# ─────────────────────────────────────────────────────────────
# FORM BUILDERS
# ─────────────────────────────────────────────────────────────

def form_intl_only(intl_train, intl_test):
    """Cold-start form on intl; seed test from intl history."""
    tr, hist = add_form_features(intl_train.sort_values("date").reset_index(drop=True))
    te, _    = add_form_features(intl_test.sort_values("date").reset_index(drop=True),
                                  seed=hist)
    return tr, te


def form_combined(intl_train, club_train, intl_test):
    """
    Separate cold-start per domain, then concatenate for joint training.
    Test seeded from intl history only (correct — test is national teams).
    """
    intl_tr, intl_hist = add_form_features(
        intl_train.sort_values("date").reset_index(drop=True)
    )
    club_tr, _ = add_form_features(
        club_train.sort_values("date").reset_index(drop=True)
    )
    combined = pd.concat([intl_tr, club_tr], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    te, _ = add_form_features(
        intl_test.sort_values("date").reset_index(drop=True),
        seed=intl_hist
    )
    return combined, te


# ─────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────

def run_experiment(name, train_df, test_df, dc_maxiter=5000, tune=True,
                   cb_depth=5, cb_lr=0.05, save_tag=None):
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  Train: {len(train_df):,}  |  Test (intl): {len(test_df):,}")
    print(f"{'='*65}")
    t0 = time.time()

    # Keep only test rows whose teams appear in this train set
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

    return metrics


# ─────────────────────────────────────────────────────────────
# BUILD FORM FEATURES
# ─────────────────────────────────────────────────────────────
print("\n--- Computing form features ---")

print("Exp A: intl only …")
train_a, test_a = form_intl_only(train_intl, base_test)

print("Exp B: intl + club …")
train_b, test_b = form_combined(train_intl, train_club, base_test)

print("Exp C: intl + sw_intl …")
combined_c = dedup_datasets(sw_intl, train_intl)
train_c, test_c = form_intl_only(
    combined_c.sort_values("date").reset_index(drop=True), base_test
)

print("Exp D: intl + club + sw_intl + sw_club …")
combined_d_intl = dedup_datasets(sw_intl, train_intl)
combined_d_club = dedup_datasets(sw_club, train_club)
train_d, test_d = form_combined(combined_d_intl, combined_d_club, base_test)

print(f"\n  Train rows — A:{len(train_a):,}  B:{len(train_b):,}  C:{len(train_c):,}  D:{len(train_d):,}")
print(f"  Test  rows — A:{len(test_a):,}  B:{len(test_b):,}  C:{len(test_c):,}  D:{len(test_d):,}")


# ─────────────────────────────────────────────────────────────
# RUN ALL EXPERIMENTS
# ─────────────────────────────────────────────────────────────
results = {}

results["A_intl_only"] = run_experiment(
    "Exp A — Intl Only", train_a, test_a,
    dc_maxiter=5000, tune=True, save_tag="exp_a_intl_only"
)

results["B_intl_club"] = run_experiment(
    "Exp B — Intl + Club", train_b, test_b,
    dc_maxiter=5000, tune=True, save_tag="exp_b_intl_club"
)

results["C_intl_sw"] = run_experiment(
    "Exp C — Intl + SW Intl", train_c, test_c,
    dc_maxiter=5000, tune=True, save_tag="exp_c_intl_sw"
)

results["D_full"] = run_experiment(
    "Exp D — Intl + Club + SW (Full)", train_d, test_d,
    dc_maxiter=5000, tune=True, save_tag="exp_d_full"
)


# ─────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
LABELS = {
    "A_intl_only": "A: Intl Only",
    "B_intl_club": "B: Intl+Club",
    "C_intl_sw":   "C: Intl+SW",
    "D_full":      "D: Full",
}
TRAIN_SIZES = {
    "A_intl_only": len(train_a),
    "B_intl_club": len(train_b),
    "C_intl_sw":   len(train_c),
    "D_full":      len(train_d),
}

print("\n" + "="*80)
print(f"{'PRODUCTION MODEL SELECTION — test on intl 2023-2024':^80}")
print("="*80)
keys = list(results.keys())
print(f"{'Metric':<22}", end="")
for k in keys:
    print(f"{LABELS[k]:>14}", end="")
print()
print("-"*80)

for m in ["log_loss", "rps", "accuracy", "exact_score_pct"]:
    vals = [results[k][m] for k in keys]
    best = min(vals) if m in ("log_loss", "rps") else max(vals)
    print(f"  {m:<20}", end="")
    for v in vals:
        flag = " ◄" if v == best else "  "
        print(f"{v:>12.6f}{flag}", end="")
    print()

print("-"*80)
print(f"  {'Train rows':<20}", end="")
for k in keys:
    print(f"{TRAIN_SIZES[k]:>14,}", end="")
print()
print("="*80)

print("\n  ◄ = best in row")
print("\nInterpretation:")
print("  log_loss / rps ↓ better  |  accuracy / exact_score ↑ better")
print("  If B > A: club data helps intl prediction (transfer learning works)")
print("  If C > A: recent intl data (soccerway) helps")
print("  If D > C: club xG data generalises to intl even via separate form features")

# Save
summary_df = pd.DataFrame(results).T
summary_df.index.name = "experiment"
summary_df.to_csv("comparison_results.csv")
print("\nSaved: comparison_results.csv")
