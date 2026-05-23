"""
train_final_s3.py — Final production training for the S3 ensemble
==================================================================
Trains two component models on ALL available historical data:
  S1 (final_s1_intl): intl-only, DC + CatBoost, no CB decay
  P2 (final_p2_sw):   intl + Soccerway (xG), DC + CatBoost, no CB decay

S3 ensemble is formed at inference by averaging their lambda predictions.
Do NOT combine the training sets — ensemble benefit comes from
independently-trained models with different feature sets (P2 has xG).
"""

import os, pickle, time
import pandas as pd
from catboost import CatBoostRegressor
from dc_cat_v3 import add_form_features, run_pipeline, save_pipeline, SAVE_DIR
from preprocessor_v2 import dedup_datasets

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data ...")
train_intl = pd.read_csv("train_intl_v2.csv")
test_intl  = pd.read_csv("test_intl_v2.csv")
sw_intl    = pd.read_csv("sw_intl.csv")

# S1: ALL intl data (train + test splits combined)
full_intl = (
    pd.concat([train_intl, test_intl])
    .drop_duplicates(subset=["date", "home_team", "away_team"])
    .sort_values("date")
    .reset_index(drop=True)
)
print(f"  train_intl : {len(train_intl):,} rows")
print(f"  test_intl  : {len(test_intl):,} rows")
print(f"  full_intl  : {len(full_intl):,} rows  (combined, deduplicated)")

# P2: sw_intl (priority — has xG) + full_intl (filler), deduplicated
full_p2_raw = (
    dedup_datasets(sw_intl, full_intl)
    .sort_values("date")
    .reset_index(drop=True)
)
print(f"  sw_intl    : {len(sw_intl):,} rows")
print(f"  full_p2    : {len(full_p2_raw):,} rows  (SW priority + intl filler)")

# ─────────────────────────────────────────────────────────────────────────────
# FORM FEATURES
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing form features ...")
train_full_s1, _ = add_form_features(full_intl)
train_full_p2, _ = add_form_features(full_p2_raw)
print(f"  S1 train rows : {len(train_full_s1):,}")
print(f"  P2 train rows : {len(train_full_p2):,}")

# Last 200 rows of each set as a live sanity-check test (not held-out)
test_s1 = train_full_s1.tail(200).reset_index(drop=True)
test_p2 = train_full_p2.tail(200).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN S1 FINAL  (intl-only, no CB decay)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  S1 Final — International only, no CB decay")
print(f"  Train: {len(train_full_s1):,}  |  Sanity test: {len(test_s1):,}")
print("=" * 70)
t0 = time.time()

metrics_s1, m_h, m_a, dc_params, t2i, _, _ = run_pipeline(
    train_full_s1, test_s1,
    dc_maxiter=5000, tune=True, cb_decay_lambda=0.0,
)
print(f"  Elapsed : {(time.time() - t0) / 60:.1f} min")
print(f"  Metrics : {metrics_s1}")
save_pipeline("final_s1_intl", m_h, m_a, dc_params, t2i)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN P2 FINAL  (intl + Soccerway xG, no CB decay)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  P2 Final — Intl + Soccerway (xG), no CB decay")
print(f"  Train: {len(train_full_p2):,}  |  Sanity test: {len(test_p2):,}")
print("=" * 70)
t0 = time.time()

metrics_p2, m_h, m_a, dc_params, t2i, _, _ = run_pipeline(
    train_full_p2, test_p2,
    dc_maxiter=5000, tune=True, cb_decay_lambda=0.0,
)
print(f"  Elapsed : {(time.time() - t0) / 60:.1f} min")
print(f"  Metrics : {metrics_p2}")
save_pipeline("final_p2_sw", m_h, m_a, dc_params, t2i)

# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK — reload and verify all 6 artifacts
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SANITY CHECK — reloading saved artifacts")
print("=" * 70)
all_ok = True
for tag in ["final_s1_intl", "final_p2_sw"]:
    try:
        m = CatBoostRegressor()
        m.load_model(f"{SAVE_DIR}/{tag}_home.cbm")
        print(f"  [OK] {SAVE_DIR}/{tag}_home.cbm  ({m.tree_count_} trees)")
        m.load_model(f"{SAVE_DIR}/{tag}_away.cbm")
        print(f"  [OK] {SAVE_DIR}/{tag}_away.cbm  ({m.tree_count_} trees)")
        with open(f"{SAVE_DIR}/{tag}_dc_params.pkl", "rb") as f:
            saved = pickle.load(f)
        print(f"  [OK] {SAVE_DIR}/{tag}_dc_params.pkl  ({len(saved['team_to_idx'])} teams)")
    except Exception as e:
        print(f"  [FAIL] {tag}: {e}")
        all_ok = False

print()
if all_ok:
    print("  All 6 model artifacts verified.")
else:
    print("  WARNING: some artifacts failed to load — check errors above.")

print()
print("  Bundle for external site:")
for tag in ["final_s1_intl", "final_p2_sw"]:
    for ext in ["_home.cbm", "_away.cbm", "_dc_params.pkl"]:
        print(f"    {SAVE_DIR}/{tag}{ext}")
print("    dc_cat_v3.py  preprocessor_v2.py  poisson_predict.py")
print("    group_fixtures.csv  knockout_slots.csv")
print("    train_intl_v2.csv  test_intl_v2.csv  sw_intl.csv")
print("    predict_wc2026.py")
