# WC 2026 Prediction — Deployment Guide

Everything needed to run `predict_wc2026.py` on an external machine and regenerate
all five output CSVs.

---

## 1. Python Scripts (copy all 5)

| File | Size | Purpose |
|---|---|---|
| `predict_wc2026.py` | 39 KB | **Main entry point.** Loads models, runs 10 000-sim Monte Carlo, writes all output files. |
| `dc_cat_v3.py` | 30 KB | Dixon-Coles + CatBoost model definition. Exposes `add_form_features`, `get_dc_log_lambdas`, `build_features`, `poisson_prob_matrix`, `outcome_probs`, `SAVE_DIR`. |
| `preprocessor_v2.py` | 17 KB | Data pipeline. Exposes `dedup_datasets` (merges intl + club datasets for form features). |
| `poisson_predict.py` | 8 KB | Exposes `qualified_teams` dict (48-team name map) and `normalize_team_name`. |
| `predict_cards_corners.py` | 5 KB | Recency-weighted cards/corners predictor. Reads `match_cards_corners.csv` and returns per-match predictions. |

---

## 2. Model Artifacts — `models_dc_cat_v3/` (copy entire folder)

Only **two** ensembles are loaded at runtime. The others are experimental/unused runs.

### Required (used in production)

| Files | Size | Description |
|---|---|---|
| `final_s1_intl_home.cbm` | 438 KB | CatBoost home goals — S1 international-only model |
| `final_s1_intl_away.cbm` | 271 KB | CatBoost away goals — S1 international-only model |
| `final_s1_intl_dc_params.pkl` | 5 KB | Dixon-Coles α/β/γ parameters + `team_to_idx` mapping for S1 |
| `final_p2_sw_home.cbm` | 1.8 MB | CatBoost home goals — P2 Soccerway-augmented model |
| `final_p2_sw_away.cbm` | 306 KB | CatBoost away goals — P2 Soccerway-augmented model |
| `final_p2_sw_dc_params.pkl` | 7 KB | Dixon-Coles parameters for P2 |

**Total required model size: ~2.9 MB**

### Optional (not loaded by `predict_wc2026.py`, safe to leave out)

`phase1_*`, `phase2_*`, `exp_a_*`, `scenario1_*`, `scenario2_*`, `scenario5_*` —
these are training experiments / ablations. Copy them only if you also want to run
`dc_cat_v3.py` standalone or `test_scenarios.py`.

---

## 3. Input Data Files (copy all 6)

| File | Size | Description |
|---|---|---|
| `group_fixtures.csv` | 5 KB | 72 group stage matches: match_id, group, home_team, away_team |
| `knockout_slots.csv` | 3 KB | 32 KO match slots with round labels and multipliers |
| `match_cards_corners.csv` | 388 KB | Per-match historical cards + corners (2006–2025 international matches). Used by `predict_cards_corners.py`. |
| `tm_national_games.csv` | 1 MB | TransferMarkt historical international results. Used by `poisson_predict.py` for recent form. |
| `International_match_stats.csv` | 13 MB | Full international match stats dataset. Used by `preprocessor_v2.dedup_datasets`. |
| `Soccerway.csv` | 8 MB | Soccerway results. Used to build form features for the P2 model. |

> **Note:** `Club_game_stats.csv` (3 MB) and `club_basic.csv` (74 KB) are only needed
> if you re-run `preprocessor_v2.py` standalone to regenerate training splits.
> They are **not** required at inference time by `predict_wc2026.py`.

---

## 4. Path Changes Required on the External Machine

All paths in the scripts are **relative** — the scripts assume they are run from the
same directory that contains the files above. No absolute paths are hardcoded, but
check these three points:

### 4a. `SAVE_DIR` in `dc_cat_v3.py` (line 509)

```python
SAVE_DIR = "models_dc_cat_v3"
```

This is relative to the **current working directory** when you run the script.
Run the script from the folder that contains `models_dc_cat_v3/`, or change the value:

```python
SAVE_DIR = "/path/to/your/models_dc_cat_v3"   # absolute path on the server
```

### 4b. CSV paths in `predict_wc2026.py` (lines 96–97)

```python
fixtures       = pd.read_csv("group_fixtures.csv")
knockout_slots = pd.read_csv("knockout_slots.csv")
```

And in `predict_cards_corners.py` (called with):

```python
build_predictions_for_fixtures(fixtures, cards_path="match_cards_corners.csv", ...)
```

And in `poisson_predict.py` (line 195):

```python
national_history = load_nat_history("tm_national_games.csv")
```

All are relative paths. **Run from the directory containing all files**, or wrap
the script with:

```bash
cd /path/to/deployment/folder && python predict_wc2026.py
```

### 4c. `preprocessor_v2.py` hard-coded paths (lines 364–367)

```python
INTL_PATH  = "International_match_stats.csv"
CGS_PATH   = "Club_game_stats.csv"
BASIC_PATH = "club_basic.csv"
SW_PATH    = "Soccerway.csv"
```

These only matter when `preprocessor_v2.py` is run **standalone** (not imported).
`predict_wc2026.py` imports only `dedup_datasets()` and passes paths as arguments,
so no change needed for inference.

---

## 5. Dependencies

```
python >= 3.10
catboost >= 1.2
numpy
pandas
scipy          # used internally by poisson_prob_matrix
```

Install on the server:

```bash
pip install catboost numpy pandas scipy
```

---

## 6. Minimal File Tree for Deployment

```
deploy/
├── predict_wc2026.py
├── dc_cat_v3.py
├── preprocessor_v2.py
├── poisson_predict.py
├── predict_cards_corners.py
├── group_fixtures.csv
├── knockout_slots.csv
├── match_cards_corners.csv
├── tm_national_games.csv
├── International_match_stats.csv
├── Soccerway.csv
└── models_dc_cat_v3/
    ├── final_s1_intl_home.cbm
    ├── final_s1_intl_away.cbm
    ├── final_s1_intl_dc_params.pkl
    ├── final_p2_sw_home.cbm
    ├── final_p2_sw_away.cbm
    └── final_p2_sw_dc_params.pkl
```

Run:

```bash
python predict_wc2026.py
```

Outputs written to the same directory:

```
wc2026_group_predictions.csv    — 72 group match predictions
wc2026_group_standings.csv      — expected final group standings
wc2026_knockout_bracket.csv     — most likely KO bracket
wc2026_win_probabilities.csv    — P(reach round) per team
wc2026_submission.csv           — 104-row competition submission file
```

---

## 7. Files NOT Needed for Inference

These exist in the repo but are not required to regenerate predictions:

| File / Folder | Reason |
|---|---|
| `tm_game_events.csv` (149 MB) | Raw TM event source — already processed into `tm_games_with_events.csv` |
| `tm_games.csv` (25 MB) | Raw TM games source — already processed |
| `tm_games_with_events.csv` | Intermediate — already merged into `match_cards_corners.csv` |
| `sb_international_corners.csv` | Intermediate — already merged into `match_cards_corners.csv` |
| `statsbomb/` folder | Raw StatsBomb JSON — corners already extracted |
| `train_intl_v2.csv`, `test_intl_v2.csv`, `train_club_v2.csv`, `test_club_v2.csv` | Training splits only |
| `train_xg_dataset.csv`, `test_xg_dataset.csv` | xG training data only |
| `Club_game_stats.csv` | Only needed for retraining |
| `club_basic.csv` | Only needed for retraining |
| `penalty_shootouts.csv` | Already integrated into `match_cards_corners.csv` |
| `dc_cat.py`, `dc_cat_v2.py`, `dc_elo_form.py` | Earlier model versions, superseded |
| `model_comparison.py`, `test_scenarios.py` | Evaluation scripts, not inference |
| `train_final_s3.py` | Training script only |
| `models_dc_cat_v2/` | Superseded model version |
| `*.ipynb` | Notebook versions of the scripts |
| `catboost_info/` | CatBoost training logs |
| `tm_domestic_games.*`, `tm_other_games.*` | Domestic / club data, not used at inference |
| `sw_club.csv`, `sw_intl.csv` | Preprocessed Soccerway — regenerated on the fly |
