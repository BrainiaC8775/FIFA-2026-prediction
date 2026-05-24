"""
predict_wc2026.py — 2026 FIFA World Cup Prediction (S3 Ensemble)
================================================================
Loads final_s1_intl + final_p2_sw model artifacts (S3 ensemble).
Predicts all 72 group stage matches and simulates the full tournament
via Monte Carlo (10,000 runs), sampling Poisson goal distributions.

Outputs (written to working directory):
  wc2026_group_predictions.csv   — all 72 match probs + predicted score
  wc2026_group_standings.csv     — expected group standings
  wc2026_knockout_bracket.csv    — most likely bracket (most-common MC path)
  wc2026_win_probabilities.csv   — P(reach each round) + P(Win) per team
"""

import os, pickle, time
import numpy as np
import pandas as pd
from collections import defaultdict
from catboost import CatBoostRegressor

from dc_cat_v3 import (
    add_form_features, get_dc_log_lambdas, build_features,
    poisson_prob_matrix, outcome_probs, SAVE_DIR,
)
from preprocessor_v2 import dedup_datasets
from poisson_predict import qualified_teams, normalize_team_name
from predict_cards_corners import build_predictions_for_fixtures, predict_match as predict_cc_match

N_SIMS   = 10_000
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_model_set(tag):
    pkl_path = f"{SAVE_DIR}/{tag}_dc_params.pkl"
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Model not found: {pkl_path}\n"
            f"Run train_final_s3.py first to generate model artifacts."
        )
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    m_home = CatBoostRegressor()
    m_away = CatBoostRegressor()
    m_home.load_model(f"{SAVE_DIR}/{tag}_home.cbm")
    m_away.load_model(f"{SAVE_DIR}/{tag}_away.cbm")
    return m_home, m_away, saved["dc_params"], saved["team_to_idx"]

print("Loading S3 ensemble models ...")
models = {
    "s1": load_model_set("final_s1_intl"),
    "p2": load_model_set("final_p2_sw"),
}
print(f"  S1: {models['s1'][0].tree_count_} home trees, {len(models['s1'][3])} training teams")
print(f"  P2: {models['p2'][0].tree_count_} home trees, {len(models['p2'][3])} training teams")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD HISTORICAL DATA & BUILD FORM SEEDS
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading historical data for form seeding ...")
train_intl = pd.read_csv("train_intl_v2.csv")
test_intl  = pd.read_csv("test_intl_v2.csv")
sw_intl    = pd.read_csv("sw_intl.csv")

full_intl = (
    pd.concat([train_intl, test_intl])
    .drop_duplicates(subset=["date", "home_team", "away_team"])
    .sort_values("date").reset_index(drop=True)
)
full_sw = (
    dedup_datasets(sw_intl, full_intl)
    .sort_values("date").reset_index(drop=True)
)

_, hist_s1 = add_form_features(full_intl)
_, hist_p2 = add_form_features(full_sw)

# Last known ELO per team from historical data
last_elo = {}
for _, row in full_intl.sort_values("date").iterrows():
    last_elo[str(row["home_team"])] = float(row["home_elo"])
    last_elo[str(row["away_team"])] = float(row["away_elo"])

print(f"  Historical intl rows : {len(full_intl):,}")
print(f"  Historical SW rows   : {len(full_sw):,}")
print(f"  Teams with ELO data  : {len(last_elo)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD & RESOLVE FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

fixtures      = pd.read_csv("group_fixtures.csv")
knockout_slots = pd.read_csv("knockout_slots.csv")

# Resolve playoff placeholders → actual qualified teams
fixtures["home_team"] = (
    fixtures["home_team"].replace(qualified_teams).apply(normalize_team_name)
)
fixtures["away_team"] = (
    fixtures["away_team"].replace(qualified_teams).apply(normalize_team_name)
)

# All 48 WC teams
all_wc_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
team_idx     = {t: i for i, t in enumerate(all_wc_teams)}
n_teams      = len(all_wc_teams)

# Group membership
team_group = {}
group_teams: dict[str, list] = defaultdict(list)
for _, row in fixtures.iterrows():
    for t in (row["home_team"], row["away_team"]):
        if t not in team_group:
            team_group[t] = row["group"]
            group_teams[row["group"]].append(t)

groups = sorted(group_teams.keys())
print(f"\n{n_teams} WC teams across {len(groups)} groups")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DC LAMBDA HELPER WITH FALLBACK FOR UNSEEN TEAMS
# ─────────────────────────────────────────────────────────────────────────────

def get_dc_lambdas_safe(home, away, dc_params, team_to_idx):
    """DC log-lambdas; unseen teams get mean strength (attack=defense=0)."""
    n         = len(team_to_idx)
    attack    = dc_params[:n]    - np.mean(dc_params[:n])
    defense   = dc_params[n:2*n] - np.mean(dc_params[n:2*n])
    home_adv  = float(dc_params[-2])

    hi = team_to_idx.get(home)
    ai = team_to_idx.get(away)
    atk_h = float(attack[hi])  if hi is not None else 0.0
    def_h = float(defense[hi]) if hi is not None else 0.0
    atk_a = float(attack[ai])  if ai is not None else 0.0
    def_a = float(defense[ai]) if ai is not None else 0.0

    log_lh = float(np.clip(atk_h - def_a + home_adv, -10, 10))
    log_la = float(np.clip(atk_a - def_h,             -10, 10))
    return log_lh, log_la

# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICT HELPER — single fixture row through S3 ensemble
# ─────────────────────────────────────────────────────────────────────────────

def predict_fixture_row(frow_s1, frow_p2, home, away):
    """
    Given pre-built form-featured rows for S1 and P2 seeds,
    return (lh_ens, la_ens, p_home, p_draw, p_away, predicted_score).
    """
    lh_list, la_list = [], []
    for tag, (m_home, m_away, dc_params, t2i) in models.items():
        frow = frow_s1 if tag == "s1" else frow_p2
        log_lh, log_la = get_dc_lambdas_safe(home, away, dc_params, t2i)
        X   = build_features(frow.reset_index(drop=True),
                             np.array([log_lh]), np.array([log_la]))
        lh  = float(np.clip(m_home.predict(X), 0.05, 10)[0])
        la  = float(np.clip(m_away.predict(X), 0.05, 10)[0])
        lh_list.append(lh)
        la_list.append(la)

    lh_ens = float(np.mean(lh_list))
    la_ens = float(np.mean(la_list))
    pm     = poisson_prob_matrix(lh_ens, la_ens)
    probs  = outcome_probs(pm)
    ph, pa = np.unravel_index(np.argmax(pm), pm.shape)
    return lh_ens, la_ens, float(probs[0]), float(probs[1]), float(probs[2]), f"{ph}-{pa}"

# ─────────────────────────────────────────────────────────────────────────────
# 6. PRECOMPUTE GROUP STAGE PREDICTIONS (frozen form at training cutoff)
# ─────────────────────────────────────────────────────────────────────────────

def build_fixture_frame(home, away, date, hist, is_neutral=0):
    """Build a 1-row DataFrame suitable for add_form_features."""
    row = pd.DataFrame({
        "date":       [date],
        "home_team":  [home],
        "away_team":  [away],
        "home_goals": [0.0],
        "away_goals": [0.0],
        "home_elo":   [last_elo.get(home, 1500.0)],
        "away_elo":   [last_elo.get(away, 1500.0)],
        "home_xg":    [0.0],
        "away_xg":    [0.0],
        "xg_available": [0],
        "is_intl":    [1],
        "is_neutral": [is_neutral],
    })
    result, _ = add_form_features(row, seed=hist)
    return result

print("\nPrecomputing group stage match predictions ...")
group_preds = {}  # match_id → dict with lh, la, probs, score

for _, fx in fixtures.iterrows():
    mid  = int(fx["match_id"])
    home = fx["home_team"]
    away = fx["away_team"]
    date = str(fx["date_utc"])[:10]

    frow_s1 = build_fixture_frame(home, away, date, hist_s1)
    frow_p2 = build_fixture_frame(home, away, date, hist_p2)
    lh, la, p_h, p_d, p_a, score = predict_fixture_row(frow_s1, frow_p2, home, away)

    group_preds[mid] = {
        "match_id": mid, "group": fx["group"],
        "home_team": home, "away_team": away,
        "lh": lh, "la": la,
        "p_home": p_h, "p_draw": p_d, "p_away": p_a,
        "predicted_score": score,
    }
    print(f"  [{fx['group']}] {home:25s} vs {away:25s}  {lh:.2f}-{la:.2f}  -> {score}")

# ─────────────────────────────────────────────────────────────────────────────
# 6b. CARDS & CORNERS — GROUP STAGE
# ─────────────────────────────────────────────────────────────────────────────

print("\nPrecomputing cards & corners for group stage ...")
_cc_preds, _cc_team_stats, _cc_globals = build_predictions_for_fixtures(
    fixtures, cards_path="match_cards_corners.csv", min_year=2014
)
for mid, cc in _cc_preds.items():
    group_preds[mid].update(cc)

_cc_df = pd.DataFrame(list(_cc_preds.values()))
print(f"  Avg total yellow : {_cc_df['total_yellow'].mean():.1f}  (expect ~3.8)")
print(f"  Avg total corners: {_cc_df['total_corners'].mean():.1f}  (expect ~9.2)")
print(f"  Avg total red    : {_cc_df['total_red'].mean():.2f}  (expect ~0.1-0.3)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PRECOMPUTE KNOCKOUT MATCH LAMBDAS (all 48×48 pairs, neutral form)
# ─────────────────────────────────────────────────────────────────────────────

print("\nPrecomputing knockout match lambdas (all team pairs) ...")
KNOCKOUT_DATE = "2026-07-05"

ordered_pairs = [(h, a) for h in all_wc_teams for a in all_wc_teams if h != a]

# Build batch DataFrame for all pairs with neutral form (history-free)
batch_rows = [{
    "date": KNOCKOUT_DATE, "home_team": h, "away_team": a,
    "home_goals": 0.0, "away_goals": 0.0,
    "home_elo": last_elo.get(h, 1500.0), "away_elo": last_elo.get(a, 1500.0),
    "home_xg": 0.0, "away_xg": 0.0,
    "xg_available": 0, "is_intl": 1, "is_neutral": 1,
} for h, a in ordered_pairs]

batch_df, _ = add_form_features(pd.DataFrame(batch_rows))

# Accumulate lambdas from both models (each contributes 50%)
ko_lh = np.zeros(len(ordered_pairs))
ko_la = np.zeros(len(ordered_pairs))

for tag, (m_home, m_away, dc_params, t2i) in models.items():
    log_lh_all = np.array([get_dc_lambdas_safe(h, a, dc_params, t2i)[0] for h, a in ordered_pairs])
    log_la_all = np.array([get_dc_lambdas_safe(h, a, dc_params, t2i)[1] for h, a in ordered_pairs])
    X_batch    = build_features(batch_df.reset_index(drop=True), log_lh_all, log_la_all)
    ko_lh     += np.clip(m_home.predict(X_batch), 0.05, 10) / 2
    ko_la     += np.clip(m_away.predict(X_batch), 0.05, 10) / 2

# Build lookup matrices [home_team_idx, away_team_idx]
ko_lh_mat = np.full((n_teams, n_teams), 1.2)
ko_la_mat = np.full((n_teams, n_teams), 0.9)
for i, (h, a) in enumerate(ordered_pairs):
    hi, ai = team_idx[h], team_idx[a]
    ko_lh_mat[hi, ai] = ko_lh[i]
    ko_la_mat[hi, ai] = ko_la[i]

print(f"  {len(ordered_pairs)} matchup lambdas precomputed")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MONTE CARLO SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Pre-index group stage for fast vectorised sampling
match_ids_ordered = list(fixtures["match_id"].astype(int))
lh_vec = np.array([group_preds[m]["lh"] for m in match_ids_ordered])
la_vec = np.array([group_preds[m]["la"] for m in match_ids_ordered])

# Map match → group, home, away for standings accumulation
match_meta = {
    m: (group_preds[m]["group"],
        group_preds[m]["home_team"],
        group_preds[m]["away_team"])
    for m in match_ids_ordered
}

# Group→list of match_ids
group_match_ids: dict[str, list] = defaultdict(list)
for mid in match_ids_ordered:
    group_match_ids[group_preds[mid]["group"]].append(mid)


def rank_group(pts_dict, gd_dict, gf_dict):
    """Sort 4 teams: pts desc → GD desc → GF desc → alphabetical."""
    teams = list(pts_dict.keys())
    teams.sort(key=lambda t: (-pts_dict[t], -gd_dict[t], -gf_dict[t], t))
    return teams  # [1st, 2nd, 3rd, 4th]


def simulate_group_stage(hg_all, ag_all):
    """
    Given sampled goals for all 72 matches, return:
      group_rankings: {group: [1st, 2nd, 3rd, 4th]}
    """
    pts = {g: defaultdict(int) for g in groups}
    gd  = {g: defaultdict(int) for g in groups}
    gf  = {g: defaultdict(int) for g in groups}

    for i, mid in enumerate(match_ids_ordered):
        grp, home, away = match_meta[mid]
        hg, ag = int(hg_all[i]), int(ag_all[i])

        if hg > ag:
            pts[grp][home] += 3
        elif ag > hg:
            pts[grp][away] += 3
        else:
            pts[grp][home] += 1
            pts[grp][away] += 1

        gd[grp][home] += hg - ag;  gd[grp][away] += ag - hg
        gf[grp][home] += hg;       gf[grp][away] += ag

    # Ensure all 4 group members appear (default 0 for groups with no goals yet)
    for g, team_list in group_teams.items():
        for t in team_list:
            pts[g].setdefault(t, 0)
            gd[g].setdefault(t, 0)
            gf[g].setdefault(t, 0)

    group_rankings = {g: rank_group(pts[g], gd[g], gf[g]) for g in groups}
    return group_rankings, pts, gd, gf


def get_third_qualifiers(group_rankings, raw_pts, raw_gd, raw_gf):
    """
    Best 8 of 12 third-place teams, ranked by pts → GD → GF → group letter.
    Returns list of 8 names sorted by group letter for consistent slot assignment.
    """
    thirds = []
    for g, ranked in group_rankings.items():
        if len(ranked) >= 3:
            third = ranked[2]
            thirds.append({
                "team": third, "group": g,
                "pts": raw_pts[g][third],
                "gd":  raw_gd[g][third],
                "gf":  raw_gf[g][third],
            })
    # Rank cross-group by performance, then take best 8
    thirds.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"], x["group"]))
    best8 = thirds[:8]
    # Sort qualified 8 by group letter for deterministic slot assignment
    best8.sort(key=lambda x: x["group"])
    return [t["team"] for t in best8]


def fill_r32_bracket(group_rankings, third_qualifiers):
    """
    Map group results + 3rd-place qualifiers to R32 slot match pairs.
    third_qualifiers: list of 8 team names sorted by group letter.
    Returns {match_id: (home_team, away_team)}.
    """
    slot_map = {}
    for g, ranked in group_rankings.items():
        if len(ranked) >= 1: slot_map[f"Winner Group {g}"]  = ranked[0]
        if len(ranked) >= 2: slot_map[f"Runner-up Group {g}"] = ranked[1]

    r32_rows = (
        knockout_slots[knockout_slots["round"] == "Round of 32"]
        .sort_values("match_id")
    )
    bracket = {}
    third_ptr = 0

    def resolve(slot):
        nonlocal third_ptr
        if slot in slot_map:
            return slot_map[slot]
        if "Best 3rd" in slot:
            team = third_qualifiers[third_ptr] if third_ptr < len(third_qualifiers) else None
            third_ptr += 1
            return team
        return None

    for _, row in r32_rows.iterrows():
        mid  = int(row["match_id"])
        home = resolve(row["slot_home"])
        away = resolve(row["slot_away"])
        if home and away:
            bracket[mid] = (home, away)

    return bracket


def simulate_knockout_match(home, away):
    """
    Sample Poisson goals for one knockout match.
    If tied after 90 min → 50/50 penalty.
    Returns (winner, home_goals, away_goals, on_pens: bool).
    """
    hi, ai = team_idx.get(home, 0), team_idx.get(away, 0)
    lh = ko_lh_mat[hi, ai]
    la = ko_la_mat[hi, ai]
    hg = np.random.poisson(lh)
    ag = np.random.poisson(la)
    if hg > ag:
        return home, hg, ag, False
    elif ag > hg:
        return away, hg, ag, False
    else:
        winner = home if np.random.random() < 0.5 else away
        return winner, hg, ag, True


def run_one_simulation():
    """
    Full tournament simulation.
    Returns:
      match_winner   {match_id: team}       — all 32 knockout match winners
      match_loser    {match_id: team}       — all 32 knockout match losers
      match_score    {match_id: "H-A"}
      match_pens     {match_id: bool}       — True if went to penalties
      group_rankings {group: [1st,..,4th]}
    """
    # ── Group stage ──────────────────────────────────────────────────────────
    hg_all = np.random.poisson(lh_vec)
    ag_all = np.random.poisson(la_vec)
    group_rankings, raw_pts, raw_gd, raw_gf = simulate_group_stage(hg_all, ag_all)

    third_qualifiers = get_third_qualifiers(group_rankings, raw_pts, raw_gd, raw_gf)
    r32_bracket      = fill_r32_bracket(group_rankings, third_qualifiers)

    # ── Knockout rounds ───────────────────────────────────────────────────────
    current_matches = dict(r32_bracket)  # match_id → (home, away)
    match_winner     = {}
    match_loser      = {}
    match_score      = {}
    match_pens       = {}
    match_home_teams = {}  # track which team was designated home in each KO slot

    round_groups = {
        "Round of 32": list(
            knockout_slots[knockout_slots["round"] == "Round of 32"]["match_id"].astype(int)
        ),
        "Round of 16": list(
            knockout_slots[knockout_slots["round"] == "Round of 16"]["match_id"].astype(int)
        ),
        "Quarter-final": list(
            knockout_slots[knockout_slots["round"] == "Quarter-final"]["match_id"].astype(int)
        ),
        "Semi-final": list(
            knockout_slots[knockout_slots["round"] == "Semi-final"]["match_id"].astype(int)
        ),
        "Third-place playoff": [103],
        "Final": [104],
    }

    for round_name, round_mids in round_groups.items():
        for mid in round_mids:
            if mid not in current_matches:
                continue
            home, away       = current_matches[mid]
            w, hg, ag, pens  = simulate_knockout_match(home, away)
            loser            = away if w == home else home
            match_winner[mid]     = w
            match_loser[mid]      = loser
            match_score[mid]      = f"{hg}-{ag}" + (" (pens)" if pens else "")
            match_pens[mid]       = pens
            match_home_teams[mid] = home

        # Propagate winners/losers to next round slots
        next_rows = knockout_slots[
            knockout_slots["round"].isin(["Round of 16", "Quarter-final",
                                          "Semi-final", "Final",
                                          "Third-place playoff"])
        ]
        for _, row in next_rows.iterrows():
            mid = int(row["match_id"])
            def parse_team(slot):
                if slot.startswith("Winner Match "):
                    ref = int(slot.split()[-1])
                    return match_winner.get(ref)
                if slot.startswith("Loser Match "):
                    ref = int(slot.split()[-1])
                    return match_loser.get(ref)
                return None
            h = parse_team(row["slot_home"])
            a = parse_team(row["slot_away"])
            if h and a:
                current_matches[mid] = (h, a)

    return match_winner, match_loser, match_score, match_pens, group_rankings, match_home_teams

# ─────────────────────────────────────────────────────────────────────────────
# 9. RUN MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nRunning Monte Carlo simulation ({N_SIMS:,} runs) ...")
t0 = time.time()

ROUNDS = ["Group Stage", "Round of 32", "Round of 16", "Quarter-final",
          "Semi-final", "Final", "Winner"]
advance_counts = {r: defaultdict(int) for r in ROUNDS}
ko_bracket_wins = defaultdict(lambda: defaultdict(int))  # match_id → team → count
group_rank_counts = defaultdict(lambda: defaultdict(int))  # team → "1st"/"2nd"/"3rd"/"4th" → count

ko_pens_counts = defaultdict(int)
ko_home_counts = defaultdict(lambda: defaultdict(int))
ko_away_counts = defaultdict(lambda: defaultdict(int))

all_ko_mid = list(knockout_slots["match_id"].astype(int))
r32_mids   = list(knockout_slots[knockout_slots["round"] == "Round of 32"]["match_id"].astype(int))
r16_mids   = list(knockout_slots[knockout_slots["round"] == "Round of 16"]["match_id"].astype(int))
qf_mids    = list(knockout_slots[knockout_slots["round"] == "Quarter-final"]["match_id"].astype(int))
sf_mids    = list(knockout_slots[knockout_slots["round"] == "Semi-final"]["match_id"].astype(int))

for sim in range(N_SIMS):
    win, lose, score, pens, g_rank, home_teams = run_one_simulation()

    # Track group stage qualifiers
    for g, ranked in g_rank.items():
        for pos, team in enumerate(ranked, 1):
            advance_counts["Group Stage"][team] += 1
            group_rank_counts[team][str(pos)] += 1

    # Track knockout advancement

    for mid in r32_mids:
        if mid in win: advance_counts["Round of 32"][win[mid]] += 1
    for mid in r16_mids:
        if mid in win: advance_counts["Round of 16"][win[mid]] += 1
    for mid in qf_mids:
        if mid in win: advance_counts["Quarter-final"][win[mid]] += 1
    for mid in sf_mids:
        if mid in win: advance_counts["Semi-final"][win[mid]] += 1
    if 104 in win:
        advance_counts["Final"][win[104]] += 1
        advance_counts["Winner"][win[104]] += 1

    # Track most common bracket path
    for mid in all_ko_mid:
        if mid in win:
            ko_bracket_wins[mid][win[mid]] += 1

    # Track penalty frequency and home/away identity per KO slot
    for mid in all_ko_mid:
        if pens.get(mid, False):
            ko_pens_counts[mid] += 1
        if mid in home_teams and mid in win:
            h = home_teams[mid]
            a = lose[mid] if win[mid] == h else win[mid]
            ko_home_counts[mid][h] += 1
            ko_away_counts[mid][a] += 1

    if (sim + 1) % 2000 == 0:
        elapsed = time.time() - t0
        print(f"  {sim+1:,}/{N_SIMS:,}  ({elapsed:.0f}s)")

elapsed = time.time() - t0
print(f"  Complete: {N_SIMS:,} runs in {elapsed:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 10. OUTPUT: GROUP PREDICTIONS CSV
# ─────────────────────────────────────────────────────────────────────────────

group_pred_rows = []
for mid in match_ids_ordered:
    p = group_preds[mid]

    _h, _a = map(int, p["predicted_score"].split("-"))
    if _h > _a:
        pred_winner = "home"
    elif _a > _h:
        pred_winner = "away"
    else:
        pred_winner = "draw"

    group_pred_rows.append({
        "match_id":            mid,
        "group":               p["group"],
        "home_team":           p["home_team"],
        "away_team":           p["away_team"],
        "expected_home_goals": round(p["lh"], 3),
        "expected_away_goals": round(p["la"], 3),
        "predicted_score":     p["predicted_score"],
        "prob_home_win":       round(p["p_home"], 4),
        "prob_draw":           round(p["p_draw"], 4),
        "prob_away_win":       round(p["p_away"], 4),
        "predicted_winner":         pred_winner,
        "predicted_home_yellow":    p.get("home_yellow",  0),
        "predicted_away_yellow":    p.get("away_yellow",  0),
        "predicted_total_yellow":   p.get("total_yellow", 0),
        "predicted_home_red":       p.get("home_red",     0),
        "predicted_away_red":       p.get("away_red",     0),
        "predicted_total_red":      p.get("total_red",    0),
        "predicted_home_corners":   p.get("home_corners", 0),
        "predicted_away_corners":   p.get("away_corners", 0),
        "predicted_total_corners":  p.get("total_corners",0),
    })

group_pred_df = pd.DataFrame(group_pred_rows)
group_pred_df.to_csv("wc2026_group_predictions.csv", index=False)
print("\nSaved: wc2026_group_predictions.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 11. OUTPUT: GROUP STANDINGS CSV (from MC)
# ─────────────────────────────────────────────────────────────────────────────

standing_rows = []
for team in all_wc_teams:
    grp = team_group.get(team, "?")
    rc  = group_rank_counts[team]
    standing_rows.append({
        "group":        grp,
        "team":         team,
        "p_1st":        round(rc.get("1", 0) / N_SIMS, 4),
        "p_2nd":        round(rc.get("2", 0) / N_SIMS, 4),
        "p_3rd":        round(rc.get("3", 0) / N_SIMS, 4),
        "p_4th":        round(rc.get("4", 0) / N_SIMS, 4),
        "p_qualify":    round((rc.get("1", 0) + rc.get("2", 0)) / N_SIMS, 4),
    })

standing_df = (
    pd.DataFrame(standing_rows)
    .sort_values(["group", "p_1st"], ascending=[True, False])
)
standing_df.to_csv("wc2026_group_standings.csv", index=False)
print("Saved: wc2026_group_standings.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 12. OUTPUT: KNOCKOUT BRACKET CSV (most likely path from MC)
# ─────────────────────────────────────────────────────────────────────────────

bracket_rows = []
for _, row in knockout_slots.sort_values("match_id").iterrows():
    mid = int(row["match_id"])
    most_likely_winner = max(ko_bracket_wins[mid], key=ko_bracket_wins[mid].get, default="TBD")
    win_pct = round(ko_bracket_wins[mid].get(most_likely_winner, 0) / N_SIMS, 4) if ko_bracket_wins[mid] else 0.0

    # Most likely home/away team from MC-tracked identity
    pred_home_team = max(ko_home_counts[mid], key=ko_home_counts[mid].get) if ko_home_counts[mid] else "TBD"
    pred_away_team = max(ko_away_counts[mid], key=ko_away_counts[mid].get) if ko_away_counts[mid] else "TBD"

    # Deterministic MAP score for the most likely matchup (clean H-A, no pens suffix)
    if pred_home_team != "TBD" and pred_away_team != "TBD":
        lh_d = ko_lh_mat[team_idx.get(pred_home_team, 0), team_idx.get(pred_away_team, 0)]
        la_d = ko_la_mat[team_idx.get(pred_home_team, 0), team_idx.get(pred_away_team, 0)]
        pm_d = poisson_prob_matrix(lh_d, la_d)
        ph_d, pa_d = np.unravel_index(np.argmax(pm_d), pm_d.shape)
        det_score = f"{ph_d}-{pa_d}"
    else:
        det_score = "TBD"

    # Penalty prediction from MC frequency
    p_pens = round(ko_pens_counts[mid] / N_SIMS, 4)
    pred_penalties = p_pens > 0.35   # threshold above historical ~28% mean; False dominates EV

    # Winner side
    if most_likely_winner == pred_home_team:
        pred_winner_side = "home"
    elif most_likely_winner == pred_away_team:
        pred_winner_side = "away"
    else:
        pred_winner_side = "home"

    # Cards & corners for this matchup
    if pred_home_team != "TBD" and pred_away_team != "TBD":
        ko_cc = predict_cc_match(pred_home_team, pred_away_team, _cc_team_stats, _cc_globals)
    else:
        ko_cc = {"total_yellow": 4, "total_red": 0, "total_corners": 9}

    bracket_rows.append({
        "match_id":              mid,
        "round":                 row["round"],
        "date_utc":              row["date_utc"],
        "venue":                 row["venue"],
        "slot_home":             row["slot_home"],
        "slot_away":             row["slot_away"],
        "predicted_home_team":   pred_home_team,
        "predicted_away_team":   pred_away_team,
        "predicted_winner":      most_likely_winner,
        "predicted_winner_side": pred_winner_side,
        "winner_pct":            win_pct,
        "predicted_score":       det_score,
        "p_penalties":           p_pens,
        "predicted_penalties":   pred_penalties,
        "predicted_total_yellow":  ko_cc.get("total_yellow",  4),
        "predicted_total_red":     ko_cc.get("total_red",     0),
        "predicted_total_corners": ko_cc.get("total_corners", 9),
    })

bracket_df = pd.DataFrame(bracket_rows)
bracket_df.to_csv("wc2026_knockout_bracket.csv", index=False)
print("Saved: wc2026_knockout_bracket.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 13. OUTPUT: WIN PROBABILITIES CSV
# ─────────────────────────────────────────────────────────────────────────────

prob_rows = []
for team in all_wc_teams:
    prob_rows.append({
        "team":            team,
        "group":           team_group.get(team, "?"),
        "p_qualify_group": round((group_rank_counts[team].get("1", 0) +
                                  group_rank_counts[team].get("2", 0)) / N_SIMS, 4),
        "p_r32":           round(advance_counts["Round of 32"][team] / N_SIMS, 4),
        "p_r16":           round(advance_counts["Round of 16"][team] / N_SIMS, 4),
        "p_qf":            round(advance_counts["Quarter-final"][team] / N_SIMS, 4),
        "p_sf":            round(advance_counts["Semi-final"][team] / N_SIMS, 4),
        "p_final":         round(advance_counts["Final"][team] / N_SIMS, 4),
        "p_win":           round(advance_counts["Winner"][team] / N_SIMS, 4),
    })

prob_df = (
    pd.DataFrame(prob_rows)
    .sort_values("p_win", ascending=False)
    .reset_index(drop=True)
)
prob_df.to_csv("wc2026_win_probabilities.csv", index=False)
print("Saved: wc2026_win_probabilities.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 14. FULL COMPETITION SUBMISSION (104 matches)
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating wc2026_submission.csv (104 matches) ...")

submission_rows = []

# ── Group stage (matches 1–72) ───────────────────────────────────────────────
for mid in match_ids_ordered:
    p = group_preds[mid]

    _h, _a = map(int, p["predicted_score"].split("-"))
    if _h > _a:
        pred_winner = "home"
    elif _a > _h:
        pred_winner = "away"
    else:
        pred_winner = "draw"

    submission_rows.append({
        "match_id":           mid,
        "round":              "Group Stage",
        "multiplier":         1,
        "home_team":          p["home_team"],
        "away_team":          p["away_team"],
        "score":              p["predicted_score"],
        "total_corners":      p.get("total_corners", 9),
        "total_yellow_cards": p.get("total_yellow",  4),
        "total_red_cards":    p.get("total_red",     0),
        "winning_team":       pred_winner,
        "matchup_home":       "",
        "matchup_away":       "",
        "match_winner":       "",
        "penalties":          "",
    })

# ── Knockout stage (matches 73–104) ─────────────────────────────────────────
for _, brow in bracket_df.iterrows():
    mid       = int(brow["match_id"])
    pred_home = brow["predicted_home_team"]
    pred_away = brow["predicted_away_team"]
    winner    = brow["predicted_winner"]

    if winner == pred_home:
        winner_side = "home"
    elif winner == pred_away:
        winner_side = "away"
    else:
        winner_side = "home"

    mult_row = knockout_slots.loc[knockout_slots["match_id"] == mid, "multiplier"]
    mult = int(mult_row.iloc[0]) if not mult_row.empty else 1

    submission_rows.append({
        "match_id":           mid,
        "round":              brow["round"],
        "multiplier":         mult,
        "home_team":          pred_home,
        "away_team":          pred_away,
        "score":              brow["predicted_score"],
        "total_corners":      int(brow["predicted_total_corners"]),
        "total_yellow_cards": int(brow["predicted_total_yellow"]),
        "total_red_cards":    int(brow["predicted_total_red"]),
        "winning_team":       "",
        "matchup_home":       pred_home,
        "matchup_away":       pred_away,
        "match_winner":       winner_side,
        "penalties":          str(brow["predicted_penalties"]),
    })

submission_df = (
    pd.DataFrame(submission_rows)
    .sort_values("match_id")
    .reset_index(drop=True)
)
submission_df.to_csv("wc2026_submission.csv", index=False)
print(f"  Saved: wc2026_submission.csv  ({len(submission_df)} rows)")

# ── Assertions ────────────────────────────────────────────────────────────────
assert len(submission_df) == 104, f"Expected 104 rows, got {len(submission_df)}"
grp_rows = submission_df[submission_df["round"] == "Group Stage"]
ko_rows  = submission_df[submission_df["round"] != "Group Stage"]
assert len(grp_rows) == 72,  f"Expected 72 group rows, got {len(grp_rows)}"
assert len(ko_rows)  == 32,  f"Expected 32 KO rows, got {len(ko_rows)}"
assert grp_rows["winning_team"].isin(["home", "draw", "away"]).all()
assert ko_rows["match_winner"].isin(["home", "away"]).all()
assert ko_rows["penalties"].isin(["True", "False"]).all()
assert ko_pens_counts[103] > 0, "Third-place (mid=103) was never simulated"

cc_check = pd.DataFrame(list(_cc_preds.values()))
assert cc_check["total_yellow"].between(0, 12).all()
assert cc_check["total_corners"].between(3, 20).all()

pen_rate = (ko_rows["penalties"] == "True").mean()
print(f"  KO penalty rate  : {pen_rate:.1%}  (expect 20-35%)")
print(f"  P(Final pens)    : {ko_pens_counts[104]/N_SIMS:.1%}")
print("  All assertions passed.")
print("Saved: wc2026_submission.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 15. CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  TOURNAMENT WINNER PREDICTION  (top 10 by P(Win))")
print("=" * 70)
print(f"  {'Team':<28} {'Group':<8} {'P(QF)':<9} {'P(SF)':<9} {'P(Final)':<10} {'P(Win)'}")
print("-" * 70)
for _, row in prob_df.head(10).iterrows():
    print(f"  {row['team']:<28} {row['group']:<8} "
          f"{row['p_qf']:<9.1%} {row['p_sf']:<9.1%} "
          f"{row['p_final']:<10.1%} {row['p_win']:.1%}")

print("\n  Predicted winner:", prob_df.iloc[0]["team"],
      f"(P={prob_df.iloc[0]['p_win']:.1%})")

# Identify most likely finalists from SF winners
sf_winner_counts = defaultdict(int)
for mid in sf_mids:
    for team, cnt in ko_bracket_wins[mid].items():
        sf_winner_counts[team] += cnt
top_finalists = sorted(sf_winner_counts, key=lambda t: -sf_winner_counts[t])[:2]
finalist1 = top_finalists[0] if len(top_finalists) > 0 else "TBD"
finalist2 = top_finalists[1] if len(top_finalists) > 1 else "TBD"
final_winner = prob_df.iloc[0]["team"]
final_score_row = bracket_df[bracket_df["match_id"] == 104]
final_score = final_score_row.iloc[0]["predicted_score"] if not final_score_row.empty else "?-?"
print(f"  Predicted final : {finalist1} vs {finalist2} -> {final_winner} {final_score}")

print("\n  Output files:")
print("    wc2026_group_predictions.csv")
print("    wc2026_group_standings.csv")
print("    wc2026_knockout_bracket.csv")
print("    wc2026_win_probabilities.csv")
print("    wc2026_submission.csv")
print("=" * 70)
