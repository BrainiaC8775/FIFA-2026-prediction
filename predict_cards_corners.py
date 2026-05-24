"""
predict_cards_corners.py
========================
Recency-weighted team-average predictor for corners, yellow cards, and red cards.
Used by predict_wc2026.py to fill in the non-goal competition fields.

Entry point: build_predictions_for_fixtures(fixtures_df, cards_path, min_year)
"""

import numpy as np
import pandas as pd

INTL_COMPS = [
    "national_team_competition",
    "FIFA World Cup",
    "UEFA Euro",
    "Copa America",
    "African Cup of Nations",
    "Asian Cup",
    "Gold Cup",
]

# Aliases for names that appear differently in match_cards_corners.csv
_CSV_TO_WC: dict[str, str] = {}  # team names already normalised upstream; kept for safety

DECAY_LAMBDA = 0.15   # per-year exponential decay (~0.16× weight for a 2014 match vs 2026)
MIN_OBS      = 3      # minimum rows before trusting a team-specific average


def load_and_filter(path: str, min_year: int = 2014) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df = df[df["competition"].isin(INTL_COMPS)].copy()
    df = df[df["year"] >= min_year].copy()
    if _CSV_TO_WC:
        df["home_team"] = df["home_team"].replace(_CSV_TO_WC)
        df["away_team"] = df["away_team"].replace(_CSV_TO_WC)
    df["years_ago"] = 2026 - df["year"]
    return df.reset_index(drop=True)


def compute_team_stats(
    df: pd.DataFrame,
    decay_lambda: float = DECAY_LAMBDA,
    min_obs: int = MIN_OBS,
) -> dict:
    all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    stats: dict = {}

    def weighted_avg(rows: pd.DataFrame, col: str):
        valid = rows[rows[col].notna()]
        n = len(valid)
        if n < min_obs:
            return None, n
        w = np.exp(-decay_lambda * valid["years_ago"].values)
        return float(np.average(valid[col].values, weights=w)), n

    for team in all_teams:
        hr = df[df["home_team"] == team]
        ar = df[df["away_team"] == team]

        hy, n_hy = weighted_avg(hr, "home_yellow_cards")
        hrd, n_hr = weighted_avg(hr, "home_red_cards")
        hc, n_hc  = weighted_avg(hr, "home_corners")
        ay, n_ay  = weighted_avg(ar, "away_yellow_cards")
        ard, n_ar = weighted_avg(ar, "away_red_cards")
        ac, n_ac  = weighted_avg(ar, "away_corners")

        stats[team] = {
            "home_yellow":  hy,
            "away_yellow":  ay,
            "home_red":     hrd,
            "away_red":     ard,
            "home_corners": hc,
            "away_corners": ac,
        }

    return stats


def global_stats(df: pd.DataFrame) -> dict:
    cards   = df.dropna(subset=["home_yellow_cards"])
    corners = df.dropna(subset=["home_corners"])
    return {
        "home_yellow":  float(cards["home_yellow_cards"].mean()),
        "away_yellow":  float(cards["away_yellow_cards"].mean()),
        "home_red":     float(cards["home_red_cards"].mean()),
        "away_red":     float(cards["away_red_cards"].mean()),
        "home_corners": float(corners["home_corners"].mean()),
        "away_corners": float(corners["away_corners"].mean()),
    }


def predict_match(
    home: str,
    away: str,
    team_stats: dict,
    globals_dict: dict,
) -> dict:
    def get(team: str, side: str, metric: str) -> float:
        val = team_stats.get(team, {}).get(f"{side}_{metric}")
        return val if val is not None else globals_dict[f"{side}_{metric}"]

    hy = get(home, "home", "yellow")
    ay = get(away, "away", "yellow")
    hr = get(home, "home", "red")
    ar = get(away, "away", "red")
    hc = get(home, "home", "corners")
    ac = get(away, "away", "corners")

    return {
        "home_yellow":   int(round(hy)),
        "away_yellow":   int(round(ay)),
        "home_red":      int(round(hr)),
        "away_red":      int(round(ar)),
        "home_corners":  int(round(hc)),
        "away_corners":  int(round(ac)),
        "total_yellow":  int(round(hy + ay)),
        "total_red":     int(round(hr + ar)),
        "total_corners": int(round(hc + ac)),
    }


def build_predictions_for_fixtures(
    fixtures: pd.DataFrame,
    cards_path: str = "match_cards_corners.csv",
    min_year: int = 2014,
) -> tuple[dict, dict, dict]:
    """
    Entry point for predict_wc2026.py.

    Returns:
        preds       — {match_id (int): predict_match(...) result}
        team_stats  — per-team averages (for KO re-use)
        gstats      — global fallback averages
    """
    df     = load_and_filter(cards_path, min_year)
    tstats = compute_team_stats(df)
    gstats = global_stats(df)

    preds: dict = {}
    for _, row in fixtures.iterrows():
        mid  = int(row["match_id"])
        home = str(row["home_team"])
        away = str(row["away_team"])
        preds[mid] = predict_match(home, away, tstats, gstats)

    return preds, tstats, gstats
