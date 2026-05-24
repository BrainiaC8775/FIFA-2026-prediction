import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── team name normalisers ──────────────────────────────────────────────────────
TM_MAP = {
    "Ivory Coast": "Côte d'Ivoire",
    "United States": "USA",
}

SB_MAP = {
    "Cape Verde Islands": "Cabo Verde",
    "United States": "USA",
}

# ── load ───────────────────────────────────────────────────────────────────────
tm = pd.read_csv("tm_games_with_events.csv")
sb = pd.read_csv("sb_international_corners.csv")

# ── normalise TM ───────────────────────────────────────────────────────────────
tm["home_team"] = tm["home_club_name"].replace(TM_MAP)
tm["away_team"] = tm["away_club_name"].replace(TM_MAP)
tm["date"] = pd.to_datetime(tm["date"]).dt.date
tm["competition"] = tm["competition_type"]

tm_clean = tm[[
    "date", "competition", "home_team", "away_team",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
]].copy()

# ── normalise SB ───────────────────────────────────────────────────────────────
sb["home_team"] = sb["home_team_name"].replace(SB_MAP)
sb["away_team"] = sb["away_team_name"].replace(SB_MAP)
sb["date"] = pd.to_datetime(sb["match_date"]).dt.date

sb_clean = sb[[
    "date", "competition_name", "home_team", "away_team",
    "home_corners", "away_corners",
]].rename(columns={"competition_name": "sb_competition"})

# ── join on (date, home_team, away_team) ───────────────────────────────────────
merged = tm_clean.merge(
    sb_clean, on=["date", "home_team", "away_team"], how="outer"
)

# Handle SB rows where home/away are swapped vs TM
unmatched_sb = sb_clean[
    ~sb_clean.set_index(["date", "home_team", "away_team"]).index.isin(
        merged.dropna(subset=["sb_competition"])
        .set_index(["date", "home_team", "away_team"]).index
    )
].copy()

# Swap and try again against TM
unmatched_sb_swapped = unmatched_sb.rename(
    columns={
        "home_team": "away_team", "away_team": "home_team",
        "home_corners": "away_corners", "away_corners": "home_corners",
    }
)
merged2 = tm_clean.merge(
    unmatched_sb_swapped[["date", "home_team", "away_team", "home_corners", "away_corners", "sb_competition"]],
    on=["date", "home_team", "away_team"], how="inner"
)

# Update merged with swapped-match corners
merged = merged.merge(
    merged2[["date", "home_team", "away_team", "home_corners", "away_corners"]],
    on=["date", "home_team", "away_team"],
    how="left",
    suffixes=("", "_swapped"),
)
merged["home_corners"] = merged["home_corners"].fillna(merged["home_corners_swapped"])
merged["away_corners"] = merged["away_corners"].fillna(merged["away_corners_swapped"])
merged.drop(columns=["home_corners_swapped", "away_corners_swapped"], inplace=True)

# Fill missing competition from SB
merged["competition"] = merged["competition"].fillna(merged["sb_competition"])
merged.drop(columns=["sb_competition"], inplace=True)

# ── final columns & types ──────────────────────────────────────────────────────
for col in ["home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards",
            "home_corners", "away_corners"]:
    merged[col] = pd.to_numeric(merged[col], errors="coerce")

merged = merged[[
    "date", "competition", "home_team", "away_team",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_corners", "away_corners",
]].sort_values(["date", "home_team"]).reset_index(drop=True)

merged.to_csv("match_cards_corners.csv", index=False)

print(f"Total rows: {len(merged)}")
print(f"With card data:   {merged['home_yellow_cards'].notna().sum()}")
print(f"With corner data: {merged['home_corners'].notna().sum()}")
print(f"With both:        {(merged['home_yellow_cards'].notna() & merged['home_corners'].notna()).sum()}")
print()
print(merged[[
    "date", "competition", "home_team", "away_team",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_corners", "away_corners",
]].head(10).to_string())
