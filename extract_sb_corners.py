import json
import os
import pandas as pd

BASE = "statsbomb/data"

# International competition+season pairs
comps = json.load(open(f"{BASE}/competitions.json", encoding="utf-8"))
intl_pairs = {
    (c["competition_id"], c["season_id"]): c
    for c in comps
    if c["competition_international"]
}

records = []

for (comp_id, season_id), comp_info in intl_pairs.items():
    match_file = f"{BASE}/matches/{comp_id}/{season_id}.json"
    if not os.path.exists(match_file):
        continue

    matches = json.load(open(match_file, encoding="utf-8"))

    for m in matches:
        match_id = m["match_id"]
        home_team_id = m["home_team"]["home_team_id"]
        home_team_name = m["home_team"]["home_team_name"]
        away_team_id = m["away_team"]["away_team_id"]
        away_team_name = m["away_team"]["away_team_name"]

        event_file = f"{BASE}/events/{match_id}.json"
        if not os.path.exists(event_file):
            home_corners = away_corners = None
        else:
            events = json.load(open(event_file, encoding="utf-8"))
            home_corners = sum(
                1 for e in events
                if e.get("type", {}).get("name") == "Pass"
                and e.get("pass", {}).get("type", {}).get("name") == "Corner"
                and e.get("team", {}).get("id") == home_team_id
            )
            away_corners = sum(
                1 for e in events
                if e.get("type", {}).get("name") == "Pass"
                and e.get("pass", {}).get("type", {}).get("name") == "Corner"
                and e.get("team", {}).get("id") == away_team_id
            )

        records.append({
            "match_id": match_id,
            "competition_id": comp_id,
            "competition_name": comp_info["competition_name"],
            "season_id": season_id,
            "season_name": comp_info["season_name"],
            "match_date": m["match_date"],
            "home_team_id": home_team_id,
            "home_team_name": home_team_name,
            "away_team_id": away_team_id,
            "away_team_name": away_team_name,
            "home_score": m["home_score"],
            "away_score": m["away_score"],
            "home_corners": home_corners,
            "away_corners": away_corners,
        })

df = pd.DataFrame(records)
df.to_csv("sb_international_corners.csv", index=False)
print(f"Saved {len(df)} matches to sb_international_corners.csv")
print(f"Matches with event data: {df['home_corners'].notna().sum()}")
print()
print(df[["competition_name", "season_name", "home_team_name", "away_team_name",
          "home_corners", "away_corners"]].head(10).to_string())
