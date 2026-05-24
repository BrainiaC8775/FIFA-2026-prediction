import pandas as pd

events = pd.read_csv("tm_game_events.csv")
games = pd.read_csv("tm_games.csv")

desc = events["description"].fillna("")
events["is_yellow"] = (
    (events["type"] == "Cards")
    & desc.str.contains("yellow card", case=False)
    & ~desc.str.contains("second yellow", case=False)
)
events["is_red"] = (events["type"] == "Cards") & (
    desc.str.contains("red card", case=False)
    | desc.str.contains("second yellow", case=False)
)
# "corner_assists" = goals where the assist type was a corner kick
events["is_corner"] = (events["type"] == "Goals") & desc.str.contains(
    "corner", case=False
)

agg = events.groupby(["game_id", "club_id"], as_index=False).agg(
    yellow_cards=("is_yellow", "sum"),
    red_cards=("is_red", "sum"),
    corner_assists=("is_corner", "sum"),
)

filtered_games = games[
    games["competition_type"].isin(["national_team_competition", "international_cup"])
].copy()

home_stats = agg.rename(
    columns={
        "club_id": "home_club_id",
        "yellow_cards": "home_yellow_cards",
        "red_cards": "home_red_cards",
        "corner_assists": "home_corner_assists",
    }
)
away_stats = agg.rename(
    columns={
        "club_id": "away_club_id",
        "yellow_cards": "away_yellow_cards",
        "red_cards": "away_red_cards",
        "corner_assists": "away_corner_assists",
    }
)

result = (
    filtered_games.merge(home_stats, on=["game_id", "home_club_id"], how="left")
    .merge(away_stats, on=["game_id", "away_club_id"], how="left")
)

for col in ["home_yellow_cards", "home_red_cards", "home_corner_assists",
            "away_yellow_cards", "away_red_cards", "away_corner_assists"]:
    result[col] = result[col].fillna(0).astype(int)

result.to_csv("tm_games_with_events.csv", index=False)
print(f"Saved {len(result)} rows to tm_games_with_events.csv")
print(result[["game_id", "home_club_name", "away_club_name",
              "home_yellow_cards", "away_yellow_cards",
              "home_red_cards", "away_red_cards",
              "home_corner_assists", "away_corner_assists"]].head(10).to_string())
