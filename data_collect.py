import numpy as np
import pandas as pd
from scipy.stats import poisson

group_fixtures = pd.read_csv('group_fixtures.csv')

qualified_teams={"UEFA Playoff A":"Bosnia and Herzegovina",
                "UEFA Playoff B":"Sweden",
                "UEFA Playoff C":"Turkey",
                "UEFA Playoff D":"Czechia",
                "FIFA Playoff 1":"Congo DR",
                "FIFA Playoff 2":"Iraq"} 

group_fixtures["home_team"]=group_fixtures["home_team"].replace(qualified_teams)
group_fixtures["away_team"]=group_fixtures["away_team"].replace(qualified_teams)

teams=pd.concat([group_fixtures["home_team"],group_fixtures["away_team"]])
unique_teams=sorted(teams.unique())
unique_teams

group_standings=group_fixtures[["group","home_team"]]
group_standings=group_standings.drop_duplicates().sort_values(["group","home_team"]).reset_index(drop=True)
group_standings["played"] = 0
group_standings["wins"] = 0
group_standings["draws"] = 0
group_standings["losses"] = 0
group_standings["goals_for"] = 0
group_standings["goals_against"] = 0
group_standings["goal_diff"] = 0
group_standings["points"] = 0

fifa_stats=pd.read_csv('fifa_wc2026_dataset.csv')
domestic_teams=pd.read_csv('tm_clubs.csv')
national_teams=pd.read_csv('tm_national_teams.csv')
competitions=pd.read_csv('tm_competitions.csv')
match_history = pd.concat([
    pd.read_csv('tm_domestic_games.csv'),
    pd.read_csv('tm_national_games.csv'),
    pd.read_csv('tm_other_games.csv')
], ignore_index=True, sort=False)
match_history = match_history.merge(
    competitions[['competition_id', 'name']].rename(columns={'name': 'competition_name'}),
    on='competition_id',
    how='left'
)

current_season = 2022
alpha = 0.88

def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum()

league_df=pd.concat([
    match_history[match_history['competition_type']=='national_team_competition'].copy(),
    match_history[match_history['competition_type']=='international_cup'].copy()
], ignore_index=True, sort=False)

train_df = league_df[league_df['season'].isin([2018,2019,2020,2021])]
train_df["weight"] = alpha ** (current_season - train_df["season"])
test_df = league_df[league_df['season']== 2022 ]

league_home_avg = weighted_mean(train_df['home_club_goals'],train_df['weight'])
league_away_avg = weighted_mean(train_df['away_club_goals'],train_df['weight'])

home_stats = train_df.groupby("home_club_id").apply(
    lambda x: pd.Series({
        "home_scored": weighted_mean(x["home_club_goals"], x["weight"]),
        "home_conceded": weighted_mean(x["away_club_goals"], x["weight"]),
        "home_games": x["weight"].sum()
    })
)


away_stats = train_df.groupby("away_club_id").apply(
    lambda x: pd.Series({
        "away_scored": weighted_mean(x["away_club_goals"], x["weight"]),
        "away_conceded": weighted_mean(x["home_club_goals"], x["weight"]),
        "away_games": x["weight"].sum()
    })
)


team_stats = home_stats.join(away_stats, how='outer').fillna(league_home_avg)

team_stats = team_stats.fillna({
    "home_scored": league_home_avg,
    "home_conceded": league_away_avg,
    "away_scored": league_away_avg,
    "away_conceded": league_home_avg,
    "home_games": 0,
    "away_games": 0
})

k = 2   #---> smoothing factor

team_stats["home_scored"] = (
    (team_stats["home_scored"] * team_stats["home_games"]) +
    (k * league_home_avg)
) / (team_stats["home_games"] + k)

team_stats["home_conceded"] = (
    (team_stats["home_conceded"] * team_stats["home_games"]) +
    (k * league_away_avg)
) / (team_stats["home_games"] + k)

team_stats["away_scored"] = (
    (team_stats["away_scored"] * team_stats["away_games"]) +
    (k * league_away_avg)
) / (team_stats["away_games"] + k)

team_stats["away_conceded"] = (
    (team_stats["away_conceded"] * team_stats["away_games"]) +
    (k * league_home_avg)
) / (team_stats["away_games"] + k)


team_stats["home_attack"] = team_stats["home_scored"] / league_home_avg
team_stats['home_defense'] = team_stats['home_conceded'] / league_away_avg
team_stats['away_attack'] = team_stats['away_scored'] / league_away_avg
team_stats['away_defense'] = team_stats['away_conceded'] / league_home_avg

def expected_goals(home_team_id, away_team_id):

    if home_team_id not in team_stats.index or away_team_id not in team_stats.index:
        return league_home_avg, league_away_avg  # fallback

    lambda_home = (
        team_stats.loc[home_team_id, "home_attack"] *
        team_stats.loc[away_team_id, "away_defense"] *
        league_home_avg
    )

    lambda_away = (
        team_stats.loc[away_team_id, "away_attack"] *
        team_stats.loc[home_team_id, "home_defense"] *
        league_away_avg
    )
    MIN_LAMBDA = 0.25
    MAX_LAMBDA = 3.2

    lambda_home = max(MIN_LAMBDA, min(lambda_home, MAX_LAMBDA))
    lambda_away = max(MIN_LAMBDA, min(lambda_away, MAX_LAMBDA))

    return lambda_home, lambda_away

def dixon_coles_correction(h, a, lambda_home, lambda_away, rho):

    if h == 0 and a == 0:
        return 1 - (lambda_home * lambda_away * rho)

    elif h == 0 and a == 1:
        return 1 + (lambda_home * rho)

    elif h == 1 and a == 0:
        return 1 + (lambda_away * rho)

    elif h == 1 and a == 1:
        return 1 - rho

    else:
        return 1

def score_probs(home_team_id, away_team_id, max_goals=7, rho=0):

    lambda_home, lambda_away = expected_goals(home_team_id, away_team_id)

    probs = {}

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            probs[(h, a)] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            base_prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)

            # ✅ Apply Dixon-Coles adjustment
            tau = dixon_coles_correction(h, a, lambda_home, lambda_away, rho)

            probs[(h, a)] = base_prob * tau
    
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    return probs

results = []

for _, row in test_df.iterrows():

    probs = score_probs(row["home_club_id"], row["away_club_id"])

    best_score = max(probs, key=probs.get)

    results.append({
        "actual_home_goals": row["home_club_goals"],
        "actual_away_goals": row["away_club_goals"],
        "probs": probs
    })

for i, r in enumerate(results[:5]):
    print(f"\nMatch {i+1}")
    top5 = sorted(r["probs"].items(), key=lambda x: x[1], reverse=True)[:5]
    print(top5)

def outcome_probs(probs):
    home_win = 0
    draw = 0
    away_win = 0

    for (h, a), p in probs.items():
        if h > a:
            home_win += p
        elif h == a:
            draw += p
        else:
            away_win += p

    return home_win, draw, away_win


correct = 0

for r in results:
    home_win, draw, away_win = outcome_probs(r["probs"])

    predicted = max(
        ["H", "D", "A"],
        key=lambda x: {"H": home_win, "D": draw, "A": away_win}[x]
    )

    actual = (
        "H" if r["actual_home_goals"] > r["actual_away_goals"]
        else "D" if r["actual_home_goals"] == r["actual_away_goals"]
        else "A"
    )

    if predicted == actual:
        correct += 1

accuracy = correct / len(results)
print("Outcome accuracy:", accuracy)

log_losses = []

for r in results:
    home_win, draw, away_win = outcome_probs(r["probs"])

    actual = (
        0 if r["actual_home_goals"] > r["actual_away_goals"]
        else 1 if r["actual_home_goals"] == r["actual_away_goals"]
        else 2
    )

    probs_vec = [home_win, draw, away_win]

    prob = probs_vec[actual]
    prob = max(prob, 1e-10)  # avoid log(0)

    log_losses.append(-np.log(prob))

print("Log Loss:", np.mean(log_losses))


correct_scores = 0

for r in results:
    predicted_score = max(r["probs"], key=r["probs"].get)

    actual_score = (
        r["actual_home_goals"],
        r["actual_away_goals"]
    )

    if predicted_score == actual_score:
        correct_scores += 1

print("Exact score accuracy:", correct_scores / len(results))

top3_correct = 0

for r in results:
    top3 = sorted(r["probs"], key=r["probs"].get, reverse=True)[:3]

    actual = (r["actual_home_goals"], r["actual_away_goals"])

    if actual in top3:
        top3_correct += 1

print("Top-3 accuracy:", top3_correct / len(results))