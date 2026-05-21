import math
import numpy as np
import pandas as pd

qualified_teams = {
    "UEFA Playoff A": "Bosnia and Herzegovina",
    "UEFA Playoff B": "Sweden",
    "UEFA Playoff C": "Turkey",
    "UEFA Playoff D": "Czechia",
    "FIFA Playoff 1": "Congo DR",
    "FIFA Playoff 2": "Iraq",
}

name_map = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Congo DR": "Democratic Republic of the Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "United States": "United States",
    "South Korea": "South Korea",
}

# Additional fallback names that are present in this dataset but differ from common FIFA names.
alias_map = {
    "Bosnia-Herzegovina": "Bosnia-Herzegovina",
    "Ivory Coast": "Ivory Coast",
    "Czech Republic": "Czech Republic",
    "South Korea": "South Korea",
    "United States": "United States",
    "Democratic Republic of the Congo": "Democratic Republic of the Congo",
}


def normalize_team_name(name: str) -> str:
    if name in name_map:
        return name_map[name]
    if name in alias_map:
        return alias_map[name]
    return name


def load_nat_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["competition_type"] == "national_team_competition"].copy()
    df["home_team"] = df["home_club_name"].astype(str).map(normalize_team_name)
    df["away_team"] = df["away_club_name"].astype(str).map(normalize_team_name)
    df = df[["home_team", "away_team", "home_club_goals", "away_club_goals"]]
    df = df.rename(columns={
        "home_club_goals": "home_goals",
        "away_club_goals": "away_goals",
    })
    return df


def prepare_fixtures(path: str) -> pd.DataFrame:
    fixtures = pd.read_csv(path)
    fixtures["home_team"] = fixtures["home_team"].replace(qualified_teams).map(normalize_team_name)
    fixtures["away_team"] = fixtures["away_team"].replace(qualified_teams).map(normalize_team_name)
    return fixtures


def poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_pmf_vector(lam: float, max_goals: int) -> np.ndarray:
    return np.array([poisson_pmf(lam, k) for k in range(max_goals)], dtype=float)


def train_poisson_model(df: pd.DataFrame, iterations: int = 400, learning_rate: float = 0.02):
    home_teams = df["home_team"].tolist()
    away_teams = df["away_team"].tolist()
    home_goals = df["home_goals"].astype(float).values
    away_goals = df["away_goals"].astype(float).values

    unique_teams = sorted(set(home_teams) | set(away_teams))
    team_index = {team: idx for idx, team in enumerate(unique_teams)}
    n_teams = len(unique_teams)

    home_idx = np.array([team_index[t] for t in home_teams], dtype=int)
    away_idx = np.array([team_index[t] for t in away_teams], dtype=int)

    mean_home = home_goals.mean()
    mean_away = away_goals.mean()
    home_adv = float(np.log(mean_home / mean_away))

    attack = np.zeros(n_teams, dtype=float)
    defense = np.zeros(n_teams, dtype=float)

    # Simple initial values based on average goals.
    home_scored = np.bincount(home_idx, weights=home_goals, minlength=n_teams)
    away_scored = np.bincount(away_idx, weights=away_goals, minlength=n_teams)
    home_games = np.bincount(home_idx, minlength=n_teams)
    away_games = np.bincount(away_idx, minlength=n_teams)

    with np.errstate(divide='ignore', invalid='ignore'):
        attack = np.log((home_scored + away_scored + 0.5) / (home_games + away_games + 0.5))
        defense = np.log((home_games + away_games + 0.5) / (home_scored + away_scored + 0.5))
    attack = np.nan_to_num(attack)
    defense = np.nan_to_num(defense)
    attack -= attack.mean()
    defense -= defense.mean()

    sum_penalty = 0.5
    param_penalty = 1e-4
    n_matches = len(df)

    for iteration in range(1, iterations + 1):
        eta_home = home_adv + attack[home_idx] + defense[away_idx]
        eta_away = attack[away_idx] + defense[home_idx]
        lambda_home = np.exp(eta_home)
        lambda_away = np.exp(eta_away)

        res_home = lambda_home - home_goals
        res_away = lambda_away - away_goals

        grad_home_adv = res_home.sum()
        grad_attack = np.bincount(home_idx, weights=res_home, minlength=n_teams) + np.bincount(away_idx, weights=res_away, minlength=n_teams)
        grad_defense = np.bincount(away_idx, weights=res_home, minlength=n_teams) + np.bincount(home_idx, weights=res_away, minlength=n_teams)

        sum_attack = attack.sum()
        sum_defense = defense.sum()
        grad_attack += 2 * sum_penalty * sum_attack + 2 * param_penalty * attack
        grad_defense += 2 * sum_penalty * sum_defense + 2 * param_penalty * defense

        home_adv -= learning_rate * grad_home_adv / n_matches
        attack -= learning_rate * grad_attack / n_matches
        defense -= learning_rate * grad_defense / n_matches

        if iteration % 50 == 0 or iteration == 1 or iteration == iterations:
            nll = np.sum(lambda_home - home_goals * eta_home + lambda_away - away_goals * eta_away)
            nll += sum_penalty * (sum_attack ** 2 + sum_defense ** 2)
            nll += param_penalty * (np.sum(attack ** 2) + np.sum(defense ** 2))
            print(f"iteration={iteration:3d} nll={nll:.3f} home_adv={home_adv:.4f}")

    return {
        "home_adv": home_adv,
        "attack": attack,
        "defense": defense,
        "team_index": team_index,
        "teams": unique_teams,
    }


def predict_fixture(model: dict, home_team: str, away_team: str):
    team_idx = model["team_index"]
    attack = model["attack"]
    defense = model["defense"]
    home_adv = model["home_adv"]

    home_idx = team_idx.get(home_team, None)
    away_idx = team_idx.get(away_team, None)

    if home_idx is None or away_idx is None:
        missing = [t for t, idx in [(home_team, home_idx), (away_team, away_idx)] if idx is None]
        print(f"WARNING: missing team parameters for {missing}. Using average strength for them.")
        home_attack = attack.mean() if home_idx is None else attack[home_idx]
        away_attack = attack.mean() if away_idx is None else attack[away_idx]
        home_defense = defense.mean() if home_idx is None else defense[home_idx]
        away_defense = defense.mean() if away_idx is None else defense[away_idx]
    else:
        home_attack = attack[home_idx]
        away_attack = attack[away_idx]
        home_defense = defense[home_idx]
        away_defense = defense[away_idx]

    exp_home = float(np.exp(home_adv + home_attack + away_defense))
    exp_away = float(np.exp(away_attack + home_defense))

    max_goals = 8
    pmf_home = poisson_pmf_vector(exp_home, max_goals)
    pmf_away = poisson_pmf_vector(exp_away, max_goals)
    joint = np.outer(pmf_home, pmf_away)

    best_h, best_a = np.unravel_index(np.argmax(joint), joint.shape)
    prob_home = float(np.triu(joint, 1).sum())
    prob_draw = float(np.trace(joint))
    prob_away = float(np.tril(joint, -1).sum())

    return {
        "expected_home_goals": exp_home,
        "expected_away_goals": exp_away,
        "predicted_home_goals": int(best_h),
        "predicted_away_goals": int(best_a),
        "prob_home_win": prob_home,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away,
    }


def main():
    national_history = load_nat_history("tm_national_games.csv")
    fixtures = prepare_fixtures("group_fixtures.csv")

    print(f"Training on {len(national_history)} national team matches.")
    model = train_poisson_model(national_history, iterations=400, learning_rate=0.03)

    predictions = []
    for _, row in fixtures.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        prediction = predict_fixture(model, home, away)
        predictions.append({
            "match_id": row["match_id"],
            "group": row["group"],
            "home_team": home,
            "away_team": away,
            **prediction,
            "forecast_score": f"{prediction['predicted_home_goals']} - {prediction['predicted_away_goals']}",
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv("group_fixtures_predictions.csv", index=False)
    print("Saved group fixture predictions to group_fixtures_predictions.csv")
    print(predictions_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
