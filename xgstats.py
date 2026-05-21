import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

train_xg=pd.read_csv('train_xg_dataset.csv')
test_xg=pd.read_csv('test_xg_dataset.csv')

#print(train_xg[['home_xGF', 'away_xGF']].describe())
#print(train_xg[['home_goals', 'away_goals']].describe())

train_xg['total_xG'] = train_xg['home_xGF'] + train_xg['away_xGF']
train_xg['total_goals'] = train_xg['home_goals'] + train_xg['away_goals']

#print(train_xg[['total_xG', 'total_goals']].corr())

# Calculate rolling avgs of xG on train dataset
train_xg = train_xg.sort_values("date")

train_xg["home_xGF_roll"] = train_xg.groupby("home_team")["home_xGF"]\
    .transform(lambda x: x.rolling(5, min_periods=1).mean())

train_xg["away_xGF_roll"] = train_xg.groupby("away_team")["away_xGF"]\
    .transform(lambda x: x.rolling(5, min_periods=1).mean())

train_xg['home_xGF_roll'] = train_xg['home_xGF_roll'].clip(0.2, 4)
train_xg['away_xGF_roll'] = train_xg['away_xGF_roll'].clip(0.2, 4)

# Calculate rolling avgs of xG on test dataset
test_xg = test_xg.sort_values("date")

test_xg["home_xGF_roll"] = test_xg.groupby("home_team")["home_xGF"]\
    .transform(lambda x: x.rolling(5, min_periods=1).mean())

test_xg["away_xGF_roll"] = test_xg.groupby("away_team")["away_xGF"]\
    .transform(lambda x: x.rolling(5, min_periods=1).mean())

test_xg['home_xGF_roll'] = test_xg['home_xGF_roll'].clip(0.2, 4)
test_xg['away_xGF_roll'] = test_xg['away_xGF_roll'].clip(0.2, 4)

# prepare unique teams
teams = pd.concat([train_xg['home_team'], train_xg['away_team']]).unique()
team_to_idx = {team: i for i, team in enumerate(teams)}
n_teams = len(teams)

# create lambda using Rolling xG
train_xg['lambda_home'] = 1.05 * train_xg['home_xGF_roll']
train_xg['lambda_away'] = 1.05 * train_xg['away_xGF_roll']

# time decaying weights
train_xg['date'] = pd.to_datetime(train_xg['date'])
train_xg['days_ago'] = (train_xg['date'].max() - train_xg['date']).dt.days
train_xg['weight'] = np.exp(-0.002 * train_xg['days_ago'])

train_xg['home_idx'] = train_xg['home_team'].map(team_to_idx)
train_xg['away_idx'] = train_xg['away_team'].map(team_to_idx)

train_sample = train_xg.sample(1000, random_state=42)

# Dixon-Coles correction function
def dc_correction(x, y, lambda_h, lambda_a, rho):
    if x == 0 and y == 0:
        return 1 - (lambda_h * lambda_a * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_h * rho)
    elif x == 1 and y == 0:
        return 1 + (lambda_a * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1

def log_likelihood(params, df):

    alpha, home_adv, rho = params

    ll = 0

    for row in df.itertuples():

        lambda_h = np.exp(home_adv) * alpha * row.home_xGF_roll
        lambda_a = alpha * row.away_xGF_roll

        x = row.home_goals
        y = row.away_goals

        p = poisson.pmf(x, lambda_h) * poisson.pmf(y, lambda_a)
        p *= dc_correction(x, y, lambda_h, lambda_a, rho)

        if p > 0:
            ll += row.weight * np.log(p)

    return -ll   # minimize

init_params = [1.0, 0.2, -0.05]  # alpha, home_adv, rho

bounds = [
    (0.5, 2.0),     # alpha
    (-0.5, 1.0),    # home_adv
    (-0.2, 0.2)     # rho
]

result = minimize(
    log_likelihood,
    init_params,
    args=(train_xg,),
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 100}
)

print("Success:", result.success)
print("Params:", result.x)


params = result.x
home_adv = params[-2]
rho = params[-1]

print("Home advantage:", home_adv)
print("Rho:", rho)

def predict_match(home_xg, away_xg, params):

    alpha, home_adv, rho = params

    lambda_h = np.exp(home_adv) * alpha * home_xg
    lambda_a = alpha * away_xg

    max_goals = 10
    probs = np.zeros((max_goals+1, max_goals+1))

    for x in range(max_goals+1):
        for y in range(max_goals+1):

            p = poisson.pmf(x, lambda_h) * poisson.pmf(y, lambda_a)
            p *= dc_correction(x, y, lambda_h, lambda_a, rho)

            probs[x, y] = p

    return probs / probs.sum()

def log_loss_test(df, params):

    total_ll = 0

    for row in df.itertuples():

        probs = predict_match(
            row.home_xGF_roll,
            row.away_xGF_roll,
            params
        )

        x = min(row.home_goals, 10)
        y = min(row.away_goals, 10)

        p = probs[x, y]

        if p > 0:
            total_ll += np.log(p)

    return -total_ll / len(df)


print("Test Log Loss:", log_loss_test(test_xg, result.x))
