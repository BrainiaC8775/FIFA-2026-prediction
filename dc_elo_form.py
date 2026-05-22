import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from itertools import product

train_club=pd.read_csv('train_club_df.csv')
test_club=pd.read_csv('test_club_df.csv')
train_intl=pd.read_csv('train_intl_df.csv')
test_intl=pd.read_csv('test_intl_df.csv')

train_teams_c = set(train_club['home_team']).union(set(train_club['away_team']))

test_club = test_club[
    test_club['home_team'].isin(train_teams_c) &
    test_club['away_team'].isin(train_teams_c)
]

train_teams_i = set(train_intl['home_team']).union(set(train_intl['away_team']))

test_intl = test_intl[
    test_intl['home_team'].isin(train_teams_i) &
    test_intl['away_team'].isin(train_teams_i)
]
def add_form_features(df, window=5):
    
    df = df.sort_values('date').copy()
    
    # Initialize columns
    df['home_form_goals_for'] = 0.0
    df['home_form_goals_against'] = 0.0
    df['away_form_goals_for'] = 0.0
    df['away_form_goals_against'] = 0.0
    
    team_history = {}
    
    for idx, row in df.iterrows():
        
        home = row['home_team']
        away = row['away_team']
        
        team_history.setdefault(home, [])
        team_history.setdefault(away, [])
        
        # --- HOME ---
        home_hist = team_history[home][-window:]
        if home_hist:
            df.at[idx, 'home_form_goals_for'] = np.mean([m[0] for m in home_hist])
            df.at[idx, 'home_form_goals_against'] = np.mean([m[1] for m in home_hist])
        
        # --- AWAY ---
        away_hist = team_history[away][-window:]
        if away_hist:
            df.at[idx, 'away_form_goals_for'] = np.mean([m[0] for m in away_hist])
            df.at[idx, 'away_form_goals_against'] = np.mean([m[1] for m in away_hist])
        
        # Update AFTER computing features (no leakage)
        team_history[home].append((row['home_goals'], row['away_goals']))
        team_history[away].append((row['away_goals'], row['home_goals']))
    
    return df

train_intl = add_form_features(train_intl)
test_intl  = add_form_features(test_intl)

train_club = add_form_features(train_club)
test_club  = add_form_features(test_club)

def encode_teams(df):
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    team_to_idx = {t: i for i, t in enumerate(teams)}
    
    df['home_idx'] = df['home_team'].map(team_to_idx)
    df['away_idx'] = df['away_team'].map(team_to_idx)
    
    return df, teams, team_to_idx

def build_features(df):
    
    return {
        "home_idx": df['home_idx'].values,
        "away_idx": df['away_idx'].values,
        "home_goals": df['home_goals'].values,
        "away_goals": df['away_goals'].values,
        "elo_diff": (df['home_elo'] - df['away_elo']).values,
        "hf_att": df['home_form_goals_for'].values,
        "hf_def": df['home_form_goals_against'].values,
        "af_att": df['away_form_goals_for'].values,
        "af_def": df['away_form_goals_against'].values,
    }

def dc_likelihood_full(params, data, n_teams, rho, decay_lambda, reg):
    
    # -----------------------------
    # UNPACK PARAMETERS
    # -----------------------------
    attack = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    
    gamma = params[-7]
    delta_intl = params[-6]
    
    theta1, theta2, theta3, theta4 = params[-5:-1]
    base_home_adv = params[-1]
    
    # -----------------------------
    # IDENTIFIABILITY CONSTRAINTS
    # -----------------------------
    attack = attack - np.mean(attack)
    defense = defense - np.mean(defense)
    
    # -----------------------------
    # INDEXING
    # -----------------------------
    hi = data['home_idx']
    ai = data['away_idx']
    
    x = data['home_goals']
    y = data['away_goals']
    
    # -----------------------------
    # FEATURES
    # -----------------------------
    elo_diff = data['elo_diff']
    is_intl = data['is_intl']
    
    # Time decay
    weights = np.exp(-decay_lambda * data['days_since'].astype(float))
    
    # -----------------------------
    # LOG LAMBDA (CLIPPED)
    # -----------------------------
    log_lambda_home = (
        attack[hi] - defense[ai] +
        gamma * elo_diff +
        theta1 * data['hf_att'] +
        theta2 * data['af_def'] +
        base_home_adv +
        delta_intl * is_intl
    )
    
    log_lambda_away = (
        attack[ai] - defense[hi] -
        gamma * elo_diff +
        theta3 * data['af_att'] +
        theta4 * data['hf_def']
    )
    
    # 🔴 CRITICAL: prevent overflow
    log_lambda_home = np.clip(log_lambda_home, -10, 10)
    log_lambda_away = np.clip(log_lambda_away, -10, 10)
    
    lambda_home = np.exp(log_lambda_home)
    lambda_away = np.exp(log_lambda_away)
    
    # Extra safety
    eps = 1e-10
    lambda_home = np.clip(lambda_home, eps, None)
    lambda_away = np.clip(lambda_away, eps, None)
    
    # -----------------------------
    # BASE LOG LIKELIHOOD (POISSON)
    # -----------------------------
    log_p = (
        -lambda_home + x * np.log(lambda_home) - gammaln(x + 1) +
        -lambda_away + y * np.log(lambda_away) - gammaln(y + 1)
    )
    
    # -----------------------------
    # SAFE DIXON-COLES CORRECTION
    # -----------------------------
    if rho != 0:
        mask_00 = (x == 0) & (y == 0)
        mask_01 = (x == 0) & (y == 1)
        mask_10 = (x == 1) & (y == 0)
        mask_11 = (x == 1) & (y == 1)
        
        val_00 = 1 - lambda_home[mask_00] * lambda_away[mask_00] * rho
        val_01 = 1 + lambda_home[mask_01] * rho
        val_10 = 1 + lambda_away[mask_10] * rho
        val_11 = 1 - rho
        
        # 🔴 CRITICAL: prevent log of negative/zero
        val_00 = np.clip(val_00, 1e-10, None)
        val_01 = np.clip(val_01, 1e-10, None)
        val_10 = np.clip(val_10, 1e-10, None)
        val_11 = max(val_11, 1e-10)
        
        log_p[mask_00] += np.log(val_00)
        log_p[mask_01] += np.log(val_01)
        log_p[mask_10] += np.log(val_10)
        log_p[mask_11] += np.log(val_11)
    
    # -----------------------------
    # APPLY TIME DECAY
    # -----------------------------
    weighted_log_p = weights * log_p
    
    # -----------------------------
    # REGULARIZATION (STABILITY)
    # -----------------------------
    reg_term = reg * np.sum(params**2)
    
    # -----------------------------
    # FINAL LOSS
    # -----------------------------
    return -np.sum(weighted_log_p) + reg_term

def add_time_decay_feature(df):
    df = df.copy()
    
    # 🔴 FORCE datetime conversion (fix)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop bad rows if any
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').copy()
    
    # reference = latest date in training set
    ref_date = df['date'].max()
    
    df['days_since'] = (ref_date - df['date']).dt.days
    
    return df

def fit_full_model(df, rho=0.0, decay_lambda=0.0005, reg=0.0001):
    
    # Encode teams
    df, teams, team_to_idx = encode_teams(df)
    
    # Time decay feature
    df = add_time_decay_feature(df)
    
    # Build data dict
    data = {
        "home_idx": df['home_idx'].values,
        "away_idx": df['away_idx'].values,
        "home_goals": df['home_goals'].values,
        "away_goals": df['away_goals'].values,
        "elo_diff": (df['home_elo'] - df['away_elo']).values,  # 🔴 scaled (important)
        "hf_att": df['home_form_goals_for'].values,
        "hf_def": df['home_form_goals_against'].values,
        "af_att": df['away_form_goals_for'].values,
        "af_def": df['away_form_goals_against'].values,
        "is_intl": df['is_intl'].values,
        "days_since": df['days_since'].values
    }
    
    n_teams = len(teams)
    
    # Parameters:
    # attack (n)
    # defense (n)
    # gamma (1)
    # delta_intl (1)
    # theta1–4 (4)
    # home_adv (1)
    
    n_params = 2 * n_teams + 7
    
    # 🔴 Clean initialization (no warm start)
    init_params = np.zeros(n_params)
    init_params[-1] = 0.1  # home advantage
    
    # 🔴 Optional but HIGHLY recommended bounds (stability)
    bounds = (
        [(None, None)] * (2 * n_teams) +   # attack, defense
        [(-1, 1),   # gamma (elo effect)
         (-1, 1),   # delta_intl
         (-1, 1),   # theta1
         (-1, 1),   # theta2
         (-1, 1),   # theta3
         (-1, 1),   # theta4
         (-2, 2)]   # home_adv
    )
    
    result = minimize(
        dc_likelihood_full,
        init_params,
        args=(data, n_teams, rho, decay_lambda, reg),  # 🔴 make sure reg is passed
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 500,
            'ftol': 1e-6,
            'gtol': 1e-5
        }
    )
    
    return result, teams, team_to_idx


def poisson_matrix(lambda_home, lambda_away, rho=0.0, max_goals=10):
    
    eps = 1e-10
    lambda_home = np.clip(lambda_home, eps, 1e4)
    lambda_away = np.clip(lambda_away, eps, 1e4)
    
    goals = np.arange(0, max_goals + 1)
    
    # -----------------------------
    # Poisson PMFs (log-space)
    # -----------------------------
    log_p_home = -lambda_home + goals * np.log(lambda_home) - gammaln(goals + 1)
    log_p_away = -lambda_away + goals * np.log(lambda_away) - gammaln(goals + 1)
    
    p_home = np.exp(log_p_home)
    p_away = np.exp(log_p_away)
    
    probs = np.outer(p_home, p_away)
    
    # -----------------------------
    # Dixon-Coles correction (CORRECTED)
    # -----------------------------
    if rho != 0:
        
        # Create goal grid
        G = max_goals + 1
        i, j = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
        
        mask_00 = (i == 0) & (j == 0)
        mask_01 = (i == 0) & (j == 1)
        mask_10 = (i == 1) & (j == 0)
        mask_11 = (i == 1) & (j == 1)
        
        val_00 = 1 - lambda_home * lambda_away * rho
        val_01 = 1 + lambda_home * rho
        val_10 = 1 + lambda_away * rho
        val_11 = 1 - rho
        
        # clip for safety
        val_00 = max(val_00, 1e-10)
        val_01 = max(val_01, 1e-10)
        val_10 = max(val_10, 1e-10)
        val_11 = max(val_11, 1e-10)
        
        probs[mask_00] *= val_00
        probs[mask_01] *= val_01
        probs[mask_10] *= val_10
        probs[mask_11] *= val_11
    
    # -----------------------------
    # Normalize
    # -----------------------------
    total = probs.sum()
    
    if total <= 0 or not np.isfinite(total):
        probs = np.ones_like(probs)
        total = probs.sum()
    
    probs /= total
    
    return probs

def outcome_probs(probs):
    home_win = np.tril(probs, -1).sum()
    draw = np.trace(probs)
    away_win = np.triu(probs, 1).sum()
    return np.array([home_win, draw, away_win])

def rps(pred_probs, actual_outcome):
    """
    pred_probs: [P(H), P(D), P(A)]
    actual_outcome: one-hot [1,0,0] etc.
    """
    cum_pred = np.cumsum(pred_probs)
    cum_actual = np.cumsum(actual_outcome)
    
    return np.sum((cum_pred - cum_actual) ** 2) / 2

def log_loss(pred_probs, actual_outcome, eps=1e-15):
    pred_probs = np.clip(pred_probs, eps, 1 - eps)
    return -np.sum(actual_outcome * np.log(pred_probs))

def get_actual_outcome(home_goals, away_goals):
    if home_goals > away_goals:
        return np.array([1, 0, 0])
    elif home_goals < away_goals:
        return np.array([0, 0, 1])
    else:
        return np.array([0, 1, 0])


def predict_score(probs):
    return np.unravel_index(np.argmax(probs), probs.shape)


def predict_outcome(pred_probs):
    return np.argmax(pred_probs)  # 0=H,1=D,2=A

def evaluate_model(df, params, team_to_idx, rho=0.0, max_goals=10):
    
    log_losses = []
    rps_scores = []
    
    correct_outcomes = 0
    correct_scores = 0
    
    total = len(df)
    
    for _, row in df.iterrows():
        
        # --- compute lambdas (reuse your model logic) ---
        i = team_to_idx[row['home_team']]
        j = team_to_idx[row['away_team']]
        
        n_teams = len(team_to_idx)
        
        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        
        gamma = params[-7]
        delta_intl = params[-6]
        theta1, theta2, theta3, theta4 = params[-5:-1]
        home_adv = params[-1]
        
        elo_diff = (row['home_elo'] - row['away_elo'])
        
        log_lambda_home = (
            attack[i] - defense[j] +
            gamma * elo_diff +
            theta1 * row['home_form_goals_for'] +
            theta2 * row['away_form_goals_against'] +
            home_adv +
            delta_intl * row['is_intl']
        )
            
        log_lambda_away = (
            attack[j] - defense[i] -
            gamma * elo_diff +
            theta3 * row['away_form_goals_for'] +
            theta4 * row['home_form_goals_against']
        )
                
        # SAME clipping as training
        log_lambda_home = np.clip(log_lambda_home, -10, 10)
        log_lambda_away = np.clip(log_lambda_away, -10, 10)
                
        lambda_home = np.exp(log_lambda_home)
        lambda_away = np.exp(log_lambda_away)
        
        # --- probabilities ---
        probs = poisson_matrix(lambda_home, lambda_away, rho, max_goals)
        pred_probs = outcome_probs(probs)
        
        actual_out = get_actual_outcome(row['home_goals'], row['away_goals'])
        
        # --- metrics ---
        log_losses.append(log_loss(pred_probs, actual_out))
        rps_scores.append(rps(pred_probs, actual_out))
        
        # --- accuracy ---
        if predict_outcome(pred_probs) == np.argmax(actual_out):
            correct_outcomes += 1
        
        # --- exact score ---
        pred_score = predict_score(probs)
        if pred_score == (row['home_goals'], row['away_goals']):
            correct_scores += 1
    
    return {
        "log_loss": np.mean(log_losses),
        "rps": np.mean(rps_scores),
        "accuracy": correct_outcomes / total,
        "exact_score_pct": correct_scores / total
    }

def run_experiments(
    train_df,
    test_df,
    decay_grid,
    rho_grid,
    reg=0.0001,
    max_goals=8
):
    
    results = []
    
    for decay_lambda in decay_grid:
        best_params = None  # warm start
        for rho in rho_grid:
            print(f"\nRunning: decay={decay_lambda}, rho={rho}")
            try:
                result, teams, team_to_idx = fit_full_model(
                    train_df,
                    rho=rho,
                    decay_lambda=decay_lambda,
                    reg=reg,
                    init_params=best_params   # 🔴 warm start
                )
            
                best_params = result.x  # update warm start
            
                if not result.success:
                    print("⚠️ Optimization did not converge")
            
                params = result.x
            
                # Evaluate
                metrics = evaluate_model(
                    test_df,
                    params=params,
                    team_to_idx=team_to_idx,
                    rho=rho,
                    max_goals=max_goals
                )
            
                # Store results
                results.append({
                    "decay_lambda": decay_lambda,
                    "rho": rho,
                    "log_loss": metrics["log_loss"],
                    "rps": metrics["rps"],
                    "accuracy": metrics["accuracy"],
                    "exact_score_pct": metrics["exact_score_pct"],
                    "converged": result.success
                })
        
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append({
                    "decay_lambda": decay_lambda,
                    "rho": rho,
                    "log_loss": None,
                    "rps": None,
                    "accuracy": None,
                    "exact_score_pct": None,
                    "converged": False
            })
    
    return pd.DataFrame(results)

### Main execution
### training  
result, teams, team_to_idx = fit_full_model(train_intl)
print(result.success, result.message)
params = result.x

### evaluation
metrics = evaluate_model(
    test_intl,
    params=params,
    team_to_idx=team_to_idx,
    rho=0.0,
    max_goals=8   # tune this later
)

print(metrics)

## Testing different decay and rho values
#decay_grid = [0.0002, 0.0005, 0.0008]
#rho_grid = [0.0]

#results_df = run_experiments(
#    train_intl,
#    test_intl,
#    decay_grid,
#    rho_grid,
#    reg=0.0001
#)

#print(results_df.sort_values("log_loss"))
#print(results_df.head(10))