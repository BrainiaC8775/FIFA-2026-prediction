import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor

# =================================
# DATA LOAD AND PREP  
# =================================
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


def dc_likelihood_full(params, data, n_teams, rho, decay_lambda, reg, use_elo, use_form, use_intl):
    
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
        attack[hi]
        - defense[ai]
        + base_home_adv
    )
    log_lambda_away = (
        attack[ai] - defense[hi]
    )

    if use_elo:
        log_lambda_home += gamma * elo_diff
        log_lambda_away += gamma * elo_diff

    if use_form:
        log_lambda_home += (
            theta1 * data['hf_att'] +
            theta2 * data['af_def']    
        )
        log_lambda_away += (
            theta3 * data['af_att'] +
            theta4 * data['hf_def']
        )

    if use_intl:
        log_lambda_home += delta_intl * is_intl
        log_lambda_away += delta_intl * is_intl
    
    
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

def get_team_elo(df):
    # use latest Elo per team
    df_sorted = df.sort_values("date")
    
    team_elo = {}
    
    for _, row in df_sorted.iterrows():
        team_elo[row['home_team']] = row['home_elo']
        team_elo[row['away_team']] = row['away_elo']
    
    return team_elo
    
def elo_to_strength(elo, base=1500):
    return (elo - base) / 400.0

def fit_full_model(
    df,
    rho=0.0,
    decay_lambda=0.0005,
    reg=0.0001,
    use_elo=True,
    use_form=True,
    use_intl=True
):
    
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
    #init_params = np.zeros(n_params)
    team_elo = get_team_elo(df)
    attack_init = np.zeros(n_teams)
    defense_init = np.zeros(n_teams)
    for team, idx in team_to_idx.items():
        elo = team_elo.get(team, 1500)
        strength = elo_to_strength(elo)

        attack_init[idx] = strength
        defense_init[idx] = -0.5 * strength  # defensive strength (tunable)

    init_params = np.concatenate([
        attack_init,
        defense_init,
        np.zeros(7)  # other params
    ])
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
        args=(data, n_teams, rho, decay_lambda, reg, use_elo, use_form, use_intl),
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

# =========================================
# STEP 1: Generate base lambdas (DC model)
# =========================================

def generate_base_lambdas(df, params, team_to_idx):
    
    n_teams = len(team_to_idx)
    
    attack = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[-1]
    
    home_idx = df['home_team'].map(team_to_idx).values
    away_idx = df['away_team'].map(team_to_idx).values
    
    log_lambda_home = (
        attack[home_idx]
        - defense[away_idx]
        + home_adv
    )
    
    log_lambda_away = (
        attack[away_idx]
        - defense[home_idx]
    )
    
    log_lambda_home = np.clip(log_lambda_home, -5, 5)
    log_lambda_away = np.clip(log_lambda_away, -5, 5)
    
    return np.exp(log_lambda_home), np.exp(log_lambda_away)


# =========================================
# STEP 2: Build ML dataset
# =========================================

def build_ml_dataset(df, lambda_home, lambda_away):
    
    X = pd.DataFrame({
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "elo_diff": df["home_elo"] - df["away_elo"],
        "goal_diff_base": lambda_home - lambda_away,
        "is_intl": df["is_intl"],
        
        # OPTIONAL (enable later safely)
        "home_team": df["home_team"],
        "away_team": df["away_team"]
    })
    
    y_home_delta = df["home_goals"] - lambda_home
    y_away_delta = df["away_goals"] - lambda_away
    
    return X, y_home_delta, y_away_delta


# =========================================
# STEP 3: Train CatBoost models
# =========================================

def train_catboost_models(X, y_home_delta, y_away_delta):
    
    cat_features = ["home_team", "away_team"]
    
    model_home = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='RMSE',
        verbose=0
    )
    
    model_away = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='RMSE',
        verbose=0
    )
    
    model_home.fit(X, y_home_delta, cat_features=cat_features)
    model_away.fit(X, y_away_delta, cat_features=cat_features)
    
    return model_home, model_away


# =========================================
# STEP 4: Apply hybrid correction
# =========================================

def apply_hybrid_model(df, params, team_to_idx, model_home, model_away):
    
    lambda_home_base, lambda_away_base = generate_base_lambdas(
        df, params, team_to_idx
    )
    
    X, _, _ = build_ml_dataset(df, lambda_home_base, lambda_away_base)
    
    delta_home = model_home.predict(X)
    delta_away = model_away.predict(X)
    
    # Clip corrections (critical)
    delta_home = np.clip(delta_home, -2, 2)
    delta_away = np.clip(delta_away, -2, 2)
    
    # Multiplicative correction (stable)
    lambda_home = lambda_home_base * np.exp(delta_home)
    lambda_away = lambda_away_base * np.exp(delta_away)
    
    # Final safety clip
    lambda_home = np.clip(lambda_home, 0.05, 5)
    lambda_away = np.clip(lambda_away, 0.05, 5)
    
    return lambda_home, lambda_away


# =========================================
# STEP 5: Poisson probability matrix
# =========================================

def poisson_matrix(lambda_home, lambda_away, max_goals=8):
    
    x = np.arange(max_goals + 1)
    y = np.arange(max_goals + 1)
    
    log_p_home = (
        -lambda_home[:, None]
        + x * np.log(lambda_home[:, None])
        - gammaln(x + 1)
    )
    
    log_p_away = (
        -lambda_away[:, None]
        + y * np.log(lambda_away[:, None])
        - gammaln(y + 1)
    )
    
    log_p = log_p_home[:, :, None] + log_p_away[:, None, :]
    p = np.exp(log_p)
    
    p /= p.sum(axis=(1, 2), keepdims=True)
    
    return p


# =========================================
# STEP 6: Evaluation
# =========================================

def evaluate_hybrid(df, lambda_home, lambda_away, max_goals=8):
    
    probs = poisson_matrix(lambda_home, lambda_away, max_goals)
    
    log_loss = 0
    correct = 0
    exact = 0
    
    for i in range(len(df)):
        hg = df.iloc[i]["home_goals"]
        ag = df.iloc[i]["away_goals"]
        
        p = probs[i]
        
        prob = p[hg, ag] if hg <= max_goals and ag <= max_goals else 1e-10
        log_loss += -np.log(prob + 1e-12)
        
        pred_h, pred_a = np.unravel_index(np.argmax(p), p.shape)
        
        if pred_h == hg and pred_a == ag:
            exact += 1
        
        if (pred_h > pred_a and hg > ag) or \
           (pred_h == pred_a and hg == ag) or \
           (pred_h < pred_a and hg < ag):
            correct += 1
    
    n = len(df)
    
    return {
        "log_loss": log_loss / n,
        "accuracy": correct / n,
        "exact_score_pct": exact / n
    }


# =========================================
# STEP 7: FULL PIPELINE
# =========================================

def run_hybrid_pipeline(train_df, test_df):
    
    print("Training DC model...")
    result, teams, team_to_idx = fit_full_model(train_df)
    params = result.x
    
    print("Generating train lambdas...")
    lambda_home_train, lambda_away_train = generate_base_lambdas(
        train_df, params, team_to_idx
    )
    
    print("Building ML dataset...")
    X_train, y_home_delta, y_away_delta = build_ml_dataset(
        train_df, lambda_home_train, lambda_away_train
    )
    
    print("Training CatBoost models...")
    model_home, model_away = train_catboost_models(
        X_train, y_home_delta, y_away_delta
    )
    
    print("Applying hybrid model...")
    lambda_home_test, lambda_away_test = apply_hybrid_model(
        test_df, params, team_to_idx, model_home, model_away
    )
    
    print("Evaluating...")
    metrics = evaluate_hybrid(
        test_df, lambda_home_test, lambda_away_test
    )
    
    return metrics

# =========================================
# Execute pipeline
# =========================================
metrics = run_hybrid_pipeline(train_intl, test_intl)
print(metrics)