import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.special import gammaln
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
train_club = pd.read_csv('train_club_df.csv')
test_club  = pd.read_csv('test_club_df.csv')
train_intl = pd.read_csv('train_intl_df.csv')
test_intl  = pd.read_csv('test_intl_df.csv')

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

# ─────────────────────────────────────────────────────────────
# FORM FEATURES (rolling window, no leakage)
# ─────────────────────────────────────────────────────────────
def add_form_features(df, window=5, seed=None):
    """
    Rolling pre-match features — strictly no leakage.
    History tuple per team: (gf, ga, pts, elo)

    seed: dict with keys 'all', 'home', 'away', 'last_date' from a prior call.
      Training history slides out naturally after `window` test matches per team.
    """
    import copy
    df = df.sort_values('date').copy()
    df['date'] = pd.to_datetime(df['date'])

    for col in [
        'home_form_goals_for',  'home_form_goals_against',
        'away_form_goals_for',  'away_form_goals_against',
        'home_form_points',     'away_form_points',
        'home_clean_sheet_rate','away_clean_sheet_rate',
        'home_elo_momentum',    'away_elo_momentum',
        # venue-split: home team's home-only form / away team's away-only form
        'home_form_gf_h', 'home_form_ga_h',
        'away_form_gf_a', 'away_form_ga_a',
        # days since last match (any venue)
        'home_days_rest', 'away_days_rest',
    ]:
        df[col] = 0.0

    if seed:
        all_h  = {t: copy.copy(h[-window:]) for t, h in seed['all'].items()}
        home_h = {t: copy.copy(h[-window:]) for t, h in seed['home'].items()}
        away_h = {t: copy.copy(h[-window:]) for t, h in seed['away'].items()}
        last_d = dict(seed['last_date'])
    else:
        all_h = {}; home_h = {}; away_h = {}; last_d = {}

    DEFAULT_REST = 30  # days assumed when a team has no prior match on record

    for idx, row in df.iterrows():
        home, away   = row['home_team'], row['away_team']
        hg, ag       = float(row['home_goals']), float(row['away_goals'])
        h_elo, a_elo = row['home_elo'], row['away_elo']
        match_date   = row['date']

        for t in (home, away):
            all_h.setdefault(t, [])
            home_h.setdefault(t, [])
            away_h.setdefault(t, [])

        h_pts, a_pts = (3, 0) if hg > ag else (0, 3) if hg < ag else (1, 1)

        # ── days rest ────────────────────────────────────────────────
        df.at[idx, 'home_days_rest'] = (
            (match_date - last_d[home]).days if home in last_d else DEFAULT_REST
        )
        df.at[idx, 'away_days_rest'] = (
            (match_date - last_d[away]).days if away in last_d else DEFAULT_REST
        )

        # ── all-venue form ───────────────────────────────────────────
        hh = all_h[home][-window:]
        if hh:
            df.at[idx, 'home_form_goals_for']    = np.mean([m[0] for m in hh])
            df.at[idx, 'home_form_goals_against'] = np.mean([m[1] for m in hh])
            df.at[idx, 'home_form_points']        = np.mean([m[2] for m in hh])
            df.at[idx, 'home_clean_sheet_rate']   = np.mean([m[1] == 0 for m in hh])
            df.at[idx, 'home_elo_momentum']       = h_elo - hh[0][3]

        ah = all_h[away][-window:]
        if ah:
            df.at[idx, 'away_form_goals_for']    = np.mean([m[0] for m in ah])
            df.at[idx, 'away_form_goals_against'] = np.mean([m[1] for m in ah])
            df.at[idx, 'away_form_points']        = np.mean([m[2] for m in ah])
            df.at[idx, 'away_clean_sheet_rate']   = np.mean([m[1] == 0 for m in ah])
            df.at[idx, 'away_elo_momentum']       = a_elo - ah[0][3]

        # ── venue-split form ─────────────────────────────────────────
        hv = home_h[home][-window:]          # home team when playing at home
        if hv:
            df.at[idx, 'home_form_gf_h'] = np.mean([m[0] for m in hv])
            df.at[idx, 'home_form_ga_h'] = np.mean([m[1] for m in hv])

        av = away_h[away][-window:]          # away team when playing away
        if av:
            df.at[idx, 'away_form_gf_a'] = np.mean([m[0] for m in av])
            df.at[idx, 'away_form_ga_a'] = np.mean([m[1] for m in av])

        # ── update histories ─────────────────────────────────────────
        all_h[home].append((hg, ag, h_pts, h_elo))
        all_h[away].append((ag, hg, a_pts, a_elo))
        home_h[home].append((hg, ag, h_pts, h_elo))   # home team's home record
        away_h[away].append((ag, hg, a_pts, a_elo))   # away team's away record
        last_d[home] = match_date
        last_d[away] = match_date

    history = {'all': all_h, 'home': home_h, 'away': away_h, 'last_date': last_d}
    return df, history


def build_form_features(train_df, test_df, window=5):
    """
    Train form: cold-start (correct — no future data available).
    Test form: seeded with last `window` train results per team.
    Training entries slide out of the window after `window` test matches per team.
    """
    train_df, train_hist = add_form_features(train_df, window=window)
    test_df,  _          = add_form_features(test_df,  window=window, seed=train_hist)
    return train_df, test_df


train_intl, test_intl = build_form_features(train_intl, test_intl)
train_club, test_club = build_form_features(train_club, test_club)


# ─────────────────────────────────────────────────────────────
# STAGE 1: DC BASE MODEL
# Learns per-team attack/defense + home advantage.
# Keeps ELO and form out so CatBoost handles non-linear effects.
# ─────────────────────────────────────────────────────────────

def encode_teams(df):
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    team_to_idx = {t: i for i, t in enumerate(teams)}
    df = df.copy()
    df['home_idx'] = df['home_team'].map(team_to_idx)
    df['away_idx'] = df['away_team'].map(team_to_idx)
    return df, teams, team_to_idx

def add_time_decay(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').copy()
    df['days_since'] = (df['date'].max() - df['date']).dt.days
    return df

def get_team_elo(df):
    team_elo = {}
    for _, row in df.sort_values('date').iterrows():
        team_elo[row['home_team']] = row['home_elo']
        team_elo[row['away_team']] = row['away_elo']
    return team_elo

def elo_to_strength(elo, base=1500):
    return (elo - base) / 400.0

def dc_neg_log_likelihood(params, data, n_teams, decay_lambda, reg):
    attack   = params[:n_teams]        - np.mean(params[:n_teams])
    defense  = params[n_teams:2*n_teams] - np.mean(params[n_teams:2*n_teams])
    home_adv = params[-2]
    rho      = params[-1]   # Dixon-Coles low-score correlation

    hi, ai = data['home_idx'], data['away_idx']
    x,  y  = data['home_goals'], data['away_goals']
    w      = np.exp(-decay_lambda * data['days_since'].astype(float))

    log_lh = np.clip(attack[hi] - defense[ai] + home_adv, -10, 10)
    log_la = np.clip(attack[ai] - defense[hi],             -10, 10)
    lh = np.clip(np.exp(log_lh), 1e-10, None)
    la = np.clip(np.exp(log_la), 1e-10, None)

    log_p = (
        -lh + x * np.log(lh) - gammaln(x + 1) +
        -la + y * np.log(la) - gammaln(y + 1)
    )

    # Dixon-Coles τ correction — adjusts for the observed negative correlation
    # between home and away goals on low-score lines (0-0, 1-0, 0-1, 1-1).
    tau = np.ones(len(x))
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    tau[m00] = np.clip(1 - lh[m00] * la[m00] * rho, 1e-10, None)
    tau[m10] = np.clip(1 + la[m10] * rho,            1e-10, None)
    tau[m01] = np.clip(1 + lh[m01] * rho,            1e-10, None)
    tau[m11] = np.clip(1 - rho,                       1e-10, None)

    return -np.sum(w * (log_p + np.log(tau))) + reg * np.sum(params**2)

def fit_dc_base(df, decay_lambda=0.0005, reg=0.0001, maxiter=2000):
    df, teams, team_to_idx = encode_teams(df)
    df = add_time_decay(df)

    data = {k: df[k].values for k in ('home_idx', 'away_idx', 'home_goals', 'away_goals', 'days_since')}
    n = len(teams)

    team_elo = get_team_elo(df)
    atk  = np.array([elo_to_strength(team_elo.get(t, 1500)) for t in teams])
    defs = -0.5 * atk
    # params layout: [attack×n, defense×n, home_adv, rho]
    init   = np.concatenate([atk, defs, [0.1, -0.1]])
    bounds = [(None, None)] * (2 * n) + [(-2, 2), (-0.5, 0.5)]

    print(f"  DC: {n} teams, {len(df)} rows, maxiter={maxiter}")
    result = minimize(
        dc_neg_log_likelihood, init,
        args=(data, n, decay_lambda, reg),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    rho_fit = result.x[-1]
    print(f"  DC rho (τ correlation) = {rho_fit:.4f}")
    return result, teams, team_to_idx

def get_dc_log_lambdas(df, params, team_to_idx):
    n = len(team_to_idx)
    attack   = params[:n]    - np.mean(params[:n])
    defense  = params[n:2*n] - np.mean(params[n:2*n])
    home_adv = params[-2]    # rho is params[-1], not used for lambda

    hi = df['home_team'].map(team_to_idx).values
    ai = df['away_team'].map(team_to_idx).values

    log_lh = np.clip(attack[hi] - defense[ai] + home_adv, -10, 10)
    log_la = np.clip(attack[ai] - defense[hi],             -10, 10)
    return log_lh, log_la


# ─────────────────────────────────────────────────────────────
# STAGE 2: CATBOOST POISSON REGRESSION
#
# Key design:
#   - loss_function='Poisson' uses log link → predict() returns λ (expected count)
#   - DC log-lambdas included as features: CatBoost learns how much to trust/adjust them
#   - ELO, form, team IDs capture non-linear residuals the DC model misses
#   - Early stopping on 10% validation split prevents overfitting
# ─────────────────────────────────────────────────────────────

def build_features(df, log_lh, log_la):
    df = df.reset_index(drop=True)
    return pd.DataFrame({
        # DC base lambdas (Stage 1 signal)
        "log_lambda_home_dc":    log_lh,
        "log_lambda_away_dc":    log_la,
        # ELO
        "elo_diff":              (df["home_elo"] - df["away_elo"]).values,
        "home_elo":              df["home_elo"].values,
        "away_elo":              df["away_elo"].values,
        # All-venue form
        "home_form_gf":          df["home_form_goals_for"].values,
        "home_form_ga":          df["home_form_goals_against"].values,
        "away_form_gf":          df["away_form_goals_for"].values,
        "away_form_ga":          df["away_form_goals_against"].values,
        "home_form_pts":         df["home_form_points"].values,
        "away_form_pts":         df["away_form_points"].values,
        "home_cs_rate":          df["home_clean_sheet_rate"].values,
        "away_cs_rate":          df["away_clean_sheet_rate"].values,
        "home_elo_momentum":     df["home_elo_momentum"].values,
        "away_elo_momentum":     df["away_elo_momentum"].values,
        # Venue-split form (home team's home-only / away team's away-only)
        "home_form_gf_h":        df["home_form_gf_h"].values,
        "home_form_ga_h":        df["home_form_ga_h"].values,
        "away_form_gf_a":        df["away_form_gf_a"].values,
        "away_form_ga_a":        df["away_form_ga_a"].values,
        # Days since last match (fatigue / rest signal)
        "home_days_rest":        df["home_days_rest"].values,
        "away_days_rest":        df["away_days_rest"].values,
        # Cross attack-vs-defense matchup (all-venue and venue-specific)
        "home_atk_vs_away_def":  (df["home_form_goals_for"]  - df["away_form_goals_against"]).values,
        "away_atk_vs_home_def":  (df["away_form_goals_for"]  - df["home_form_goals_against"]).values,
        "home_atk_vs_away_def_v":(df["home_form_gf_h"] - df["away_form_ga_a"]).values,
        "away_atk_vs_home_def_v":(df["away_form_gf_a"] - df["home_form_ga_h"]).values,
        # Match context
        "is_intl":               df["is_intl"].values,
        "is_neutral":            df["is_neutral"].values,
        # Categorical team IDs
        "home_team":             df["home_team"].values,
        "away_team":             df["away_team"].values,
    })

def _cb_val_loss(model):
    return model.best_score_["validation"]["Poisson"]

def _make_cb_params(depth, lr):
    return dict(
        iterations=2000,
        depth=depth,
        learning_rate=lr,
        loss_function='Poisson',
        eval_metric='Poisson',
        early_stopping_rounds=75,
        use_best_model=True,
        l2_leaf_reg=5,
        min_data_in_leaf=10,
        verbose=0,
        random_seed=42,
    )

def train_catboost_poisson(X_train, y_home, y_away, tune=True, depth=5, lr=0.05):
    """
    tune=True  → grid search over depth × lr (use on smaller datasets like intl).
    tune=False → train directly with supplied depth/lr (use on large datasets).
    """
    cat_features = ["home_team", "away_team"]
    X_tr, X_val, yh_tr, yh_val, ya_tr, ya_val = train_test_split(
        X_train, y_home, y_away, test_size=0.1, random_state=42
    )

    if tune:
        depths = [3, 4, 5]
        lrs    = [0.02, 0.05, 0.10]
        print("  Tuning CatBoost (depth × lr grid) …")
        best_home = best_away = None
        best_loss_h = best_loss_a = float("inf")
        best_params_h = best_params_a = {}

        for d in depths:
            for l in lrs:
                p  = _make_cb_params(d, l)
                mh = CatBoostRegressor(**p)
                ma = CatBoostRegressor(**p)
                mh.fit(X_tr, yh_tr, cat_features=cat_features, eval_set=(X_val, yh_val))
                ma.fit(X_tr, ya_tr, cat_features=cat_features, eval_set=(X_val, ya_val))
                lh, la = _cb_val_loss(mh), _cb_val_loss(ma)
                print(f"    depth={d} lr={l:.2f}  →  home_val={lh:.4f}  away_val={la:.4f}")
                if lh < best_loss_h: best_loss_h, best_home, best_params_h = lh, mh, p
                if la < best_loss_a: best_loss_a, best_away, best_params_a = la, ma, p

        print(f"  Best home: depth={best_params_h['depth']} lr={best_params_h['learning_rate']}  val={best_loss_h:.4f}")
        print(f"  Best away: depth={best_params_a['depth']} lr={best_params_a['learning_rate']}  val={best_loss_a:.4f}")
        return best_home, best_away

    else:
        print(f"  Training CatBoost (depth={depth}, lr={lr}, no grid) …")
        p  = _make_cb_params(depth, lr)
        mh = CatBoostRegressor(**p)
        ma = CatBoostRegressor(**p)
        mh.fit(X_tr, yh_tr, cat_features=cat_features, eval_set=(X_val, yh_val))
        ma.fit(X_tr, ya_tr, cat_features=cat_features, eval_set=(X_val, ya_val))
        print(f"  home_val={_cb_val_loss(mh):.4f}  away_val={_cb_val_loss(ma):.4f}")
        return mh, ma


# ─────────────────────────────────────────────────────────────
# EVALUATION
# Uses outcome probabilities (H/D/A) — same as dc_elo_form.py
# ─────────────────────────────────────────────────────────────

def poisson_prob_matrix(lh, la, max_goals=8):
    lh = max(float(lh), 1e-10)
    la = max(float(la), 1e-10)
    g  = np.arange(max_goals + 1)
    ph = np.exp(-lh + g * np.log(lh) - gammaln(g + 1))
    pa = np.exp(-la + g * np.log(la) - gammaln(g + 1))
    p  = np.outer(ph, pa)
    return p / p.sum()

def outcome_probs(p):
    return np.array([np.tril(p, -1).sum(), np.trace(p), np.triu(p, 1).sum()])

def one_hot_outcome(hg, ag):
    if hg > ag:  return np.array([1, 0, 0])
    if hg < ag:  return np.array([0, 0, 1])
    return np.array([0, 1, 0])

def evaluate(df, lh_arr, la_arr, max_goals=8):
    df = df.reset_index(drop=True)
    log_losses, rps_scores = [], []
    n_correct = n_exact = 0

    for i in range(len(df)):
        hg = int(df.at[i, 'home_goals'])
        ag = int(df.at[i, 'away_goals'])

        pm     = poisson_prob_matrix(lh_arr[i], la_arr[i], max_goals)
        pred   = outcome_probs(pm)
        actual = one_hot_outcome(hg, ag)

        pred_c = np.clip(pred, 1e-15, 1 - 1e-15)
        log_losses.append(-np.sum(actual * np.log(pred_c)))
        rps_scores.append(np.sum((np.cumsum(pred) - np.cumsum(actual))**2) / 2)

        n_correct += int(np.argmax(pred) == np.argmax(actual))
        ph, pa = np.unravel_index(np.argmax(pm), pm.shape)
        n_exact += int(ph == hg and pa == ag)

    n = len(df)
    return {
        "log_loss":       round(np.mean(log_losses), 6),
        "rps":            round(np.mean(rps_scores),  6),
        "accuracy":       round(n_correct / n,         6),
        "exact_score_pct": round(n_exact  / n,         6),
    }


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(train_df, test_df, decay_lambda=0.0005, reg=0.0001,
                 dc_maxiter=2000, tune=True, cb_depth=5, cb_lr=0.05):

    print("─── Stage 1: DC base model ───")
    result, teams, team_to_idx = fit_dc_base(train_df, decay_lambda, reg, maxiter=dc_maxiter)
    print(f"  converged={result.success}   loss={result.fun:.4f}")

    log_lh_train, log_la_train = get_dc_log_lambdas(train_df, result.x, team_to_idx)
    log_lh_test,  log_la_test  = get_dc_log_lambdas(test_df,  result.x, team_to_idx)

    X_train = build_features(train_df, log_lh_train, log_la_train)
    X_test  = build_features(test_df,  log_lh_test,  log_la_test)

    y_home = train_df["home_goals"].values.astype(float)
    y_away = train_df["away_goals"].values.astype(float)

    print("─── Stage 2: CatBoost Poisson ───")
    m_home, m_away = train_catboost_poisson(X_train, y_home, y_away,
                                            tune=tune, depth=cb_depth, lr=cb_lr)
    print(f"  home best iter: {m_home.best_iteration_}  |  away best iter: {m_away.best_iteration_}")

    # CatBoost Poisson loss: predict() returns expected count λ (not log λ)
    lh_train = np.clip(m_home.predict(X_train), 0.05, 10)
    la_train = np.clip(m_away.predict(X_train), 0.05, 10)
    lh_test  = np.clip(m_home.predict(X_test),  0.05, 10)
    la_test  = np.clip(m_away.predict(X_test),  0.05, 10)

    # Sanity check: typical football lambda should be 0.5–3
    print(f"  λ_home sample: {lh_test[:5].round(3)}  (expect ~1–3)")
    print(f"  λ_away sample: {la_test[:5].round(3)}  (expect ~1–3)")

    print("─── Evaluation ───")
    train_metrics = evaluate(train_df, lh_train, la_train)
    test_metrics  = evaluate(test_df,  lh_test,  la_test)

    print("  [TRAIN]", {k: v for k, v in train_metrics.items()})
    print("  [TEST] ", {k: v for k, v in test_metrics.items()})

    return test_metrics, m_home, m_away, result.x, team_to_idx


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
import os, pickle

SAVE_DIR = "models_dc_cat_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_pipeline(tag, m_home, m_away, dc_params, team_to_idx):
    m_home.save_model(f"{SAVE_DIR}/{tag}_home.cbm")
    m_away.save_model(f"{SAVE_DIR}/{tag}_away.cbm")
    with open(f"{SAVE_DIR}/{tag}_dc_params.pkl", "wb") as f:
        pickle.dump({"dc_params": dc_params, "team_to_idx": team_to_idx}, f)
    print(f"  Saved → {SAVE_DIR}/{tag}_{{home,away}}.cbm + {tag}_dc_params.pkl")

all_results = {}

# ── International ──────────────────────────────────────────────
print("=== DC + CatBoost Poisson Hybrid (v2) — International ===\n")
metrics, m_home, m_away, dc_params, team_to_idx = run_pipeline(train_intl, test_intl)
all_results["intl"] = metrics
save_pipeline("intl", m_home, m_away, dc_params, team_to_idx)

# ── Club ───────────────────────────────────────────────────────
print("\n=== DC + CatBoost Poisson Hybrid (v2) — Club ===\n")
# Club: 1135 teams → DC is an initialiser only (maxiter=300 ~3 min).
# CatBoost uses best params from intl grid search (depth=5, lr=0.05), no re-tuning.
metrics, m_home, m_away, dc_params, team_to_idx = run_pipeline(
    train_club, test_club, dc_maxiter=300, tune=False, cb_depth=5, cb_lr=0.05
)
all_results["club"] = metrics
save_pipeline("club", m_home, m_away, dc_params, team_to_idx)

# ── Summary ────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Metric':<22} {'Intl':>10} {'Club':>10}")
print("-"*55)
for k in all_results["intl"]:
    print(f"  {k:<20} {all_results['intl'][k]:>10} {all_results['club'][k]:>10}")
print("="*55)

# Baseline reference (intl, from model_comparison.py):
#   DC_only  → log_loss=0.9096, rps=0.1812, accuracy=0.5996
#   DC_Elo   → log_loss=0.9166, rps=0.1820, accuracy=0.5975
