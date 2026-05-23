import os, pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.special import gammaln
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# FORM FEATURES  (v3: extends history tuple with xG)
#
# History tuple per team: (gf, ga, pts, elo, xgf, xga)
#   [0]=gf  [1]=ga  [2]=pts  [3]=elo  [4]=xgf  [5]=xga
#
# xG fallback: when xg_available=0, rolling xGF/xGA falls back
# to goals scored/conceded — semantically correct for intl data.
# ─────────────────────────────────────────────────────────────

XG_FORM_COLS = [
    'home_form_xgf', 'home_form_xga',
    'away_form_xgf', 'away_form_xga',
    'home_form_xgf_h', 'home_form_xga_h',
    'away_form_xgf_a', 'away_form_xga_a',
]

def add_form_features(df, window=5, seed=None):
    """
    Rolling pre-match features — strictly no leakage.
    History tuple per team: (gf, ga, pts, elo, xgf, xga)

    seed: dict with keys 'all', 'home', 'away', 'last_date' from a prior call.
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
        'home_form_gf_h', 'home_form_ga_h',
        'away_form_gf_a', 'away_form_ga_a',
        'home_days_rest', 'away_days_rest',
        # v3 additions — rolling xG
        'home_form_xgf', 'home_form_xga',
        'away_form_xgf', 'away_form_xga',
        'home_form_xgf_h', 'home_form_xga_h',
        'away_form_xgf_a', 'away_form_xga_a',
    ]:
        df[col] = 0.0

    # Ensure xG columns exist (backward-compatible with old-format inputs)
    for col in ['home_xg', 'away_xg', 'xg_available']:
        if col not in df.columns:
            df[col] = 0

    if seed:
        all_h  = {t: copy.copy(h[-window:]) for t, h in seed['all'].items()}
        home_h = {t: copy.copy(h[-window:]) for t, h in seed['home'].items()}
        away_h = {t: copy.copy(h[-window:]) for t, h in seed['away'].items()}
        last_d = dict(seed['last_date'])
    else:
        all_h = {}; home_h = {}; away_h = {}; last_d = {}

    DEFAULT_REST = 30

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

        # xG values — fall back to goals when xG is unavailable
        xg_avail = int(row.get('xg_available', 0))
        h_xgf = float(row['home_xg']) if xg_avail == 1 and pd.notna(row['home_xg']) else hg
        a_xgf = float(row['away_xg']) if xg_avail == 1 and pd.notna(row['away_xg']) else ag

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
            df.at[idx, 'home_form_xgf']           = np.mean([m[4] for m in hh])
            df.at[idx, 'home_form_xga']           = np.mean([m[5] for m in hh])

        ah = all_h[away][-window:]
        if ah:
            df.at[idx, 'away_form_goals_for']    = np.mean([m[0] for m in ah])
            df.at[idx, 'away_form_goals_against'] = np.mean([m[1] for m in ah])
            df.at[idx, 'away_form_points']        = np.mean([m[2] for m in ah])
            df.at[idx, 'away_clean_sheet_rate']   = np.mean([m[1] == 0 for m in ah])
            df.at[idx, 'away_elo_momentum']       = a_elo - ah[0][3]
            df.at[idx, 'away_form_xgf']           = np.mean([m[4] for m in ah])
            df.at[idx, 'away_form_xga']           = np.mean([m[5] for m in ah])

        # ── venue-split form ─────────────────────────────────────────
        hv = home_h[home][-window:]
        if hv:
            df.at[idx, 'home_form_gf_h']  = np.mean([m[0] for m in hv])
            df.at[idx, 'home_form_ga_h']  = np.mean([m[1] for m in hv])
            df.at[idx, 'home_form_xgf_h'] = np.mean([m[4] for m in hv])
            df.at[idx, 'home_form_xga_h'] = np.mean([m[5] for m in hv])

        av = away_h[away][-window:]
        if av:
            df.at[idx, 'away_form_gf_a']  = np.mean([m[0] for m in av])
            df.at[idx, 'away_form_ga_a']  = np.mean([m[1] for m in av])
            df.at[idx, 'away_form_xgf_a'] = np.mean([m[4] for m in av])
            df.at[idx, 'away_form_xga_a'] = np.mean([m[5] for m in av])

        # ── update histories (6-element tuple) ───────────────────────
        all_h[home].append((hg, ag, h_pts, h_elo, h_xgf, a_xgf))
        all_h[away].append((ag, hg, a_pts, a_elo, a_xgf, h_xgf))
        home_h[home].append((hg, ag, h_pts, h_elo, h_xgf, a_xgf))
        away_h[away].append((ag, hg, a_pts, a_elo, a_xgf, h_xgf))
        last_d[home] = match_date
        last_d[away] = match_date

    history = {'all': all_h, 'home': home_h, 'away': away_h, 'last_date': last_d}
    return df, history


def build_form_features(train_df, test_df, window=5):
    train_df, train_hist = add_form_features(train_df, window=window)
    test_df,  _          = add_form_features(test_df,  window=window, seed=train_hist)
    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# STAGE 1: DC BASE MODEL
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
    attack   = params[:n_teams]          - np.mean(params[:n_teams])
    defense  = params[n_teams:2*n_teams] - np.mean(params[n_teams:2*n_teams])
    home_adv = params[-2]
    rho      = params[-1]

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

    # Dixon-Coles τ correction for low-score lines
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
    home_adv = params[-2]

    hi = df['home_team'].map(team_to_idx).values
    ai = df['away_team'].map(team_to_idx).values

    log_lh = np.clip(attack[hi] - defense[ai] + home_adv, -10, 10)
    log_la = np.clip(attack[ai] - defense[hi],             -10, 10)
    return log_lh, log_la


# ─────────────────────────────────────────────────────────────
# STAGE 2: CATBOOST POISSON REGRESSION
#
# v3 adds 11 xG-related features (26 → 37 total):
#   - rolling xGF/xGA (all-venue and venue-split)
#   - xG matchup interactions
#   - xg_available flag (teaches CatBoost to weight differently)
# ─────────────────────────────────────────────────────────────

def build_features(df, log_lh, log_la):
    df = df.reset_index(drop=True)

    # Column guards — backward-compatible with old-format DataFrames
    for col in ['home_xg', 'away_xg', 'xg_available']:
        if col not in df.columns:
            df[col] = 0
    for c in XG_FORM_COLS:
        if c not in df.columns:
            df[c] = 0.0
    FIFA_COLS = ['home_avg_attack', 'home_avg_defense', 'away_avg_attack',
                 'away_avg_defense', 'home_avg_overall', 'away_avg_overall']
    for c in FIFA_COLS:
        if c not in df.columns:
            df[c] = np.nan

    return pd.DataFrame({
        # DC base lambdas (Stage 1 signal)
        "log_lambda_home_dc":     log_lh,
        "log_lambda_away_dc":     log_la,
        # ELO
        "elo_diff":               (df["home_elo"] - df["away_elo"]).values,
        "home_elo":               df["home_elo"].values,
        "away_elo":               df["away_elo"].values,
        # All-venue goals form
        "home_form_gf":           df["home_form_goals_for"].values,
        "home_form_ga":           df["home_form_goals_against"].values,
        "away_form_gf":           df["away_form_goals_for"].values,
        "away_form_ga":           df["away_form_goals_against"].values,
        "home_form_pts":          df["home_form_points"].values,
        "away_form_pts":          df["away_form_points"].values,
        "home_cs_rate":           df["home_clean_sheet_rate"].values,
        "away_cs_rate":           df["away_clean_sheet_rate"].values,
        "home_elo_momentum":      df["home_elo_momentum"].values,
        "away_elo_momentum":      df["away_elo_momentum"].values,
        # Venue-split goals form
        "home_form_gf_h":         df["home_form_gf_h"].values,
        "home_form_ga_h":         df["home_form_ga_h"].values,
        "away_form_gf_a":         df["away_form_gf_a"].values,
        "away_form_ga_a":         df["away_form_ga_a"].values,
        # Days rest
        "home_days_rest":         df["home_days_rest"].values,
        "away_days_rest":         df["away_days_rest"].values,
        # Goals matchup interactions
        "home_atk_vs_away_def":   (df["home_form_goals_for"]  - df["away_form_goals_against"]).values,
        "away_atk_vs_home_def":   (df["away_form_goals_for"]  - df["home_form_goals_against"]).values,
        "home_atk_vs_away_def_v": (df["home_form_gf_h"] - df["away_form_ga_a"]).values,
        "away_atk_vs_home_def_v": (df["away_form_gf_a"] - df["home_form_ga_h"]).values,
        # Match context
        "is_intl":                df["is_intl"].values,
        "is_neutral":             df["is_neutral"].values,
        # ── v3: rolling xG form (all-venue) ──────────────────────
        "home_form_xgf":          df["home_form_xgf"].values,
        "home_form_xga":          df["home_form_xga"].values,
        "away_form_xgf":          df["away_form_xgf"].values,
        "away_form_xga":          df["away_form_xga"].values,
        # v3: venue-split xG form
        "home_form_xgf_h":        df["home_form_xgf_h"].values,
        "home_form_xga_h":        df["home_form_xga_h"].values,
        "away_form_xgf_a":        df["away_form_xgf_a"].values,
        "away_form_xga_a":        df["away_form_xga_a"].values,
        # v3: xG matchup interactions
        "home_xgf_vs_away_xga":   (df["home_form_xgf"] - df["away_form_xga"]).values,
        "away_xgf_vs_home_xga":   (df["away_form_xgf"] - df["home_form_xga"]).values,
        # v3: data-source reliability flag
        "xg_available":           df["xg_available"].values,
        # ── v3: FIFA squad ratings (intl only; 0.0 for club via fillna) ─
        "home_avg_attack":        df["home_avg_attack"].fillna(0.0).values,
        "home_avg_defense":       df["home_avg_defense"].fillna(0.0).values,
        "away_avg_attack":        df["away_avg_attack"].fillna(0.0).values,
        "away_avg_defense":       df["away_avg_defense"].fillna(0.0).values,
        "home_avg_overall":       df["home_avg_overall"].fillna(0.0).values,
        "away_avg_overall":       df["away_avg_overall"].fillna(0.0).values,
        "fifa_attack_diff":       (df["home_avg_attack"] - df["away_avg_attack"]).fillna(0.0).values,
        "fifa_defense_diff":      (df["home_avg_defense"] - df["away_avg_defense"]).fillna(0.0).values,
        # Categorical team IDs
        "home_team":              df["home_team"].values,
        "away_team":              df["away_team"].values,
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

def train_catboost_poisson(X_train, y_home, y_away, tune=True, depth=5, lr=0.05, sample_weight=None):
    cat_features = ["home_team", "away_team"]
    split_inputs = [X_train, y_home, y_away]
    if sample_weight is not None:
        split_inputs.append(sample_weight)
    split = train_test_split(*split_inputs, test_size=0.1, random_state=42)
    if sample_weight is not None:
        X_tr, X_val, yh_tr, yh_val, ya_tr, ya_val, w_tr, _ = split
    else:
        X_tr, X_val, yh_tr, yh_val, ya_tr, ya_val = split
        w_tr = None

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
                mh.fit(X_tr, yh_tr, cat_features=cat_features, eval_set=(X_val, yh_val), sample_weight=w_tr)
                ma.fit(X_tr, ya_tr, cat_features=cat_features, eval_set=(X_val, ya_val), sample_weight=w_tr)
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
        mh.fit(X_tr, yh_tr, cat_features=cat_features, eval_set=(X_val, yh_val), sample_weight=w_tr)
        ma.fit(X_tr, ya_tr, cat_features=cat_features, eval_set=(X_val, ya_val), sample_weight=w_tr)
        print(f"  home_val={_cb_val_loss(mh):.4f}  away_val={_cb_val_loss(ma):.4f}")
        return mh, ma


# ─────────────────────────────────────────────────────────────
# EVALUATION
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
        "log_loss":        round(np.mean(log_losses), 6),
        "rps":             round(np.mean(rps_scores),  6),
        "accuracy":        round(n_correct / n,         6),
        "exact_score_pct": round(n_exact  / n,          6),
    }


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(train_df, test_df, decay_lambda=0.0005, reg=0.0001,
                 dc_maxiter=2000, tune=True, cb_depth=5, cb_lr=0.05, cb_decay_lambda=0.0):

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
    if cb_decay_lambda > 0:
        dates = pd.to_datetime(train_df['date'], errors='coerce')
        days_since_cb = (dates.max() - dates).dt.days.fillna(0).values.astype(float)
        cb_weights = np.exp(-cb_decay_lambda * days_since_cb)
        print(f"  CB time-decay: λ={cb_decay_lambda}  weights [{cb_weights.min():.3f}, {cb_weights.max():.3f}]  mean={cb_weights.mean():.3f}")
    else:
        cb_weights = None
    m_home, m_away = train_catboost_poisson(X_train, y_home, y_away,
                                            tune=tune, depth=cb_depth, lr=cb_lr,
                                            sample_weight=cb_weights)
    print(f"  home best iter: {m_home.best_iteration_}  |  away best iter: {m_away.best_iteration_}")

    lh_train = np.clip(m_home.predict(X_train), 0.05, 10)
    la_train = np.clip(m_away.predict(X_train), 0.05, 10)
    lh_test  = np.clip(m_home.predict(X_test),  0.05, 10)
    la_test  = np.clip(m_away.predict(X_test),  0.05, 10)

    print(f"  λ_home sample: {lh_test[:5].round(3)}  (expect ~1–3)")
    print(f"  λ_away sample: {la_test[:5].round(3)}  (expect ~1–3)")

    print("─── Evaluation ───")
    train_metrics = evaluate(train_df, lh_train, la_train)
    test_metrics  = evaluate(test_df,  lh_test,  la_test)

    print("  [TRAIN]", {k: v for k, v in train_metrics.items()})
    print("  [TEST] ", {k: v for k, v in test_metrics.items()})

    return test_metrics, m_home, m_away, result.x, team_to_idx, lh_test, la_test


# ─────────────────────────────────────────────────────────────
# SAVE HELPER  (module-level so test_scenarios.py can import it)
# ─────────────────────────────────────────────────────────────
SAVE_DIR = "models_dc_cat_v3"

def save_pipeline(tag, m_home, m_away, dc_params, team_to_idx, save_dir=None):
    d = save_dir or SAVE_DIR
    os.makedirs(d, exist_ok=True)
    m_home.save_model(f"{d}/{tag}_home.cbm")
    m_away.save_model(f"{d}/{tag}_away.cbm")
    with open(f"{d}/{tag}_dc_params.pkl", "wb") as f:
        pickle.dump({"dc_params": dc_params, "team_to_idx": team_to_idx}, f)
    print(f"  Saved → {d}/{tag}_{{home,away}}.cbm + {tag}_dc_params.pkl")


if __name__ == "__main__":
    from preprocessor_v2 import dedup_datasets

    # ── Data loading ──────────────────────────────────────────────
    train_club = pd.read_csv('train_club_v2.csv')
    test_club  = pd.read_csv('test_club_v2.csv')
    train_intl = pd.read_csv('train_intl_v2.csv')
    test_intl  = pd.read_csv('test_intl_v2.csv')

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

    # ── Form features ─────────────────────────────────────────────
    train_intl, test_intl = build_form_features(train_intl, test_intl)
    train_club, test_club = build_form_features(train_club, test_club)

    all_results = {}

    # ── Phase 1: International ────────────────────────────────────
    print("=== DC + CatBoost Poisson v3 — Phase 1: International ===\n")
    metrics, m_home, m_away, dc_params, team_to_idx, _, _ = run_pipeline(
        train_intl, test_intl, dc_maxiter=5000
    )
    all_results["intl"] = metrics
    save_pipeline("phase1_intl", m_home, m_away, dc_params, team_to_idx)

    # ── Phase 1: Club ─────────────────────────────────────────────
    print("\n=== DC + CatBoost Poisson v3 — Phase 1: Club ===\n")
    metrics, m_home, m_away, dc_params, team_to_idx, _, _ = run_pipeline(
        train_club, test_club, dc_maxiter=500, tune=False, cb_depth=5, cb_lr=0.05
    )
    all_results["club"] = metrics
    save_pipeline("phase1_club", m_home, m_away, dc_params, team_to_idx)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Metric':<22} {'Intl':>12} {'Club':>12}")
    print("-"*60)
    for k in all_results["intl"]:
        print(f"  {k:<20} {all_results['intl'][k]:>12} {all_results['club'][k]:>12}")
    print("="*60)

    # v2 reference (intl, 2022 test window):
    #   DC_only  → log_loss=0.9096, rps=0.1812, accuracy=0.5996
    #   DC_Elo   → log_loss=0.9166, rps=0.1820, accuracy=0.5975
    #   DC_Cat_v2 (intl) → see models_dc_cat_v2/ for baseline

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: Soccerway enrichment
    # ─────────────────────────────────────────────────────────────
    if os.path.exists("sw_intl.csv") and os.path.exists("sw_club.csv"):
        print("\n=== Phase 2: Soccerway Enrichment ===\n")

        sw_club = pd.read_csv("sw_club.csv")
        sw_intl = pd.read_csv("sw_intl.csv")

        # Soccerway is priority (has xG); intl_stats is fallback
        combined_intl = dedup_datasets(sw_intl, train_intl)
        combined_club = dedup_datasets(sw_club, train_club)

        combined_intl = combined_intl.sort_values('date').reset_index(drop=True)
        combined_club = combined_club.sort_values('date').reset_index(drop=True)

        combined_intl_form, intl_hist = add_form_features(combined_intl)
        combined_club_form, club_hist = add_form_features(combined_club)

        # Test form seeded from combined training history (no leakage)
        test_intl_p2, _ = add_form_features(test_intl, seed=intl_hist)
        test_club_p2,  _ = add_form_features(test_club,  seed=club_hist)

        print("=== Phase 2: International ===\n")
        metrics_p2, mh2, ma2, dcp2, t2i2, _, _ = run_pipeline(
            combined_intl_form, test_intl_p2, dc_maxiter=5000
        )
        all_results["intl_p2"] = metrics_p2
        save_pipeline("phase2_intl", mh2, ma2, dcp2, t2i2)

        print("\n=== Phase 2: Club ===\n")
        metrics_p2c, mh2c, ma2c, dcp2c, t2i2c, _, _ = run_pipeline(
            combined_club_form, test_club_p2, dc_maxiter=500, tune=False, cb_depth=5, cb_lr=0.05
        )
        all_results["club_p2"] = metrics_p2c
        save_pipeline("phase2_club", mh2c, ma2c, dcp2c, t2i2c)

        print("\n" + "="*70)
        print(f"{'Metric':<22} {'Intl P1':>10} {'Intl P2':>10} {'Club P1':>10} {'Club P2':>10}")
        print("-"*70)
        for k in all_results["intl"]:
            p1i = all_results['intl'].get(k, '-')
            p2i = all_results.get('intl_p2', {}).get(k, '-')
            p1c = all_results['club'].get(k, '-')
            p2c = all_results.get('club_p2', {}).get(k, '-')
            print(f"  {k:<20} {p1i:>10} {p2i:>10} {p1c:>10} {p2c:>10}")
        print("="*70)
    else:
        print("\n[Phase 2 skipped — sw_intl.csv / sw_club.csv not found]")
        print("Run preprocessor_v2.py first, then re-run dc_cat_v3.py for Phase 2.")
