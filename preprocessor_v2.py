import numpy as np
import pandas as pd

#############################################
# CONSTANTS
#############################################

TEAM_ALIASES = {
    "Bosnia & Herzegovina":               "Bosnia-Herzegovina",
    "Bosnia and Herzegovina":             "Bosnia-Herzegovina",
    "Côte d'Ivoire":                      "Ivory Coast",
    "Cote d'Ivoire":                      "Ivory Coast",
    "USA":                                "United States",
    "Cabo Verde":                         "Cape Verde",
    "Czechia":                            "Czech Republic",
    "DR Congo":                           "Democratic Republic of the Congo",
    "Congo DR":                           "Democratic Republic of the Congo",
    "Congo, DR":                          "Democratic Republic of the Congo",
    "Atlético Madrid":                    "Atletico Madrid",
    "Inter Milan":                        "Inter",
    "Curaçao":                            "Curacao",
    "São Tomé e Príncipe":                "Sao Tome and Principe",
    "Eswatini":                           "Swaziland",
    "North Macedonia":                    "Macedonia",
    "Republic of Ireland":                "Ireland",
    "Korea Republic":                     "South Korea",
    "Korea DPR":                          "North Korea",
    "Chinese Taipei":                     "Taiwan",
    "Türkiye":                            "Turkey",
    "Turkiye":                            "Turkey",
}

# Keywords that identify international (national-team) competitions in league_division
INTL_KEYWORDS = [
    'world cup',
    'nations league',
    'copa america',
    'africa cup of nations',
    'asian cup',
    'gold cup',
    'concacaf championship',
    'olympic games',
    'afcon',
    'euro 20',          # "Euro 2024", "Euro 2020" — avoids matching "UEFA Europa League"
    'european championship',
]

# Standardized output schema shared across all loaders
SCHEMA_COLS = [
    'date',
    'home_team',
    'away_team',
    'home_goals',
    'away_goals',
    'is_neutral',
    'is_intl',
    'home_xg',
    'away_xg',
    'xg_available',
    'home_elo',
    'away_elo',
    # FIFA squad ratings — populated from intl_stats; NaN for club / soccerway rows
    'home_avg_attack',
    'home_avg_defense',
    'away_avg_attack',
    'away_avg_defense',
    'home_avg_overall',
    'away_avg_overall',
]

# FIFA rating columns (NaN for non-intl sources)
FIFA_RATING_COLS = [
    'home_avg_attack', 'home_avg_defense',
    'away_avg_attack', 'away_avg_defense',
    'home_avg_overall', 'away_avg_overall',
]


#############################################
# TEAM NAME NORMALIZATION
#############################################

def normalize_team_name(name):
    if not isinstance(name, str):
        return name
    return TEAM_ALIASES.get(name.strip(), name.strip())


#############################################
# ELO (neutral-aware, reused from v1)
#############################################

def compute_elo(df, k=20, home_adv=50):
    df = df.copy()
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    elo = {team: 1500 for team in teams}

    home_elo_list = []
    away_elo_list = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        home_elo = elo[home]
        away_elo = elo[away]

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

        ha = 0 if row['is_neutral'] == 1 else home_adv
        exp_home = 1 / (1 + 10 ** ((away_elo - (home_elo + ha)) / 400))

        if row['home_goals'] > row['away_goals']:
            score_home = 1
        elif row['home_goals'] < row['away_goals']:
            score_home = 0
        else:
            score_home = 0.5

        elo[home] += k * (score_home - exp_home)
        elo[away] += k * ((1 - score_home) - (1 - exp_home))

    df['home_elo'] = home_elo_list
    df['away_elo'] = away_elo_list
    return df


#############################################
# SELECT FINAL COLUMNS
#############################################

def select_columns_v2(df):
    return df[SCHEMA_COLS].copy()


#############################################
# TIME-BASED TRAIN / TEST SPLIT (reused from v1)
#############################################

def train_test_split_time(df, train_end_date, test_start_date, test_end_date):
    train = df[df['date'] <= train_end_date].copy()
    test  = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()
    return train, test


#############################################
# LOADER 1: INTERNATIONAL MATCH STATS
#   Source: International_match_stats.csv
#   ELO:    passthrough (pre-computed Elo-World ratings)
#   xG:     unavailable — NaN + xg_available=0
#############################################

def load_intl_stats(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        '_home_team':  'home_team',
        '_away_team':  'away_team',
        '_date':       'date',
        '_tournament': 'tournament',
    })

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])

    # Drop friendlies
    if 'tournament' in df.columns:
        df = df[~df['tournament'].str.contains('friendly', case=False, na=False)]

    df['home_team'] = df['home_team'].apply(normalize_team_name)
    df['away_team'] = df['away_team'].apply(normalize_team_name)

    df['is_intl']   = 1
    df['is_neutral'] = df['is_neutral'].fillna(0).astype(int)

    # No xG in this dataset — use goal-fallback is handled at model layer via xg_available=0
    df['home_xg']       = np.nan
    df['away_xg']       = np.nan
    df['xg_available']  = 0

    # home_elo / away_elo are already present as pre-match ratings — passthrough

    # FIFA squad ratings — available in this dataset; rename to match schema
    rating_rename = {
        'home_avg_attack':  'home_avg_attack',
        'home_avg_defense': 'home_avg_defense',
        'away_avg_attack':  'away_avg_attack',
        'away_avg_defense': 'away_avg_defense',
        'home_avg_overall': 'home_avg_overall',
        'away_avg_overall': 'away_avg_overall',
    }
    for col in FIFA_RATING_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values('date').reset_index(drop=True)
    print(f"  [intl_stats] {len(df):,} rows after filtering friendlies")
    return select_columns_v2(df)


#############################################
# LOADER 2: CLUB GAME STATS (understat)
#   Source: Club_game_stats.csv + club_basic.csv
#   ELO:    recomputed from scratch (no pre-existing ELO)
#   xG:     h_xg / a_xg — full coverage, xg_available=1
#############################################

def load_club_game_stats(cgs_path, basic_path):
    cgs   = pd.read_csv(cgs_path)
    basic = pd.read_csv(basic_path)

    cgs = cgs.rename(columns={
        'team_h':  'home_team',
        'team_a':  'away_team',
        'h_goals': 'home_goals',
        'a_goals': 'away_goals',
        'h_xg':    'home_xg',
        'a_xg':    'away_xg',
    })

    # Parse date (ISO with time component — strip to date only)
    cgs['date'] = pd.to_datetime(cgs['date'], errors='coerce').dt.normalize()
    cgs = cgs.dropna(subset=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])

    cgs['home_team'] = cgs['home_team'].apply(normalize_team_name)
    cgs['away_team'] = cgs['away_team'].apply(normalize_team_name)

    # Join season-level avg_xG from club_basic (background feature, not used directly in schema)
    # Merge for home side
    basic_h = basic[['club_id', 'season', 'avg_xG']].rename(
        columns={'club_id': 'h_id', 'avg_xG': 'home_season_avg_xg'})
    # Merge for away side
    basic_a = basic[['club_id', 'season', 'avg_xG']].rename(
        columns={'club_id': 'a_id', 'avg_xG': 'away_season_avg_xg'})

    cgs = cgs.merge(basic_h, on=['h_id', 'season'], how='left')
    cgs = cgs.merge(basic_a, on=['a_id', 'season'], how='left')

    cgs['is_intl']    = 0
    cgs['is_neutral'] = 0
    # FIFA ratings not available for club data
    for col in FIFA_RATING_COLS:
        cgs[col] = np.nan
    cgs['xg_available'] = 1

    cgs = cgs.sort_values('date').reset_index(drop=True)
    cgs = compute_elo(cgs)

    print(f"  [club_game_stats] {len(cgs):,} rows, xg_available=1 for all")
    return select_columns_v2(cgs)


#############################################
# LOADER 3: SOCCERWAY (club + intl, 2024–2025)
#   Source: Soccerway.csv
#   ELO:    recomputed separately for club and intl splits
#   xG:     home_xg / away_xg — ~80% fill rate for club, ~50% for intl
#   Returns: (club_df, intl_df)
#############################################

def _is_intl_division(div):
    if not isinstance(div, str):
        return False
    div_lower = div.lower()
    # Explicitly exclude club European competitions that contain partial keyword matches
    if 'europa league' in div_lower or 'conference league' in div_lower:
        return False
    return any(kw in div_lower for kw in INTL_KEYWORDS)

def load_soccerway(path):
    df = pd.read_csv(path, low_memory=False)

    # Parse date — handles ISO ("2024-08-24") and English month format ("August 24, 2024")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Second pass for any rows that didn't parse (e.g. "August 24, 2024" format)
    unparsed = df['date'].isna() & df['date'].astype(str).ne('NaT')
    if unparsed.any():
        df.loc[df['date'].isna(), 'date'] = pd.to_datetime(
            df.loc[df['date'].isna(), 'date'], format='%B %d, %Y', errors='coerce'
        )

    df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])

    df['home_team'] = df['home_team'].apply(normalize_team_name)
    df['away_team'] = df['away_team'].apply(normalize_team_name)

    # Filter out friendlies
    ld_lower = df['league_division'].fillna('').str.lower()
    df = df[~ld_lower.str.contains('friendly', na=False)].copy()

    # Classify intl vs club
    df['_is_intl'] = df['league_division'].apply(_is_intl_division)

    # is_neutral: WC tournament proper (non-qualification) = neutral site; everything else = 0
    df['is_neutral'] = 0
    wc_proper = df['league_division'].fillna('').str.lower().str.match(
        r'^world cup$|^world cup - (group|round|quarter|semi|final|third)'
    )
    df.loc[wc_proper, 'is_neutral'] = 1

    # xG availability (leave NaN as NaN — do not zero-fill)
    df['xg_available'] = df['home_xg'].notna().astype(int)

    # ── Split ─────────────────────────────────────────────────────
    intl_df = df[df['_is_intl']].copy()
    club_df = df[~df['_is_intl']].copy()

    intl_df['is_intl'] = 1
    club_df['is_intl'] = 0
    club_df['is_neutral'] = 0

    # Compute ELO independently for each split
    for split_df in (intl_df, club_df):
        split_df.sort_values('date', inplace=True)
        split_df.reset_index(drop=True, inplace=True)

    # FIFA ratings not available in Soccerway
    for col in FIFA_RATING_COLS:
        intl_df[col] = np.nan
        club_df[col] = np.nan

    intl_df = compute_elo(intl_df)
    club_df = compute_elo(club_df)

    print(f"  [soccerway] club={len(club_df):,}  intl={len(intl_df):,}")
    print(f"    club xg_available:  {club_df['xg_available'].value_counts().to_dict()}")
    print(f"    intl xg_available:  {intl_df['xg_available'].value_counts().to_dict()}")

    return select_columns_v2(club_df), select_columns_v2(intl_df)


#############################################
# DEDUPLICATION HELPER
#   Used when merging intl_stats + soccerway
#   Prefers soccerway rows (have xG); falls back to intl_stats
#############################################

def dedup_datasets(primary_df, secondary_df):
    """
    Merge two DataFrames, keeping primary rows when duplicates exist.
    Dedup key: (date, home_team, away_team)
    Falls back to (date, home_goals, away_goals) for unresolved name mismatches.
    """
    combined = pd.concat([primary_df, secondary_df], ignore_index=True)
    # Normalize date column — handles mixed str/Timestamp types from CSV reads
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined = combined.sort_values(['date', 'home_team', 'away_team'])

    # Exact team-name dedup (keep first = primary / preferred source)
    combined = combined.drop_duplicates(
        subset=['date', 'home_team', 'away_team'], keep='first'
    )
    return combined.sort_values('date').reset_index(drop=True)


#############################################
# MAIN
#############################################

if __name__ == "__main__":

    INTL_PATH  = "International_match_stats.csv"
    CGS_PATH   = "Club_game_stats.csv"
    BASIC_PATH = "club_basic.csv"
    SW_PATH    = "Soccerway.csv"

    # ── International ─────────────────────────────────────────────
    print("=== Loading International Match Stats ===")
    intl_df = load_intl_stats(INTL_PATH)

    train_intl, test_intl = train_test_split_time(
        intl_df,
        train_end_date="2022-12-31",
        test_start_date="2023-01-01",
        test_end_date="2024-12-31",
    )
    print(f"  train_intl={len(train_intl):,}  test_intl={len(test_intl):,}")

    # ── Club ──────────────────────────────────────────────────────
    print("\n=== Loading Club Game Stats (understat) ===")
    club_df = load_club_game_stats(CGS_PATH, BASIC_PATH)

    train_club, test_club = train_test_split_time(
        club_df,
        train_end_date="2022-12-31",
        test_start_date="2023-01-01",
        test_end_date="2024-12-31",
    )
    print(f"  train_club={len(train_club):,}  test_club={len(test_club):,}")

    # ── Soccerway ─────────────────────────────────────────────────
    print("\n=== Loading Soccerway ===")
    sw_club, sw_intl = load_soccerway(SW_PATH)

    # ── Save Phase 1 CSVs ─────────────────────────────────────────
    train_intl.to_csv("train_intl_v2.csv", index=False)
    test_intl.to_csv("test_intl_v2.csv",   index=False)
    train_club.to_csv("train_club_v2.csv", index=False)
    test_club.to_csv("test_club_v2.csv",   index=False)
    sw_club.to_csv("sw_club.csv",           index=False)
    sw_intl.to_csv("sw_intl.csv",           index=False)

    print("\n=== Saved Phase 1 CSVs ===")
    print(f"  train_intl_v2.csv   {len(train_intl):,} rows")
    print(f"  test_intl_v2.csv    {len(test_intl):,} rows")
    print(f"  train_club_v2.csv   {len(train_club):,} rows")
    print(f"  test_club_v2.csv    {len(test_club):,} rows")
    print(f"  sw_club.csv         {len(sw_club):,} rows")
    print(f"  sw_intl.csv         {len(sw_intl):,} rows")

    # ── Spot-check ────────────────────────────────────────────────
    print("\n=== Spot Checks ===")

    # intl: ELO populated, xg_available=0
    sample_wc = train_intl[train_intl['date'].dt.year == 2018].head(3)
    print("\ntrain_intl 2018 sample:")
    print(sample_wc[['date', 'home_team', 'away_team', 'home_elo', 'xg_available']].to_string())

    # club: xg for Man Utd vs Tottenham Aug 2015
    sample_club = train_club[
        (train_club['home_team'] == 'Manchester United') &
        (train_club['date'].dt.year == 2015)
    ].head(3)
    print("\ntrain_club Man Utd 2015 sample:")
    print(sample_club[['date', 'home_team', 'away_team', 'home_xg', 'away_xg', 'xg_available']].to_string())

    print("\n=== Summary ===")
    for name, df in [("train_intl", train_intl), ("test_intl", test_intl),
                     ("train_club", train_club), ("test_club", test_club),
                     ("sw_club",    sw_club),    ("sw_intl",   sw_intl)]:
        xg_rate = df['xg_available'].mean()
        print(f"  {name:<18} rows={len(df):>6,}  xg_rate={xg_rate:.0%}"
              f"  date_range={df['date'].min().date()} → {df['date'].max().date()}")
