import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')



import numpy as np
import pandas as pd

#############################################
# 1. LOAD + STANDARDIZE
#############################################

def load_and_standardize(df, dataset_type="club"):
    
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    col_map = {
        '_date': 'date',
        'matchdate':'date',
        'hometeam': 'home_team',
        'awayteam': 'away_team',
        '_home_team': 'home_team',
        '_away_team': 'away_team',
        'fthome': 'home_goals',
        'ftaway': 'away_goals',
        '_tournament':'tournament'
    }
    
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if dataset_type == "intl":
        print(df['tournament'].unique())

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])
    
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


#############################################
# 2. REMOVE FRIENDLIES (INTL)
#############################################

def remove_friendlies(df):
    
    if 'tournament' in df.columns:
        df = df[~df['tournament'].str.contains("friendly", case=False, na=False)]
    
    return df


#############################################
# 3. ADD STANDARD FLAGS
#############################################

def add_standard_flags(df, dataset_type="club"):
    
    df = df.copy()
    
    # is_intl flag
    df['is_intl'] = 1 if dataset_type == "intl" else 0
    
    # is_neutral handling
    if 'is_neutral' in df.columns:
        df['is_neutral'] = df['is_neutral'].astype(int)
    else:
        # club matches default = not neutral
        df['is_neutral'] = 0
    
    return df


#############################################
# 4. SELECT FINAL COLUMNS (STRICT SCHEMA)
#############################################

def select_columns(df):
    
    cols = [
        'date',
        'home_team',
        'away_team',
        'home_goals',
        'away_goals',
        'is_neutral',
        'is_intl'
    ]
    
    return df[cols].copy()


#############################################
# 5. ELO (PRE-MATCH, NEUTRAL-AWARE)
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
        
        # Remove home advantage if neutral
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
# 6. MASTER PIPELINE
#############################################

def build_dataset(df, dataset_type="club"):
    
    df = load_and_standardize(df, dataset_type)
    
    if dataset_type == "intl":
        df = remove_friendlies(df)
    
    df = add_standard_flags(df, dataset_type)
    
    df = select_columns(df)
    
    df = compute_elo(df)
    
    return df


#############################################
# 7. TRAIN / TEST SPLIT (TIME-BASED)
#############################################

def train_test_split(df, train_end_date, test_start_date, test_end_date):
    """
    Time-based split to avoid leakage.
    """
    
    train = df[df['date'] <= train_end_date].copy()
    test = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()
    
    return train, test

#old_club_df = pd.read_csv("/kaggle/input/datasets/adamgbor/club-football-match-data-2000-2025/Matches.csv")  # old dataset with club matches that i used for the model
intl_game_stats = pd.read_csv("/kaggle/input/datasets/lchikry/international-football-match-features-and-statistics/teams_match_features.csv") # old dataset with international matches
club_basic = pd.read_csv("/kaggle/input/datasets/codytipton/player-stats-per-game-understat/clubs.csv") # basic club and league info from understat
club_game_stats = pd.read_csv("/kaggle/input/datasets/codytipton/player-stats-per-game-understat/general_game_stats.csv") # club match data with xG from 2014 - 2026
soccerway = pd.read_csv("/kaggle/input/datasets/omarameen99/football-matches-data-from-soccerway/scraped_dataset.csv") # all match stats from 2024 - current

# Build datasets
club_clean = build_dataset(club_df, "club")
intl_clean = build_dataset(intl_df, "intl")

# Splits
# Club
train_club, test_club = train_test_split(
     club_clean,
     train_end_date="2021-12-31",
     test_start_date="2022-01-01",
     test_end_date="2022-12-31"
)

# International
train_intl, test_intl = train_test_split(
     intl_clean,
     train_end_date="2020-12-31",
     test_start_date="2022-01-01",
     test_end_date="2022-12-31"
)
train_club.to_csv("train_club_df.csv", index=False)
test_club.to_csv("test_club_df.csv", index=False)
train_intl.to_csv("train_intl_df.csv", index=False)
test_intl.to_csv("test_intl_df.csv", index=False)
print(train_club.describe())
print(test_club.describe())
print(train_intl.describe())
print(test_intl.describe())