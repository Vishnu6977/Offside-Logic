import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Constants
NUM_ESTIMATORS = 400
MIN_SAMPLES_SPLIT = 30
CUTOFF_DATE = '2024-03-18'

def load_data(file_path: str) -> pd.DataFrame:
    """Load the matches.csv file."""
    return pd.read_csv(file_path)

def preprocess_data(matches: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    matches["Date"] = pd.to_datetime(matches["Date"], errors='coerce')
    matches.dropna(subset=["Date"], inplace=True)
    matches["Venue"] = matches["Venue"].astype("category").cat.codes
    matches["Opponent"] = matches["Opponent"].astype("category")
    matches["Opponent_code"] = matches["Opponent"].cat.codes
    matches["target"] = matches["Result"].apply(lambda x: 2 if x == 'W' else (1 if x == 'D' else 0)).astype(int)
    
    team_strength = calculate_team_strength(matches)
    home_win_percentage = calculate_home_win_percentage(matches)
    away_win_percentage = calculate_away_win_percentage(matches)
    
    matches = matches.merge(team_strength, on="Team", suffixes=("", "_strength"))
    matches = matches.merge(home_win_percentage, on="Team", suffixes=("", "_home_win_percentage"))
    matches = matches.merge(away_win_percentage, on="Team", suffixes=("", "_away_win_percentage"))
    
    return matches

def create_rolling_features(matches: pd.DataFrame, feature_columns: list[str], rolling_window: int = 5) -> pd.DataFrame:
    """Create rolling features for each team."""
    rolling_features = []
    
    for team in matches["Team"].unique():
        team_matches = matches[matches["Team"] == team].sort_values("Date")
        
        # Check if the team has any matches
        if not team_matches.empty:
            # Calculate rolling averages for main features
            team_rolling = team_matches[feature_columns].rolling(rolling_window, closed='left').mean()
            
            # Calculate rolling averages for win percentages
            team_rolling["home_win_percentage_rolling"] = team_matches["home_win_percentage"].rolling(rolling_window, closed='left').mean()
            team_rolling["away_win_percentage_rolling"] = team_matches["away_win_percentage"].rolling(rolling_window, closed='left').mean()
            
            team_rolling["Team"] = team
            team_rolling["Date"] = team_matches["Date"].values
            rolling_features.append(team_rolling)
    
    # Check if there are any rolling features to concatenate
    if rolling_features:
        rolling_data = pd.concat(rolling_features, ignore_index=True)
        rolling_data = rolling_data.dropna(subset=feature_columns + ["home_win_percentage_rolling", "away_win_percentage_rolling"])
        matches = matches.merge(rolling_data, on=["Team", "Date"], suffixes=("", "_rolling"))
    else:
        # If there are no rolling features, create dummy columns
        for column in feature_columns:
            matches[f"{column}_rolling"] = 0
        matches["home_win_percentage_rolling"] = 0
        matches["away_win_percentage_rolling"] = 0
    
    return matches

def train_random_forest_model(data: pd.DataFrame, predictors: list[str], target_column: str = "target") -> pd.DataFrame:
    """Train a random forest model."""
    train = data[data["Date"] < CUTOFF_DATE]
    test = data[data["Date"] >= CUTOFF_DATE]
    rf = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, min_samples_split=MIN_SAMPLES_SPLIT, random_state=1)
    rf.fit(train[predictors], train[target_column])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame({"actual": test[target_column], "predicted": preds}, index=test.index)
    return combined

def evaluate_model(combined: pd.DataFrame) -> float:
    """Evaluate the model using accuracy score."""
    return accuracy_score(combined["actual"], combined["predicted"])

def calculate_team_strength(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the overall win percentage for each team."""
    team_strength = matches.groupby("Team")["target"].apply(lambda x: (x == 2).sum() / len(x)).reset_index(name="strength")
    return team_strength

def calculate_home_win_percentage(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the home win percentage for each team."""
    home_matches = matches[matches["Venue"] == 0]  # Assuming '0' is 'Home' after encoding
    home_win_percentage = home_matches.groupby("Team")["target"].apply(lambda x: (x == 2).sum() / len(x)).reset_index(name="home_win_percentage")
    return home_win_percentage

def calculate_away_win_percentage(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the away win percentage for each team."""
    away_matches = matches[matches["Venue"] == 1]  # Assuming '1' is 'Away' after encoding
    away_win_percentage = away_matches.groupby("Team")["target"].apply(lambda x: (x == 2).sum() / len(x)).reset_index(name="away_win_percentage")
    return away_win_percentage

def main():
    file_path = "matches.csv"
    matches = load_data(file_path)
    matches = preprocess_data(matches)
    feature_columns = ["xG", "xGA", "Poss", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
    matches = create_rolling_features(matches, feature_columns)
    rolling_feature_columns = [f"{c}_rolling" for c in feature_columns]
    predictors = feature_columns + rolling_feature_columns + ["strength", "home_win_percentage", "away_win_percentage", "home_win_percentage_rolling", "away_win_percentage_rolling"]
    combined = train_random_forest_model(matches, predictors)
    accuracy = evaluate_model(combined)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
