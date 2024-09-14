import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
import numpy as np
import warnings

# Constants
NUM_ESTIMATORS = 400
MIN_SAMPLES_SPLIT = 30
CUTOFF_DATE = '2023-11-01'

def cross_validate_model(data: pd.DataFrame, predictors: list[str], target_column: str = "target"):
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, min_samples_split=MIN_SAMPLES_SPLIT, random_state=1)
    
    X = data[predictors]
    y = data[target_column]
    
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        scores.append(accuracy)
    
    print(f"Cross-validated accuracy: {np.mean(scores):.2f}")

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        
        if not team_matches.empty:
            # Calculate rolling averages for main features
            team_rolling = team_matches[feature_columns].rolling(rolling_window, closed='left').mean()
            
            # Calculate rolling averages for win percentages
            team_rolling["home_win_percentage_rolling"] = team_matches["home_win_percentage"].rolling(rolling_window, closed='left').mean()
            team_rolling["away_win_percentage_rolling"] = team_matches["away_win_percentage"].rolling(rolling_window, closed='left').mean()
            
            team_rolling["Team"] = team
            team_rolling["Date"] = team_matches["Date"].values
            rolling_features.append(team_rolling)
    
    if rolling_features:
        rolling_data = pd.concat(rolling_features, ignore_index=True)
        rolling_data = rolling_data.dropna(subset=feature_columns + ["home_win_percentage_rolling", "away_win_percentage_rolling"])
        matches = matches.merge(rolling_data, on=["Team", "Date"], suffixes=("", "_rolling"))
    else:
        for column in feature_columns:
            matches[f"{column}_rolling"] = 0
        matches["home_win_percentage_rolling"] = 0
        matches["away_win_percentage_rolling"] = 0
    
    return matches

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

def calculate_goal_differential(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the goal differential for each match."""
    matches["goal_differential"] = matches["GF"] - matches["GA"]
    return matches

def calculate_shot_conversion_rate(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the shot conversion rate for each team."""
    matches["shot_conversion_rate"] = matches["GF"] / matches["Sh"]
    matches["shot_conversion_rate"] = matches["shot_conversion_rate"].replace([float('inf'), -float('inf'), np.nan], 0)
    return matches

def calculate_shooting_accuracy(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the shooting accuracy for each team."""
    matches["shooting_accuracy"] = matches["SoT"] / matches["Sh"]
    matches["shooting_accuracy"] = matches["shooting_accuracy"].replace([float('inf'), -float('inf'), np.nan], 0)
    return matches

def calculate_possession_efficiency(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the possession efficiency for each team."""
    matches["possession_efficiency"] = matches["GF"] / matches["Poss"]
    matches["possession_efficiency"] = matches["possession_efficiency"].replace([float('inf'), -float('inf'), np.nan], 0)
    return matches

def calculate_defense_efficiency(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the defense efficiency for each team."""
    # Calculate the average shots on target by each opponent
    matches["Opponent_SoT"] = matches.groupby("Opponent")["SoT"].transform("mean")
    
    # Calculate defense efficiency
    matches["defense_efficiency"] = matches["GA"] / matches["Opponent_SoT"]
    matches["defense_efficiency"] = matches["defense_efficiency"].replace([float('inf'), -float('inf'), np.nan], 0)
    
    return matches

def check_data_leakage(matches: pd.DataFrame, predictors: list[str], target_column: str = "target"):
    print("Predictor columns:", predictors)
    print("Target column:", target_column)
    print("Columns in dataset:", matches.columns)
    
    # Check if target column is in predictors
    if target_column in predictors:
        raise ValueError("Target column is included in predictors, which indicates data leakage.")

def verify_date_split(data: pd.DataFrame, cutoff_date: str):
    train = data[data["Date"] < cutoff_date]
    test = data[data["Date"] >= cutoff_date]
    
    print(f"Training data period: {train['Date'].min()} to {train['Date'].max()}")
    print(f"Test data period: {test['Date'].min()} to {test['Date'].max()}")
    
    if train['Date'].max() >= test['Date'].min():
        raise ValueError("Date-based split is incorrect; there is overlap between training and test sets.")

def train_random_forest_model(data: pd.DataFrame, predictors: list[str], target_column: str = "target") -> pd.DataFrame:
    """Train a random forest model."""
    train = data[data["Date"] < CUTOFF_DATE]
    test = data[data["Date"] >= CUTOFF_DATE]
    
    if train.empty or test.empty:
        raise ValueError("No data available for training or testing.")
    
    rf = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, min_samples_split=MIN_SAMPLES_SPLIT, random_state=1)
    rf.fit(train[predictors], train[target_column])
    
    # Make predictions
    train_preds = rf.predict(train[predictors])
    test_preds = rf.predict(test[predictors])
    
    train_accuracy = accuracy_score(train[target_column], train_preds)
    test_accuracy = accuracy_score(test[target_column], test_preds)
    
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    combined = pd.DataFrame({"actual": test[target_column], "predicted": test_preds}, index=test.index)
    return combined

def main():
    file_path = "matches.csv"
    matches = load_data(file_path)
    matches = preprocess_data(matches)
    feature_columns = ["xG", "xGA", "Poss", "Sh", "SoT", "FK", "PK", "PKatt"]
    matches = create_rolling_features(matches, feature_columns)
    matches = calculate_goal_differential(matches)
    matches = calculate_shot_conversion_rate(matches)
    matches = calculate_shooting_accuracy(matches)
    matches = calculate_possession_efficiency(matches)
    matches = calculate_defense_efficiency(matches)
    
    rolling_feature_columns = [f"{c}_rolling" for c in feature_columns]
    predictors = feature_columns + rolling_feature_columns + ["strength", "home_win_percentage", "away_win_percentage", "goal_differential", "shot_conversion_rate", "shooting_accuracy", "possession_efficiency", "defense_efficiency"]
    
    # Perform checks
    check_data_leakage(matches, predictors)
    verify_date_split(matches, CUTOFF_DATE)
    
    # Train the model and evaluate accuracy
    combined = train_random_forest_model(matches, predictors)
    accuracy = accuracy_score(combined["actual"], combined["predicted"])
    print(f"Accuracy: {accuracy:.2f}")
    
    # Cross-validation
    cross_validate_model(matches, predictors)

if __name__ == "__main__":
    main()
